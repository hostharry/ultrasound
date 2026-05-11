"""Differentiable single-angle DAS beamformer for training-time image-domain loss.

Mirrors the evaluation NumPy DAS (das.py, das_pw_rf) but in PyTorch,
so gradients flow back through the RF input to the reconstruction network.

First version limitations:
  - Single plane-wave angle per call (no compound)
  - Fixed c, fixed scan grid, fixed probe geometry
  - Per-batch-element processing (angles may differ across batch)
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# --------------- helpers ---------------

def _tukey25_torch(distance, aperture, alpha=0.25):
    """Tukey-25 % spatial receive apodization (matches das.py _tukey25)."""
    r = 2.0 * distance / (aperture + 1e-10)
    out = torch.zeros_like(r)
    flat = r <= (1.0 - alpha)
    out[flat] = 1.0
    taper = (r > (1.0 - alpha)) & (r <= 1.0)
    out[taper] = 0.5 * (1.0 + torch.cos(
        (math.pi / alpha) * (r[taper] - 1.0 + alpha)
    ))
    return out


def _hilbert_filter_axial(N, device, dtype):
    """Frequency-domain Hilbert filter for axial FFT."""
    h = torch.zeros(N, device=device, dtype=dtype)
    h[0] = 1.0
    h[1:(N + 1) // 2] = 2.0
    if N % 2 == 0:
        h[N // 2] = 1.0
    return h


# --------------- core module ---------------

class DASForwardSingleAngle(nn.Module):
    """Differentiable single-angle plane-wave DAS beamformer.

    All geometry is pre-computed and stored as non-trainable buffers.
    Forward: (B, n_elem, n_samp) + (B,) angles -> (B, n_z, n_x) envelope.
    Gradients flow through RF via differentiable linear interpolation + Hilbert.

    Parameters
    ----------
    probe_x, probe_z : array-like (n_elem,)
        Element x / z positions in metres.
    x_axis : array-like (n_x,)
        Lateral pixel grid in metres.
    z_axis : array-like (n_z,)
        Axial pixel grid in metres.
    initial_time : float
        Time of first RF sample (s).
    fs : float
        Sampling frequency (Hz).
    c : float
        Speed of sound (m/s).
    rx_f_number : float
        Receive f-number for dynamic aperture (default 1.75).
    chunk_size : int
        Pixels per processing chunk (controls peak GPU memory).
    use_checkpoint : bool
        If True, use gradient checkpointing per chunk to save memory
        at the cost of ~2x forward compute.
    """

    def __init__(self, probe_x, probe_z, x_axis, z_axis,
                 initial_time, fs, c,
                 rx_f_number=1.75, chunk_size=10000,
                 use_checkpoint=False):
        super().__init__()

        x = torch.as_tensor(x_axis, dtype=torch.float64)
        z = torch.as_tensor(z_axis, dtype=torch.float64)

        # numpy: X, Z = np.meshgrid(x_axis, z_axis) -> shape (n_z, n_x)
        X, Z = torch.meshgrid(x, z, indexing='xy')  # X(n_z, n_x), Z(n_z, n_x)

        xf = X.reshape(-1).contiguous()
        zf = Z.reshape(-1).contiguous()
        self.register_buffer('xf', xf)
        self.register_buffer('zf', zf)

        px = torch.as_tensor(probe_x, dtype=torch.float64)
        pz = torch.as_tensor(probe_z, dtype=torch.float64)
        self.register_buffer('px', px)
        self.register_buffer('pz', pz)

        rx_delay = torch.sqrt(
            (px[None, :] - xf[:, None]) ** 2
            + (pz[None, :] - zf[:, None]) ** 2
        )
        self.register_buffer('rx_delay', rx_delay)          # (n_pix, n_elem)

        lat_dist = torch.abs(xf[:, None] - px[None, :])
        aperture = zf / rx_f_number
        apod = _tukey25_torch(lat_dist, aperture[:, None])
        self.register_buffer('apod', apod)                   # (n_pix, n_elem)

        self.t0 = initial_time
        self.fs = fs
        self.c = c
        self.n_z = len(z_axis)
        self.n_x = len(x_axis)
        self.n_pix = self.n_z * self.n_x
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint

    # ---- public API ----

    def forward(self, rf, angles):
        """
        Parameters
        ----------
        rf     : (B, n_elem, n_samp) float – single-angle RF data
        angles : (B,) float – steering angle per sample (rad)

        Returns
        -------
        envelope : (B, n_z, n_x) float – DAS envelope image
        """
        B = rf.shape[0]
        all_bf = []
        for b in range(B):
            all_bf.append(self._beamform_one(rf[b], float(angles[b])))
        bf = torch.stack(all_bf, dim=0)                       # (B, n_z, n_x)
        return self._hilbert_envelope_axial(bf)

    # ---- internals ----

    def _beamform_one(self, rf_b, angle):
        """Beamform one batch element.

        rf_b  : (n_elem, n_samp) – requires_grad
        angle : float – steering angle (rad)
        Returns: (n_z, n_x) beamformed RF (before envelope)
        """
        n_samp = rf_b.shape[1]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        tx = self.zf * cos_a + self.xf * sin_a               # (n_pix,) float64

        chunks = []
        for cs in range(0, self.n_pix, self.chunk_size):
            ce = min(cs + self.chunk_size, self.n_pix)
            tx_c   = tx[cs:ce]
            rx_c   = self.rx_delay[cs:ce]
            apod_c = self.apod[cs:ce]

            if self.use_checkpoint and rf_b.requires_grad:
                chunk = grad_checkpoint(
                    self._beamform_chunk,
                    rf_b, tx_c, rx_c, apod_c,
                    n_samp, self.c, self.t0, self.fs,
                    use_reentrant=False,
                )
            else:
                chunk = self._beamform_chunk(
                    rf_b, tx_c, rx_c, apod_c,
                    n_samp, self.c, self.t0, self.fs,
                )
            chunks.append(chunk)

        return torch.cat(chunks, dim=0).reshape(self.n_z, self.n_x)

    @staticmethod
    def _beamform_chunk(rf_b, tx_c, rx_c, apod_c, n_samp, c, t0, fs):
        """Beamform one pixel chunk for one sample.

        rf_b   : (n_elem, n_samp) float – the only tensor carrying gradient
        tx_c   : (nc,) float64 – TX delay
        rx_c   : (nc, n_elem) float64 – RX delay
        apod_c : (nc, n_elem) float64 – apodization weights
        Returns: (nc,) float – beamformed values
        """
        sf = ((tx_c[:, None] + rx_c) / c - t0) * fs          # (nc, n_elem) f64

        i0 = sf.floor().long()
        i1 = i0 + 1
        frac = (sf - i0.double()).float()                     # (nc, n_elem) f32

        ok = ((i0 >= 0) & (i1 < n_samp)).float()             # (nc, n_elem)
        i0c = i0.clamp(0, n_samp - 1)
        i1c = i1.clamp(0, n_samp - 1)

        # gather along sample dim: rf_b (n_elem, n_samp), index (n_elem, nc)
        v0 = torch.gather(rf_b, 1, i0c.T)                    # (n_elem, nc)
        v1 = torch.gather(rf_b, 1, i1c.T)
        v = v0 * (1.0 - frac.T) + v1 * frac.T               # (n_elem, nc)
        v = v * ok.T

        return (apod_c.T.float() * v).sum(dim=0)             # (nc,)

    @staticmethod
    def _hilbert_envelope_axial(bf):
        """Analytic-signal envelope along axial dimension.

        bf : (B, n_z, n_x) float -> (B, n_z, n_x) envelope
        """
        N = bf.shape[1]
        X_f = torch.fft.fft(bf, dim=1)
        h = _hilbert_filter_axial(N, bf.device, bf.dtype)
        analytic = torch.fft.ifft(X_f * h[None, :, None], dim=1)
        return torch.sqrt(analytic.real ** 2 + analytic.imag ** 2 + 1e-8)


class PostDASEnvelope(nn.Module):
    """Post-DAS 包络提取: 对已波束成形的 RF 做 Hilbert 包络.

    用于 post-DAS 数据 (compress_mode="post_das") 的图像域损失计算,
    替代完整的 DASForwardSingleAngle.

    输入 beamformed RF 存储为 (B, n_x, n_z), 其中:
      - dim=1 (n_x): lateral scanline
      - dim=2 (n_z): axial samples
    Hilbert 变换沿 dim=-1 (axial) 执行.
    """

    def forward(self, bf_rf, angles=None):
        """
        Parameters
        ----------
        bf_rf  : (B, n_x, n_z) float – beamformed RF (squeezed from model output)
        angles : ignored, kept for API compatibility with DASForwardSingleAngle

        Returns
        -------
        envelope : (B, n_x, n_z) float
        """
        N = bf_rf.shape[-1]
        X_f = torch.fft.fft(bf_rf, dim=-1)
        h = _hilbert_filter_axial(N, bf_rf.device, bf_rf.dtype)
        analytic = torch.fft.ifft(X_f * h[None, None, :], dim=-1)
        return torch.sqrt(analytic.real ** 2 + analytic.imag ** 2 + 1e-8)
