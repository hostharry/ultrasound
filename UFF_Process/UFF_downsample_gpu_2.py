"""UFF -> (FDBF subsampled time-domain input, DAS target) dataset builder.

This is a cleaned + fixed version of the previously shared script.

Key points aligned with the paper:
  - Build the *subsampled beamformed* signal in the time domain by:
      (i) computing subsampled Fourier coefficients of the beamformed signal (via FDBF)
      (ii) zero-padding / restoring the negative spectrum to keep the original length
      (iii) inverse Fourier transform -> aliased time-domain input

The heavy part is precomputing Q_{k,m}[n] (geometry-only) and it should be done once.
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np
import torch


# -----------------------------
# 1) geometry -> delta_m (meters)
# -----------------------------

def infer_delta_m_from_geometry(geom: np.ndarray) -> np.ndarray:
    """Infer element x-positions from geometry and return delta wrt center element.

    Your UFF geometry appears as (7, C). We pick the row with the largest span as x.
    If values look like mm, we convert to meters.
    """
    geom = np.asarray(geom)
    spans = geom.max(axis=1) - geom.min(axis=1)
    row = int(np.argmax(spans))
    x = geom[row].astype(np.float64)
    if np.nanmax(np.abs(x)) > 1.0:  # looks like mm
        x *= 1e-3
    m0 = len(x) // 2
    return (x - x[m0]).astype(np.float64)


# -----------------------------
# 2) pick mu (rfft bins) by maximum energy window inside [fmin,fmax]
# -----------------------------

@torch.no_grad()
def estimate_avg_rfft_mag_from_uff(dset, fs_hz: float, num_lines: int = 32,
                                  device: str = "cuda", seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Average |rfft| over a few random lines (and over channels) to get a stable spectrum."""
    L, M, N = dset.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(L, size=min(num_lines, L), replace=False)

    acc = None
    for i in idx:
        rf = dset[i, ...].astype(np.float32)        # (M,N) on CPU
        rf_t = torch.from_numpy(rf).to(device)      # (M,N) on GPU
        R = torch.fft.rfft(rf_t, dim=-1)            # (M,N//2+1)
        m = R.abs().mean(dim=0)                     # (N//2+1)
        acc = m if acc is None else (acc + m)

    mag = (acc / float(len(idx))).detach().cpu().numpy()
    freqs = np.fft.rfftfreq(N, d=1.0 / fs_hz)
    return freqs, mag


def pick_best_contiguous_mu(freqs_hz: np.ndarray, mag: np.ndarray, K: int,
                            fmin_hz: float, fmax_hz: float, avoid_dc: bool = True) -> np.ndarray:
    """Pick a contiguous window of length K within [fmin,fmax] that maximizes energy."""
    Nf = len(freqs_hz)
    K = int(K)
    if K <= 1 or K > Nf:
        raise ValueError(f"K={K} invalid for rfft bins {Nf}")

    s_min = int(np.searchsorted(freqs_hz, fmin_hz, side="left"))
    s_max = int(np.searchsorted(freqs_hz, fmax_hz, side="right")) - K
    s_min = max(s_min, 0)
    s_max = min(s_max, Nf - K)
    if avoid_dc:
        s_min = max(s_min, 1)
    if s_min > s_max:
        raise ValueError("No valid window in the given band; widen [fmin,fmax].")

    score = np.convolve(mag.astype(np.float64), np.ones(K, dtype=np.float64), mode="valid")
    s = int(np.argmax(score[s_min:s_max + 1]) + s_min)
    mu = np.arange(s, s + K, dtype=np.int64)
    if avoid_dc:
        mu = mu[mu != 0]
    return mu


def K_like_paper(N: int, rate: int) -> int:
    """Match paper's 1920->230 (8x) and 1920->130 (15x). For others, use N/r."""
    if rate == 8:
        return int(round(N / 8.0))
    if rate == 15:
        return int(round(N / 15.0))
    return int(round(N / float(rate)))


# -----------------------------
# 3) tau_m(t,theta) (paper eq (1))
# -----------------------------

def compute_tau(delta_m: torch.Tensor, t: torch.Tensor, theta: float, c: float) -> torch.Tensor:
    """Return tau(m,t) in seconds."""
    dm = delta_m[:, None]
    tt = t[None, :]
    sin_th = torch.sin(torch.tensor(theta, device=t.device, dtype=t.dtype))
    inside = tt**2 - 4.0 * (dm / c) * tt * sin_th + 4.0 * (dm / c) ** 2
    inside = torch.clamp(inside, min=0.0)
    return 0.5 * (tt + torch.sqrt(inside))


# -----------------------------
# 4) Q precompute (GPU) — vectorized (no explicit loop over n)
# -----------------------------

@torch.no_grad()
def precompute_Q_fast(delta_m_m: np.ndarray, fs: float, N: int, mu_union: np.ndarray,
                      N1: int = 10, N2: int = 10, theta: float = 0.0, c: float = 1540.0,
                      chunk_k: int = 32, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Q_{k,m}[n] numerically on GPU.

    Q shape: (M, K, Nn) complex64, where:
      - K = len(mu_union)
      - Nn = N1+N2+1, n in [-N1..N2]

    This uses the decomposition:
      exp(j2π(k-n)τ/T) exp(-j2π k t/T)
        = [exp(j2π k τ/T) exp(-j2π k t/T)] * exp(-j2π n τ/T)

    Then Q for each m is a batch matmul over time samples.
    """
    delta_m = torch.tensor(delta_m_m, device=device, dtype=torch.float32)
    t = torch.arange(N, device=device, dtype=torch.float32) / fs
    T = float(N / fs)
    dt_over_T = float((1.0 / fs) / T)

    tau = compute_tau(delta_m, t, theta=theta, c=c)  # (M,N)

    n_list = torch.arange(-N1, N2 + 1, device=device, dtype=torch.float32)  # (Nn,)
    mu_k = torch.tensor(mu_union.astype(np.int64), device=device, dtype=torch.float32)  # (K,)

    M = int(delta_m.numel())
    K = int(mu_k.numel())
    Nn = int(n_list.numel())

    # B[m, t, n] = exp(-j 2π n τ_m(t)/T)
    two_pi_over_T = float(2.0 * np.pi / T)
    B = torch.exp(-1j * (two_pi_over_T * tau[:, :, None] * n_list[None, None, :])).to(torch.complex64)  # (M,N,Nn)

    Q = torch.empty((M, K, Nn), device=device, dtype=torch.complex64)

    for s in range(0, K, chunk_k):
        k_chunk = mu_k[s:s + chunk_k]  # (Kc,)
        # e^{-j2π k t/T}: (Kc,N)
        e_kt = torch.exp(-1j * (two_pi_over_T * k_chunk[:, None] * t[None, :])).to(torch.complex64)
        # exp(+j2π k τ/T): (M,Kc,N)
        e_k_tau = torch.exp(1j * (two_pi_over_T * tau[:, None, :] * k_chunk[None, :, None])).to(torch.complex64)
        A = e_k_tau * e_kt[None, :, :]  # (M,Kc,N)
        # batch matmul over time: (M,Kc,N) @ (M,N,Nn) -> (M,Kc,Nn)
        Q[:, s:s + k_chunk.numel(), :] = torch.bmm(A, B) * dt_over_T

    return Q, (n_list.to(torch.int64))


# -----------------------------
# 5) FDBF coefficients on selected k (paper eq (7))
# -----------------------------

@torch.no_grad()
def fdbf_coeffs_from_Q(rf_ch: torch.Tensor, Q: torch.Tensor,
                       mu_union: np.ndarray, n_list: torch.Tensor) -> torch.Tensor:
    """Return c[k] for k in mu_union.

    rf_ch: (M,N) float32
    Q: (M,K,Nn) complex64
    mu_union: (K,) rfft bins
    n_list: (Nn,) int64 values in [-N1..N2]
    """
    M, N = rf_ch.shape
    # cm[k] = FFT(x)[k] / N  (Fourier series coeffs)
    Cm = torch.fft.fft(rf_ch.to(torch.float32), dim=-1).to(torch.complex64) / float(N)  # (M,N)

    mu_k = torch.tensor(mu_union.astype(np.int64), device=rf_ch.device, dtype=torch.int64)
    kk = (mu_k[None, :, None] - n_list[None, None, :]) % N  # (1,K,Nn)
    kk = kk.expand(M, -1, -1)                                # (M,K,Nn)

    Cm_exp = Cm[:, None, :].expand(-1, mu_k.numel(), -1)     # (M,K,N)
    Cm_sel = torch.take_along_dim(Cm_exp, kk, dim=-1)        # (M,K,Nn)

    c_hat_mk = (Cm_sel * Q).sum(dim=-1)  # (M,K)
    return c_hat_mk.mean(dim=0)          # (K,)


# -----------------------------
# 6) time-domain DAS using the SAME tau model (paper-style) for GT
# -----------------------------

@torch.no_grad()
def das_time_domain_paper(rf_ch: torch.Tensor, delta_m_m: np.ndarray,
                          fs: float, theta: float = 0.0, c: float = 1540.0) -> torch.Tensor:
    """y(t) = mean_m phi_m( tau_m(t,theta) ) with linear interpolation."""
    device = rf_ch.device
    M, N = rf_ch.shape
    delta_m = torch.tensor(delta_m_m, device=device, dtype=torch.float32)
    t = torch.arange(N, device=device, dtype=torch.float32) / fs
    tau = compute_tau(delta_m, t, theta=theta, c=c)  # (M,N)

    idx = tau * fs
    i0 = torch.floor(idx).to(torch.int64)
    a = (idx - i0.to(idx.dtype)).to(torch.float32)

    i0c = torch.clamp(i0, 0, N - 1)
    i1c = torch.clamp(i0 + 1, 0, N - 1)

    rf0 = torch.gather(rf_ch, 1, i0c)
    rf1 = torch.gather(rf_ch, 1, i1c)
    aligned = (1.0 - a) * rf0 + a * rf1
    return aligned.mean(dim=0).to(torch.float32)


# -----------------------------
# 7) build dataset
# -----------------------------

@torch.no_grad()
def build_dataset_fdbf_gpu(uff_path: str, out_npz: str,
                           reductions: Iterable[int] = (8, 9, 15),
                           N1: int = 10, N2: int = 10,
                           theta: float = 0.0, c: float = 1540.0,
                           device: str | None = None,
                           q_cache: str | None = None,
                           spec_num_lines: int = 32) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    reductions = tuple(int(r) for r in reductions)

    with h5py.File(uff_path, "r") as f:
        dset = f["channel_data/data"]
        geom = f["channel_data/probe/geometry"][...]
        fs = float(np.squeeze(f["channel_data/sampling_frequency"][...]))
        fc = float(np.squeeze(f["channel_data/pulse/center_frequency"][...]))

        L, M, N = dset.shape
        print(f"UFF: L={L}, M={M}, N={N}, fs={fs/1e6:.2f} MHz, fc={fc/1e6:.2f} MHz")

        delta_m = infer_delta_m_from_geometry(geom)

        # ---- pick mu for each reduction based on averaged spectrum ----
        freqs, mag = estimate_avg_rfft_mag_from_uff(dset, fs_hz=fs, num_lines=spec_num_lines, device=device)
        fmin, fmax = 0.5 * fc, 1.5 * fc

        mu_dict: Dict[int, np.ndarray] = {}
        for r in reductions:
            K = K_like_paper(N, r)
            mu_dict[r] = pick_best_contiguous_mu(freqs, mag, K, fmin_hz=fmin, fmax_hz=fmax, avoid_dc=True)

        mu_union = np.unique(np.concatenate([mu_dict[r] for r in reductions])).astype(np.int64)
        mu_union.sort()
        print("|mu_union| =", len(mu_union), "  per-rate:", {r: len(mu_dict[r]) for r in reductions})

        # ---- Q cache (optional) ----
        Q = None
        n_list = None
        if q_cache is not None and os.path.exists(q_cache):
            ckpt = torch.load(q_cache, map_location=device)
            if (np.array_equal(ckpt["mu_union"], mu_union)
                    and int(ckpt["N"]) == int(N)
                    and int(ckpt["N1"]) == int(N1)
                    and int(ckpt["N2"]) == int(N2)
                    and float(ckpt["theta"]) == float(theta)
                    and float(ckpt["fs"]) == float(fs)):
                Q = ckpt["Q"].to(device)
                n_list = ckpt["n_list"].to(device)
                print("Loaded Q from cache:", q_cache)

        if Q is None:
            print("Precomputing Q (this is the heavy step)...")
            Q, n_list = precompute_Q_fast(delta_m, fs, N, mu_union, N1=N1, N2=N2, theta=theta, c=c, device=device)
            if q_cache is not None:
                torch.save({
                    "Q": Q.detach().cpu(),
                    "n_list": n_list.detach().cpu(),
                    "mu_union": mu_union,
                    "N": int(N),
                    "N1": int(N1),
                    "N2": int(N2),
                    "theta": float(theta),
                    "fs": float(fs),
                }, q_cache)
                print("Saved Q cache:", q_cache)

        # precompute index maps for each reduction
        idx_in_union = {
            r: torch.tensor(np.searchsorted(mu_union, mu_dict[r]).astype(np.int64), device=device, dtype=torch.long)
            for r in reductions
        }
        mu_torch = {r: torch.tensor(mu_dict[r].astype(np.int64), device=device, dtype=torch.long) for r in reductions}

        # allocate outputs (CPU)
        Y = np.zeros((L, N), np.float32)
        X = {r: np.zeros((L, N), np.float32) for r in reductions}
        Yk = {}
        mask = {}

        # main loop
        for i in range(L):
            rf = dset[i, ...].astype(np.float32)          # (M,N)
            rf_t = torch.from_numpy(rf).to(device)

            y = das_time_domain_paper(rf_t, delta_m, fs, theta=theta, c=c)
            y_np = y.detach().cpu().numpy()
            Y[i] = y_np
            Yf = np.fft.rfft(y_np).astype(np.complex64)

            if i == 0:
                for r in reductions:
                    mu_r = mu_dict[r]
                    Yk[r] = np.zeros((L, len(mu_r)), np.complex64)
                    mask_r = np.zeros((N // 2 + 1,), np.uint8)
                    mask_r[mu_r] = 1
                    mask[r] = mask_r
            for r in reductions:
                Yk[r][i] = Yf[mu_dict[r]]

            c_union = fdbf_coeffs_from_Q(rf_t, Q, mu_union, n_list)  # (Kunion,) complex

            for r in reductions:
                # pick the subset of coefficients for this reduction
                c_sub = torch.index_select(c_union, 0, idx_in_union[r])
                # build rfft spectrum then irfft -> time-domain input
                spec = torch.zeros((N // 2 + 1,), device=device, dtype=torch.complex64)
                spec.scatter_(0, mu_torch[r], c_sub.to(torch.complex64))
                x = torch.fft.irfft(spec, n=N) * float(N)
                X[r][i] = x.to(torch.float32).detach().cpu().numpy()

            if (i + 1) % 16 == 0:
                print(f"[{i+1}/{L}] done")

    # save
    save_dict = {
        "Y": Y,
        "fs": np.float32(fs),
        "fc": np.float32(fc),
        "c": np.float32(c),
        "theta": np.float32(theta),
        "N1": np.int32(N1),
        "N2": np.int32(N2),
    }
    for r in reductions:
        save_dict[f"X{r}"] = X[r]
        save_dict[f"mu{r}"] = mu_dict[r].astype(np.int32)
        if r in Yk:
            save_dict[f"Y{r}_k"] = Yk[r]
        if r in mask:
            save_dict[f"mask{r}"] = mask[r]
    np.savez_compressed(out_npz, **save_dict)
    print("saved:", out_npz)
    for r in reductions:
        if r in Yk:
            print(f"Y{r}_k shape:", Yk[r].shape, "dtype:", Yk[r].dtype)
        if r in mask:
            print(f"mask{r} shape:", mask[r].shape, "dtype:", mask[r].dtype)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--uff",
        default="Alpinion_L3-8_FI_hyperechoic_scatterers.uff",
        help="UFF input path",
    )
    p.add_argument(
        "--out",
        default="dataset_fdbf_energy_mu_8_9_15.npz",
        help="Output npz path",
    )
    p.add_argument("--reductions", nargs="+", type=int, default=[8, 9, 15])
    p.add_argument("--N1", type=int, default=10)
    p.add_argument("--N2", type=int, default=10)
    p.add_argument("--theta", type=float, default=0.0)
    p.add_argument("--c", type=float, default=1540.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--q_cache", type=str, default=None)
    p.add_argument("--spec_num_lines", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset_fdbf_gpu(
        uff_path=args.uff,
        out_npz=args.out,
        reductions=args.reductions,
        N1=args.N1,
        N2=args.N2,
        theta=args.theta,
        c=args.c,
        device=args.device,
        q_cache=args.q_cache,
        spec_num_lines=args.spec_num_lines,
    )
