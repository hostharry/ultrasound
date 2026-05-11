"""FISTA-DWT-Lite: 轻量化 Prox 版本 (1D).

核心改造 (相对 FISTA_Transformer_DWT.py):
  - HASA 权重网络: 保留 Transformer (全局条件建模)
  - TV prox: 全局 Transformer → Conv1d 残差块 (局部平滑/边缘保持)
  - DWT prox: 全局 Transformer → 子带内 Conv1d 残差块 + gate/shrink
  - DFFM: 不变
  - 复杂度: O(L^2) → O(L*k), 显存大幅下降
"""

import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "Utils")
_ADMM_DIR = os.path.join(os.path.dirname(__file__), "..", "Admm_net")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
if _ADMM_DIR not in sys.path:
    sys.path.append(_ADMM_DIR)

from ops import soft_threshold
from FISTA_Transformer import HASAWeightTransformer1D
from FISTA_Transformer_DFFM import DFFM1D
from FISTA_Transformer_DWT import HaarDWT1d, HaarIDWT1d


# ═══════════════════════ Conv 基础组件 ═══════════════════════


class ConvResBlock1D(nn.Module):
    """两层 Conv1d + 残差: (B, C, L) → (B, C, L)."""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
        )

    def forward(self, x):
        return x + self.block(x)


# ═══════════════════════ Conv TV Branch ═══════════════════════


class ConvProxTV1D(nn.Module):
    """Conv1d 残差块替代 Transformer 的 TV 近端分支.

    数据流: z → proj → N×ConvRes (enc) → soft_threshold → N×ConvRes (dec) → proj → x_tv
    """

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5):
        super().__init__()
        self.proj_in = nn.Conv1d(1, d_model, 1)
        self.enc = nn.Sequential(
            *[ConvResBlock1D(d_model, kernel_size) for _ in range(num_blocks)])
        self.dec = nn.Sequential(
            *[ConvResBlock1D(d_model, kernel_size) for _ in range(num_blocks)])
        self.proj_out = nn.Conv1d(d_model, 1, 1)

    def forward(self, z, thr_tv):
        """
        z:      (B, 1, L)
        thr_tv: (B, 1, L)
        Returns: x_tv (B, 1, L), feat_tv (B, L, d_model)
        """
        feat = self.proj_in(z)               # (B, d_model, L)
        feat = self.enc(feat)                # (B, d_model, L)
        feat_enc = feat

        thr_expanded = thr_tv.expand_as(feat)
        feat_shrink = soft_threshold(feat, thr_expanded)

        feat = self.dec(feat_shrink)         # (B, d_model, L)
        x_tv = self.proj_out(feat)           # (B, 1, L)

        return x_tv, feat_enc.permute(0, 2, 1)


# ═══════════════════════ Conv DWT Branch ═══════════════════════


class DWTConvBranch(nn.Module):
    """子带内 Conv1d 替代全局 Transformer 的 DWT 多尺度小波分支.

    数据流:
      z → pad → DWT → {cA, cD_J, ..., cD_1}
        → 每个子带: Conv1d proj → N×ConvRes (enc)
        → cA: 可学习门控 / cD: per-(subband×position) soft-threshold
        → 每个子带: N×ConvRes (dec) → Conv1d proj
        → IDWT → crop → x_wav
    """

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5, J=3):
        super().__init__()
        self.d_model = d_model
        self.J = J
        self.n_bands = J + 1
        self._divisor = 2 ** J

        self.dwt = HaarDWT1d(J)
        self.idwt = HaarIDWT1d(J)

        self.proj_in = nn.ModuleList(
            [nn.Conv1d(1, d_model, 1) for _ in range(self.n_bands)])
        self.enc = nn.ModuleList([
            nn.Sequential(*[ConvResBlock1D(d_model, kernel_size)
                            for _ in range(num_blocks)])
            for _ in range(self.n_bands)])
        self.dec = nn.ModuleList([
            nn.Sequential(*[ConvResBlock1D(d_model, kernel_size)
                            for _ in range(num_blocks)])
            for _ in range(self.n_bands)])
        self.proj_out = nn.ModuleList(
            [nn.Conv1d(d_model, 1, 1) for _ in range(self.n_bands)])

        self.approx_gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.Sigmoid(),
        )
        self.subband_scale = nn.Parameter(torch.ones(J))

    def _pad(self, x):
        L = x.shape[-1]
        rem = L % self._divisor
        if rem == 0:
            return x, L
        pad_len = self._divisor - rem
        return F.pad(x, (0, pad_len)), L

    @staticmethod
    def _downsample_thr(thr_wav, lengths, L_padded):
        out = []
        for s_len in lengths:
            ratio = L_padded // s_len
            if ratio == 1:
                out.append(thr_wav)
            else:
                out.append(F.avg_pool1d(thr_wav, kernel_size=ratio, stride=ratio))
        return out

    def _realign_to_spatial(self, feats, lengths, L_target):
        """各子带 encoder 特征上采样到 L_target 后均值融合."""
        aligned = []
        for f in feats:
            f_up = F.interpolate(
                f, size=L_target, mode='linear', align_corners=False)
            aligned.append(f_up)
        return torch.stack(aligned, dim=0).mean(dim=0)

    def forward(self, z, thr_wav):
        """
        z:       (B, 1, L)
        thr_wav: (B, 1, L)
        Returns: x_wav (B, 1, L), feat_spatial (B, L_padded, d_model)
        """
        B, _, L_orig = z.shape
        z_pad, _ = self._pad(z)
        thr_pad, _ = self._pad(thr_wav)
        L_padded = z_pad.shape[-1]

        cA, details = self.dwt(z_pad)
        subbands = [cA] + details
        lengths = [s.shape[-1] for s in subbands]
        thr_per_band = self._downsample_thr(thr_pad, lengths, L_padded)

        enc_feats = []
        for band, proj, enc_blk in zip(subbands, self.proj_in, self.enc):
            feat = proj(band)                # (B, d_model, band_len)
            feat = enc_blk(feat)
            enc_feats.append(feat)

        processed = []
        gated = enc_feats[0] * self.approx_gate(enc_feats[0])
        processed.append(gated)

        for j, (feat_j, thr_j) in enumerate(
                zip(enc_feats[1:], thr_per_band[1:])):
            scale_j = F.softplus(self.subband_scale[j])
            thr_expanded = (scale_j * thr_j).expand_as(feat_j)
            processed.append(soft_threshold(feat_j, thr_expanded))

        rec_bands = []
        for p, dec_blk, proj_out in zip(processed, self.dec, self.proj_out):
            decoded = dec_blk(p)
            rec_bands.append(proj_out(decoded))

        x_wav = self.idwt(rec_bands[0], rec_bands[1:])
        x_wav = x_wav[..., :L_orig]

        feat_spatial = self._realign_to_spatial(enc_feats, lengths, L_padded)
        return x_wav, feat_spatial.permute(0, 2, 1)


# ═══════════════════════ Lite Prox / Block / Net ═══════════════════════


class ISTAProxDWT1D_Lite(nn.Module):
    """双分支 ISTA 近端 (Lite): TV (Conv1d) + Wav (DWT + 子带内 Conv1d)."""

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5, J=3):
        super().__init__()
        self.tv_branch = ConvProxTV1D(
            d_model=d_model, num_blocks=num_blocks, kernel_size=kernel_size)
        self.wav_branch = DWTConvBranch(
            d_model=d_model, num_blocks=num_blocks, kernel_size=kernel_size, J=J)

    def forward(self, z, thr_tv, thr_wav, alpha=None):
        L = z.shape[-1]
        x_tv, feat_tv = self.tv_branch(z, thr_tv)
        x_wav, feat_wav_full = self.wav_branch(z, thr_wav)

        feat_wav = feat_wav_full[:, :L, :]

        if alpha is None:
            x_next = 0.5 * (x_tv + x_wav)
            feat_fused = 0.5 * (feat_tv + feat_wav)
        else:
            alpha_seq = alpha.permute(0, 2, 1).expand_as(feat_tv)
            x_next = alpha * x_tv + (1.0 - alpha) * x_wav
            feat_fused = alpha_seq * feat_tv + (1.0 - alpha_seq) * feat_wav

        return x_next, feat_fused


class FISTA_DWT_Lite_Block(nn.Module):
    """单层 FISTA-Lite block: 梯度下降 → HASA(Transformer) → Lite Prox → momentum."""

    def __init__(self, weight_ctor, d_model=32, nhead=4,
                 num_transformer_layers=1, num_conv_blocks=2,
                 conv_ks=5, J=3):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProxDWT1D_Lite(
            d_model=d_model, num_blocks=num_conv_blocks,
            kernel_size=conv_ks, J=J,
        )

    def forward(self, x_prev, v, y, op):
        r = op.A(v) - y
        g = op.At(r)
        z = v - F.softplus(self.rho) * g

        lambda_tv, lambda_wav, alpha = self.weight(z)
        thr_tv = F.softplus(self.soft_thr) * lambda_tv
        thr_wav = F.softplus(self.soft_thr) * lambda_wav

        x_next, feat_fused = self.prox(z, thr_tv, thr_wav, alpha=alpha)

        beta = torch.tanh(self.beta)
        v_next = x_next + beta * (x_next - x_prev)

        zeros = torch.zeros_like(x_next)
        aux = {
            "rho1": F.softplus(self.rho).detach(),
            "rho2": F.softplus(self.rho).detach(),
            "eta": F.softplus(self.soft_thr).detach(),
            "gamma": torch.tensor(0.0, device=x_next.device),
            "lambda_tv": lambda_tv.detach(),
            "lambda_wav": lambda_wav.detach(),
            "alpha": alpha.detach(),
            "sym_tv": None,
            "sym_wav": None,
            "constraint_wav": zeros.detach(),
            "constraint_tv": zeros.detach(),
        }
        return x_next, v_next, feat_fused, aux


class FISTA_DWT_Lite_Net(nn.Module):
    """1D FISTA-DWT-Lite: HASA(Transformer) + Lite Prox(Conv) + DFFM."""

    def __init__(self, layer_num=4, hasa_ctor=None,
                 d_model=32, nhead=4, num_transformer_layers=1,
                 num_conv_blocks=2, conv_ks=5, J=3):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.tensor(0.2))

        if hasa_ctor is None:
            hasa_ctor = lambda: HASAWeightTransformer1D(
                d_model=d_model, nhead=nhead, num_layers=num_transformer_layers,
            )

        self.blocks = nn.ModuleList([
            FISTA_DWT_Lite_Block(
                hasa_ctor, d_model=d_model, nhead=nhead,
                num_transformer_layers=num_transformer_layers,
                num_conv_blocks=num_conv_blocks, conv_ks=conv_ks, J=J,
            )
            for _ in range(layer_num)
        ])

        self.dffm = DFFM1D(layer_num, d_model)

    def forward(self, y, op, x0=None,
                return_aux=False, return_symloss=False, return_branches=False):
        if x0 is None:
            x0 = op.At(y)

        scale = x0.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        x = x0 / scale
        v = x0 / scale
        y_scaled = y / scale.view(scale.shape[0], 1)

        stage_outputs = []
        feat_sum = 0
        aux_list = []

        for blk in self.blocks:
            x, v, feat_fused, aux = blk(x, v, y_scaled, op)
            stage_outputs.append(x.squeeze(1))
            feat_sum = feat_sum + feat_fused.permute(0, 2, 1)

            if return_aux or return_symloss or return_branches:
                aux_list.append(aux)

        x_cat = torch.stack(stage_outputs, dim=-1)
        x_fused = self.dffm(x_cat, feat_sum, self.weight)
        x_out = x_fused * scale

        if return_aux or return_symloss or return_branches:
            return x_out, aux_list
        return x_out
