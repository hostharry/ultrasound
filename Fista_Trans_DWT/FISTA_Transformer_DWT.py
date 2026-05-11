"""FISTA-Transformer + DWT 多尺度小波分支 + DFFM 跨层融合 (1D).

核心改造:
  - Wav 分支: 信号先做 J 级 Haar DWT, 各子带独立投影 + scale PE + depth PE,
    经 Transformer Encoder/Decoder 处理后:
      * 细节子带 (cD): 按 "子带×位置" 做 soft-threshold
      * 近似子带 (cA): 可学习门控, 不做硬截断
    最后 IDWT 重建.
  - TV 分支: 与原始 FISTA-Transformer 相同, 全局 Transformer 编解码 + 统一阈值.
  - DFFM: 收集所有展开层的输出和编码器特征, 跨层融合.
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
from FISTA_Transformer import TransformerBlock, _sinusoidal_pe, HASAWeightTransformer1D
from FISTA_Transformer_DFFM import DFFM1D


# ═══════════════════════ Haar DWT / IDWT ═══════════════════════


class HaarDWT1d(nn.Module):
    """J 级 Haar 小波分解: (B,1,L) → cA_J, [cD_J, ..., cD_1]."""

    def __init__(self, J=3):
        super().__init__()
        self.J = J
        lo = torch.tensor([[1.0, 1.0]]) / math.sqrt(2)
        hi = torch.tensor([[1.0, -1.0]]) / math.sqrt(2)
        self.register_buffer("lo", lo.unsqueeze(0))   # (1, 1, 2)
        self.register_buffer("hi", hi.unsqueeze(0))

    def forward(self, x):
        details = []
        for _ in range(self.J):
            cA = F.conv1d(x, self.lo, stride=2)
            cD = F.conv1d(x, self.hi, stride=2)
            details.append(cD)
            x = cA
        return x, details[::-1]  # cA_J, [cD_J, cD_{J-1}, ..., cD_1]


class HaarIDWT1d(nn.Module):
    """J 级 Haar 小波重建: cA_J, [cD_J, ..., cD_1] → (B,1,L)."""

    def __init__(self, J=3):
        super().__init__()
        self.J = J
        lo_r = torch.tensor([[1.0, 1.0]]) / math.sqrt(2)
        hi_r = torch.tensor([[1.0, -1.0]]) / math.sqrt(2)
        self.register_buffer("lo_r", lo_r.unsqueeze(0))
        self.register_buffer("hi_r", hi_r.unsqueeze(0))

    def forward(self, cA, details):
        x = cA
        for cD in details:
            x = F.conv_transpose1d(x, self.lo_r, stride=2) + \
                F.conv_transpose1d(cD, self.hi_r, stride=2)
        return x


# ═══════════════════════ DWT Wav Branch ═══════════════════════


class DWTWavBranch(nn.Module):
    """DWT 多尺度小波代理分支.

    数据流:
      z → pad(2^J) → DWT → {cA_J, cD_J, ..., cD_1}
        → 子带独立投影 + scale_embed + depth_PE
        → Transformer Encoder
        → cA: 可学习门控 / cD: per-(subband×position) soft-threshold
        → Transformer Decoder
        → 子带投影回 1D → IDWT → crop → x_wav
    """

    def __init__(self, d_model=16, nhead=4, num_layers=1, J=3, mlp_ratio=2.0):
        super().__init__()
        self.d_model = d_model
        self.J = J
        self.n_bands = J + 1
        self._divisor = 2 ** J

        self.dwt = HaarDWT1d(J)
        self.idwt = HaarIDWT1d(J)

        self.proj_in = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(self.n_bands)])
        self.proj_out = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(self.n_bands)])

        self.scale_embed = nn.Embedding(self.n_bands, d_model)

        self.enc = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio)
             for _ in range(num_layers)])
        self.enc_norm = nn.LayerNorm(d_model)

        self.dec = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio)
             for _ in range(num_layers)])
        self.dec_norm = nn.LayerNorm(d_model)

        self.approx_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.subband_scale = nn.Parameter(torch.ones(J))

    def _pad(self, x):
        """右侧零填充使长度能被 2^J 整除, 返回 (x_padded, orig_len)."""
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

    def _make_depth_pe(self, lengths, L_padded, device):
        """计算每个子带 token 在原始信号中的深度中心, 生成深度位置编码.

        cA_J / cD_j 的 token i 对应原始信号的中心深度 = (i+0.5) * stride_j.
        """
        strides = [self._divisor]                         # cA_J
        for k in range(1, self.n_bands):                  # cD_J, cD_{J-1}, ..., cD_1
            strides.append(2 ** (self.J - k + 1))

        depth_centers = []
        for band_len, stride in zip(lengths, strides):
            centers = (torch.arange(band_len, device=device).float() + 0.5) * stride
            depth_centers.append(centers)

        all_pos = torch.cat(depth_centers).unsqueeze(1)   # (T, 1)
        d = self.d_model
        i = torch.arange(d, device=device).float()
        angle = all_pos / (10000.0 ** (2 * (i // 2) / d))
        pe = torch.zeros(1, all_pos.shape[0], d, device=device)
        pe[0, :, 0::2] = torch.sin(angle[:, 0::2])
        pe[0, :, 1::2] = torch.cos(angle[:, 1::2])
        return pe

    def _realign_to_spatial(self, feat, lengths, L_target):
        """将子带拼接的编码器特征重采样到空间域 (B, L_target, d_model).

        每个子带特征线性上采样到 L_target, 再取均值, 保证与 TV 分支特征坐标系一致.
        """
        parts = torch.split(feat, lengths, dim=1)
        aligned = []
        for p in parts:
            p_up = F.interpolate(
                p.permute(0, 2, 1),            # (B, d_model, band_len)
                size=L_target, mode='linear', align_corners=False,
            ).permute(0, 2, 1)                 # (B, L_target, d_model)
            aligned.append(p_up)
        return torch.stack(aligned, dim=0).mean(dim=0)

    def forward(self, z, thr_wav):
        """
        z:       (B, 1, L)
        thr_wav: (B, 1, L) — per-position threshold from HASA

        Returns:
            x_wav:        (B, 1, L)
            feat_spatial: (B, L_padded, d_model)  spatially-aligned encoder features
        """
        B, _, L_orig = z.shape

        z_pad, _ = self._pad(z)
        thr_pad, _ = self._pad(thr_wav)
        L_padded = z_pad.shape[-1]

        cA, details = self.dwt(z_pad)
        subbands = [cA] + details
        lengths = [s.shape[-1] for s in subbands]

        thr_per_band = self._downsample_thr(thr_pad, lengths, L_padded)

        tok_list, scale_list = [], []
        for i, (band, proj) in enumerate(zip(subbands, self.proj_in)):
            tok = proj(band.permute(0, 2, 1))
            band_len = tok.shape[1]
            tok = tok + _sinusoidal_pe(band_len, self.d_model, z.device)
            tok_list.append(tok)
            scale_list.append(
                torch.full((band_len,), i, device=z.device, dtype=torch.long))

        tok_all = torch.cat(tok_list, dim=1)
        scale_ids = torch.cat(scale_list)
        tok_all = tok_all + self.scale_embed(scale_ids)
        tok_all = tok_all + self._make_depth_pe(lengths, L_padded, z.device)

        for blk in self.enc:
            tok_all = blk(tok_all)
        feat = self.enc_norm(tok_all)

        parts = torch.split(feat, lengths, dim=1)

        gated = parts[0] * self.approx_gate(parts[0])

        shrunk = [gated]
        for j, (part_j, thr_j) in enumerate(
                zip(parts[1:], thr_per_band[1:])):
            scale_j = F.softplus(self.subband_scale[j])
            thr_expanded = (scale_j * thr_j).permute(0, 2, 1).expand_as(part_j)
            shrunk.append(soft_threshold(part_j, thr_expanded))

        feat_shrunk = torch.cat(shrunk, dim=1)

        for blk in self.dec:
            feat_shrunk = blk(feat_shrunk)
        tok_dec = self.dec_norm(feat_shrunk)

        dec_parts = torch.split(tok_dec, lengths, dim=1)
        rec_bands = [proj(p).permute(0, 2, 1)
                     for p, proj in zip(dec_parts, self.proj_out)]

        x_wav = self.idwt(rec_bands[0], rec_bands[1:])
        x_wav = x_wav[..., :L_orig]

        feat_spatial = self._realign_to_spatial(feat, lengths, L_padded)
        return x_wav, feat_spatial


# ═══════════════════════ Prox / Block / Net ═══════════════════════


class ISTAProxDWT1D(nn.Module):
    """双分支 ISTA 近端: TV (全局 Transformer) + Wav (DWT 多尺度)."""

    def __init__(self, d_model=16, nhead=4, num_layers=1,
                 J=3, mlp_ratio=2.0):
        super().__init__()
        self.d_model = d_model

        # TV branch (unchanged)
        self.enc_proj_tv = nn.Linear(1, d_model)
        self.enc_tv = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio)
             for _ in range(num_layers)])
        self.enc_norm_tv = nn.LayerNorm(d_model)
        self.dec_tv = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio)
             for _ in range(num_layers)])
        self.dec_norm_tv = nn.LayerNorm(d_model)
        self.dec_proj_tv = nn.Linear(d_model, 1)

        # DWT Wav branch
        self.wav_branch = DWTWavBranch(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            J=J, mlp_ratio=mlp_ratio,
        )

    def _tv_forward(self, z, thr_tv):
        tok = self.enc_proj_tv(z.permute(0, 2, 1))
        tok = tok + _sinusoidal_pe(tok.shape[1], self.d_model, z.device)
        for blk in self.enc_tv:
            tok = blk(tok)
        feat_tv = self.enc_norm_tv(tok)

        feat_shrink = soft_threshold(
            feat_tv, thr_tv.permute(0, 2, 1).expand_as(feat_tv))

        tok = feat_shrink
        for blk in self.dec_tv:
            tok = blk(tok)
        tok = self.dec_norm_tv(tok)
        x_tv = self.dec_proj_tv(tok).permute(0, 2, 1)
        return x_tv, feat_tv

    def forward(self, z, thr_tv, thr_wav, alpha=None):
        L = z.shape[-1]
        x_tv, feat_tv = self._tv_forward(z, thr_tv)
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


class FISTA_DWT_Block(nn.Module):
    """单层 FISTA block: 梯度下降 → HASA → DWT Prox → FISTA momentum."""

    def __init__(self, weight_ctor, d_model=16, nhead=4,
                 num_layers=1, J=3):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProxDWT1D(
            d_model=d_model, nhead=nhead, num_layers=num_layers, J=J,
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


class FISTA_Transformer_DWT_Net(nn.Module):
    """1D FISTA-Transformer + DWT 多尺度小波分支 + DFFM 跨层融合."""

    def __init__(self, layer_num=4, hasa_ctor=None,
                 d_model=16, nhead=4, num_layers=1, J=3):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.tensor(0.2))

        if hasa_ctor is None:
            hasa_ctor = lambda: HASAWeightTransformer1D(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
            )

        self.blocks = nn.ModuleList([
            FISTA_DWT_Block(
                hasa_ctor, d_model=d_model, nhead=nhead,
                num_layers=num_layers, J=J,
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
