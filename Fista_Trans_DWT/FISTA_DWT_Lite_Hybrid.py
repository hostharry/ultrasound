"""FISTA-DWT-Lite-Hybrid: 在 Lite 基础上增强 DWT 分支跨子带协同 (1D).

相对 FISTA_DWT_Lite.py 的改动:
  - DWT 分支新增 CrossSubbandMixer: 子带编码后做轻量 band-axis attention
  - DWT 分支新增 LearnableBandFusion: 替代简单平均的可学习多子带特征融合
  - TV 分支 / HASA / DFFM / FISTA 外层 均不变
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


def _inverse_tanh_scalar(x: float) -> float:
    x = max(min(x, 0.999), -0.999)
    return 0.5 * math.log((1.0 + x) / (1.0 - x))


# ═══════════════════════ Conv 基础组件 ═══════════════════════


class ConvResBlock1D(nn.Module):
    """两层 Conv1d + 残差: (B, C, L) -> (B, C, L)."""

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
    """Conv1d 残差块替代 Transformer 的 TV 近端分支 (与 Lite 版相同)."""

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5):
        super().__init__()
        self.proj_in = nn.Conv1d(1, d_model, 1)
        self.enc = nn.Sequential(
            *[ConvResBlock1D(d_model, kernel_size) for _ in range(num_blocks)])
        self.dec = nn.Sequential(
            *[ConvResBlock1D(d_model, kernel_size) for _ in range(num_blocks)])
        self.proj_out = nn.Conv1d(d_model, 1, 1)

    def forward(self, z, thr_tv):
        feat = self.proj_in(z)
        feat = self.enc(feat)
        feat_enc = feat
        thr_expanded = thr_tv.expand_as(feat)
        feat_shrink = soft_threshold(feat, thr_expanded)
        feat = self.dec(feat_shrink)
        x_tv = self.proj_out(feat)
        return x_tv, feat_enc.permute(0, 2, 1)


# ═══════════════════════ 跨子带交互模块 ═══════════════════════


class CrossSubbandMixer(nn.Module):
    """轻量 band-axis attention: 在每个空间位置让 J+1 个子带 token 互相注意.

    输入: aligned_feats list of (B, D, L_padded), 长度 n_bands
    输出: mixed list of (B, D, L_padded), 长度 n_bands

    复杂度只与子带数 n_bands 相关 (通常 4), 不依赖序列长度.
    注意: gamma 门控已移至 DWTConvBranchHybrid, 此模块只负责 attention + FFN.
    """

    def __init__(self, d_model, n_bands, nhead=4, mlp_ratio=2.0):
        super().__init__()
        self.n_bands = n_bands
        self.d_model = d_model
        self.band_embed = nn.Embedding(n_bands, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, aligned_feats):
        """
        aligned_feats: list of (B, D, L) tensors, len == n_bands
        Returns: list of (B, D, L) tensors, len == n_bands
        """
        B, D, L = aligned_feats[0].shape

        x = torch.stack(aligned_feats, dim=1)              # (B, n_bands, D, L)
        x = x.permute(0, 3, 1, 2).contiguous()             # (B, L, n_bands, D)
        x_flat = x.view(B * L, self.n_bands, D)             # (B*L, n_bands, D)

        band_ids = torch.arange(self.n_bands, device=x.device)
        x_flat = x_flat + self.band_embed(band_ids)

        x_normed = self.norm1(x_flat)
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x_flat = x_flat + attn_out

        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        out = x_flat.view(B, L, self.n_bands, D)            # (B, L, n_bands, D)
        out = out.permute(0, 2, 3, 1)                       # (B, n_bands, D, L)

        return [out[:, i] for i in range(self.n_bands)]


# ═══════════════════════ 可学习子带融合 ═══════════════════════


class LearnableBandFusion(nn.Module):
    """将对齐后的多子带特征通过 concat + 1x1 Conv 融合为单一特征图.

    替代原版简单平均, 让模型学习不同位置上各子带的重要性.
    """

    def __init__(self, d_model, n_bands):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(n_bands * d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, aligned_feats):
        """
        aligned_feats: list of (B, D, L) tensors
        Returns: (B, D, L)
        """
        x = torch.cat(aligned_feats, dim=1)  # (B, n_bands*D, L)
        return self.fuse(x)                   # (B, D, L)


# ═══════════════════════ Hybrid DWT Branch ═══════════════════════


class DWTConvBranchHybrid(nn.Module):
    """增强版 DWT 多尺度小波分支: 加入跨子带交互 + 可学习融合.

    数据流:
      z -> pad -> DWT -> {cA, cD_J, ..., cD_1}
        -> 子带独立编码 (enc_feats, 各子带原分辨率)
        -> 对齐到 L_padded -> CrossSubbandMixer -> 恢复到原长度 (mixer_delta)
        -> gamma 门控混合: enc_feats + tanh(gamma) * (mixer_delta - enc_feats)
           (gamma=small_init 时近似原版 Lite, 无插值损伤)
        -> cA gate / cD soft-threshold
        -> 子带独立解码 -> IDWT -> crop -> x_wav

    并行地:
      mixer 对齐输出 -> LearnableBandFusion -> feat_spatial
    """

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5, J=3,
                 mixer_nhead=4, mixer_mlp_ratio=2.0,
                 mixer_gamma_init=1e-2, detail_thr_gain=1.0):
        super().__init__()
        self.d_model = d_model
        self.J = J
        self.n_bands = J + 1
        self._divisor = 2 ** J

        if d_model % mixer_nhead != 0:
            raise ValueError(
                f"mixer_nhead={mixer_nhead} must divide d_model={d_model}"
            )

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
        self.detail_thr_gain = detail_thr_gain

        # === 新增模块 ===
        self.cross_band_mixer = CrossSubbandMixer(
            d_model=d_model,
            n_bands=self.n_bands,
            nhead=mixer_nhead,
            mlp_ratio=mixer_mlp_ratio,
        )
        self.feature_fuser = LearnableBandFusion(
            d_model=d_model,
            n_bands=self.n_bands,
        )
        self.gamma = nn.Parameter(torch.tensor(_inverse_tanh_scalar(mixer_gamma_init)))

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

    @staticmethod
    def _align_feats(feats, L_target):
        """各子带 encoder 特征上采样到 L_target."""
        aligned = []
        for f in feats:
            if f.shape[-1] == L_target:
                aligned.append(f)
            else:
                f_up = F.interpolate(
                    f, size=L_target, mode='linear', align_corners=False)
                aligned.append(f_up)
        return aligned

    @staticmethod
    def _restore_lengths(aligned_feats, lengths):
        """将统一长度的 mixed 特征恢复到各子带原长度."""
        restored = []
        for feat, tgt_len in zip(aligned_feats, lengths):
            if feat.shape[-1] == tgt_len:
                restored.append(feat)
            else:
                restored.append(F.interpolate(
                    feat, size=tgt_len, mode='linear', align_corners=False))
        return restored

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

        # ---- 子带独立编码 ----
        enc_feats = []
        for band, proj, enc_blk in zip(subbands, self.proj_in, self.enc):
            feat = proj(band)
            feat = enc_blk(feat)
            enc_feats.append(feat)

        # ---- 跨子带交互 (在 L_padded 上做 attention) ----
        aligned_feats = self._align_feats(enc_feats, L_padded)
        mixed_aligned = self.cross_band_mixer(aligned_feats)
        mixer_restored = self._restore_lengths(mixed_aligned, lengths)

        # ---- gamma 门控: 在原分辨率上混合, 避免插值平滑直接替换 enc_feats ----
        g = torch.tanh(self.gamma)
        blended = [enc + g * (mix - enc)
                   for enc, mix in zip(enc_feats, mixer_restored)]

        # ---- gate / shrink ----
        processed = []
        gated = blended[0] * self.approx_gate(blended[0])
        processed.append(gated)

        for j, (feat_j, thr_j) in enumerate(
                zip(blended[1:], thr_per_band[1:])):
            scale_j = self.detail_thr_gain * F.softplus(self.subband_scale[j])
            thr_expanded = (scale_j * thr_j).expand_as(feat_j)
            processed.append(soft_threshold(feat_j, thr_expanded))

        # ---- 子带独立解码 + IDWT ----
        rec_bands = []
        for p, dec_blk, proj_out in zip(processed, self.dec, self.proj_out):
            decoded = dec_blk(p)
            rec_bands.append(proj_out(decoded))

        x_wav = self.idwt(rec_bands[0], rec_bands[1:])
        x_wav = x_wav[..., :L_orig]

        # ---- 可学习融合 (同样受 gamma 门控, 与重建路径一致) ----
        blended_aligned = [a + g * (m - a)
                           for a, m in zip(aligned_feats, mixed_aligned)]
        feat_spatial = self.feature_fuser(blended_aligned)
        return x_wav, feat_spatial.permute(0, 2, 1)


# ═══════════════════════ Hybrid Prox / Block / Net ═══════════════════════


class ISTAProxDWT1D_Hybrid(nn.Module):
    """双分支 ISTA 近端 (Hybrid): TV (Conv1d) + Wav (DWT + 跨子带交互)."""

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5, J=3,
                 mixer_nhead=4, mixer_mlp_ratio=2.0,
                 mixer_gamma_init=1e-2, detail_thr_gain=1.0):
        super().__init__()
        self.tv_branch = ConvProxTV1D(
            d_model=d_model, num_blocks=num_blocks, kernel_size=kernel_size)
        self.wav_branch = DWTConvBranchHybrid(
            d_model=d_model, num_blocks=num_blocks, kernel_size=kernel_size,
            J=J, mixer_nhead=mixer_nhead, mixer_mlp_ratio=mixer_mlp_ratio,
            mixer_gamma_init=mixer_gamma_init, detail_thr_gain=detail_thr_gain)

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


class FISTA_DWT_Lite_Hybrid_Block(nn.Module):
    """单层 FISTA-Hybrid block: 梯度下降 -> HASA(Transformer) -> Hybrid Prox -> momentum."""

    def __init__(self, weight_ctor, d_model=32, nhead=4,
                 num_transformer_layers=1, num_conv_blocks=2,
                 conv_ks=5, J=3, mixer_nhead=4, mixer_mlp_ratio=2.0,
                 mixer_gamma_init=1e-2, detail_thr_gain=1.0):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProxDWT1D_Hybrid(
            d_model=d_model, num_blocks=num_conv_blocks,
            kernel_size=conv_ks, J=J,
            mixer_nhead=mixer_nhead, mixer_mlp_ratio=mixer_mlp_ratio,
            mixer_gamma_init=mixer_gamma_init, detail_thr_gain=detail_thr_gain,
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
            "gamma": torch.tanh(self.prox.wav_branch.gamma).detach(),
            "lambda_tv": lambda_tv.detach(),
            "lambda_wav": lambda_wav.detach(),
            "alpha": alpha.detach(),
            "sym_tv": None,
            "sym_wav": None,
            "constraint_wav": zeros.detach(),
            "constraint_tv": zeros.detach(),
        }
        return x_next, v_next, feat_fused, aux


class FISTA_DWT_Lite_Hybrid_Net(nn.Module):
    """1D FISTA-DWT-Lite-Hybrid: HASA(Transformer) + Hybrid Prox(Conv+CrossBand) + DFFM."""

    def __init__(self, layer_num=4, hasa_ctor=None,
                 d_model=32, nhead=4, num_transformer_layers=1,
                 num_conv_blocks=2, conv_ks=5, J=3,
                 mixer_nhead=4, mixer_mlp_ratio=2.0,
                 mixer_gamma_init=1e-2, detail_thr_gain=1.0):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.tensor(0.2))

        if hasa_ctor is None:
            hasa_ctor = lambda: HASAWeightTransformer1D(
                d_model=d_model, nhead=nhead, num_layers=num_transformer_layers,
            )

        self.blocks = nn.ModuleList([
            FISTA_DWT_Lite_Hybrid_Block(
                hasa_ctor, d_model=d_model, nhead=nhead,
                num_transformer_layers=num_transformer_layers,
                num_conv_blocks=num_conv_blocks, conv_ks=conv_ks, J=J,
                mixer_nhead=mixer_nhead, mixer_mlp_ratio=mixer_mlp_ratio,
                mixer_gamma_init=mixer_gamma_init, detail_thr_gain=detail_thr_gain,
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
