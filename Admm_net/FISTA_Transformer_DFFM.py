"""FISTA-Transformer + DFFM 跨层融合 (1D).

在 FISTA_Transformer.py 的基础上加入 HUNet 的 DFFM 机制:
收集所有展开层的 1-ch 输出和编码器特征，通过 DFFM 融合为最终重建，
替代仅使用最后一层输出的策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from admm_ops import soft_threshold
from FISTA_Transformer import (
    TransformerBlock,
    _sinusoidal_pe,
    HASAWeightTransformer1D,
)


# ═══════════════════════ DFFM 组件 ═══════════════════════


class Tail1D(nn.Module):
    """(B, d_model, L) -> (B, 1, L)."""

    def __init__(self, d_model):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model // 3 or 1, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 3 or 1, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class SELayer1D(nn.Module):
    """Squeeze-and-Excitation for 1D: (B, C, L) -> (B, C, L)."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        mid = max(channel // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1)
        return x * y.expand_as(x)


class DMlp1D(nn.Module):
    """Depth-wise MLP for 1D: (B, C, L) -> (B, C, L)."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.PReLU(),
            nn.Conv1d(dim, 4 * dim, 1),
            nn.PReLU(),
            nn.Conv1d(4 * dim, dim, 1),
        )

    def forward(self, x):
        return self.block(x)


class DFFM1D(nn.Module):
    """Dual-path Feature Fusion Module (1D).

    Fuses all layer outputs (1-ch each) and accumulated encoder features.
    """

    def __init__(self, num_layers, d_model):
        super().__init__()
        self.conv = nn.Conv1d(num_layers, d_model, 3, padding=1)
        self.dmlp = DMlp1D(d_model)
        self.se = SELayer1D(d_model)
        self.tail = Tail1D(d_model)

    def forward(self, x_stages, feat_sum, weight):
        """
        x_stages: (B, L, num_layers) — stacked 1-ch outputs
        feat_sum: (B, d_model, L) — accumulated encoder features
        weight: scalar parameter
        """
        x = self.conv(x_stages.permute(0, 2, 1))     # (B, d_model, L)
        shortcut = x
        x = self.dmlp(x)
        x = self.se(x)
        x = x + shortcut
        return self.tail(x + weight * feat_sum)       # (B, 1, L)


# ═══════════════════════ 修改的 Prox / Block / Net ═══════════════════════


class ISTAProxTransformer1D_Dual_DFFM(nn.Module):
    """Transformer 双分支 ISTA Prox (1D), 额外返回编码器特征."""

    def __init__(self, d_model=32, nhead=4, num_layers=2, mlp_ratio=2.0):
        super().__init__()
        self.d_model = d_model

        self.enc_proj_tv = nn.Linear(1, d_model)
        self.enc_tv = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)])
        self.enc_norm_tv = nn.LayerNorm(d_model)
        self.dec_tv = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)])
        self.dec_norm_tv = nn.LayerNorm(d_model)
        self.dec_proj_tv = nn.Linear(d_model, 1)

        self.enc_proj_wav = nn.Linear(1, d_model)
        self.enc_wav = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)])
        self.enc_norm_wav = nn.LayerNorm(d_model)
        self.dec_wav = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)])
        self.dec_norm_wav = nn.LayerNorm(d_model)
        self.dec_proj_wav = nn.Linear(d_model, 1)

    def _encode(self, z, proj, enc_blocks, norm):
        tok = proj(z.permute(0, 2, 1))
        tok = tok + _sinusoidal_pe(tok.shape[1], self.d_model, z.device)
        for blk in enc_blocks:
            tok = blk(tok)
        return norm(tok)

    @staticmethod
    def _decode(feat, dec_blocks, norm, proj):
        tok = feat
        for blk in dec_blocks:
            tok = blk(tok)
        tok = norm(tok)
        return proj(tok).permute(0, 2, 1)

    @staticmethod
    def _expand_thr(thr, feat):
        return thr.permute(0, 2, 1).expand_as(feat)

    def _branch_forward(self, z, thr, proj, enc, enc_norm,
                        dec, dec_norm, dec_proj, return_symloss=False):
        feat = self._encode(z, proj, enc, enc_norm)
        feat_shrink = soft_threshold(feat, self._expand_thr(thr, feat))
        x_next = self._decode(feat_shrink, dec, dec_norm, dec_proj)

        sym = None
        if return_symloss:
            x_est = self._decode(feat, dec, dec_norm, dec_proj)
            sym = x_est - z

        return x_next, feat, sym

    def forward(self, z, thr_tv, thr_wav, alpha=None,
                return_symloss=False, return_branches=False):
        x_tv, feat_tv, sym_tv = self._branch_forward(
            z, thr_tv,
            self.enc_proj_tv, self.enc_tv, self.enc_norm_tv,
            self.dec_tv, self.dec_norm_tv, self.dec_proj_tv,
            return_symloss,
        )
        x_wav, feat_wav, sym_wav = self._branch_forward(
            z, thr_wav,
            self.enc_proj_wav, self.enc_wav, self.enc_norm_wav,
            self.dec_wav, self.dec_norm_wav, self.dec_proj_wav,
            return_symloss,
        )

        if alpha is None:
            x_next = 0.5 * (x_tv + x_wav)
            feat_fused = 0.5 * (feat_tv + feat_wav)
        else:
            alpha_seq = alpha.permute(0, 2, 1).expand_as(feat_tv)
            x_next = alpha * x_tv + (1.0 - alpha) * x_wav
            feat_fused = alpha_seq * feat_tv + (1.0 - alpha_seq) * feat_wav

        if not return_branches:
            return x_next, feat_fused, sym_tv, sym_wav, None, None
        return x_next, feat_fused, sym_tv, sym_wav, x_tv, x_wav


class HASA_FISTA_Transformer_Block_1D_DFFM(nn.Module):
    """单层 Transformer-FISTA block (1D), 返回编码器特征."""

    def __init__(self, weight_ctor, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProxTransformer1D_Dual_DFFM(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
        )

    def forward(self, x_prev, v, y, op,
                return_symloss=False, return_branches=False):
        r = op.A(v) - y
        g = op.At(r)
        z = v - F.softplus(self.rho) * g

        lambda_tv, lambda_wav, alpha = self.weight(z)
        thr_tv = F.softplus(self.soft_thr) * lambda_tv
        thr_wav = F.softplus(self.soft_thr) * lambda_wav

        x_next, feat_fused, sym_tv, sym_wav, x_tv, x_wav = self.prox(
            z, thr_tv, thr_wav, alpha=alpha,
            return_symloss=return_symloss, return_branches=return_branches,
        )

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
            "sym_tv": sym_tv,
            "sym_wav": sym_wav,
            "constraint_wav": zeros.detach(),
            "constraint_tv": zeros.detach(),
        }
        if return_branches:
            aux["x_tv"] = x_tv
            aux["x_wav"] = x_wav

        return x_next, v_next, feat_fused, aux


class HASA_FISTA_Transformer_Net_1D_DFFM(nn.Module):
    """1D Transformer-based HASA-FISTA + DFFM 跨层融合网络."""

    def __init__(self, layer_num=12, hasa_ctor=None,
                 d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.tensor(0.2))

        if hasa_ctor is None:
            hasa_ctor = lambda: HASAWeightTransformer1D(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
            )

        self.blocks = nn.ModuleList([
            HASA_FISTA_Transformer_Block_1D_DFFM(
                hasa_ctor, d_model=d_model, nhead=nhead, num_layers=num_layers,
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
            x, v, feat_fused, aux = blk(
                x, v, y_scaled, op,
                return_symloss=return_symloss,
                return_branches=return_branches,
            )
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
