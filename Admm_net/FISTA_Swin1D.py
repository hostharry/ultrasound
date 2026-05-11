"""Swin-style 1D Window Attention HASA-FISTA 网络.

复用 FISTA_Transformer.py 的 FISTA 展开框架，将全局自注意力替换为
1D 局部窗口注意力（Swin Transformer 思路），复杂度从 O(L²) 降为 O(L·W)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from admm_ops import soft_threshold
from FISTA_Transformer import _sinusoidal_pe


# ═══════════════════════ 1D Window Attention 组件 ═══════════════════════


def _window_partition_1d(x, ws):
    """将 1D 序列划分为不重叠窗口: (B, L, D) → (B*nW, ws, D)."""
    B, L, D = x.shape
    nW = L // ws
    return x.view(B, nW, ws, D).reshape(B * nW, ws, D)


def _window_reverse_1d(windows, ws, L):
    """窗口还原为 1D 序列: (B*nW, ws, D) → (B, L, D)."""
    nW = L // ws
    B = windows.shape[0] // nW
    D = windows.shape[-1]
    return windows.view(B, nW, ws, D).reshape(B, L, D)


def _compute_swin_mask_1d(L, ws, shift_size, device):
    """为 shifted window attention 计算 1D 注意力掩码: (nW, ws, ws)."""
    mask = torch.zeros(1, L, 1, device=device)
    slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
    cnt = 0
    for s in slices:
        mask[:, s, :] = cnt
        cnt += 1
    mask_windows = _window_partition_1d(mask, ws).squeeze(-1)   # (nW, ws)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class WindowAttention1D(nn.Module):
    """1D 窗口内多头自注意力."""

    def __init__(self, d_model, nhead=4, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B*nW, ws, D), mask: (nW, ws, ws) or None."""
        BnW, L, D = x.shape
        nh, hd = self.nhead, self.head_dim

        qkv = self.qkv(x).reshape(BnW, L, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(-1, nW, nh, L, L)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(BnW, nh, L, L)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BnW, L, D)
        return self.proj(x)


class SwinBlock1D(nn.Module):
    """1D Swin Transformer block: (shifted) window attention + FFN.

    输入输出: (B, L, D)。自动 pad 到 window_size 整数倍。
    """

    def __init__(self, d_model, nhead=4, mlp_ratio=2.0, dropout=0.0,
                 window_size=64, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = WindowAttention1D(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )
        self._mask_cache = {}

    def _get_mask(self, Lp, device):
        key = (Lp, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = _compute_swin_mask_1d(
                Lp, self.window_size, self.shift_size, device,
            )
        return self._mask_cache[key]

    def forward(self, x):
        """x: (B, L, D)."""
        B, L, D = x.shape
        ws = self.window_size

        pad_r = (ws - L % ws) % ws
        if pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r))
        Lp = L + pad_r

        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)
            mask = self._get_mask(Lp, x.device)
        else:
            mask = None

        windows = _window_partition_1d(x, ws)
        windows = self.attn(windows, mask=mask)
        x = _window_reverse_1d(windows, ws, Lp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)

        x = shortcut + x
        x = x + self.ffn(self.norm2(x))

        if pad_r > 0:
            x = x[:, :L, :].contiguous()
        return x


def _make_swin_blocks_1d(num_layers, d_model, nhead, mlp_ratio, window_size):
    """创建交替 regular / shifted 窗口的 1D Swin block 序列."""
    return nn.ModuleList([
        SwinBlock1D(d_model, nhead, mlp_ratio,
                    window_size=window_size, shift=(i % 2 == 1))
        for i in range(num_layers)
    ])


# ═══════════════════════ Swin 1D 权重 & Prox ═══════════════════════


class HASAWeightSwin1D(nn.Module):
    """Swin Window Attention HASA 权重网络 (1D)."""

    def __init__(self, d_model=32, nhead=4, num_layers=2,
                 mlp_ratio=2.0, window_size=64):
        super().__init__()
        self.d_model = d_model
        self.proj_in = nn.Linear(1, d_model)
        self.blocks = _make_swin_blocks_1d(
            num_layers, d_model, nhead, mlp_ratio, window_size,
        )
        self.norm = nn.LayerNorm(d_model)

        self.head_tv = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.head_wav = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.head_alpha = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        nn.init.constant_(self.head_tv[0].bias, -4.0)
        nn.init.constant_(self.head_wav[0].bias, -4.0)
        nn.init.constant_(self.head_alpha[0].bias, 0.0)

    def forward(self, x):
        B, _, L = x.shape
        tok = self.proj_in(x.permute(0, 2, 1))                     # (B, L, D)
        tok = tok + _sinusoidal_pe(L, self.d_model, x.device)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)

        tv    = self.head_tv(tok).permute(0, 2, 1)                 # (B, 1, L)
        wav   = self.head_wav(tok).permute(0, 2, 1)
        alpha = self.head_alpha(tok).permute(0, 2, 1)
        return tv, wav, alpha


class ISTAProxSwin1D_Dual(nn.Module):
    """Swin Window Attention 双分支 ISTA Prox (1D)."""

    def __init__(self, d_model=32, nhead=4, num_layers=2,
                 mlp_ratio=2.0, window_size=64):
        super().__init__()
        self.d_model = d_model

        self.enc_proj_tv = nn.Linear(1, d_model)
        self.enc_tv = _make_swin_blocks_1d(
            num_layers, d_model, nhead, mlp_ratio, window_size)
        self.enc_norm_tv = nn.LayerNorm(d_model)
        self.dec_tv = _make_swin_blocks_1d(
            num_layers, d_model, nhead, mlp_ratio, window_size)
        self.dec_norm_tv = nn.LayerNorm(d_model)
        self.dec_proj_tv = nn.Linear(d_model, 1)

        self.enc_proj_wav = nn.Linear(1, d_model)
        self.enc_wav = _make_swin_blocks_1d(
            num_layers, d_model, nhead, mlp_ratio, window_size)
        self.enc_norm_wav = nn.LayerNorm(d_model)
        self.dec_wav = _make_swin_blocks_1d(
            num_layers, d_model, nhead, mlp_ratio, window_size)
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

        if not return_symloss:
            return x_next, None
        x_est = self._decode(feat, dec, dec_norm, dec_proj)
        return x_next, x_est - z

    def forward(self, z, thr_tv, thr_wav, alpha=None,
                return_symloss=False, return_branches=False):
        x_tv, sym_tv = self._branch_forward(
            z, thr_tv,
            self.enc_proj_tv, self.enc_tv, self.enc_norm_tv,
            self.dec_tv, self.dec_norm_tv, self.dec_proj_tv,
            return_symloss,
        )
        x_wav, sym_wav = self._branch_forward(
            z, thr_wav,
            self.enc_proj_wav, self.enc_wav, self.enc_norm_wav,
            self.dec_wav, self.dec_norm_wav, self.dec_proj_wav,
            return_symloss,
        )

        if alpha is None:
            x_next = 0.5 * (x_tv + x_wav)
        else:
            x_next = alpha * x_tv + (1.0 - alpha) * x_wav

        if not return_branches:
            return x_next, sym_tv, sym_wav, None, None
        return x_next, sym_tv, sym_wav, x_tv, x_wav


# ═══════════════════════ FISTA 展开网络 ═══════════════════════


class HASA_FISTA_Swin_Block_1D(nn.Module):
    """单层 Swin-FISTA block (1D)."""

    def __init__(self, weight_ctor, d_model=32, nhead=4,
                 num_layers=2, window_size=64):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProxSwin1D_Dual(
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, window_size=window_size,
        )

    def forward(self, x_prev, v, y, op,
                return_symloss=False, return_branches=False):
        r = op.A(v) - y
        g = op.At(r)
        z = v - F.softplus(self.rho) * g

        lambda_tv, lambda_wav, alpha = self.weight(z)
        thr_tv = F.softplus(self.soft_thr) * lambda_tv
        thr_wav = F.softplus(self.soft_thr) * lambda_wav

        x_next, sym_tv, sym_wav, x_tv, x_wav = self.prox(
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
        return x_next, v_next, aux


class HASA_FISTA_Swin_Net_1D(nn.Module):
    """1D Swin Window Attention HASA-FISTA 网络."""

    def __init__(self, layer_num=9, hasa_ctor=None,
                 d_model=32, nhead=4, num_layers=2, window_size=64):
        super().__init__()
        if hasa_ctor is None:
            hasa_ctor = lambda: HASAWeightSwin1D(
                d_model=d_model, nhead=nhead,
                num_layers=num_layers, window_size=window_size,
            )

        self.blocks = nn.ModuleList([
            HASA_FISTA_Swin_Block_1D(
                hasa_ctor, d_model=d_model, nhead=nhead,
                num_layers=num_layers, window_size=window_size,
            )
            for _ in range(layer_num)
        ])

    def forward(self, y, op, x0=None,
                return_aux=False, return_symloss=False, return_branches=False):
        if x0 is None:
            x0 = op.At(y)

        scale = x0.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        x = x0 / scale
        v = x0 / scale
        y_scaled = y / scale.view(scale.shape[0], 1)

        aux_list = []
        for blk in self.blocks:
            x, v, aux = blk(
                x, v, y_scaled, op,
                return_symloss=return_symloss,
                return_branches=return_branches,
            )
            if return_aux or return_symloss or return_branches:
                aux_list.append(aux)

        x = x * scale

        if return_aux or return_symloss or return_branches:
            return x, aux_list
        return x
