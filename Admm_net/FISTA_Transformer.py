"""Transformer-based HASA-FISTA (1D & 2D).

将 FISTA_Baseline.py / FISTA_Baseline_2D.py 中的卷积算子替换为 Transformer。
核心改动:
  - HASAWeight:  Conv → Transformer encoder + per-position linear heads
  - ISTAProx:    Conv encoder/decoder → Transformer encoder/decoder
  - 保持 soft-threshold, FISTA momentum, data consistency 逻辑不变

1D: 直接将信号每个位置作为 token, 使用全局 self-attention。
2D: 使用 patch embedding 将图像切分为不重叠 patch, 作为 token 序列处理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from admm_ops import soft_threshold


# ═══════════════════════ Transformer 基础组件 ═══════════════════════


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN → MHSA → residual → LN → FFN → residual."""

    def __init__(self, d_model, nhead=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


def _sinusoidal_pe(seq_len, d_model, device):
    """生成 1D 正弦位置编码 (1, seq_len, d_model)."""
    pe = torch.zeros(1, seq_len, d_model, device=device)
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    i = torch.arange(d_model, device=device).float()
    angle = pos / (10000.0 ** (2 * (i // 2) / d_model))
    pe[0, :, 0::2] = torch.sin(angle[:, 0::2])
    pe[0, :, 1::2] = torch.cos(angle[:, 1::2])
    return pe


def _sinusoidal_pe_2d(h, w, d_model, device):
    """生成 2D 正弦位置编码 (1, h*w, d_model), 前半编码行, 后半编码列."""
    half = d_model // 2
    rest = d_model - half
    pe_row = _sinusoidal_pe(h, half, device).squeeze(0)
    pe_col = _sinusoidal_pe(w, rest, device).squeeze(0)
    pe = torch.cat(
        [pe_row.unsqueeze(1).expand(-1, w, -1),
         pe_col.unsqueeze(0).expand(h, -1, -1)],
        dim=-1,
    )
    return pe.reshape(1, h * w, d_model)


# ═══════════════════════════ 1D 版本 ═══════════════════════════


class HASAWeightTransformer1D(nn.Module):
    """Transformer-based HASA 权重网络 (1D): 输出 lambda_tv / lambda_wav / alpha."""

    def __init__(self, d_model=32, nhead=4, num_layers=2, mlp_ratio=2.0):
        super().__init__()
        self.d_model = d_model
        self.proj_in = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        self.head_tv = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.head_wav = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.head_alpha = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        nn.init.constant_(self.head_tv[0].bias, -4.0)
        nn.init.constant_(self.head_wav[0].bias, -4.0)
        nn.init.constant_(self.head_alpha[0].bias, 0.0)

    def forward(self, x):
        # x: (B, 1, L)
        B, _, L = x.shape
        tok = self.proj_in(x.permute(0, 2, 1))                 # (B, L, D)
        tok = tok + _sinusoidal_pe(L, self.d_model, x.device)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)                                    # (B, L, D)

        tv    = self.head_tv(tok).permute(0, 2, 1)             # (B, 1, L)
        wav   = self.head_wav(tok).permute(0, 2, 1)
        alpha = self.head_alpha(tok).permute(0, 2, 1)
        return tv, wav, alpha


class ISTAProxTransformer1D_Dual(nn.Module):
    """Transformer-based 双分支 ISTA Prox (1D).

    每个分支: Linear embed → Transformer enc → soft-threshold → Transformer dec → Linear proj
    """

    def __init__(self, d_model=32, nhead=4, num_layers=2, mlp_ratio=2.0):
        super().__init__()
        self.d_model = d_model

        # TV branch
        self.enc_proj_tv = nn.Linear(1, d_model)
        self.enc_tv = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)]
        )
        self.enc_norm_tv = nn.LayerNorm(d_model)
        self.dec_tv = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)]
        )
        self.dec_norm_tv = nn.LayerNorm(d_model)
        self.dec_proj_tv = nn.Linear(d_model, 1)

        # WAV branch
        self.enc_proj_wav = nn.Linear(1, d_model)
        self.enc_wav = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)]
        )
        self.enc_norm_wav = nn.LayerNorm(d_model)
        self.dec_wav = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio) for _ in range(num_layers)]
        )
        self.dec_norm_wav = nn.LayerNorm(d_model)
        self.dec_proj_wav = nn.Linear(d_model, 1)

    def _encode(self, z, proj, enc_blocks, norm):
        """(B, 1, L) → (B, L, D)."""
        tok = proj(z.permute(0, 2, 1))                         # (B, L, D)
        tok = tok + _sinusoidal_pe(tok.shape[1], self.d_model, z.device)
        for blk in enc_blocks:
            tok = blk(tok)
        return norm(tok)

    @staticmethod
    def _decode(feat, dec_blocks, norm, proj):
        """(B, L, D) → (B, 1, L)."""
        tok = feat
        for blk in dec_blocks:
            tok = blk(tok)
        tok = norm(tok)
        return proj(tok).permute(0, 2, 1)                      # (B, 1, L)

    @staticmethod
    def _expand_thr(thr, feat):
        # thr: (B, 1, L), feat: (B, L, D) → (B, L, D)
        return thr.permute(0, 2, 1).expand_as(feat)

    def _branch_forward(self, z, thr, proj, enc, enc_norm,
                        dec, dec_norm, dec_proj, return_symloss=False):
        feat = self._encode(z, proj, enc, enc_norm)             # (B, L, D)
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


class HASA_FISTA_Transformer_Block_1D(nn.Module):
    """单层 Transformer-FISTA block (1D)."""

    def __init__(self, weight_ctor, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProxTransformer1D_Dual(
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


class HASA_FISTA_Transformer_Net_1D(nn.Module):
    """1D Transformer-based HASA-FISTA 网络."""

    def __init__(self, layer_num=12, hasa_ctor=None,
                 d_model=32, nhead=4, num_layers=2):
        super().__init__()
        if hasa_ctor is None:
            hasa_ctor = lambda: HASAWeightTransformer1D(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
            )

        self.blocks = nn.ModuleList([
            HASA_FISTA_Transformer_Block_1D(
                hasa_ctor, d_model=d_model, nhead=nhead, num_layers=num_layers,
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


# ═══════════════════════ 2D Swin Window Attention 组件 ═══════════════════════


def _window_partition(x, ws):
    """将 2D 特征图划分为不重叠窗口: (B, H, W, D) → (B*nW, ws*ws, D)."""
    B, H, W, D = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, D)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, ws * ws, D)


def _window_reverse(windows, ws, H, W):
    """将窗口合并还原为 2D 特征图: (B*nW, ws*ws, D) → (B, H, W, D)."""
    nH, nW = H // ws, W // ws
    B = windows.shape[0] // (nH * nW)
    D = windows.shape[-1]
    x = windows.view(B, nH, nW, ws, ws, D)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, D)


def _compute_swin_mask(H, W, ws, shift_size, device):
    """为 shifted window attention 计算注意力掩码: (nW, ws*ws, ws*ws)."""
    img_mask = torch.zeros(1, H, W, 1, device=device)
    h_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h_s in h_slices:
        for w_s in w_slices:
            img_mask[:, h_s, w_s, :] = cnt
            cnt += 1
    mask_windows = _window_partition(img_mask, ws).squeeze(-1)   # (nW, ws*ws)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class WindowAttention(nn.Module):
    """窗口内多头自注意力 (支持 shifted window mask)."""

    def __init__(self, d_model, nhead=4, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B*nW, ws*ws, D), mask: (nW, ws*ws, ws*ws) or None."""
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


class SwinBlock2D(nn.Module):
    """Swin Transformer block: (shifted) window attention + FFN.

    输入输出均为 (B, N, D), 需要额外传入 h, w (patch grid 尺寸).
    内部自动 pad 到 win_size 整数倍, 处理后裁剪回原尺寸.
    """

    def __init__(self, d_model, nhead=4, mlp_ratio=2.0, dropout=0.0,
                 win_size=8, shift=False):
        super().__init__()
        self.win_size = win_size
        self.shift_size = win_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = WindowAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )
        self._mask_cache = {}

    def _get_mask(self, Hp, Wp, device):
        key = (Hp, Wp, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = _compute_swin_mask(
                Hp, Wp, self.win_size, self.shift_size, device,
            )
        return self._mask_cache[key]

    def forward(self, x, h, w):
        """x: (B, h*w, D)."""
        B, N, D = x.shape
        ws = self.win_size

        x_2d = x.view(B, h, w, D)

        pad_b = (ws - h % ws) % ws
        pad_r = (ws - w % ws) % ws
        if pad_b > 0 or pad_r > 0:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = h + pad_b, w + pad_r

        shortcut = x_2d
        x_2d = self.norm1(x_2d)

        if self.shift_size > 0:
            x_2d = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = self._get_mask(Hp, Wp, x.device)
        else:
            mask = None

        windows = _window_partition(x_2d, ws)
        windows = self.attn(windows, mask=mask)
        x_2d = _window_reverse(windows, ws, Hp, Wp)

        if self.shift_size > 0:
            x_2d = torch.roll(x_2d, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x_2d = shortcut + x_2d
        x_2d = x_2d + self.ffn(self.norm2(x_2d))

        if pad_b > 0 or pad_r > 0:
            x_2d = x_2d[:, :h, :w, :].contiguous()

        return x_2d.reshape(B, N, D)


def _make_swin_blocks(num_layers, d_model, nhead, mlp_ratio, win_size):
    """创建交替 regular / shifted 窗口的 Swin block 序列."""
    return nn.ModuleList([
        SwinBlock2D(d_model, nhead, mlp_ratio,
                    win_size=win_size, shift=(i % 2 == 1))
        for i in range(num_layers)
    ])


# ═══════════════════════════ 2D 版本 ═══════════════════════════


class PatchEmbed2D(nn.Module):
    """2D Patch Embedding: (B, 1, H, W) → (B, N, D).

    自动 pad 到 patch_size 的整数倍。
    """

    def __init__(self, patch_size=4, d_model=64):
        super().__init__()
        self.P = patch_size
        self.proj = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        P = self.P
        _, _, H, W = x.shape
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        return x.flatten(2).permute(0, 2, 1)


class PatchUnembed2D(nn.Module):
    """2D Patch Unembedding: (B, N, D) → (B, 1, H, W)."""

    def __init__(self, patch_size=4, d_model=64):
        super().__init__()
        self.P = patch_size
        self.proj = nn.Linear(d_model, patch_size * patch_size)

    def forward(self, tok, h, w, orig_H, orig_W):
        P = self.P
        B = tok.shape[0]
        x = self.proj(tok)                                      # (B, N, P*P)
        x = x.reshape(B, h, w, P, P)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, 1, h * P, w * P)
        return x[:, :, :orig_H, :orig_W]


class HASAWeightTransformer2D(nn.Module):
    """Swin Transformer-based HASA 权重网络 (2D, window attention)."""

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 mlp_ratio=2.0, patch_size=2, win_size=8):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        self.embed = PatchEmbed2D(patch_size, d_model)
        self.blocks = _make_swin_blocks(num_layers, d_model, nhead, mlp_ratio, win_size)
        self.norm = nn.LayerNorm(d_model)

        pp = patch_size * patch_size
        self.head_tv = nn.Sequential(nn.Linear(d_model, pp), nn.Softplus())
        self.head_wav = nn.Sequential(nn.Linear(d_model, pp), nn.Softplus())
        self.head_alpha = nn.Sequential(nn.Linear(d_model, pp), nn.Sigmoid())

        nn.init.constant_(self.head_tv[0].bias, -4.0)
        nn.init.constant_(self.head_wav[0].bias, -4.0)
        nn.init.constant_(self.head_alpha[0].bias, 0.0)

    def _tokens_to_image(self, tok, h, w, orig_H, orig_W):
        """(B, N, P*P) → (B, 1, H, W), 裁剪到原始尺寸."""
        P = self.patch_size
        B = tok.shape[0]
        x = tok.reshape(B, h, w, P, P)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, 1, h * P, w * P)
        return x[:, :, :orig_H, :orig_W]

    def forward(self, x):
        B, _, H, W = x.shape
        P = self.patch_size
        h = (H + P - 1) // P
        w = (W + P - 1) // P

        tok = self.embed(x)
        tok = tok + _sinusoidal_pe_2d(h, w, self.d_model, x.device)
        for blk in self.blocks:
            tok = blk(tok, h, w)
        tok = self.norm(tok)

        tv    = self._tokens_to_image(self.head_tv(tok), h, w, H, W)
        wav   = self._tokens_to_image(self.head_wav(tok), h, w, H, W)
        alpha = self._tokens_to_image(self.head_alpha(tok), h, w, H, W)
        return tv, wav, alpha


class ISTAProxTransformer2D_Dual(nn.Module):
    """Swin Transformer-based 双分支 ISTA Prox (2D, window attention).

    每个分支: PatchEmbed → Swin enc → soft-threshold → Swin dec → PatchUnembed
    阈值通过 avg_pool 池化到 patch 级别后在特征维度展开。
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 mlp_ratio=2.0, patch_size=2, win_size=8):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # TV branch
        self.embed_tv = PatchEmbed2D(patch_size, d_model)
        self.enc_tv = _make_swin_blocks(num_layers, d_model, nhead, mlp_ratio, win_size)
        self.enc_norm_tv = nn.LayerNorm(d_model)
        self.dec_tv = _make_swin_blocks(num_layers, d_model, nhead, mlp_ratio, win_size)
        self.dec_norm_tv = nn.LayerNorm(d_model)
        self.unembed_tv = PatchUnembed2D(patch_size, d_model)

        # WAV branch
        self.embed_wav = PatchEmbed2D(patch_size, d_model)
        self.enc_wav = _make_swin_blocks(num_layers, d_model, nhead, mlp_ratio, win_size)
        self.enc_norm_wav = nn.LayerNorm(d_model)
        self.dec_wav = _make_swin_blocks(num_layers, d_model, nhead, mlp_ratio, win_size)
        self.dec_norm_wav = nn.LayerNorm(d_model)
        self.unembed_wav = PatchUnembed2D(patch_size, d_model)

    def _encode(self, z, embed, enc_blocks, norm):
        """(B, 1, H, W) → (B, N, D), h, w."""
        P = self.patch_size
        _, _, H, W = z.shape
        h = (H + P - 1) // P
        w = (W + P - 1) // P
        tok = embed(z)
        tok = tok + _sinusoidal_pe_2d(h, w, self.d_model, z.device)
        for blk in enc_blocks:
            tok = blk(tok, h, w)
        return norm(tok), h, w

    @staticmethod
    def _decode(feat, dec_blocks, norm, unembed, h, w, orig_H, orig_W):
        """(B, N, D) → (B, 1, H, W)."""
        tok = feat
        for blk in dec_blocks:
            tok = blk(tok, h, w)
        tok = norm(tok)
        return unembed(tok, h, w, orig_H, orig_W)

    def _pool_thr(self, thr, h, w):
        """将像素级阈值池化到 patch 级: (B,1,H,W) → (B,N,1)."""
        P = self.patch_size
        _, _, H, W = thr.shape
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            thr = F.pad(thr, (0, pad_w, 0, pad_h), mode="replicate")
        pooled = F.avg_pool2d(thr, kernel_size=P, stride=P)
        return pooled.flatten(2).permute(0, 2, 1)

    def _branch_forward(self, z, thr, embed, enc, enc_norm,
                        dec, dec_norm, unembed, return_symloss=False):
        _, _, H, W = z.shape
        feat, h, w = self._encode(z, embed, enc, enc_norm)
        thr_tok = self._pool_thr(thr, h, w)
        feat_shrink = soft_threshold(feat, thr_tok.expand_as(feat))
        x_next = self._decode(
            feat_shrink, dec, dec_norm, unembed, h, w, H, W,
        )

        if not return_symloss:
            return x_next, None
        x_est = self._decode(feat, dec, dec_norm, unembed, h, w, H, W)
        return x_next, x_est - z

    def forward(self, z, thr_tv, thr_wav, alpha=None,
                return_symloss=False, return_branches=False):
        x_tv, sym_tv = self._branch_forward(
            z, thr_tv,
            self.embed_tv, self.enc_tv, self.enc_norm_tv,
            self.dec_tv, self.dec_norm_tv, self.unembed_tv,
            return_symloss,
        )
        x_wav, sym_wav = self._branch_forward(
            z, thr_wav,
            self.embed_wav, self.enc_wav, self.enc_norm_wav,
            self.dec_wav, self.dec_norm_wav, self.unembed_wav,
            return_symloss,
        )

        if alpha is None:
            x_next = 0.5 * (x_tv + x_wav)
        else:
            x_next = alpha * x_tv + (1.0 - alpha) * x_wav

        if not return_branches:
            return x_next, sym_tv, sym_wav, None, None
        return x_next, sym_tv, sym_wav, x_tv, x_wav


class HASA_FISTA_Transformer_Block_2D(nn.Module):
    """单层 Swin-FISTA block (2D)."""

    def __init__(self, weight_ctor, d_model=64, nhead=4,
                 num_layers=2, patch_size=2, win_size=8):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProxTransformer2D_Dual(
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, patch_size=patch_size, win_size=win_size,
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


class HASA_FISTA_Transformer_Net_2D(nn.Module):
    """2D Swin Transformer-based HASA-FISTA 网络."""

    def __init__(self, layer_num=12, hasa_ctor=None,
                 d_model=64, nhead=4, num_layers=2,
                 patch_size=2, win_size=8):
        super().__init__()
        if hasa_ctor is None:
            hasa_ctor = lambda: HASAWeightTransformer2D(
                d_model=d_model, nhead=nhead,
                num_layers=num_layers, patch_size=patch_size,
                win_size=win_size,
            )

        self.blocks = nn.ModuleList([
            HASA_FISTA_Transformer_Block_2D(
                hasa_ctor, d_model=d_model, nhead=nhead,
                num_layers=num_layers, patch_size=patch_size,
                win_size=win_size,
            )
            for _ in range(layer_num)
        ])

    def forward(self, y, op, x0=None,
                return_aux=False, return_symloss=False, return_branches=False):
        if x0 is None:
            x0 = op.At(y)

        scale = x0.abs().amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-6)
        x = x0 / scale
        v = x0 / scale
        y_scaled = y / scale.view(scale.shape[0], 1, 1)

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
