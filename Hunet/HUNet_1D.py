"""HUNet-1D: Homotopy Unfolding Network for 1D Ultrasound CS Reconstruction.

改造自 HUNet (CVPR 2025) 的 2D 图像 CS 网络，适配 1D 超声信号。
核心保留:
  - Homotopy 展开: γ 逐阶段衰减的软阈值
  - 多尺度 U-Net encoder/decoder (1D Conv + SwinBlock1D)
  - DFFM 跨阶段融合 (concat + DMlp + SE)
替换:
  - 可学习采样 Φ → 固定 MaskedRFFT1D
  - 2D Conv/Patch → 1D Conv
  - PSA → SwinBlock1D (局部窗口注意力)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Utils"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import soft_threshold


# ═══════════════════════ 1D Swin 窗口注意力 ═══════════════════════


def _window_partition_1d(x, ws):
    """(B, L, D) -> (B*nW, ws, D)"""
    B, L, D = x.shape
    nW = L // ws
    return x.view(B, nW, ws, D).reshape(B * nW, ws, D)


def _window_reverse_1d(windows, ws, L):
    """(B*nW, ws, D) -> (B, L, D)"""
    nW = L // ws
    B = windows.shape[0] // nW
    D = windows.shape[-1]
    return windows.view(B, nW, ws, D).reshape(B, L, D)


def _compute_swin_mask_1d(L, ws, shift_size, device):
    mask = torch.zeros(1, L, 1, device=device)
    slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
    cnt = 0
    for s in slices:
        mask[:, s, :] = cnt
        cnt += 1
    mask_windows = _window_partition_1d(mask, ws).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class WindowAttention1D(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
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
    def __init__(self, d_model, nhead=4, mlp_ratio=2.0, dropout=0.0,
                 window_size=32, shift=False):
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
                Lp, self.window_size, self.shift_size, device)
        return self._mask_cache[key]

    def forward(self, x):
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


def _make_swin_blocks(num_layers, d_model, nhead, mlp_ratio, window_size):
    return nn.ModuleList([
        SwinBlock1D(d_model, nhead, mlp_ratio,
                    window_size=window_size, shift=(i % 2 == 1))
        for i in range(num_layers)
    ])


# ═══════════════════════ 多尺度 U-Net 组件 ═══════════════════════


class Head1D(nn.Module):
    """(B, 1, L) -> (B, embed_dim, L) with learnable residual scaling."""

    def __init__(self, embed_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(1, embed_dim // 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim // 3, embed_dim // 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim // 3, embed_dim, 3, padding=1),
        )
        self.alpha = nn.Parameter(1e-2 * torch.ones(1, embed_dim, 1))

    def forward(self, x):
        return x + self.alpha * self.block(x)


class Tail1D(nn.Module):
    """(B, embed_dim, L) -> (B, 1, L)."""

    def __init__(self, embed_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim // 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim // 3, embed_dim // 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim // 3, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1D(nn.Module):
    """(B, L, C) -> (B, L//2, 2C).

    1D analog of HUNet's 2D Downsample:
    Conv1d(C, C, 3) then interleave-merge adjacent samples to double channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, bias=False)

    def forward(self, x):
        B, L, C = x.shape
        x = self.conv(x.permute(0, 2, 1))          # (B, C, L)
        x = x.reshape(B, C, L // 2, 2)              # split adjacent
        x = x.permute(0, 1, 3, 2).reshape(B, C * 2, L // 2)
        return x.permute(0, 2, 1)                    # (B, L//2, 2C)


class Upsample1D(nn.Module):
    """(B, L, C) -> (B, 2L, C//2).

    1D analog of HUNet's 2D Upsample:
    Conv1d(C, C, 3) then deinterleave channels to double spatial length.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, bias=False)

    def forward(self, x):
        B, L, C = x.shape
        x = self.conv(x.permute(0, 2, 1))            # (B, C, L)
        x = x.reshape(B, C // 2, 2, L)               # split channels
        x = x.permute(0, 1, 3, 2).reshape(B, C // 2, L * 2)
        return x.permute(0, 2, 1)                     # (B, 2L, C//2)


# ═══════════════════════ SE + DMlp ═══════════════════════


class SELayer1D(nn.Module):
    """Squeeze-and-Excitation for 1D: (B, C, L) -> (B, C, L)."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        mid = max(channel // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(mid, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(channel, mid, bias=False)

    def forward(self, x):
        B, C, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(self.proj(y)).view(B, C, 1)
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


# ═══════════════════════ Homotopy Stage ═══════════════════════


class HomotopyStage1D(nn.Module):
    """Single homotopy unfolding stage.

    Gradient step -> Head -> multi-scale Encoder -> soft threshold -> Decoder -> Tail -> residual.
    """

    def __init__(self, stage_no, embed_dim=32, depth=4, nhead=4,
                 window_size=32, mlp_ratio=2.0, swin_depth=2):
        super().__init__()
        self.depth = depth
        self.n_downsample = depth // 2 - 1

        self.lambda_ = nn.Parameter(torch.tensor(0.5))

        bottleneck_dim = embed_dim * 2 ** self.n_downsample
        self.gamma = nn.Parameter(
            torch.full((1, 1, bottleneck_dim), 0.1 ** (stage_no + 1)))

        self.head = Head1D(embed_dim)
        self.tail = Tail1D(embed_dim)

        self.encoder = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.up_sample = nn.ModuleList()

        for i in range(depth):
            dim_i = embed_dim * 2 ** (i // 2)
            self.encoder.append(
                _make_swin_blocks(swin_depth, dim_i, nhead, mlp_ratio, window_size))
            if i % 2 == 1 and i != depth - 1:
                self.down_sample.append(Downsample1D(dim_i))

        dec_list = []
        up_list = []
        for i in range(depth):
            dim_i = embed_dim * 2 ** (i // 2)
            dec_list.append(
                _make_swin_blocks(swin_depth, dim_i, nhead, mlp_ratio, window_size))
            if i % 2 == 1 and i != 1:
                up_list.append(Upsample1D(dim_i))

        up_list.reverse()
        dec_list.reverse()
        self.decoder_blocks = nn.ModuleList(dec_list)
        self.up_sample = nn.ModuleList(up_list)

    def forward(self, x, y, op):
        """
        x: (B, 1, L) current reconstruction
        y: (B, K) complex measurements
        op: MaskedRFFT1D
        Returns: x_next (B, 1, L), x_multi (B, L, embed_dim)
        """
        res = x
        lam = F.softplus(self.lambda_)
        r = x - lam * op.At(op.A(x) - y)

        feat = self.head(r)                          # (B, embed_dim, L)
        feat_seq = feat.permute(0, 2, 1)             # (B, L, embed_dim)
        L = feat_seq.shape[1]

        factor = 2 ** max(self.n_downsample, 1)
        pad_r = (factor - L % factor) % factor
        if pad_r > 0:
            feat_seq = F.pad(feat_seq, (0, 0, 0, pad_r))

        # ---- Encoder ----
        x_ms = []
        cnt = 0
        for i, enc_blocks in enumerate(self.encoder):
            for blk in enc_blocks:
                feat_seq = blk(feat_seq)
            x_ms.append(feat_seq)
            if i % 2 == 1 and i != self.depth - 1:
                feat_seq = self.down_sample[cnt](feat_seq)
                cnt += 1

        # ---- Decoder (reversed) ----
        out = torch.zeros_like(x_ms[-1])
        x_ms.reverse()
        cnt = 0
        for i, (x_e, dec_blocks) in enumerate(zip(x_ms, self.decoder_blocks)):
            if i == 0:
                B_sz = x_e.shape[0]
                x_e = soft_threshold(
                    x_e, self.gamma.abs().expand(B_sz, -1, -1))
            combined = out + x_e
            for blk in dec_blocks:
                combined = blk(combined)
            out = combined
            if i % 2 == 1 and i != self.depth - 1:
                out = self.up_sample[cnt](out)
                cnt += 1

        out = out[:, :L, :]
        x_multi = out                                # (B, L, embed_dim)
        out_1d = self.tail(out.permute(0, 2, 1))     # (B, 1, L)
        x_next = out_1d + res

        return x_next, x_multi


# ═══════════════════════ DFFM 跨阶段融合 ═══════════════════════


class DFFM1D(nn.Module):
    """Dual-path Feature Fusion Module (1D).

    Fuses all stage outputs (1-ch each) and multi-scale features into final reconstruction.
    """

    def __init__(self, num_stages, embed_dim):
        super().__init__()
        self.conv = nn.Conv1d(num_stages, embed_dim, 3, padding=1)
        self.dmlp = DMlp1D(embed_dim)
        self.se = SELayer1D(embed_dim)
        self.tail = Tail1D(embed_dim)

    def forward(self, x_stages, x_multi_sum, weight):
        """
        x_stages: (B, L, num_stages)
        x_multi_sum: (B, embed_dim, L)
        weight: scalar parameter
        """
        x = self.conv(x_stages.permute(0, 2, 1))     # (B, embed_dim, L)
        shortcut = x
        x = self.dmlp(x)
        x = self.se(x)
        x = x + shortcut
        return self.tail(x + weight * x_multi_sum)    # (B, 1, L)


# ═══════════════════════ HUNet1D ═══════════════════════


class HUNet1D(nn.Module):
    """HUNet-1D: Homotopy Unfolding Network for 1D Ultrasound CS.

    Interface: forward(y, op, return_aux=True) -> (x_hat, aux_list),
    compatible with CombinedLoss and the shared training pipeline.
    """

    def __init__(self, num_stages=7, depth=4, embed_dim=32,
                 nhead=4, window_size=32, mlp_ratio=2.0, swin_depth=2):
        super().__init__()
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.tensor(0.2))

        self.stages = nn.ModuleList([
            HomotopyStage1D(
                stage_no=i, embed_dim=embed_dim, depth=depth,
                nhead=nhead, window_size=window_size,
                mlp_ratio=mlp_ratio, swin_depth=swin_depth,
            )
            for i in range(num_stages)
        ])

        self.dffm = DFFM1D(num_stages, embed_dim)

    def forward(self, y, op, x0=None, return_aux=False, **kwargs):
        if x0 is None:
            x0 = op.At(y)

        scale = x0.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        x = x0 / scale
        y_scaled = y / scale.view(scale.shape[0], 1)

        stage_outputs = []
        x_multi_sum = 0
        aux_list = []

        for i, stage in enumerate(self.stages):
            x, x_multi = stage(x, y_scaled, op)
            stage_outputs.append(x.squeeze(1))
            x_multi_sum = x_multi_sum + x_multi.permute(0, 2, 1)

            if return_aux:
                aux_list.append({
                    "rho1": F.softplus(stage.lambda_).detach(),
                    "rho2": F.softplus(stage.lambda_).detach(),
                    "eta": stage.gamma.abs().mean().detach(),
                    "gamma": torch.tensor(0.0, device=x.device),
                    "constraint_wav": torch.zeros(1, device=x.device),
                    "constraint_tv": torch.zeros(1, device=x.device),
                })

        x_cat = torch.stack(stage_outputs, dim=-1)
        x_fused = self.dffm(x_cat, x_multi_sum, self.weight)
        x_out = x_fused * scale

        if return_aux:
            return x_out, aux_list
        return x_out
