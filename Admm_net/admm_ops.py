"""HASA-ADMM-Net 共享算子

提供 1D / 2D 通用的基础运算，避免跨模块重复定义。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_threshold(x, thr):
    """逐元素软阈值: sign(x) * max(|x| - thr, 0)"""
    return torch.sign(x) * F.relu(torch.abs(x) - thr)


# ======================== Hilbert 包络 ========================

_hilbert_filter_cache: dict = {}


def _hilbert_filter(N: int, device, dtype):
    """构造 Hilbert 频域滤波器 h[k], 按 (N, device, dtype) 缓存."""
    key = (N, device, dtype)
    h = _hilbert_filter_cache.get(key)
    if h is None:
        h = torch.zeros(N, device=device, dtype=dtype)
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0
        if N % 2 == 0:
            h[N // 2] = 1.0
        _hilbert_filter_cache[key] = h
    return h


def hilbert_envelope(x):
    """希尔伯特包络, 自动适配 1D (B,1,N) 和 2D (B,1,H,W)."""
    if x.dim() == 3:
        return _hilbert_envelope_1d(x)
    return _hilbert_envelope_2d(x)


def _hilbert_envelope_1d(x):
    """x: (B, 1, N) → (B, 1, N)"""
    x_flat = x.squeeze(1)
    X = torch.fft.fft(x_flat, dim=-1)
    h = _hilbert_filter(X.shape[-1], x.device, x.dtype)
    analytic = torch.fft.ifft(X * h.unsqueeze(0), dim=-1)
    envelope = torch.sqrt(analytic.real ** 2 + analytic.imag ** 2 + 1e-8)
    return envelope.unsqueeze(1)


def _hilbert_envelope_2d(x):
    """x: (B, 1, H, W) → (B, 1, H, W)  逐行 Hilbert."""
    B, C, H, W = x.shape
    x_flat = x.reshape(B * H, W)
    X_f = torch.fft.fft(x_flat, dim=-1)
    h = _hilbert_filter(W, x.device, x.dtype)
    analytic = torch.fft.ifft(X_f * h.unsqueeze(0), dim=-1)
    envelope = torch.sqrt(analytic.real ** 2 + analytic.imag ** 2 + 1e-8)
    return envelope.reshape(B, C, H, W)


# ======================== HASA 自适应权重工厂 ========================

def build_hasa_weight(ndim: int, hidden_ch: int = 16, num_layers: int = 2,
                      inner_ks: int = 5):
    """构建 HASA 自适应权重生成器.

    Args:
        ndim: 1 → Conv1d, 2 → Conv2d
        hidden_ch: 隐藏通道数
        num_layers: 卷积层数
        inner_ks: 第 2 层起的卷积核大小 (默认 5, 与原始架构一致)

    第 1 层固定 3×3 (输入投影), 后续层使用 inner_ks.
    默认 (hidden_ch=16, num_layers=2, inner_ks=5) 与原始 HASA 完全一致.
    """
    Conv = nn.Conv1d if ndim == 1 else nn.Conv2d
    layers = [Conv(1, hidden_ch, 3, padding=1), nn.ReLU(inplace=True)]
    inner_pad = inner_ks // 2
    for _ in range(num_layers - 1):
        layers.extend([
            Conv(hidden_ch, hidden_ch, inner_ks, padding=inner_pad),
            nn.ReLU(inplace=True),
        ])
    feat_net = nn.Sequential(*layers)
    head_wav = nn.Sequential(Conv(hidden_ch, 1, 1), nn.Softplus())
    head_tv = nn.Sequential(Conv(hidden_ch, 1, 1), nn.Softplus())
    for head in (head_wav, head_tv):
        nn.init.constant_(head[0].bias, -4.0)

    class _HASAWeight(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_net = feat_net
            self.head_wav = head_wav
            self.head_tv = head_tv

        def forward(self, x):
            feat = self.feat_net(x)
            return self.head_wav(feat), self.head_tv(feat)

    return _HASAWeight()
