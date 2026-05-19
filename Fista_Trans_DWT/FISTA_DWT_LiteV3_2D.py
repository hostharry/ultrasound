"""FISTA-DWT-LiteV3-2D (帧/patch).

V3 = Lite-2D 的独立子带 WAV enc/dec + Pre-Norm GroupNorm.

V3 相对 Lite-2D 的主要结构改动:
  - ConvResBlock2D 从 (Conv → GELU → Conv) + skip
                  改为 (GN → GELU → Conv → GN → GELU → Conv) + skip  (Pre-Norm)
  - WAV 分支恢复为 4 个 DWT 子带各自独立 enc/dec; HASA / DFFM / proj 层均不动.

设计动机 (详见 docs):
  V2 中 WAV 共享 dec 在 ep5 因单 batch 梯度异常被 Adam 二阶矩永久放大
  (676× 跳跃, effective lr 掉 34×). 根因是 ConvResBlock 无任何 normalization,
  激活幅度可随训练自由漂移, 配合 4-cascade FISTA 形成 cascade 放大.
  GN(Pre-Norm) 限制激活 magnitude 同时保留残差路径的 scale 守恒, 切断该链路.

参数对比 (4 层, d_model=32, num_blocks=2, conv_ks=5, num_groups=8):
  Lite_2D    : 4.18M  (4 套独立 enc/dec, 无 norm)
  LiteV2_2D  : 1.73M  (共享 enc/dec, 无 norm)
  LiteV3_2D  : ~4.2M  (4 套独立 enc/dec, GN)

参见: FISTA_DWT_LiteV2_2D.py (V2 参考实现).
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "Utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from ops import smooth_soft_threshold

_N_BANDS = 4  # LL, LH, HL, HH

LAMBDA_FLOOR = 5e-3
DEFAULT_GN_GROUPS = 8


def _hasa_groups(channels, target_groups=4):
    """选择能整除 channels 的最大 GN 组数, 不超过 target_groups."""
    g = min(target_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class MultiScaleHASA2D(nn.Module):
    """重参数化 HASA: 双分支 + 分支内 GN + tanh 调制.

    输出: lam = softplus(log_base) * (1 + scale * tanh(head(feat))) + floor
      - log_base 是可学标量 (per-layer per-head), 控制全局阈值水平
      - tanh ∈ [-1, 1], 配合 scale=0.5 让 (1 + 0.5*tanh) ∈ [0.5, 1.5] 必正
      - feat 经 GN 控尺度, head 无 bias 不可走捷径
      - alpha head 已删除 (V3 prox 不读 alpha, 始终走 0.5 等权)
    """

    def __init__(self, hidden_ch=16, num_layers=2, inner_ks=5, context_ks=3,
                 context_dilation=3, base_init=-2.0, scale=0.5):
        super().__init__()
        inner_pad = inner_ks // 2
        ctx_pad = context_dilation * (context_ks // 2)
        gn_g = _hasa_groups(hidden_ch, target_groups=4)

        local_layers = [
            nn.Conv2d(1, hidden_ch, 3, padding=1),
            nn.GroupNorm(gn_g, hidden_ch),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            local_layers.extend([
                nn.Conv2d(hidden_ch, hidden_ch, inner_ks, padding=inner_pad),
                nn.GroupNorm(gn_g, hidden_ch),
                nn.GELU(),
            ])
        self.local_net = nn.Sequential(*local_layers)

        ctx_layers = [
            nn.Conv2d(1, hidden_ch, 3, padding=1),
            nn.GroupNorm(gn_g, hidden_ch),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            ctx_layers.extend([
                nn.Conv2d(hidden_ch, hidden_ch, context_ks,
                          padding=ctx_pad, dilation=context_dilation),
                nn.GroupNorm(gn_g, hidden_ch),
                nn.GELU(),
            ])
        self.context_net = nn.Sequential(*ctx_layers)

        self.merge = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch, 1),
            nn.GroupNorm(gn_g, hidden_ch),
            nn.GELU(),
        )

        # head: 1x1, 无 bias, 小初始化, head 后绝不接 norm
        self.head_tv = nn.Conv2d(hidden_ch, 1, 1, bias=False)
        self.head_wav = nn.Conv2d(hidden_ch, 1, 1, bias=False)
        nn.init.normal_(self.head_tv.weight, std=0.1)
        nn.init.normal_(self.head_wav.weight, std=0.1)

        # 全局阈值水平 (可学标量)
        self.log_base_tv = nn.Parameter(torch.tensor(float(base_init)))
        self.log_base_wav = nn.Parameter(torch.tensor(float(base_init)))
        # 调制幅度 (固定标量, 第一版不学)
        self.scale = float(scale)

    def forward(self, x):
        f_local = self.local_net(x)
        f_ctx = self.context_net(x)
        feat = self.merge(torch.cat([f_local, f_ctx], dim=1))

        mod_tv = torch.tanh(self.head_tv(feat))
        mod_wav = torch.tanh(self.head_wav(feat))
        base_tv = F.softplus(self.log_base_tv)
        base_wav = F.softplus(self.log_base_wav)
        lam_tv = base_tv * (1.0 + self.scale * mod_tv) + LAMBDA_FLOOR
        lam_wav = base_wav * (1.0 + self.scale * mod_wav) + LAMBDA_FLOOR
        return lam_tv, lam_wav


class _UConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3):
        super().__init__()
        pad = ks // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, ks, padding=pad)
        self.conv2 = nn.Conv2d(out_ch, out_ch, ks, padding=pad)
        self.act = nn.GELU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.act(self.conv2(self.act(self.conv1(x))))


class MiniUNetHASA2D(nn.Module):
    """重参数化版: U-Net 主体 (复用 _UConvBlock) + head 改造同 MultiScaleHASA2D.

    注: U-Net 内部 _UConvBlock 不加 GN, 因为 _UConvBlock 是其它模块共享的简单块,
    避免侵入式改动. MiniUNetHASA 自身 skip + 双 down-up 路径已经提供了足够的尺度稳定性.
    """

    def __init__(self, base_ch=16, base_init=-2.0, scale=0.5):
        super().__init__()
        c1, c2, c4 = base_ch, base_ch * 2, base_ch * 4

        self.enc1 = _UConvBlock(1, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = _UConvBlock(c2, c2)
        self.down2 = nn.Conv2d(c2, c4, 3, stride=2, padding=1)

        self.bottleneck = nn.Sequential(
            _UConvBlock(c4, c4),
            _UConvBlock(c4, c4),
        )

        self.up_reduce1 = nn.Conv2d(c4, c2, 1)
        self.dec1 = _UConvBlock(c2 * 2, c2)
        self.up_reduce2 = nn.Conv2d(c2, c1, 1)
        self.dec2 = _UConvBlock(c1 * 2, c1)

        self.head_tv = nn.Conv2d(c1, 1, 1, bias=False)
        self.head_wav = nn.Conv2d(c1, 1, 1, bias=False)
        nn.init.normal_(self.head_tv.weight, std=0.1)
        nn.init.normal_(self.head_wav.weight, std=0.1)

        self.log_base_tv = nn.Parameter(torch.tensor(float(base_init)))
        self.log_base_wav = nn.Parameter(torch.tensor(float(base_init)))
        self.scale = float(scale)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(self.down1(f1))
        f3 = self.down2(f2)

        b = self.bottleneck(f3)

        u2 = F.interpolate(b, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.dec1(torch.cat([self.up_reduce1(u2), f2], dim=1))

        u1 = F.interpolate(u2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        u1 = self.dec2(torch.cat([self.up_reduce2(u1), f1], dim=1))

        mod_tv = torch.tanh(self.head_tv(u1))
        mod_wav = torch.tanh(self.head_wav(u1))
        base_tv = F.softplus(self.log_base_tv)
        base_wav = F.softplus(self.log_base_wav)
        lam_tv = base_tv * (1.0 + self.scale * mod_tv) + LAMBDA_FLOOR
        lam_wav = base_wav * (1.0 + self.scale * mod_wav) + LAMBDA_FLOOR
        return lam_tv, lam_wav


class HaarDWT2D(nn.Module):
    """正交 2D Haar 单层: (B,1,H,W) -> (B,4,H',W'), 逆变换对称."""

    def __init__(self):
        super().__init__()
        s = 0.5
        filters = torch.tensor([
            [[s,  s], [s,  s]],
            [[s,  s], [-s, -s]],
            [[s, -s], [s, -s]],
            [[s, -s], [-s, s]],
        ]).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x):
        _, _, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return F.conv2d(x, self.filters, stride=2)

    def inverse(self, w, out_shape=None):
        result = F.conv_transpose2d(w, self.filters, stride=2)
        if out_shape is not None:
            result = result[..., :out_shape[0], :out_shape[1]]
        return result


class ConvResBlock2D(nn.Module):
    """Pre-Norm 残差块: x + Conv(GELU(GN(Conv(GELU(GN(x)))))).

    与 V2 的区别:
      V2:  x + Conv(GELU(Conv(x)))                   ← 无 norm
      V3:  x + Conv(GELU(GN(Conv(GELU(GN(x))))))     ← Pre-Norm GroupNorm
    """

    def __init__(self, channels, kernel_size=5, num_groups=DEFAULT_GN_GROUPS):
        super().__init__()
        pad = kernel_size // 2
        eff_g = min(num_groups, channels)
        while channels % eff_g != 0:
            eff_g -= 1
        self.norm1 = nn.GroupNorm(eff_g, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(eff_g, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.act = nn.GELU()

        # Fixup / LayerScale 风格的"残差路径零初始化":
        # init 时 h(x) ≡ 0, ResBlock 表现为 identity (x + 0 = x).
        # 训练过程中 conv2 的权重从 0 长出, h 的幅度由数据要求决定,
        # 避免 Pre-Norm 残差块在 init 时 h 比 skip 路径量级大百倍的失衡.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class ConvProxTV2D(nn.Module):
    """TV-like learned prox (接口与 V2 一致, ConvResBlock 自动带 GN).

    forward 返回 (delta_tv, feat_enc, constraint):
      constraint = dec(enc(proj_in(z))) - proj_in(z)  仅当 compute_constraint=True
    """

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5,
                 prox_tau: float = 0.01, num_groups=DEFAULT_GN_GROUPS,
                 layerscale_init: float = 1e-2):
        super().__init__()
        self.prox_tau = float(prox_tau)
        self.proj_in = nn.Conv2d(1, d_model, 1)
        self.enc = nn.Sequential(
            *[ConvResBlock2D(d_model, kernel_size, num_groups=num_groups)
              for _ in range(num_blocks)])
        self.dec = nn.Sequential(
            *[ConvResBlock2D(d_model, kernel_size, num_groups=num_groups)
              for _ in range(num_blocks)])
        self.proj_out = nn.Conv2d(d_model, 1, 1)

        # 不再使用零初始化 proj_out: 改用 LayerScale 缩小 delta 幅度.
        # 优势: 内部权重保持 Kaiming 量级 -> enc/dec 梯度通畅, soft_thresh 输入幅度
        # 完整, lam_tv 收到的梯度信号强; LS_tv 可学, 训练过程自适应放大 delta.
        # 注: ConvResBlock 内的 conv2 零初始化保留 (这是残差路径 identity init,
        # 与 LayerScale 互补不冲突).
        self.layerscale_tv = nn.Parameter(torch.tensor(float(layerscale_init)))

    def forward(self, z, thr_tv, compute_constraint=False):
        feat_in = self.proj_in(z)
        feat_enc = self.enc(feat_in)
        constraint = None
        if compute_constraint:
            feat_sym = self.dec(feat_enc)
            constraint = feat_sym - feat_in
        feat = smooth_soft_threshold(feat_enc, thr_tv.expand_as(feat_enc), tau=self.prox_tau)
        feat = self.dec(feat)
        delta_tv = self.layerscale_tv * self.proj_out(feat)
        return delta_tv, feat_enc, constraint


class DWTConvBranch2D(nn.Module):
    """单层 2D Haar: 4 子带独立 Conv 编码/解码 + Pre-Norm GN.

    这是 0510 Lite-2D 的 WAV 表达形式, 但 ConvResBlock 内部带 GN.
    """

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5, J=1,
                 prox_tau: float = 0.01, num_groups=DEFAULT_GN_GROUPS,
                 layerscale_init: float = 1e-2):
        super().__init__()
        self.prox_tau = float(prox_tau)
        self.d_model = int(d_model)
        self.dwt = HaarDWT2D()

        self.proj_in = nn.ModuleList(
            [nn.Conv2d(1, d_model, 1) for _ in range(_N_BANDS)])
        self.enc = nn.ModuleList([
            nn.Sequential(*[
                ConvResBlock2D(d_model, kernel_size, num_groups=num_groups)
                for _ in range(num_blocks)
            ])
            for _ in range(_N_BANDS)
        ])
        self.dec = nn.ModuleList([
            nn.Sequential(*[
                ConvResBlock2D(d_model, kernel_size, num_groups=num_groups)
                for _ in range(num_blocks)
            ])
            for _ in range(_N_BANDS)
        ])
        self.proj_out = nn.ModuleList(
            [nn.Conv2d(d_model, 1, 1) for _ in range(_N_BANDS)])

        # 不再使用零初始化 4 个 proj_out: 改用单一 LayerScale 缩放 delta_wav.
        # 这样 4 个子带的 enc/dec/proj_out 在 init 时都有完整梯度通路,
        # 避免之前观察到的 layer 1 HL band / layer 3 dec 永久死亡现象.
        self.layerscale_wav = nn.Parameter(torch.tensor(float(layerscale_init)))

        self.approx_gate = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Sigmoid(),
        )
        self.subband_scale = nn.Parameter(torch.ones(3))

        self.band_fuse = nn.Conv2d(_N_BANDS * d_model, d_model, 1)

    @staticmethod
    def _pad_even(x):
        H, W = x.shape[-2:]
        if H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        return x, (H, W)

    def forward(self, z, thr_wav, compute_constraint=False):
        H_orig, W_orig = z.shape[-2:]
        z_pad, _ = self._pad_even(z)
        thr_pad, _ = self._pad_even(thr_wav)
        Hp, Wp = z_pad.shape[-2:]

        w = self.dwt(z_pad)
        thr_down = F.avg_pool2d(thr_pad, kernel_size=2, stride=2)

        subbands = w.chunk(_N_BANDS, dim=1)

        proj_feats = [proj(band) for band, proj in zip(subbands, self.proj_in)]
        enc_feats = [
            enc(feat) for feat, enc in zip(proj_feats, self.enc)
        ]

        constraint = None
        if compute_constraint:
            sym_feats = [
                dec(feat) for feat, dec in zip(enc_feats, self.dec)
            ]
            constraint = torch.cat(
                [sym - feat for sym, feat in zip(sym_feats, proj_feats)], dim=1)

        processed = [enc_feats[0] * self.approx_gate(enc_feats[0])]
        for j, feat_j in enumerate(enc_feats[1:]):
            scale_j = F.softplus(self.subband_scale[j])
            processed.append(smooth_soft_threshold(
                feat_j, (scale_j * thr_down).expand_as(feat_j), tau=self.prox_tau))

        rec_bands = [
            pout(dec(p)) for p, dec, pout in zip(processed, self.dec, self.proj_out)
        ]

        rec_stack = torch.cat(rec_bands, dim=1)
        delta_wav = self.dwt.inverse(rec_stack, out_shape=(Hp, Wp))
        delta_wav = delta_wav[..., :H_orig, :W_orig]
        # 单一标量缩放: 数学上等价于对每个子带的 rec_band 同等缩放再 iDWT.
        delta_wav = self.layerscale_wav * delta_wav

        target_size = (H_orig, W_orig)
        aligned = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                   for f in enc_feats]
        feat_spatial = self.band_fuse(torch.cat(aligned, dim=1))

        return delta_wav, feat_spatial, constraint


class Tail2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        mid = max(d_model // 3, 1)
        self.block = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, mid, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class SELayer2D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        mid = max(channel // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        y = x.mean(dim=(-2, -1))
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class DMlp2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.PReLU(),
            nn.Conv2d(dim, 4 * dim, 1),
            nn.PReLU(),
            nn.Conv2d(4 * dim, dim, 1),
        )

    def forward(self, x):
        return self.block(x)


class DFFM2D(nn.Module):
    def __init__(self, num_layers, d_model):
        super().__init__()
        self.conv = nn.Conv2d(num_layers, d_model, 3, padding=1)
        self.dmlp = DMlp2D(d_model)
        self.se = SELayer2D(d_model)
        self.tail = Tail2D(d_model)

    def forward(self, x_stages, feat_sum, weight):
        x = self.conv(x_stages)
        shortcut = x
        x = self.dmlp(x)
        x = self.se(x)
        x = x + shortcut
        return self.tail(x + weight * feat_sum)


class ISTAProxDWT2D_LiteV3(nn.Module):
    """双分支近端: TV + 独立子带 DWT, ConvResBlock 自动带 GN."""

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5, J=1,
                 prox_tau: float = 0.01, num_groups=DEFAULT_GN_GROUPS):
        super().__init__()
        self.tv_branch = ConvProxTV2D(
            d_model=d_model, num_blocks=num_blocks, kernel_size=kernel_size,
            prox_tau=prox_tau, num_groups=num_groups)
        self.wav_branch = DWTConvBranch2D(
            d_model=d_model, num_blocks=num_blocks, kernel_size=kernel_size, J=J,
            prox_tau=prox_tau, num_groups=num_groups)

    def forward(self, z, thr_tv, thr_wav, alpha=None, compute_constraints=False):
        # alpha 参数保留向后兼容; HASA 已删除 alpha head, V3 始终走 0.5 等权路径.
        delta_tv, feat_tv, constraint_tv = self.tv_branch(
            z, thr_tv, compute_constraint=compute_constraints)
        delta_wav, feat_wav, constraint_wav = self.wav_branch(
            z, thr_wav, compute_constraint=compute_constraints)
        if alpha is None:
            delta = 0.5 * (delta_tv + delta_wav)
            feat_fused = 0.5 * (feat_tv + feat_wav)
        else:
            delta = alpha * delta_tv + (1.0 - alpha) * delta_wav
            feat_fused = alpha * feat_tv + (1.0 - alpha) * feat_wav
        x_next = z + delta
        return x_next, feat_fused, constraint_tv, constraint_wav


class FISTA_DWT_LiteV3_Block_2D(nn.Module):
    """单层: 梯度步 -> HASA -> LiteV3 Prox2D (残差) -> FISTA 动量."""

    def __init__(self, weight_ctor, d_model=32, num_conv_blocks=2,
                 conv_ks=5, J=1, prox_tau: float = 0.01,
                 num_groups=DEFAULT_GN_GROUPS):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProxDWT2D_LiteV3(
            d_model=d_model, num_blocks=num_conv_blocks,
            kernel_size=conv_ks, J=J, prox_tau=prox_tau, num_groups=num_groups,
        )

    def forward(self, x_prev, v, y_scaled, op, compute_constraints=False):
        rho_val = F.softplus(self.rho)
        eta_val = F.softplus(self.soft_thr)

        grad = op.At(op.A(v) - y_scaled)
        z = v - rho_val * grad

        # HASA 重参数化版只返回 (lam_tv, lam_wav), 不再有 alpha
        lambda_tv, lambda_wav = self.weight(z)
        x_next, feat_fused, constraint_tv, constraint_wav = self.prox(
            z, eta_val * lambda_tv, eta_val * lambda_wav,
            compute_constraints=compute_constraints)

        v_next = x_next + torch.tanh(self.beta) * (x_next - x_prev)

        _zero = x_next.new_zeros(())
        aux = {
            "rho1": rho_val.detach(),
            "rho2": rho_val.detach(),
            "eta": eta_val.detach(),
            "lambda_tv": lambda_tv.detach(),
            "lambda_wav": lambda_wav.detach(),
            "constraint_wav": constraint_wav if constraint_wav is not None else _zero,
            "constraint_tv": constraint_tv if constraint_tv is not None else _zero,
        }
        return x_next, v_next, feat_fused, aux


class FISTA_DWT_LiteV3_2D_Net(nn.Module):
    """2D FISTA-DWT-LiteV3 + DFFM 跨层融合."""

    def __init__(self, layer_num=4, hasa_ctor=None,
                 d_model=32, num_conv_blocks=2, conv_ks=5, J=1,
                 prox_tau: float = 0.01, num_groups=DEFAULT_GN_GROUPS):
        super().__init__()
        self.dffm_weight = nn.Parameter(torch.tensor(0.2))
        self.prox_tau = float(prox_tau)
        self.num_groups = int(num_groups)

        if hasa_ctor is None:
            hasa_ctor = lambda: MultiScaleHASA2D()

        self.blocks = nn.ModuleList([
            FISTA_DWT_LiteV3_Block_2D(
                hasa_ctor, d_model=d_model,
                num_conv_blocks=num_conv_blocks, conv_ks=conv_ks, J=J,
                prox_tau=prox_tau, num_groups=num_groups,
            )
            for _ in range(layer_num)
        ])

        self.dffm = DFFM2D(layer_num, d_model)

    def forward(self, y, op, x0=None, return_aux=False,
                compute_constraints=False, **_kw):
        if x0 is None:
            x0 = op.At(y)

        scale = x0.abs().amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-6)
        x = x0 / scale
        v = x0 / scale
        y_scaled = y / scale.view(scale.shape[0], 1, 1)

        stage_outputs = []
        feat_sum = 0
        aux_list = []

        for blk in self.blocks:
            x, v, feat_fused, aux = blk(
                x, v, y_scaled, op, compute_constraints=compute_constraints)
            stage_outputs.append(x.squeeze(1))
            feat_sum = feat_sum + feat_fused
            if return_aux:
                aux_list.append(aux)

        x_stages = torch.stack(stage_outputs, dim=1)
        x_fused = self.dffm(x_stages, feat_sum, self.dffm_weight)
        x_out = x_fused * scale

        if return_aux:
            return x_out, aux_list
        return x_out
