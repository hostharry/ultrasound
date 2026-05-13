"""FISTA-DWT-Lite-2D (帧/patch).

按原版 1D Lite 结构对齐:
  - 多尺度 Conv HASA (MultiScaleHASA2D): local + context 双分支, 增强条件建模
  - ConvProxTV2D + 单层 2D Haar DWT (J=1, 4 子带 LL/LH/HL/HH)
  - DWT 子带 learnable fusion (concat + 1x1 Conv, 替代 mean)
  - Prox 残差形式: x_next = z + delta
  - DFFM2D 跨层融合

参见: FISTA_DWT_Lite_2D_按1D原版修改说明.md
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

# HASA λ_tv / λ_wav 下限: 防止 softplus 左侧梯度死区 + 训练中漂到近零
LAMBDA_FLOOR = 5e-3


# ═══════════════════════ Multi-scale HASA 2D ═══════════════════════


class MultiScaleHASA2D(nn.Module):
    """多尺度 Conv2d HASA: local + context 双分支, 替代单尺度 HASAWeightFISTA2D.

    local 分支: 小卷积核 (ks=3), 捕获局部边缘/纹理
    context 分支: 大卷积核 + 空洞卷积, 捕获更大范围的区域统计
    合并后输出 lambda_tv / lambda_wav / alpha.
    """

    def __init__(self, hidden_ch=16, num_layers=2, inner_ks=5, context_ks=3,
                 context_dilation=3):
        super().__init__()
        inner_pad = inner_ks // 2
        ctx_pad = context_dilation * (context_ks // 2)

        # local branch
        local_layers = [nn.Conv2d(1, hidden_ch, 3, padding=1), nn.GELU()]
        for _ in range(num_layers - 1):
            local_layers.extend([
                nn.Conv2d(hidden_ch, hidden_ch, inner_ks, padding=inner_pad),
                nn.GELU(),
            ])
        self.local_net = nn.Sequential(*local_layers)

        # context branch: dilated convs for larger receptive field
        ctx_layers = [nn.Conv2d(1, hidden_ch, 3, padding=1), nn.GELU()]
        for _ in range(num_layers - 1):
            ctx_layers.extend([
                nn.Conv2d(hidden_ch, hidden_ch, context_ks,
                          padding=ctx_pad, dilation=context_dilation),
                nn.GELU(),
            ])
        self.context_net = nn.Sequential(*ctx_layers)

        # merge: concat (2*hidden_ch) -> hidden_ch
        self.merge = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch, 1),
            nn.GELU(),
        )

        self.head_tv = nn.Sequential(nn.Conv2d(hidden_ch, 1, 1), nn.Softplus())
        self.head_wav = nn.Sequential(nn.Conv2d(hidden_ch, 1, 1), nn.Softplus())
        self.head_alpha = nn.Sequential(nn.Conv2d(hidden_ch, 1, 1), nn.Sigmoid())

        nn.init.constant_(self.head_tv[0].bias, -2.0)
        nn.init.constant_(self.head_wav[0].bias, -2.0)
        nn.init.constant_(self.head_alpha[0].bias, 0.0)

    def forward(self, x):
        f_local = self.local_net(x)
        f_ctx = self.context_net(x)
        feat = self.merge(torch.cat([f_local, f_ctx], dim=1))
        lam_tv = self.head_tv(feat) + LAMBDA_FLOOR
        lam_wav = self.head_wav(feat) + LAMBDA_FLOOR
        return lam_tv, lam_wav, self.head_alpha(feat)


# ═══════════════════════ Mini-U-Net HASA 2D ═══════════════════════


class _UConvBlock(nn.Module):
    """Conv(in→out) + GELU + Conv(out→out) + GELU + 1x1 skip."""

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
    """Mini-U-Net HASA: 两次下采样的轻量 encoder-decoder 权重预测器.

    比 MultiScaleHASA2D 有更强的区域理解能力 (多尺度 + skip),
    适合 simu_cont 这种依赖区域统计的任务.
    输出接口与 MultiScaleHASA2D 完全一致: (lambda_tv, lambda_wav, alpha).
    """

    def __init__(self, base_ch=16):
        super().__init__()
        c1, c2, c4 = base_ch, base_ch * 2, base_ch * 4

        # encoder
        self.enc1 = _UConvBlock(1, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = _UConvBlock(c2, c2)
        self.down2 = nn.Conv2d(c2, c4, 3, stride=2, padding=1)

        # bottleneck
        self.bottleneck = nn.Sequential(
            _UConvBlock(c4, c4),
            _UConvBlock(c4, c4),
        )

        # decoder
        self.up_reduce1 = nn.Conv2d(c4, c2, 1)
        self.dec1 = _UConvBlock(c2 * 2, c2)       # concat(up, skip2) -> c2
        self.up_reduce2 = nn.Conv2d(c2, c1, 1)
        self.dec2 = _UConvBlock(c1 * 2, c1)       # concat(up, skip1) -> c1

        # heads
        self.head_tv = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Softplus())
        self.head_wav = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Softplus())
        self.head_alpha = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())

        nn.init.constant_(self.head_tv[0].bias, -2.0)
        nn.init.constant_(self.head_wav[0].bias, -2.0)
        nn.init.constant_(self.head_alpha[0].bias, 0.0)

    def forward(self, x):
        # encoder
        f1 = self.enc1(x)                         # (B, c1, H, W)
        f2 = self.enc2(self.down1(f1))             # (B, c2, H/2, W/2)
        f3 = self.down2(f2)                        # (B, c4, H/4, W/4)

        # bottleneck
        b = self.bottleneck(f3)                    # (B, c4, H/4, W/4)

        # decoder
        u2 = F.interpolate(b, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.dec1(torch.cat([self.up_reduce1(u2), f2], dim=1))

        u1 = F.interpolate(u2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        u1 = self.dec2(torch.cat([self.up_reduce2(u1), f1], dim=1))

        lam_tv = self.head_tv(u1) + LAMBDA_FLOOR
        lam_wav = self.head_wav(u1) + LAMBDA_FLOOR
        return lam_tv, lam_wav, self.head_alpha(u1)


# ═══════════════════════ 2D Haar (单层, 4 子带) ═══════════════════════


class HaarDWT2D(nn.Module):
    """正交 2D Haar 单层: (B,1,H,W) -> (B,4,H',W'), 逆变换对称."""

    def __init__(self):
        super().__init__()
        s = 0.5
        filters = torch.tensor([
            [[s,  s], [s,  s]],   # LL
            [[s,  s], [-s, -s]],  # LH
            [[s, -s], [s, -s]],   # HL
            [[s, -s], [-s, s]],   # HH
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


# ═══════════════════════ Conv 基础 ═══════════════════════


class ConvResBlock2D(nn.Module):
    """两层 Conv2d + 残差: (B,C,H,W) -> (B,C,H,W)."""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
        )

    def forward(self, x):
        return x + self.block(x)


# ═══════════════════════ TV 分支 ═══════════════════════


class ConvProxTV2D(nn.Module):
    """TV-like learned prox (残差输出): 1x1 -> ConvRes enc -> shrink -> ConvRes dec -> 1x1.

    返回 (delta_tv, feat_enc):
      delta_tv: (B, 1, H, W) 修正量
      feat_enc: (B, d_model, H, W) 编码器输出, 供 DFFM 使用.
    """

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5, prox_tau: float = 0.01):
        super().__init__()
        self.prox_tau = float(prox_tau)
        self.proj_in = nn.Conv2d(1, d_model, 1)
        self.enc = nn.Sequential(
            *[ConvResBlock2D(d_model, kernel_size) for _ in range(num_blocks)])
        self.dec = nn.Sequential(
            *[ConvResBlock2D(d_model, kernel_size) for _ in range(num_blocks)])
        self.proj_out = nn.Conv2d(d_model, 1, 1)

    def forward(self, z, thr_tv):
        feat = self.proj_in(z)
        feat_enc = self.enc(feat)
        feat = smooth_soft_threshold(feat_enc, thr_tv.expand_as(feat_enc), tau=self.prox_tau)
        feat = self.dec(feat)
        return self.proj_out(feat), feat_enc


# ═══════════════════════ DWT 分支 (J=1) ═══════════════════════


class DWTConvBranch2D(nn.Module):
    """单层 2D Haar: 4 子带独立 Conv 编码; LL 门控; LH/HL/HH shrink; IDWT.

    子带特征融合: concat + 1x1 Conv (learnable, 替代 mean).

    返回 (delta_wav, feat_spatial):
      delta_wav:    (B, 1, H, W) 修正量
      feat_spatial: (B, d_model, H, W) 子带特征 learnable fusion, 供 DFFM 使用.
    """

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5, J=1,
                 prox_tau: float = 0.01):
        super().__init__()
        self.prox_tau = float(prox_tau)
        self.dwt = HaarDWT2D()

        self.proj_in = nn.ModuleList(
            [nn.Conv2d(1, d_model, 1) for _ in range(_N_BANDS)])
        self.enc = nn.ModuleList([
            nn.Sequential(*[ConvResBlock2D(d_model, kernel_size)
                            for _ in range(num_blocks)])
            for _ in range(_N_BANDS)])
        self.dec = nn.ModuleList([
            nn.Sequential(*[ConvResBlock2D(d_model, kernel_size)
                            for _ in range(num_blocks)])
            for _ in range(_N_BANDS)])
        self.proj_out = nn.ModuleList(
            [nn.Conv2d(d_model, 1, 1) for _ in range(_N_BANDS)])

        self.approx_gate = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Sigmoid(),
        )
        self.subband_scale = nn.Parameter(torch.ones(3))

        # learnable band fusion: concat(4*d_model) -> 1x1 Conv -> d_model
        self.band_fuse = nn.Conv2d(_N_BANDS * d_model, d_model, 1)

    @staticmethod
    def _pad_even(x):
        H, W = x.shape[-2:]
        if H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        return x, (H, W)

    def forward(self, z, thr_wav):
        H_orig, W_orig = z.shape[-2:]
        z_pad, _ = self._pad_even(z)
        thr_pad, _ = self._pad_even(thr_wav)
        Hp, Wp = z_pad.shape[-2:]

        w = self.dwt(z_pad)
        thr_down = F.avg_pool2d(thr_pad, kernel_size=2, stride=2)

        subbands = w.chunk(_N_BANDS, dim=1)

        enc_feats = [enc(proj(band))
                     for band, proj, enc in zip(subbands, self.proj_in, self.enc)]

        # LL: gate; LH/HL/HH: smooth soft-threshold (gradient flows in dead zone)
        processed = [enc_feats[0] * self.approx_gate(enc_feats[0])]
        for j, feat_j in enumerate(enc_feats[1:]):
            scale_j = F.softplus(self.subband_scale[j])
            processed.append(smooth_soft_threshold(
                feat_j, (scale_j * thr_down).expand_as(feat_j), tau=self.prox_tau))

        rec_bands = [pout(dec(p))
                     for p, dec, pout in zip(processed, self.dec, self.proj_out)]

        rec_stack = torch.cat(rec_bands, dim=1)
        delta_wav = self.dwt.inverse(rec_stack, out_shape=(Hp, Wp))
        delta_wav = delta_wav[..., :H_orig, :W_orig]

        # learnable band fusion: 上采样各子带 -> concat -> 1x1 Conv
        target_size = (H_orig, W_orig)
        aligned = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                   for f in enc_feats]
        feat_spatial = self.band_fuse(torch.cat(aligned, dim=1))

        return delta_wav, feat_spatial


# ═══════════════════════ DFFM 2D ═══════════════════════


class Tail2D(nn.Module):
    """(B, d_model, H, W) -> (B, 1, H, W)."""

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
    """Squeeze-and-Excitation for 2D: (B, C, H, W) -> (B, C, H, W)."""

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
    """Depth-wise MLP for 2D: (B, C, H, W) -> (B, C, H, W)."""

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
    """Dual-path Feature Fusion Module (2D).

    与 DFFM1D 对称: Conv(num_layers→d_model) → DMlp → SE → 残差 → Tail.
    """

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


# ═══════════════════════ Prox / Block / Net ═══════════════════════


class ISTAProxDWT2D_Lite(nn.Module):
    """双分支近端 (残差形式): x_next = z + alpha*delta_tv + (1-alpha)*delta_wav."""

    def __init__(self, d_model=32, num_blocks=2, kernel_size=5, J=1,
                 prox_tau: float = 0.01):
        super().__init__()
        self.tv_branch = ConvProxTV2D(
            d_model=d_model, num_blocks=num_blocks, kernel_size=kernel_size,
            prox_tau=prox_tau)
        self.wav_branch = DWTConvBranch2D(
            d_model=d_model, num_blocks=num_blocks, kernel_size=kernel_size, J=J,
            prox_tau=prox_tau)

    def forward(self, z, thr_tv, thr_wav, alpha=None):
        delta_tv, feat_tv = self.tv_branch(z, thr_tv)
        delta_wav, feat_wav = self.wav_branch(z, thr_wav)
        if alpha is None:
            delta = 0.5 * (delta_tv + delta_wav)
            feat_fused = 0.5 * (feat_tv + feat_wav)
        else:
            delta = alpha * delta_tv + (1.0 - alpha) * delta_wav
            feat_fused = alpha * feat_tv + (1.0 - alpha) * feat_wav
        x_next = z + delta
        return x_next, feat_fused


class FISTA_DWT_Lite_Block_2D(nn.Module):
    """单层: 梯度步 -> HASA -> Lite Prox2D (残差) -> FISTA 动量."""

    def __init__(self, weight_ctor, d_model=32, num_conv_blocks=2,
                 conv_ks=5, J=1, prox_tau: float = 0.01):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProxDWT2D_Lite(
            d_model=d_model, num_blocks=num_conv_blocks,
            kernel_size=conv_ks, J=J, prox_tau=prox_tau,
        )

    def forward(self, x_prev, v, y_scaled, op):
        rho_val = F.softplus(self.rho)
        eta_val = F.softplus(self.soft_thr)

        grad = op.At(op.A(v) - y_scaled)
        z = v - rho_val * grad

        lambda_tv, lambda_wav, alpha = self.weight(z)
        # 不传 alpha: prox 走 0.5*(delta_tv + delta_wav); head_alpha 仍输出供诊断
        x_next, feat_fused = self.prox(z, eta_val * lambda_tv, eta_val * lambda_wav)

        v_next = x_next + torch.tanh(self.beta) * (x_next - x_prev)

        _zero = x_next.new_zeros(())
        aux = {
            "rho1": rho_val.detach(),
            "rho2": rho_val.detach(),
            "eta": eta_val.detach(),
            "lambda_tv": lambda_tv.detach(),
            "lambda_wav": lambda_wav.detach(),
            "alpha": alpha.detach(),
            "constraint_wav": _zero,
            "constraint_tv": _zero,
        }
        return x_next, v_next, feat_fused, aux


class FISTA_DWT_Lite_2D_Net(nn.Module):
    """2D FISTA-DWT-Lite + DFFM 跨层融合."""

    def __init__(self, layer_num=4, hasa_ctor=None,
                 d_model=32, num_conv_blocks=2, conv_ks=5, J=1,
                 prox_tau: float = 0.01):
        super().__init__()
        self.dffm_weight = nn.Parameter(torch.tensor(0.2))
        self.prox_tau = float(prox_tau)

        if hasa_ctor is None:
            hasa_ctor = lambda: MultiScaleHASA2D()

        self.blocks = nn.ModuleList([
            FISTA_DWT_Lite_Block_2D(
                hasa_ctor, d_model=d_model,
                num_conv_blocks=num_conv_blocks, conv_ks=conv_ks, J=J,
                prox_tau=prox_tau,
            )
            for _ in range(layer_num)
        ])

        self.dffm = DFFM2D(layer_num, d_model)

    def forward(self, y, op, x0=None, return_aux=False, **_kw):
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
            x, v, feat_fused, aux = blk(x, v, y_scaled, op)
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
