"""HASA-ADMM-Net 2D: 利用 RF 帧空间相关性的深度展开网络

将 1D ADMM 展开推广到 2D，利用 RF 帧相邻阵元间的横向空间相关性。

与 1D 版本的核心区别:
  - 小波: 1D Haar → 2D Haar (2×2 张量积, 4 个子带)
  - TV:   1D 差分 → 2D 梯度 (轴向 + 横向)
  - HASA: Conv1d → Conv2d
  - x-update: 仍为逐行频域闭式解 (测量算子逐行独立)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from admm_ops import soft_threshold, hilbert_envelope, build_hasa_weight


# ======================== 2D 有限差分 ========================

class FiniteDiff2D(nn.Module):
    """2D 有限差分: 轴向 + 横向梯度

    forward: (B,1,H,W) → (B,2,H,W)
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('kernel_w', torch.tensor([[[[1.0, -1.0]]]]))
        self.register_buffer('kernel_h', torch.tensor([[[[1.0], [-1.0]]]]))

    def forward(self, x):
        dw = F.pad(F.conv2d(x, self.kernel_w), (0, 1))
        dh = F.pad(F.conv2d(x, self.kernel_h), (0, 0, 0, 1))
        return torch.cat([dw, dh], dim=1)


# ======================== 2D Haar 小波 ========================

class HaarDWT2D(nn.Module):
    """正交 2D Haar 小波 (单层分解, 2×2 张量积)

    forward: (B, 1, H, W) → (B, 4, ⌈H/2⌉, ⌈W/2⌉)
    inverse: (B, 4, H', W') → (B, 1, H, W)
    """

    def __init__(self):
        super().__init__()
        s = 0.5
        filters = torch.tensor([
            [[s,  s], [s,  s]],
            [[s,  s], [-s, -s]],
            [[s, -s], [s, -s]],
            [[s, -s], [-s, s]],
        ]).unsqueeze(1)
        self.register_buffer('filters', filters)

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


class SparseTransform2D(nn.Module):
    """2D 稀疏变换 (当前仅 Mode A: 固定 Haar DWT)"""

    def __init__(self, mode='A'):
        super().__init__()
        if mode != 'A':
            raise NotImplementedError("2D Mode B 暂未实现")
        self.transform = HaarDWT2D()

    def forward(self, x):
        return self.transform(x)

    def inverse(self, w, out_shape=None):
        return self.transform.inverse(w, out_shape=out_shape)


# ======================== HASA 2D (向后兼容别名) ========================

class HASAWeight2D(nn.Module):
    """HASA 自适应权重生成器 (2D)."""
    def __init__(self, hidden_ch=16, num_layers=2, inner_ks=5):
        super().__init__()
        inner = build_hasa_weight(ndim=2, hidden_ch=hidden_ch,
                                  num_layers=num_layers, inner_ks=inner_ks)
        self.feat_net = inner.feat_net
        self.head_wav = inner.head_wav
        self.head_tv = inner.head_tv

    def forward(self, x):
        feat = self.feat_net(x)
        return self.head_wav(feat), self.head_tv(feat)


# ======================== 2D ADMM Block ========================

class HASA_ADMM_Block_2D(nn.Module):
    """2D ADMM 展开单层"""

    def __init__(self, hasa_ctor, W, D):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(-2.0))
        self.rho1 = nn.Parameter(torch.tensor(-1.0))
        self.rho2 = nn.Parameter(torch.tensor(-1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.hasa = hasa_ctor()
        self.W = W
        self.D = D

    @staticmethod
    def _freq_solve_2d(y, op, rhs_prior, rho):
        """逐行频域闭式解"""
        B, _, H, W = rhs_prior.shape
        N = op.N
        mu = op.mu.to(y.device)

        y_full = torch.zeros(B, H, N // 2 + 1, device=y.device, dtype=y.dtype)
        y_full.scatter_(2, mu.unsqueeze(0).unsqueeze(0).expand(B, H, -1), y)

        rhs_freq = torch.fft.rfft(rhs_prior.squeeze(1), dim=-1)
        mask_f = op.mask.float().to(y.device).reshape(1, 1, -1)

        x_freq = (y_full + rho * rhs_freq) / (mask_f + rho)
        return torch.fft.irfft(x_freq, n=N, dim=-1).unsqueeze(1)

    def forward(self, x, w, p, u1, u2, y, op):
        rho1 = F.softplus(self.rho1)
        rho2 = F.softplus(self.rho2)
        eta = F.softplus(self.eta)
        gamma = F.softplus(self.gamma)

        _, _, H, W = x.shape

        lambda_wav, lambda_tv = self.hasa(x)

        Wx = self.W.forward(x)
        thr_wav = lambda_wav / (rho1 + 1e-8)
        thr_wav = F.adaptive_avg_pool2d(thr_wav, (Wx.shape[-2], Wx.shape[-1]))
        thr_wav = thr_wav.expand_as(Wx)
        w_new = soft_threshold(Wx + u1, thr_wav)

        env = hilbert_envelope(x)
        Denv = self.D(env)
        thr_tv = lambda_tv / (rho2 + 1e-8)
        thr_tv = thr_tv.expand_as(Denv)
        p_new = soft_threshold(Denv + u2, thr_tv)

        rhs_wav = self.W.inverse(w_new - u1, out_shape=(H, W))
        x_linear = self._freq_solve_2d(y, op, rhs_wav, rho1)

        need_higher_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            xl = x_linear if x_linear.requires_grad else x_linear.detach().requires_grad_(True)
            env_xl = hilbert_envelope(xl)
            Denv_xl = self.D(env_xl)
            residual_tv = Denv_xl - p_new + u2
            loss_tv = 0.5 * (residual_tv ** 2).sum()
            grad_tv = torch.autograd.grad(loss_tv, xl, create_graph=need_higher_grad)[0]

        x_new = x_linear - eta * rho2 * grad_tv

        Wx_new_d = self.W.forward(x_new.detach())
        Denv_new_d = self.D(hilbert_envelope(x_new.detach()))
        u1_new = u1 + gamma * (Wx_new_d - w_new.detach())
        u2_new = u2 + gamma * (Denv_new_d - p_new.detach())

        aux = {
            "lambda_wav": lambda_wav.detach(), "lambda_tv": lambda_tv.detach(),
            "rho1": rho1.detach(), "rho2": rho2.detach(),
            "eta": eta.detach(), "gamma": gamma.detach(),
            "constraint_wav": (Wx_new_d - w_new.detach()).detach(),
            "constraint_tv": (Denv_new_d - p_new.detach()).detach(),
        }
        return x_new, w_new, p_new, u1_new, u2_new, aux


# ======================== 完整网络 ========================

class HASA_ADMM_Net_2D(nn.Module):
    """HASA-ADMM-Net 2D: K 层展开的 2D ADMM 深度网络"""

    def __init__(self, layer_num, hasa_ctor, W_mode='A', share_W=True):
        super().__init__()
        self.D = FiniteDiff2D()

        if share_W:
            W = SparseTransform2D(W_mode)
            self.W = W
            self.blocks = nn.ModuleList([
                HASA_ADMM_Block_2D(hasa_ctor, W, self.D) for _ in range(layer_num)
            ])
        else:
            self.blocks = nn.ModuleList()
            W_list = nn.ModuleList()
            for _ in range(layer_num):
                W_k = SparseTransform2D(W_mode)
                W_list.append(W_k)
                self.blocks.append(HASA_ADMM_Block_2D(hasa_ctor, W_k, self.D))
            self.W_list = W_list
            self.W = W_list[0]

        self.layer_num = layer_num

    def forward(self, y, op, x0=None, return_aux=False):
        if x0 is None:
            x0 = op.At(y)

        scale = x0.abs().amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-6)
        x = x0 / scale
        y_scaled = y / scale.view(scale.shape[0], 1, 1)

        W_ref = self.W
        w = W_ref.forward(x)
        p = self.D(hilbert_envelope(x))
        u1 = torch.zeros_like(w)
        u2 = torch.zeros_like(p)

        aux_list = []
        for blk in self.blocks:
            x, w, p, u1, u2, aux = blk(x, w, p, u1, u2, y_scaled, op)
            if return_aux:
                aux_list.append(aux)

        x = x * scale

        if return_aux:
            return x, aux_list
        return x
