"""HASA-ADMM-Net: 1D 深度展开 ADMM 网络

基于 ADMM 变量分裂的深度展开网络，用于超声 RF 信号压缩感知重建。

优化目标:
  min_x  1/2||Ax-y||^2 + λ1||Wx||_1 + λ2||D·Env(x)||_1

每层展开对应一次完整 ADMM 迭代:
  ① w-update: soft_threshold(Wx + u1, λ1/ρ1)
  ② p-update: soft_threshold(D·Env(x) + u2, λ2/ρ2)
  ③ x-update: 频域闭式解 + TV 梯度修正
  ④ 对偶更新: u1, u2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from admm_ops import soft_threshold, hilbert_envelope, build_hasa_weight


# ======================== 稀疏分析算子 W ========================

class HaarDWT1D(nn.Module):
    """Mode A: 固定 Haar 小波 (单层分解)"""

    def __init__(self):
        super().__init__()
        s2 = 1.0 / (2 ** 0.5)
        self.register_buffer('lo', torch.tensor([[[s2, s2]]]))
        self.register_buffer('hi', torch.tensor([[[s2, -s2]]]))

    def forward(self, x):
        a = F.conv1d(x, self.lo, stride=2)
        d = F.conv1d(x, self.hi, stride=2)
        return torch.cat([a, d], dim=-1)

    def inverse(self, w):
        N_half = w.shape[-1] // 2
        a, d = w[..., :N_half], w[..., N_half:]
        return F.conv_transpose1d(a, self.lo, stride=2) + \
               F.conv_transpose1d(d, self.hi, stride=2)


class LearnableAnalysis1D(nn.Module):
    """Mode B: 可学习稀疏分析算子"""

    def __init__(self, num_filters=2, kernel_size=8):
        super().__init__()
        pad = kernel_size // 2 - 1
        self.analysis = nn.Conv1d(1, num_filters, kernel_size,
                                  stride=2, padding=pad, bias=False)
        self.synthesis = nn.ConvTranspose1d(num_filters, 1, kernel_size,
                                            stride=2, padding=pad, bias=False)

    def forward(self, x):
        return self.analysis(x)

    def inverse(self, w):
        return self.synthesis(w)


class SparseTransform1D(nn.Module):
    """统一接口: Mode A (固定 Haar DWT) / Mode B (可学习分析算子)"""

    def __init__(self, mode='A', num_filters=2, kernel_size=8):
        super().__init__()
        self.mode = mode
        if mode == 'A':
            self.transform = HaarDWT1D()
        else:
            self.transform = LearnableAnalysis1D(num_filters, kernel_size)

    def forward(self, x):
        return self.transform(x)

    def inverse(self, w, out_len: int = None):
        result = self.transform.inverse(w)
        if out_len is not None and result.shape[-1] != out_len:
            diff = out_len - result.shape[-1]
            result = F.pad(result, (0, diff)) if diff > 0 else result[..., :out_len]
        return result


class FiniteDiff1D(nn.Module):
    """固定的一阶有限差分算子 D 及其伴随 D^T"""

    def __init__(self):
        super().__init__()
        self.register_buffer('kernel', torch.tensor([[[1.0, -1.0]]]))

    def forward(self, x):
        return F.conv1d(x, self.kernel)

    def adjoint(self, p):
        return F.conv_transpose1d(p, self.kernel)


# ======================== HASA (向后兼容别名) ========================

class HASAWeight1D(nn.Module):
    """HASA 自适应权重生成器 (1D)."""
    def __init__(self, hidden_ch=16, num_layers=2, inner_ks=5):
        super().__init__()
        inner = build_hasa_weight(ndim=1, hidden_ch=hidden_ch,
                                  num_layers=num_layers, inner_ks=inner_ks)
        self.feat_net = inner.feat_net
        self.head_wav = inner.head_wav
        self.head_tv = inner.head_tv

    def forward(self, x):
        feat = self.feat_net(x)
        return self.head_wav(feat), self.head_tv(feat)


# ======================== 单层 ADMM Block ========================

class HASA_ADMM_Block(nn.Module):
    """深度展开的单层 ADMM 迭代

    可学习参数 (逐层独立): eta, rho1, rho2, gamma, HASA 网络, W (Mode B).
    """

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
    def _freq_solve(y, op, rhs_prior, rho):
        """频域闭式解: x = F⁻¹[ (Y_full + ρ·F(rhs)) / (M + ρ) ]"""
        B = rhs_prior.shape[0]
        N = op.N
        mu = op.mu.to(y.device)
        y_full = torch.zeros(B, N // 2 + 1, device=y.device, dtype=y.dtype)
        y_full.scatter_(1, mu.unsqueeze(0).expand(B, -1), y)
        rhs_freq = torch.fft.rfft(rhs_prior.squeeze(1), dim=-1)
        mask_f = op.mask.float().to(y.device).unsqueeze(0)
        x_freq = (y_full + rho * rhs_freq) / (mask_f + rho)
        return torch.fft.irfft(x_freq, n=N, dim=-1).unsqueeze(1)

    def forward(self, x, w, p, u1, u2, y, op):
        rho1 = F.softplus(self.rho1)
        rho2 = F.softplus(self.rho2)
        eta = F.softplus(self.eta)
        gamma = F.softplus(self.gamma)

        lambda_wav, lambda_tv = self.hasa(x)

        Wx = self.W.forward(x)
        thr_wav = lambda_wav / (rho1 + 1e-8)
        if thr_wav.shape[-1] != Wx.shape[-1]:
            thr_wav = F.adaptive_avg_pool1d(thr_wav, Wx.shape[-1])
        if thr_wav.shape[1] != Wx.shape[1]:
            thr_wav = thr_wav.expand_as(Wx)
        w_new = soft_threshold(Wx + u1, thr_wav)

        env = hilbert_envelope(x)
        Denv = self.D(env)
        thr_tv = lambda_tv / (rho2 + 1e-8)
        if thr_tv.shape[-1] != Denv.shape[-1]:
            thr_tv = F.adaptive_avg_pool1d(thr_tv, Denv.shape[-1])
        p_new = soft_threshold(Denv + u2, thr_tv)

        rhs_wav = self.W.inverse(w_new - u1, out_len=x.shape[-1])
        x_linear = self._freq_solve(y, op, rhs_wav, rho1)

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

class HASA_ADMM_Net(nn.Module):
    """HASA-ADMM-Net: K 层展开的 1D ADMM 深度网络"""

    def __init__(self, layer_num, hasa_ctor, W_mode='A',
                 W_num_filters=2, W_kernel_size=8, share_W=True):
        super().__init__()
        self.D = FiniteDiff1D()

        if share_W:
            W = SparseTransform1D(W_mode, W_num_filters, W_kernel_size)
            self.W = W
            self.blocks = nn.ModuleList([
                HASA_ADMM_Block(hasa_ctor, W, self.D) for _ in range(layer_num)
            ])
        else:
            self.blocks = nn.ModuleList()
            W_list = nn.ModuleList()
            for _ in range(layer_num):
                W_k = SparseTransform1D(W_mode, W_num_filters, W_kernel_size)
                W_list.append(W_k)
                self.blocks.append(HASA_ADMM_Block(hasa_ctor, W_k, self.D))
            self.W_list = W_list
            self.W = W_list[0]

        self.layer_num = layer_num

    def forward(self, y, op, x0=None, return_aux=False):
        if x0 is None:
            x0 = op.At(y)

        scale = x0.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        x = x0 / scale
        y_scaled = y / scale.view(scale.shape[0], 1)

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
