"""ISTA-Net+ 1D 模型 (适配 admm_net 框架)

基于 ISTA-Net+ (Zhang & Ghanem, CVPR 2018) 的 1D 超声版本.
每层包含: 梯度下降步 + 多通道卷积近端映射 + 残差连接 + 对称损失.
参数量远大于论文 LISTA (~112K vs 396), 作为更强的展开基线.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ======================== 损失函数 ========================

class ISTANetPlusLoss(nn.Module):
    """ISTA-Net+ 专用损失 (与原始论文一致).

    loss = ||x_hat - x_gt||^2 + γ × Σ_k ||sym_k||^2

    参考: Train_CS_ISTA_Net_plus.py (lines 238-247)
    """

    def __init__(self, gamma_sym: float = 0.01, use_nmse: bool = False):
        super().__init__()
        self.gamma_sym = gamma_sym
        self.use_nmse = use_nmse

    def forward(self, x_pred, x_gt, aux_list=None):
        if self.use_nmse:
            loss_discrepancy = (x_pred - x_gt).pow(2).mean() / (
                x_gt.pow(2).mean().clamp(min=1e-8))
        else:
            loss_discrepancy = torch.mean(torch.pow(x_pred - x_gt, 2))

        loss_constraint = torch.tensor(0.0, device=x_pred.device)
        if aux_list and self.gamma_sym > 0:
            for aux in aux_list:
                if "sym_loss" in aux:
                    loss_constraint = loss_constraint + aux["sym_loss"]

        total = loss_discrepancy + self.gamma_sym * loss_constraint

        return total, {
            "loss_rf": loss_discrepancy.item(),
            "loss_env": 0.0,
            "loss_constraint": loss_constraint.item(),
            "loss_total": total.item(),
        }


# ======================== 网络模块 ========================

class RFFT_Mask_ForBack(nn.Module):
    """x → rFFT → mask → irFFT → x_masked  (模拟 A^T A 操作)"""

    def forward(self, x, mask):
        x_sq = x.squeeze(1)
        X_freq = torch.fft.rfft(x_sq, dim=-1)
        X_masked = X_freq * mask.unsqueeze(0)
        return torch.fft.irfft(X_masked, n=x_sq.shape[-1], dim=-1).unsqueeze(1)


class ISTANetPlusBlock(nn.Module):
    """ISTA-Net+ 单层 (1D), 与原始 BasicBlock 结构完全一致.

    梯度步:  r_k = x_k - λ * A^T(A x_k - y)
    近端步:  x_{k+1} = x_input + G(backward(S_θ(forward(D(x_input)))))
    对称损失: backward(forward(D(x_input))) - D(x_input)
    """

    def __init__(self, n_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv_D = nn.Parameter(init.xavier_normal_(
            torch.Tensor(n_channels, 1, kernel_size)))
        self.conv1_forward = nn.Parameter(init.xavier_normal_(
            torch.Tensor(n_channels, n_channels, kernel_size)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(
            torch.Tensor(n_channels, n_channels, kernel_size)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(
            torch.Tensor(n_channels, n_channels, kernel_size)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(
            torch.Tensor(n_channels, n_channels, kernel_size)))
        self.conv_G = nn.Parameter(init.xavier_normal_(
            torch.Tensor(1, n_channels, kernel_size)))

        self.padding = pad

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv1d(x_input, self.conv_D, padding=self.padding)

        x = F.conv1d(x_D, self.conv1_forward, padding=self.padding)
        x = F.relu(x)
        x_forward = F.conv1d(x, self.conv2_forward, padding=self.padding)

        x = torch.mul(torch.sign(x_forward),
                       F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv1d(x, self.conv1_backward, padding=self.padding)
        x = F.relu(x)
        x_backward = F.conv1d(x, self.conv2_backward, padding=self.padding)

        x_G = F.conv1d(x_backward, self.conv_G, padding=self.padding)
        x_pred = x_input + x_G

        x = F.conv1d(x_forward, self.conv1_backward, padding=self.padding)
        x = F.relu(x)
        x_D_est = F.conv1d(x, self.conv2_backward, padding=self.padding)
        symloss = x_D_est - x_D

        return x_pred, symloss


# ======================== 主网络 ========================

class ISTANetPlus(nn.Module):
    """ISTA-Net+ 网络 (1D).

    Args:
        layer_num: 展开层数 (默认 9).
        n_channels: 中间卷积通道数 (默认 32).
        kernel_size: 卷积核大小 (默认 3).
    """

    def __init__(self, layer_num: int = 9, n_channels: int = 32,
                 kernel_size: int = 3):
        super().__init__()
        self.layer_num = layer_num
        self.fft_forback = RFFT_Mask_ForBack()
        self.blocks = nn.ModuleList([
            ISTANetPlusBlock(n_channels, kernel_size)
            for _ in range(layer_num)
        ])

    def forward(self, y_sub, op, return_aux=False):
        mask = torch.zeros(op.N // 2 + 1, device=y_sub.device)
        mask[op.mu] = 1.0

        PhiTb = op.At(y_sub)
        x = PhiTb.clone()

        layers_sym = []
        for blk in self.blocks:
            x, layer_sym = blk(x, self.fft_forback, PhiTb, mask)
            layers_sym.append(layer_sym)

        if return_aux:
            aux_list = []
            for i, blk in enumerate(self.blocks):
                aux = {
                    "rho1": blk.lambda_step.detach(),
                    "rho2": blk.soft_thr.detach(),
                    "eta": blk.lambda_step.detach(),
                    "sym_loss": torch.mean(torch.pow(layers_sym[i], 2)),
                }
                aux_list.append(aux)
            return x, aux_list
        return x
