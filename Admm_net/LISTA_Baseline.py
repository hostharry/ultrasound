"""LISTA (Learned ISTA) 1D 模型

将 ISTA 迭代展开为深度网络, 每层使用可学习 1D 卷积替代矩阵乘法.
参考: Gregor & LeCun (ICML 2010); Mamistvalov & Eldar (TUFFC 2021).
"""

import torch
import torch.nn as nn


# ======================== 修改版软阈值 (论文 eq.30) ========================

class SigmoidShrinkage(nn.Module):
    """S_λ(x) = x / (1 + exp(-(|x| - λ)))

    比标准 soft-threshold 更平滑, 梯度不会在 |x|=λ 处截断.
    """

    def __init__(self, init_lambda: float = 0.1):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(init_lambda))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1.0 + torch.exp(-(torch.abs(x) - self.lam)))


# ======================== LISTA ========================

class LISTA_Layer(nn.Module):
    """单层 LISTA: x_{k+1} = S_{λk}(W_e(y) + W_t(x_k))"""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.W_e = nn.Conv1d(1, 1, kernel_size, padding=pad, bias=True)
        self.W_t = nn.Conv1d(1, 1, kernel_size, padding=pad, bias=True)
        self.shrinkage = SigmoidShrinkage(init_lambda=0.1)

    def forward(self, y: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        return self.shrinkage(self.W_e(y) + self.W_t(x_prev))


class LISTA_Net(nn.Module):
    """Learned ISTA 网络 (1D).

    Args:
        layer_num: 展开层数 (论文使用 30).
        kernel_size: W_e / W_t 卷积核大小 (论文使用 5).
    """

    def __init__(self, layer_num: int = 30, kernel_size: int = 5):
        super().__init__()
        self.blocks = nn.ModuleList([
            LISTA_Layer(kernel_size) for _ in range(layer_num)
        ])
        self.output_conv = nn.Conv1d(
            1, 1, kernel_size, padding=kernel_size // 2, bias=True,
        )

    def forward(self, y_sub, op, return_aux=False):
        y = op.At(y_sub)                     # (B, 1, N) aliased time-domain
        x = torch.zeros_like(y)

        aux_list = []
        for blk in self.blocks:
            x = blk(y, x)
            aux_list.append({
                "rho1": blk.shrinkage.lam.detach(),
                "rho2": blk.shrinkage.lam.detach(),
                "eta": blk.W_t.weight.data.abs().mean().detach(),
            })

        x_hat = self.output_conv(x)

        if return_aux:
            return x_hat, aux_list
        return x_hat
