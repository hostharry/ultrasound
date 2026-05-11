"""Deep-Unfolded NESTA 1D 模型

将 Nesterov 加速算法展开为深度网络, 保留测量算子 A/A^T,
学习每层的步长、阈值和动量参数.
参考: Becker, Bobin & Candès (SIAM 2011).
"""

import torch
import torch.nn as nn

from admm_ops import soft_threshold


# ======================== Deep-Unfolded NESTA ========================

class NESTA_Layer(nn.Module):
    """单层 Deep-Unfolded NESTA.

    保留测量算子 A/A^T, 学习: 步长 μ_k, 阈值 λ_k, 动量 τ_k.
    """

    def __init__(self, init_step: float = 0.5, init_lambda: float = 0.01):
        super().__init__()
        self.log_step = nn.Parameter(torch.tensor(init_step).log())
        self.log_lambda = nn.Parameter(torch.tensor(init_lambda).log())
        self.logit_tau = nn.Parameter(torch.zeros(1))

    @property
    def step_size(self):
        return self.log_step.exp()

    @property
    def threshold(self):
        return self.log_lambda.exp()

    @property
    def tau(self):
        return torch.sigmoid(self.logit_tau)

    def forward(self, y_sub, op, x_k, z_k):
        tau = self.tau

        # Nesterov momentum combination
        q_k = tau * z_k + (1.0 - tau) * x_k

        # Gradient: A^T(A q - y)
        residual = op.A(q_k) - y_sub
        grad = op.At(residual)

        # Proximal gradient on z
        z_next = soft_threshold(z_k - self.step_size * grad, self.threshold)

        # Nesterov update for x
        x_next = q_k + tau * (z_next - z_k)

        return x_next, z_next


class NESTA_Net(nn.Module):
    """Deep-Unfolded NESTA 网络 (1D).

    Args:
        layer_num: 展开层数 (对标论文 60 次迭代, 推荐 15-30).
    """

    def __init__(self, layer_num: int = 15):
        super().__init__()
        self.blocks = nn.ModuleList([
            NESTA_Layer(init_step=0.5, init_lambda=0.01)
            for _ in range(layer_num)
        ])

    def forward(self, y_sub, op, return_aux=False):
        x = op.At(y_sub)                     # (B, 1, N)
        z = x.clone()

        aux_list = []
        for blk in self.blocks:
            x, z = blk(y_sub, op, x, z)
            aux_list.append({
                "rho1": blk.step_size.detach(),
                "rho2": blk.threshold.detach(),
                "eta": blk.tau.detach().squeeze(),
            })

        if return_aux:
            return x, aux_list
        return x
