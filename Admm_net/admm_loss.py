"""可组合的压缩感知重建损失函数库.

每个损失拆为独立函数, 支持可选的深度加权;
CombinedLoss 负责按权重聚合, 不包含任何计算逻辑.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from admm_ops import hilbert_envelope


# ======================== 深度加权 ========================

def build_depth_weight(N: int, mode: str, alpha: float,
                       device: torch.device) -> torch.Tensor:
    """构建 1D 深度加权向量 (N,), 均值归一化到 1.0.

    mode:
        "linear" — w(i) = 1 + alpha * i/(N-1);  alpha=2 → 深层 3x
        "exp"    — w(i) = exp(alpha * i/(N-1));  alpha=2 → 深层 ~7x
    """
    t = torch.linspace(0, 1, N, device=device)
    if mode == "linear":
        w = 1.0 + alpha * t
    elif mode == "exp":
        w = torch.exp(alpha * t)
    else:
        raise ValueError(f"Unknown depth_weight mode: {mode}")
    return w / w.mean()


# ======================== 单项损失函数 ========================

def rf_mse(pred, gt, weight=None):
    """逐点 MSE, 可选深度加权."""
    err = (pred - gt).pow(2)
    if weight is not None:
        err = err * weight
    return err.mean()


def rf_nmse(pred, gt, weight=None):
    """归一化 MSE: ||pred-gt||² / ||gt||², 可选深度加权."""
    err = (pred - gt).pow(2)
    gt2 = gt.pow(2)
    if weight is not None:
        err = err * weight
        gt2 = gt2 * weight
    return err.mean() / gt2.mean().clamp(min=1e-8)


def envelope_l1(pred, gt, envelope_fn=None, use_log=False, weight=None):
    """包络域 L1 损失."""
    fn = envelope_fn or hilbert_envelope
    env_p, env_g = fn(pred), fn(gt)
    if use_log:
        env_p = torch.log(env_p + 1e-6)
        env_g = torch.log(env_g + 1e-6)
    err = (env_p - env_g).abs()
    if weight is not None:
        err = err * weight
    return err.mean()


def msle(pred, gt, weight=None):
    """Mean Squared Log Error: mean((log(1+|pred|) - log(1+|gt|))²).

    对弱信号的相对误差更敏感, 缓解深层信号被忽视的问题.
    """
    err = (torch.log1p(pred.abs()) - torch.log1p(gt.abs())).pow(2)
    if weight is not None:
        err = err * weight
    return err.mean()


def constraint_loss(aux_list):
    """ADMM 约束一致性 (从最后一层的 aux 取)."""
    if not aux_list:
        return torch.tensor(0.0)
    aux = aux_list[-1]
    device = aux["constraint_wav"].device
    return (
        aux["constraint_wav"].pow(2).mean()
        + aux["constraint_tv"].pow(2).mean()
    )


# ======================== 聚合器 ========================

class CombinedLoss(nn.Module):
    """按权重聚合各项损失, 自动兼容 1D (B,1,N) / 2D (B,1,H,W).

    Parameters
    ----------
    gamma_env : float          包络 L1 权重
    gamma_constraint : float   ADMM 约束权重
    gamma_msle : float         MSLE 权重, 0 表示不启用
    use_nmse : bool            RF 项用 NMSE 还是 MSE
    use_log_env : bool         包络域用 log 变换
    depth_weight : str         "none" | "linear" | "exp"
    depth_weight_alpha : float 加权强度
    envelope_fn : callable     包络提取函数
    """

    def __init__(self, gamma_env=0.1, gamma_constraint=0.01,
                 gamma_msle=0.0,
                 use_nmse=False, use_log_env=False,
                 depth_weight="none", depth_weight_alpha=2.0,
                 envelope_fn=None):
        super().__init__()
        self.gamma_env = gamma_env
        self.gamma_constraint = gamma_constraint
        self.gamma_msle = gamma_msle
        self.use_nmse = use_nmse
        self.use_log_env = use_log_env
        self.depth_weight = depth_weight
        self.depth_weight_alpha = depth_weight_alpha
        self._envelope_fn = envelope_fn or hilbert_envelope
        self._dw_cache: dict = {}

    def _get_weight(self, x):
        """仅 1D 且 depth_weight != 'none' 时返回加权向量, 否则 None."""
        if self.depth_weight == "none" or x.ndim != 3:
            return None
        N = x.shape[-1]
        key = (N, str(x.device))
        if key not in self._dw_cache:
            self._dw_cache[key] = build_depth_weight(
                N, self.depth_weight, self.depth_weight_alpha, x.device
            )
        return self._dw_cache[key]

    def forward(self, x_pred, x_gt, aux_list=None):
        w = self._get_weight(x_pred)

        # RF
        loss_rf = (rf_nmse if self.use_nmse else rf_mse)(x_pred, x_gt, w)

        # Envelope
        loss_env = envelope_l1(x_pred, x_gt, self._envelope_fn,
                               self.use_log_env, w)

        # MSLE
        loss_msle = (msle(x_pred, x_gt, w)
                     if self.gamma_msle > 0
                     else torch.tensor(0.0, device=x_pred.device))

        # Constraint
        loss_con = (constraint_loss(aux_list)
                    if aux_list and self.gamma_constraint > 0
                    else torch.tensor(0.0, device=x_pred.device))

        total = (loss_rf
                 + self.gamma_env * loss_env
                 + self.gamma_msle * loss_msle
                 + self.gamma_constraint * loss_con)

        return total, {
            "loss_rf": loss_rf.item(),
            "loss_env": loss_env.item(),
            "loss_msle": loss_msle.item(),
            "loss_constraint": loss_con.item(),
            "loss_total": total.item(),
        }
