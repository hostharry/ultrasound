"""Shared reconstruction losses for Ultrasound models.

This module is the public loss entry point used by the shared training
pipeline.  It supports the older MSE/NMSE losses and the 2D Lite settings
used by the recent in-vivo experiments (SLAE, dB-distribution KLD, DAS loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import hilbert_envelope


def build_depth_weight(N: int, mode: str, alpha: float,
                       device: torch.device, dtype=None) -> torch.Tensor:
    """Build a depth weight vector normalized to mean 1."""
    dtype = dtype or torch.float32
    t = torch.linspace(0, 1, N, device=device, dtype=dtype)
    if mode == "linear":
        w = 1.0 + alpha * t
    elif mode == "exp":
        w = torch.exp(alpha * t)
    else:
        raise ValueError(f"Unknown depth_weight mode: {mode}")
    return w / w.mean().clamp(min=1e-8)


def _apply_weight(err, weight):
    if weight is None:
        return err
    return err * weight


def rf_mse(pred, gt, weight=None):
    return _apply_weight((pred - gt).pow(2), weight).mean()


def rf_nmse(pred, gt, weight=None):
    err = (pred - gt).pow(2)
    denom = gt.pow(2)
    if weight is not None:
        err = err * weight
        denom = denom * weight
    return err.mean() / denom.mean().clamp(min=1e-8)


def _alpha_from_db(alpha_db: float) -> float:
    """Map a dB floor to a linear soft-log offset."""
    return float(10.0 ** (alpha_db / 20.0))


def _db_magnitude(x, alpha_db: float = -60.0):
    """按文献定义映射到 dB / echogenicity 空间.

    20 * log10(max(|x|, alpha)).  alpha = 10^(alpha_db/20) 作为噪声底.
    """
    alpha = _alpha_from_db(alpha_db)
    return 20.0 * torch.log10(x.abs().clamp(min=alpha))


def signed_log_l1(pred, gt, alpha_db=-60.0, weight=None):
    """Signed Log Absolute Error: 正负分离 + log10 幅度域 L1.

    受 SMSLE 启发, 先把信号拆成正/负两支, 各自在对数幅度域比较, 再平均.
    既保留极性信息, 又能避免 sign(x) 在 0 处梯度跳变, 适合高动态范围 RF.
    单像素误差范围在 [0, |alpha_db|/20] 内 (默认 alpha_db=-60 时 ≤ 3).
    """
    alpha = _alpha_from_db(alpha_db)
    pred_pos = pred.clamp(min=0.0)
    gt_pos = gt.clamp(min=0.0)
    pred_neg = (-pred).clamp(min=0.0)
    gt_neg = (-gt).clamp(min=0.0)

    err_pos = (torch.log10(pred_pos.clamp(min=alpha))
               - torch.log10(gt_pos.clamp(min=alpha))).abs()
    err_neg = (torch.log10(pred_neg.clamp(min=alpha))
               - torch.log10(gt_neg.clamp(min=alpha))).abs()
    if weight is not None:
        err_pos = err_pos * weight
        err_neg = err_neg * weight
    return 0.5 * (err_pos.mean() + err_neg.mean())


def envelope_l1(pred, gt, envelope_fn=None, use_log=False, weight=None,
                env_pred=None, env_gt=None):
    fn = envelope_fn or hilbert_envelope
    env_p = fn(pred) if env_pred is None else env_pred
    env_g = fn(gt) if env_gt is None else env_gt
    if use_log:
        env_p = torch.log(env_p + 1e-6)
        env_g = torch.log(env_g + 1e-6)
    return _apply_weight((env_p - env_g).abs(), weight).mean()


def msle(pred, gt, weight=None):
    err = (torch.log1p(pred.abs()) - torch.log1p(gt.abs())).pow(2)
    return _apply_weight(err, weight).mean()


def kld_db_distribution(pred, gt, alpha_db: float = -60.0, *,
                        low_db: float = -60.0, high_db: float = 0.0,
                        n_bins: int = 40, eta: float = 0.5,
                        eps: float = 1e-8, var_floor: float = 0.01):
    """高斯近似 KLD: 假设 dB 分布近似高斯, 用解析公式替代软直方图.

    D_KL(p || q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 1/2

    复杂度从 O(B·M·K) 降到 O(B·M), 显存和耗时降低约 K=40 倍.
    `low_db / high_db / n_bins / eta / eps` 保留以兼容老接口, 不再使用.
    """
    pred_db = _db_magnitude(pred, alpha_db=alpha_db)
    gt_db = _db_magnitude(gt.detach(), alpha_db=alpha_db)

    def _stats(v):
        flat = v.reshape(v.shape[0], -1)
        mu = flat.mean(dim=1)
        var = flat.var(dim=1).clamp(min=var_floor)
        return mu, var

    mu_p, var_p = _stats(gt_db)
    mu_q, var_q = _stats(pred_db)

    kld = (torch.log(var_q.sqrt() / var_p.sqrt())
           + (var_p + (mu_p - mu_q).pow(2)) / (2.0 * var_q)
           - 0.5)
    return kld.clamp(max=50.0).mean()


def kld_mslae_loss(pred, gt, alpha_db=-60.0, beta_kld=0.5,
                  low_db=-60.0, high_db=0.0, n_bins=40, eta=0.5):
    slae = signed_log_l1(pred, gt, alpha_db=alpha_db)
    kld = kld_db_distribution(
        pred, gt, alpha_db=alpha_db,
        low_db=low_db, high_db=high_db, n_bins=n_bins, eta=eta)
    return slae + beta_kld * kld


def constraint_loss(aux_list):
    if not aux_list:
        return torch.tensor(0.0)
    aux = aux_list[-1]
    terms = []
    if "constraint_wav" in aux:
        terms.append(aux["constraint_wav"].pow(2).mean())
    if "constraint_tv" in aux:
        terms.append(aux["constraint_tv"].pow(2).mean())
    if not terms:
        return torch.tensor(0.0)
    return sum(terms)


_SOBEL_H = torch.tensor([[-1., -2., -1.],
                         [0., 0., 0.],
                         [1., 2., 1.]]).reshape(1, 1, 3, 3) / 8.0
_SOBEL_W = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]]).reshape(1, 1, 3, 3) / 8.0


def _log_envelope_2d(x, envelope_fn, eps=1e-6):
    return torch.log(envelope_fn(x) + eps)


def _sobel_grad(u):
    sh = _SOBEL_H.to(u.device, u.dtype)
    sw = _SOBEL_W.to(u.device, u.dtype)
    return F.conv2d(u, sh, padding=1), F.conv2d(u, sw, padding=1)


def gradient_consistency_2d(pred, gt, envelope_fn=None, eps=1e-6,
                            log_env_pred=None, log_env_gt=None,
                            sobel_h=None, sobel_w=None):
    fn = envelope_fn or hilbert_envelope
    u_pred = _log_envelope_2d(pred, fn, eps) if log_env_pred is None else log_env_pred
    u_gt = _log_envelope_2d(gt, fn, eps) if log_env_gt is None else log_env_gt
    gh_p, gw_p = _sobel_grad(u_pred)
    gh_g, gw_g = _sobel_grad(u_gt)
    return (gh_p - gh_g).abs().mean() + (gw_p - gw_g).abs().mean()


def local_stat_loss_2d(pred, gt, envelope_fn=None, win=7, alpha_var=0.5,
                       eps=1e-6, log_env_pred=None, log_env_gt=None):
    fn = envelope_fn or hilbert_envelope
    u_pred = _log_envelope_2d(pred, fn, eps) if log_env_pred is None else log_env_pred
    u_gt = _log_envelope_2d(gt, fn, eps) if log_env_gt is None else log_env_gt
    pad = win // 2
    mu_p = F.avg_pool2d(u_pred, win, stride=1, padding=pad)
    mu_g = F.avg_pool2d(u_gt, win, stride=1, padding=pad)
    var_p = F.avg_pool2d(u_pred.pow(2), win, stride=1, padding=pad) - mu_p.pow(2)
    var_g = F.avg_pool2d(u_gt.pow(2), win, stride=1, padding=pad) - mu_g.pow(2)
    loss_mean = (mu_p - mu_g).abs().mean()
    loss_var = (torch.log(var_p.clamp(min=eps)) -
                torch.log(var_g.clamp(min=eps))).abs().mean()
    return loss_mean + alpha_var * loss_var


def das_image_loss(pred, gt, das_forward, angles, mode="log_l1",
                   alpha_db=-60.0, beta_kld=0.5,
                   kld_low_db=-40.0, kld_high_db=40.0,
                   kld_bins=40, kld_eta=0.5):
    """Image-domain DAS loss for a batch of single-angle RF frames."""
    if das_forward is None or angles is None:
        return pred.new_tensor(0.0), {}
    pred_env = das_forward(pred.squeeze(1), angles)
    gt_env = das_forward(gt.squeeze(1), angles)
    if mode == "kld_mslae":
        loss = kld_mslae_loss(
            pred_env, gt_env, alpha_db=alpha_db, beta_kld=beta_kld,
            low_db=kld_low_db, high_db=kld_high_db,
            n_bins=kld_bins, eta=kld_eta,
        )
    else:
        loss = (torch.log(pred_env + 1e-6) - torch.log(gt_env + 1e-6)).abs().mean()
    return loss, {"das_env_pred": pred_env.detach(), "das_env_gt": gt_env.detach()}


class CombinedLoss(nn.Module):
    """Aggregate RF, envelope, structural, distribution, and optional DAS losses."""

    def __init__(self, gamma_env=0.1, gamma_constraint=0.01,
                 gamma_msle=0.0, gamma_grad=0.0, gamma_stat=0.0,
                 gamma_kld=0.0, gamma_das=0.0,
                 stat_var_weight=0.5, stat_win=7,
                 use_nmse=False, use_log_env=False, use_slae=False,
                 alpha_db=-60.0,
                 kld_bins=40, kld_eta=0.5,
                 kld_low_db=-60.0, kld_high_db=0.0,
                 das_loss_mode="log_l1", das_alpha_db=-60.0,
                 das_beta_kld=0.5, das_kld_bins=40, das_kld_eta=0.5,
                 das_kld_low_db=-40.0, das_kld_high_db=40.0,
                 depth_weight="none", depth_weight_alpha=2.0,
                 envelope_fn=None):
        super().__init__()
        self.gamma_env = gamma_env
        self.gamma_constraint = gamma_constraint
        self.gamma_msle = gamma_msle
        self.gamma_grad = gamma_grad
        self.gamma_stat = gamma_stat
        self.gamma_kld = gamma_kld
        self.gamma_das = gamma_das
        self.stat_var_weight = stat_var_weight
        self.stat_win = stat_win
        self.use_nmse = use_nmse
        self.use_log_env = use_log_env
        self.use_slae = use_slae
        self.alpha_db = alpha_db
        self.kld_bins = kld_bins
        self.kld_eta = kld_eta
        self.kld_low_db = kld_low_db
        self.kld_high_db = kld_high_db
        self.das_loss_mode = das_loss_mode
        self.das_alpha_db = das_alpha_db
        self.das_beta_kld = das_beta_kld
        self.das_kld_bins = das_kld_bins
        self.das_kld_eta = das_kld_eta
        self.das_kld_low_db = das_kld_low_db
        self.das_kld_high_db = das_kld_high_db
        self.depth_weight = depth_weight
        self.depth_weight_alpha = depth_weight_alpha
        self._envelope_fn = envelope_fn or hilbert_envelope
        self._dw_cache = {}

    def _get_weight(self, x):
        """Return a broadcastable depth weight for 1D or 2D inputs."""
        if self.depth_weight == "none":
            return None
        N = x.shape[-1]
        key = (N, str(x.device), str(x.dtype), self.depth_weight, self.depth_weight_alpha)
        if key not in self._dw_cache:
            w = build_depth_weight(
                N, self.depth_weight, self.depth_weight_alpha, x.device, x.dtype)
            self._dw_cache[key] = w
        w = self._dw_cache[key]
        if x.ndim == 3:
            return w.view(1, 1, N)
        if x.ndim == 4:
            return w.view(1, 1, 1, N)
        return w

    def forward(self, x_pred, x_gt, aux_list=None, das_meta=None):
        w = self._get_weight(x_pred)
        is_2d = x_pred.ndim == 4

        if self.use_slae:
            loss_rf = signed_log_l1(x_pred, x_gt, self.alpha_db, w)
        else:
            loss_rf = (rf_nmse if self.use_nmse else rf_mse)(x_pred, x_gt, w)

        need_env = (self.gamma_env > 0 or self.gamma_grad > 0 or
                    self.gamma_stat > 0)
        env_pred = env_gt = log_env_pred = log_env_gt = None
        if need_env:
            env_pred = self._envelope_fn(x_pred)
            env_gt = self._envelope_fn(x_gt)
            if self.use_log_env or self.gamma_grad > 0 or self.gamma_stat > 0:
                log_env_pred = torch.log(env_pred + 1e-6)
                log_env_gt = torch.log(env_gt + 1e-6)

        loss_env = (envelope_l1(
            x_pred, x_gt, self._envelope_fn, self.use_log_env, w,
            env_pred=env_pred, env_gt=env_gt)
            if self.gamma_env > 0 else x_pred.new_tensor(0.0))
        loss_msle = (msle(x_pred, x_gt, w)
                     if self.gamma_msle > 0 else x_pred.new_tensor(0.0))
        loss_con = (constraint_loss(aux_list).to(x_pred.device)
                    if aux_list and self.gamma_constraint > 0
                    else x_pred.new_tensor(0.0))
        loss_grad = (gradient_consistency_2d(
            x_pred, x_gt, self._envelope_fn,
            log_env_pred=log_env_pred, log_env_gt=log_env_gt)
            if is_2d and self.gamma_grad > 0 else x_pred.new_tensor(0.0))
        loss_stat = (local_stat_loss_2d(
            x_pred, x_gt, self._envelope_fn, self.stat_win,
            self.stat_var_weight, log_env_pred=log_env_pred,
            log_env_gt=log_env_gt)
            if is_2d and self.gamma_stat > 0 else x_pred.new_tensor(0.0))
        loss_kld = (kld_db_distribution(
            x_pred, x_gt, alpha_db=self.alpha_db,
            low_db=self.kld_low_db, high_db=self.kld_high_db,
            n_bins=self.kld_bins, eta=self.kld_eta)
            if self.gamma_kld > 0 else x_pred.new_tensor(0.0))

        loss_das = x_pred.new_tensor(0.0)
        if self.gamma_das > 0 and das_meta:
            loss_das, _ = das_image_loss(
                x_pred, x_gt,
                das_forward=das_meta.get("das_forward"),
                angles=das_meta.get("angles"),
                mode=self.das_loss_mode,
                alpha_db=self.das_alpha_db,
                beta_kld=self.das_beta_kld,
                kld_low_db=self.das_kld_low_db,
                kld_high_db=self.das_kld_high_db,
                kld_bins=self.das_kld_bins,
                kld_eta=self.das_kld_eta,
            )

        total = (
            loss_rf
            + self.gamma_env * loss_env
            + self.gamma_msle * loss_msle
            + self.gamma_constraint * loss_con
            + self.gamma_grad * loss_grad
            + self.gamma_stat * loss_stat
            + self.gamma_kld * loss_kld
            + self.gamma_das * loss_das
        )

        return total, {
            "loss_rf": loss_rf.item(),
            "loss_env": loss_env.item(),
            "loss_msle": loss_msle.item(),
            "loss_constraint": loss_con.item(),
            "loss_grad": loss_grad.item(),
            "loss_stat": loss_stat.item(),
            "loss_kld": loss_kld.item(),
            "loss_das": loss_das.item(),
            "loss_total": total.item(),
        }


__all__ = [
    "build_depth_weight",
    "rf_mse",
    "rf_nmse",
    "signed_log_l1",
    "envelope_l1",
    "msle",
    "kld_db_distribution",
    "kld_mslae_loss",
    "constraint_loss",
    "gradient_consistency_2d",
    "local_stat_loss_2d",
    "das_image_loss",
    "CombinedLoss",
]
