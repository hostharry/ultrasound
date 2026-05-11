import numpy as np
import torch
from typing import Dict, List


def calc_snr(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    signal_power = (y_true ** 2).sum(dim=(-2, -1))
    noise_power = ((y_true - y_pred) ** 2).sum(dim=(-2, -1))
    return 10 * torch.log10(signal_power / (noise_power + 1e-10))


def calc_nmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return ((y_true - y_pred) ** 2).sum(dim=(-2, -1)) / ((y_true ** 2).sum(dim=(-2, -1)) + 1e-10)


def envelope_np(x: np.ndarray) -> np.ndarray:
    """Hilbert 包络 (numpy)"""
    from scipy.signal import hilbert as sp_hilbert
    return np.abs(sp_hilbert(x, axis=-1)).astype(np.float32)


def to_db(env: np.ndarray, dynamic_range: float = 60.0) -> np.ndarray:
    """包络 -> 对数压缩 dB"""
    env = np.clip(env, 1e-10, None)
    denom = max(float(env.max()), 1e-10)
    db = 20.0 * np.log10(env / denom)
    return np.clip(db, -dynamic_range, 0.0)


def _calc_psnr(mse: float, peak: float = 1.0) -> float:
    if mse <= 1e-12:
        return float("inf")
    return 20 * np.log10(peak / np.sqrt(mse))


def _calc_ssim_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1D SSIM 的简化实现，避免额外依赖"""
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    mu_x = np.mean(y_true)
    mu_y = np.mean(y_pred)
    var_x = np.var(y_true)
    var_y = np.var(y_pred)
    cov_xy = np.mean((y_true - mu_x) * (y_pred - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    )
    return float(ssim)


def compute_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """对单个样本计算指标 (1D RF 信号)"""
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    mse = float(np.mean((y_true - y_pred) ** 2))
    signal_power = float(np.sum(y_true ** 2))
    noise_power = float(np.sum((y_true - y_pred) ** 2))
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    nmse = noise_power / (signal_power + 1e-10)
    psnr = _calc_psnr(mse, peak=max(float(np.max(np.abs(y_true))), 1.0))
    ssim = _calc_ssim_1d(y_true, y_pred)

    env_true = envelope_np(y_true)
    env_pred = envelope_np(y_pred)
    env_mse = float(np.mean((env_true - env_pred) ** 2))
    corr = np.corrcoef(env_true.ravel(), env_pred.ravel())[0, 1]
    env_corr = float(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))

    return {
        "SNR_dB": float(snr),
        "NMSE": float(nmse),
        "MSE": mse,
        "PSNR_dB": float(psnr),
        "SSIM_1D": float(ssim),
        "Env_MSE": env_mse,
        "Env_Corr": env_corr,
    }


def summarize_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = metrics_list[0].keys() if metrics_list else []
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out
