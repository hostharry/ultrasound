"""B-mode 图像域评估工具模块.

提供:
- load_epfl_settings()      : 从 epfl/settings 加载 DAS 重建参数
- regroup_frames()          : 按 acquisition_id 重组帧为完整 RF cube
- env_to_db()               : envelope → 归一化 dB
- bmode_snr / psnr / ssim   : 图像域指标 (在 dB B-mode 上)
- auto_contrast_roi()       : 自动 contrast ROI
- auto_speckle_roi()        : 自动 homogeneous speckle ROI
- compute_bmode_metrics()   : 一次性算全部图像域指标
- plot_bmode_comparison()   : GT / Recon / Init 三幅对比图

评估方式尽量对齐 EPFL 系列论文:
- PSNR / SSIM / B-mode SNR: 在 envelope + log compression 后的 B-mode 图像上计算
- contrast: 在 envelope 图像 ROI 上计算
- speckle quality: 在 homogeneous ROI 上计算 envelope SNR 和 2D ACF 的 axial/lateral FWHM
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ====================== EPFL settings loader ======================

def load_epfl_settings(settings_dir: str | Path) -> dict:
    """从 epfl/settings 目录加载 DAS 重建所需的全部参数.

    Returns
    -------
    dict with keys:
        probe_geometry : (3, n_elem) float64  [m]
        initial_time   : float                [s]
        fs             : float                [Hz]
        c              : float                [m/s]
        x_axis         : (n_x,) float64       [m]
        z_axis         : (n_z,) float64       [m]
        angles         : (n_angles,) float64  [rad]
    """
    import h5py
    import yaml

    sd = Path(settings_dir)
    cfg_path = sd / "beamforming_settings.yaml"
    text = cfg_path.read_text(encoding="utf-8").replace("\t", "    ")
    cfg = yaml.safe_load(text)

    mat_path = sd / "sequence_verasonics_ge9ld_87pws.mat"
    with h5py.File(mat_path, "r") as f:
        elem_pos = f["preSet/Trans/ElementPos"][()]
        pdata_origin = f["preSet/PData/Origin"][()].reshape(-1)
        pdata_delta = f["preSet/PData/PDelta"][()].reshape(-1)
        pdata_size = f["preSet/PData/Size"][()].reshape(-1)

    probe_geometry = elem_pos[:3].astype(np.float64) * 1e-3

    c = float(cfg["transducer"]["c0"])
    fs = float(cfg["acquisition"]["sampling_frequency"])
    wavelength = float(cfg["transducer"]["wavelength"])

    time_axis = np.load(sd / "time_axis.npy", allow_pickle=True).astype(np.float64)
    initial_time = float(time_axis[0])
    angles = np.load(sd / "steering_angles.npy", allow_pickle=True).astype(np.float64)

    n_z = int(round(float(pdata_size[0])))
    n_x = int(round(float(pdata_size[1])))
    x0 = float(pdata_origin[0]) * wavelength
    z0 = float(pdata_origin[2]) * wavelength
    dx = float(pdata_delta[0]) * wavelength
    dz = float(pdata_delta[2]) * wavelength
    x_axis = x0 + np.arange(n_x, dtype=np.float64) * dx
    z_axis = z0 + np.arange(n_z, dtype=np.float64) * dz

    return dict(
        probe_geometry=probe_geometry,
        initial_time=initial_time,
        fs=fs, c=c,
        x_axis=x_axis, z_axis=z_axis,
        angles=angles,
    )


# ====================== frame re-grouping =========================

def regroup_frames(
    frames: np.ndarray,
    acquisition_id: np.ndarray,
    frame_angle_idx: np.ndarray,
    n_angles: int = 87,
) -> Dict[int, np.ndarray]:
    """将 (n_frames, H, W) 按 acquisition_id 重组为 {acq_id: (n_angles, H, W)}.

    只保留拥有完整 n_angles 帧的 acquisition.
    """
    acqs: Dict[int, np.ndarray] = {}
    for aid in np.unique(acquisition_id):
        mask = acquisition_id == aid
        idx = np.where(mask)[0]
        if len(idx) < n_angles:
            continue
        ang_order = np.argsort(frame_angle_idx[idx])
        acqs[int(aid)] = frames[idx[ang_order]][:n_angles]
    return acqs


# ====================== envelope / dB =============================

def env_to_db(env: np.ndarray, dynamic_range: float = 60.0) -> np.ndarray:
    """归一化 envelope → dB (range: [-dynamic_range, 0])."""
    mx = float(env.max()) if env.size else 1e-10
    if mx > 0:
        env = env / mx
    return np.clip(20.0 * np.log10(np.clip(env, 1e-10, None)), -dynamic_range, 0.0)


# ====================== image-domain metrics ======================

def bmode_snr(gt_db: np.ndarray, pred_db: np.ndarray) -> float:
    """SNR (dB) on B-mode dB images."""
    sig = np.sum(gt_db ** 2)
    noise = np.sum((gt_db - pred_db) ** 2)
    return float(10.0 * np.log10(sig / (noise + 1e-10)))


def bmode_psnr(gt_db: np.ndarray, pred_db: np.ndarray) -> float:
    """PSNR (dB) on B-mode dB images."""
    mse = float(np.mean((gt_db - pred_db) ** 2))
    peak = float(np.max(np.abs(gt_db)))
    if mse < 1e-12:
        return float("inf")
    return float(20.0 * np.log10(peak / np.sqrt(mse)))


def bmode_ssim(gt_db: np.ndarray, pred_db: np.ndarray) -> float:
    """SSIM on B-mode dB images (via scikit-image)."""
    from skimage.metrics import structural_similarity as _ssim
    dr = float(gt_db.max() - gt_db.min())
    if dr < 1e-10:
        dr = 1.0
    return float(_ssim(gt_db, pred_db, data_range=dr))


# ====================== contrast ==================================

def contrast_ratio(
    env_img: np.ndarray,
    roi_inside: Tuple[int, int, int, int],
    roi_outside: Tuple[int, int, int, int],
) -> float:
    """Contrast in dB on envelope images.

    roi = (z_start, z_end, x_start, x_end) pixel indices.
    """
    z0i, z1i, x0i, x1i = roi_inside
    z0o, z1o, x0o, x1o = roi_outside
    s_in = float(np.mean(env_img[z0i:z1i, x0i:x1i]))
    s_out = float(np.mean(env_img[z0o:z1o, x0o:x1o]))
    return float(20.0 * np.log10((s_in + 1e-10) / (s_out + 1e-10)))


def contrast_to_noise_ratio(
    env_img: np.ndarray,
    roi_inside: Tuple[int, int, int, int],
    roi_outside: Tuple[int, int, int, int],
) -> float:
    """CNR on envelope images using the same inside/outside ROIs."""
    z0i, z1i, x0i, x1i = roi_inside
    z0o, z1o, x0o, x1o = roi_outside
    patch_in = env_img[z0i:z1i, x0i:x1i]
    patch_out = env_img[z0o:z1o, x0o:x1o]
    mu_in = float(np.mean(patch_in))
    mu_out = float(np.mean(patch_out))
    var_in = float(np.var(patch_in))
    var_out = float(np.var(patch_out))
    return float(np.abs(mu_in - mu_out) / np.sqrt(var_in + var_out + 1e-10))


def auto_contrast_roi(
    bmode_db: np.ndarray,
    x_axis_mm: np.ndarray,
    z_axis_mm: np.ndarray,
    roi_radius_mm: float = 3.0,
) -> Tuple[float, dict]:
    """自动在 B-mode 最暗区域 vs 上方背景计算 contrast.

    Returns
    -------
    contrast_dB : float
    roi_info    : dict  {"roi_inside": (...), "roi_outside": (...)}
    """
    from scipy.ndimage import uniform_filter

    dx = float(np.abs(x_axis_mm[1] - x_axis_mm[0])) if len(x_axis_mm) > 1 else 1.0
    dz = float(np.abs(z_axis_mm[1] - z_axis_mm[0])) if len(z_axis_mm) > 1 else 1.0
    kx = max(1, int(round(2 * roi_radius_mm / dx)))
    kz = max(1, int(round(2 * roi_radius_mm / dz)))

    smoothed = uniform_filter(bmode_db, size=(kz, kx))
    interior = smoothed[kz:-kz, kx:-kx]
    mi = np.unravel_index(np.argmin(interior), interior.shape)
    cz, cx = mi[0] + kz, mi[1] + kx

    hz, hx = kz // 2, kx // 2
    roi_in = (cz - hz, cz + hz, cx - hx, cx + hx)
    out_cz = max(hz, cz - kz)
    roi_out = (out_cz - hz, out_cz + hz, cx - hx, cx + hx)

    proxy = float(
        np.mean(bmode_db[roi_in[0]:roi_in[1], roi_in[2]:roi_in[3]]) -
        np.mean(bmode_db[roi_out[0]:roi_out[1], roi_out[2]:roi_out[3]])
    )
    return proxy, {"roi_inside": roi_in, "roi_outside": roi_out}


# ====================== speckle ===================================

def _clip_roi(
    roi: Tuple[int, int, int, int],
    shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    z0, z1, x0, x1 = roi
    h, w = shape
    z0 = max(0, min(h - 1, z0))
    z1 = max(z0 + 1, min(h, z1))
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    return z0, z1, x0, x1


def auto_speckle_roi(
    bmode_db: np.ndarray,
    x_axis_mm: np.ndarray,
    z_axis_mm: np.ndarray,
    roi_w_mm: float = 3.0,
    roi_h_mm: float = 3.0,
) -> Tuple[Tuple[int, int, int, int], dict]:
    """自动选择 homogeneous speckle ROI.

    采用 GT B-mode 的局部低方差区域，并限制在图像中部，尽量避开边界与极暗阴影。
    """
    from scipy.ndimage import uniform_filter

    dx = float(np.abs(x_axis_mm[1] - x_axis_mm[0])) if len(x_axis_mm) > 1 else 1.0
    dz = float(np.abs(z_axis_mm[1] - z_axis_mm[0])) if len(z_axis_mm) > 1 else 1.0
    kx = max(8, int(round(roi_w_mm / dx)))
    kz = max(8, int(round(roi_h_mm / dz)))
    if kx >= bmode_db.shape[1]:
        kx = max(2, bmode_db.shape[1] // 3)
    if kz >= bmode_db.shape[0]:
        kz = max(2, bmode_db.shape[0] // 3)

    mean_map = uniform_filter(bmode_db, size=(kz, kx))
    mean_sq_map = uniform_filter(bmode_db ** 2, size=(kz, kx))
    var_map = np.maximum(mean_sq_map - mean_map ** 2, 0.0)

    h, w = bmode_db.shape
    z_margin = kz // 2
    x_margin = kx // 2
    z0 = max(z_margin, int(0.2 * h))
    z1 = min(h - z_margin, int(0.8 * h))
    x0 = max(x_margin, int(0.2 * w))
    x1 = min(w - x_margin, int(0.8 * w))

    sub_var = var_map[z0:z1, x0:x1].copy()
    sub_mean = mean_map[z0:z1, x0:x1]
    p20, p80 = np.percentile(bmode_db, [20, 80])
    invalid = (sub_mean < p20) | (sub_mean > p80)
    sub_var[invalid] = np.inf

    if not np.isfinite(sub_var).any():
        sub_var = var_map[z0:z1, x0:x1]

    cz_rel, cx_rel = np.unravel_index(np.argmin(sub_var), sub_var.shape)
    cz = z0 + cz_rel
    cx = x0 + cx_rel
    hz = kz // 2
    hx = kx // 2
    roi = _clip_roi((cz - hz, cz + hz, cx - hx, cx + hx), bmode_db.shape)
    return roi, {"center": (int(cz), int(cx)), "roi_shape_px": (int(kz), int(kx))}


def speckle_snr(env_img: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """Envelope ROI 上的 speckle SNR = mean / std."""
    z0, z1, x0, x1 = _clip_roi(roi, env_img.shape)
    patch = env_img[z0:z1, x0:x1]
    mu = float(np.mean(patch))
    sigma = float(np.std(patch))
    return float(mu / (sigma + 1e-10))


def _fwhm_from_profile(profile: np.ndarray, axis_mm: np.ndarray, level: float = 0.5) -> float:
    above = profile >= level
    if not np.any(above):
        return float("nan")
    idx = np.where(above)[0]
    return float(axis_mm[idx[-1]] - axis_mm[idx[0]])


def speckle_acf_fwhm(
    env_img: np.ndarray,
    roi: Tuple[int, int, int, int],
    x_axis_mm: np.ndarray,
    z_axis_mm: np.ndarray,
) -> Tuple[float, float]:
    """Speckle ROI 的 2D ACF 横向/轴向 FWHM (mm)."""
    z0, z1, x0, x1 = _clip_roi(roi, env_img.shape)
    patch = env_img[z0:z1, x0:x1].astype(np.float64)
    patch = patch - np.mean(patch)
    if np.allclose(patch, 0.0):
        return float("nan"), float("nan")

    h, w = patch.shape
    fft_shape = (2 * h - 1, 2 * w - 1)
    f = np.fft.fft2(patch, s=fft_shape)
    acf = np.fft.ifft2(np.abs(f) ** 2).real
    acf = np.fft.fftshift(acf)
    acf /= max(float(acf.max()), 1e-10)

    dz = float(np.abs(z_axis_mm[1] - z_axis_mm[0])) if len(z_axis_mm) > 1 else 1.0
    dx = float(np.abs(x_axis_mm[1] - x_axis_mm[0])) if len(x_axis_mm) > 1 else 1.0
    z_lag = (np.arange(acf.shape[0]) - (acf.shape[0] // 2)) * dz
    x_lag = (np.arange(acf.shape[1]) - (acf.shape[1] // 2)) * dx

    cz = acf.shape[0] // 2
    cx = acf.shape[1] // 2
    axial_profile = acf[:, cx]
    lateral_profile = acf[cz, :]

    fwhm_axial = _fwhm_from_profile(axial_profile, z_lag, level=0.5)
    fwhm_lateral = _fwhm_from_profile(lateral_profile, x_lag, level=0.5)
    return abs(fwhm_axial), abs(fwhm_lateral)


# ====================== reference cache ===========================

def compute_reference(
    gt_env: np.ndarray,
    x_axis_mm: np.ndarray,
    z_axis_mm: np.ndarray,
    dynamic_range: float = 60.0,
) -> dict:
    """Pre-compute GT-only quantities: ROIs and GT-side metrics.

    These never change across models, so compute once and reuse.
    """
    gt_db = env_to_db(gt_env, dynamic_range)
    _, contrast_roi = auto_contrast_roi(gt_db, x_axis_mm, z_axis_mm)
    speckle_roi, _ = auto_speckle_roi(gt_db, x_axis_mm, z_axis_mm)

    return {
        "gt_db": gt_db,
        "contrast_roi": contrast_roi,
        "speckle_roi": speckle_roi,
        "contrast_gt_dB": contrast_ratio(gt_env, contrast_roi["roi_inside"], contrast_roi["roi_outside"]),
        "cnr_gt": contrast_to_noise_ratio(gt_env, contrast_roi["roi_inside"], contrast_roi["roi_outside"]),
        "speckle_SNR_gt": speckle_snr(gt_env, speckle_roi),
        "speckle_FWHM_axial_gt_mm": speckle_acf_fwhm(gt_env, speckle_roi, x_axis_mm, z_axis_mm)[0],
        "speckle_FWHM_lateral_gt_mm": speckle_acf_fwhm(gt_env, speckle_roi, x_axis_mm, z_axis_mm)[1],
    }


def compute_pred_metrics(
    pred_env: np.ndarray,
    gt_env: np.ndarray,
    ref: dict,
    x_axis_mm: np.ndarray,
    z_axis_mm: np.ndarray,
    dynamic_range: float = 60.0,
) -> Dict[str, float]:
    """Compute metrics for a pred image using pre-computed reference.

    Avoids re-computing ROIs and GT-side metrics.
    """
    gt_db = ref["gt_db"]
    pred_db = env_to_db(pred_env, dynamic_range)

    roi_in = ref["contrast_roi"]["roi_inside"]
    roi_out = ref["contrast_roi"]["roi_outside"]
    sp_roi = ref["speckle_roi"]

    fwhm_ax_pred, fwhm_lat_pred = speckle_acf_fwhm(pred_env, sp_roi, x_axis_mm, z_axis_mm)

    return {
        "bmode_SNR_dB": bmode_snr(gt_db, pred_db),
        "bmode_PSNR_dB": bmode_psnr(gt_db, pred_db),
        "bmode_SSIM": bmode_ssim(gt_db, pred_db),
        "contrast_gt_dB": ref["contrast_gt_dB"],
        "contrast_pred_dB": contrast_ratio(pred_env, roi_in, roi_out),
        "cnr_gt": ref["cnr_gt"],
        "cnr_pred": contrast_to_noise_ratio(pred_env, roi_in, roi_out),
        "speckle_SNR_gt": ref["speckle_SNR_gt"],
        "speckle_SNR_pred": speckle_snr(pred_env, sp_roi),
        "speckle_FWHM_axial_gt_mm": ref["speckle_FWHM_axial_gt_mm"],
        "speckle_FWHM_axial_pred_mm": fwhm_ax_pred,
        "speckle_FWHM_lateral_gt_mm": ref["speckle_FWHM_lateral_gt_mm"],
        "speckle_FWHM_lateral_pred_mm": fwhm_lat_pred,
    }


# ====================== all-in-one metric =========================

def compute_bmode_metrics(
    gt_env: np.ndarray,
    pred_env: np.ndarray,
    x_axis_mm: np.ndarray,
    z_axis_mm: np.ndarray,
    dynamic_range: float = 60.0,
    reference: Optional[dict] = None,
) -> Dict[str, float]:
    """一次性计算全部 B-mode 图像域指标.

    如果传入 reference (由 compute_reference 生成), 则跳过 GT 侧重复计算.
    """
    if reference is not None:
        return compute_pred_metrics(pred_env, gt_env, reference,
                                    x_axis_mm, z_axis_mm, dynamic_range)

    gt_db = env_to_db(gt_env, dynamic_range)
    pred_db = env_to_db(pred_env, dynamic_range)

    _, contrast_roi = auto_contrast_roi(gt_db, x_axis_mm, z_axis_mm)
    cr_gt = contrast_ratio(gt_env, contrast_roi["roi_inside"], contrast_roi["roi_outside"])
    cr_pred = contrast_ratio(pred_env, contrast_roi["roi_inside"], contrast_roi["roi_outside"])
    cnr_gt = contrast_to_noise_ratio(gt_env, contrast_roi["roi_inside"], contrast_roi["roi_outside"])
    cnr_pred = contrast_to_noise_ratio(pred_env, contrast_roi["roi_inside"], contrast_roi["roi_outside"])

    speckle_roi, _ = auto_speckle_roi(gt_db, x_axis_mm, z_axis_mm)
    snr_gt = speckle_snr(gt_env, speckle_roi)
    snr_pred = speckle_snr(pred_env, speckle_roi)
    fwhm_ax_gt, fwhm_lat_gt = speckle_acf_fwhm(gt_env, speckle_roi, x_axis_mm, z_axis_mm)
    fwhm_ax_pred, fwhm_lat_pred = speckle_acf_fwhm(pred_env, speckle_roi, x_axis_mm, z_axis_mm)

    return {
        "bmode_SNR_dB": bmode_snr(gt_db, pred_db),
        "bmode_PSNR_dB": bmode_psnr(gt_db, pred_db),
        "bmode_SSIM": bmode_ssim(gt_db, pred_db),
        "contrast_gt_dB": cr_gt,
        "contrast_pred_dB": cr_pred,
        "cnr_gt": cnr_gt,
        "cnr_pred": cnr_pred,
        "speckle_SNR_gt": snr_gt,
        "speckle_SNR_pred": snr_pred,
        "speckle_FWHM_axial_gt_mm": fwhm_ax_gt,
        "speckle_FWHM_axial_pred_mm": fwhm_ax_pred,
        "speckle_FWHM_lateral_gt_mm": fwhm_lat_gt,
        "speckle_FWHM_lateral_pred_mm": fwhm_lat_pred,
    }


# ====================== plotting ==================================

def plot_bmode_comparison(
    gt_db: np.ndarray,
    pred_db: np.ndarray,
    init_db: np.ndarray,
    x_axis_mm: np.ndarray,
    z_axis_mm: np.ndarray,
    save_path: str,
    dynamic_range: float = 60.0,
    model_name: str = "Recon",
    metrics: Optional[Dict[str, float]] = None,
):
    """GT / Recon / Init 三幅 B-mode dB 对比图.

    Parameters
    ----------
    metrics : 如果给出, 在标题中显示 PSNR / SSIM.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [x_axis_mm[0], x_axis_mm[-1], z_axis_mm[-1], z_axis_mm[0]]
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    for ax, img, title in zip(
        axes,
        [gt_db, pred_db, init_db],
        ["Ground Truth", model_name, "Init (A\u2020y)"],
    ):
        ax.imshow(img, cmap="gray", vmin=-dynamic_range, vmax=0,
                  aspect="equal", extent=extent)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
        ax.set_title(title)

    suptitle = f"DAS B-mode | {model_name} | {dynamic_range:.0f} dB"
    if metrics:
        suptitle += (f" | PSNR={metrics.get('bmode_PSNR_dB', 0):.2f} dB"
                     f" | SSIM={metrics.get('bmode_SSIM', 0):.4f}")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# ====================== summary I/O ==============================

def save_bmode_summary(
    all_metrics: List[Dict[str, float]],
    output_dir: str,
):
    """将多个 acquisition 的 B-mode 指标汇总保存为 JSON."""
    os.makedirs(output_dir, exist_ok=True)
    per_acq_path = os.path.join(output_dir, "bmode_per_acquisition.json")
    with open(per_acq_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    if not all_metrics:
        return {}

    keys = [k for k in all_metrics[0] if k != "acquisition"]
    summary: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))

    summary_path = os.path.join(output_dir, "bmode_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
