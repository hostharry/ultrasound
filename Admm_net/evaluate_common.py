"""1D / 2D 评估脚本共享逻辑

各模型评估脚本只需提供 load_model(ckpt_path, device) -> (model, meta),
然后调用 evaluate_1d / evaluate_2d 即可。
"""

import os
import sys
_UTILS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Utils')
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
import argparse
import numpy as np
import torch

from data import UltrasoundDataset, UltrasoundFrameDataset, split_indices
from metrics import (
    compute_sample_metrics, summarize_metrics,
    calc_snr, calc_nmse, envelope_np, to_db,
)
from report import generate_evaluation_report, save_eval_summary
from visualization import (
    parse_train_log,
    plot_training_curves,
    plot_signal_comparison,
    plot_error_distribution,
    plot_envelope_comparison_2d,
)


# ======================== 推理 ========================

@torch.no_grad()
def run_inference(model, dataset, indices, batch_size=32):
    op = dataset.op
    all_pred, all_gt, all_init = [], [], []
    n = len(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        x_input, y_target, y_k = dataset.get_batch(idx)
        y_sub = y_k if y_k is not None else op.A(x_input)
        x_hat, _ = model(y_sub, op, return_aux=True)
        x_init = op.At(y_sub)
        all_pred.append(x_hat.cpu().numpy())
        all_gt.append(y_target.cpu().numpy())
        all_init.append(x_init.cpu().numpy())
    return {
        "pred": np.concatenate(all_pred, axis=0),
        "gt": np.concatenate(all_gt, axis=0),
        "init": np.concatenate(all_init, axis=0),
    }


# ======================== DAS 可视化 ========================

def _env_to_db(env, dynamic_range):
    mx = env.max()
    if mx > 0:
        env = env / mx
    return np.clip(20.0 * np.log10(np.clip(env, 1e-10, None)), -dynamic_range, 0)


def _compute_and_plot_das(dataset, pred, init, args, out_dir, model_name):
    """从 npz 元数据执行 DAS 波束形成并绘制 B-mode 对比图."""
    if getattr(dataset, "rf_data_3d", None) is None:
        return
    if getattr(dataset, "scan_x_axis", None) is None:
        print("  [跳过 DAS] npz 中无 scan 网格 (scan_x_axis)")
        return

    try:
        from das import das_pw_rf
    except ImportError:
        print("  [跳过 DAS] 缺少 das.py 模块")
        return

    rf_3d = dataset.rf_data_3d
    angles = dataset.angles
    probe_geom = dataset.probe_geometry
    t0 = dataset.initial_time
    fs_val = dataset.fs
    c_val = dataset.c
    x_axis = dataset.scan_x_axis
    z_axis = dataset.scan_z_axis
    sel = dataset.selected_angles

    if sel is not None:
        rf_gt = rf_3d[sel]
        ang_gt = angles[sel]
    else:
        rf_gt = rf_3d
        ang_gt = angles

    print("\n  计算 GT DAS B-mode ...")
    das_gt = das_pw_rf(rf_gt, ang_gt, probe_geom, t0, fs_val, c_val,
                       x_axis, z_axis)

    das_pred_img, das_init_img = None, None

    if args.eval_all:
        n_sel = rf_gt.shape[0]
        n_elem = rf_gt.shape[1]
        n_samp = rf_gt.shape[2]
        n_expected = n_sel * n_elem
        n_actual = pred.shape[0]

        if n_actual == n_expected:
            pred_3d = pred.squeeze().reshape(n_sel, n_elem, n_samp)
            init_3d = init.squeeze().reshape(n_sel, n_elem, n_samp)

            print("  计算 Recon DAS B-mode ...")
            das_pred_img = das_pw_rf(pred_3d, ang_gt, probe_geom, t0,
                                     fs_val, c_val, x_axis, z_axis)
            print("  计算 Init DAS B-mode ...")
            das_init_img = das_pw_rf(init_3d, ang_gt, probe_geom, t0,
                                     fs_val, c_val, x_axis, z_axis)
        else:
            print(f"  [跳过 Recon/Init DAS] 样本数不匹配: "
                  f"pred={n_actual}, 需要={n_expected}")
    else:
        print("  [提示] 使用 --eval_all 可生成 Recon/Init DAS 对比图")

    _plot_das_bmode(das_gt, das_pred_img, das_init_img,
                    x_axis, z_axis, out_dir,
                    getattr(args, "dynamic_range", 60.0), model_name)

    das_save = {"das_gt": das_gt, "x_axis": x_axis, "z_axis": z_axis}
    if das_pred_img is not None:
        das_save["das_pred"] = das_pred_img
        das_save["das_init"] = das_init_img
    np.savez_compressed(os.path.join(out_dir, "das_results.npz"), **das_save)
    print("  das_results.npz saved")


def _compute_and_plot_das_2d(npz_path, pred_frames, gt_frames, init_frames,
                              out_dir, dynamic_range=60.0, model_name="Recon"):
    """从 2D 帧级重建结果执行 DAS 波束形成.

    Parameters
    ----------
    npz_path : str
        帧级 npz 路径 (含 DAS 元数据: angles, probe_geometry 等)
    pred_frames : ndarray (n_frames, H, W)
    gt_frames   : ndarray (n_frames, H, W)
    init_frames : ndarray (n_frames, H, W)
    """
    data = np.load(npz_path, allow_pickle=True)

    if "scan_x_axis" not in data:
        print("  [跳过 DAS] npz 中无 scan 网格 (scan_x_axis)")
        return
    if "angles" not in data:
        print("  [跳过 DAS] npz 中无 angles")
        return

    try:
        from das import das_pw_rf
    except ImportError:
        print("  [跳过 DAS] 缺少 das.py 模块")
        return

    angles = np.asarray(data["angles"], dtype=np.float64)
    sel = data["selected_angles"] if "selected_angles" in data else None
    if sel is not None:
        sel = np.asarray(sel, dtype=int)
        if sel.size == 0:
            sel = None
    probe_geom = np.asarray(data["probe_geometry"])
    t0 = float(data["initial_time"])
    fs_val = float(data["fs"])
    c_val = float(data["c"])
    x_axis = np.asarray(data["scan_x_axis"])
    z_axis = np.asarray(data["scan_z_axis"])

    ang = angles[sel] if sel is not None else angles

    print("\n  计算 GT DAS B-mode ...")
    das_gt = das_pw_rf(gt_frames, ang, probe_geom, t0, fs_val, c_val,
                       x_axis, z_axis)

    print("  计算 Recon DAS B-mode ...")
    das_pred = das_pw_rf(pred_frames, ang, probe_geom, t0, fs_val, c_val,
                         x_axis, z_axis)

    print("  计算 Init DAS B-mode ...")
    das_init = das_pw_rf(init_frames, ang, probe_geom, t0, fs_val, c_val,
                         x_axis, z_axis)

    _plot_das_bmode(das_gt, das_pred, das_init, x_axis, z_axis,
                    out_dir, dynamic_range, model_name)

    np.savez_compressed(os.path.join(out_dir, "das_results.npz"),
                        das_gt=das_gt, das_pred=das_pred, das_init=das_init,
                        x_axis=x_axis, z_axis=z_axis)
    print("  das_results.npz saved")


def _plot_das_bmode(das_gt, das_pred, das_init, x_axis, z_axis,
                    save_dir, dynamic_range=60.0, model_name="Recon"):
    """DAS B-mode 对比图: GT / Recon / Init."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_mm = x_axis * 1e3
    z_mm = z_axis * 1e3
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    has_recon = das_pred is not None and das_init is not None
    n_cols = 3 if has_recon else 1

    fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    db_gt = _env_to_db(das_gt, dynamic_range)

    images = [("Ground Truth (DAS)", db_gt, das_gt)]
    if has_recon:
        db_pred = _env_to_db(das_pred, dynamic_range)
        db_init = _env_to_db(das_init, dynamic_range)
        images.append((f"{model_name} (DAS)", db_pred, das_pred))
        images.append(("Init / A†y (DAS)", db_init, das_init))

    for col, (title, db_img, _) in enumerate(images):
        ax = axes[0, col]
        ax.imshow(db_img, aspect="equal", cmap="gray",
                  vmin=-dynamic_range, vmax=0, extent=extent)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")

    if has_recon:
        err_recon = np.abs(das_gt - das_pred)
        err_init = np.abs(das_gt - das_init)
        e_max = max(np.percentile(err_init, 99), 1e-10)

        ax_e1 = axes[1, 0]
        im_e = ax_e1.imshow(err_recon, aspect="equal", cmap="hot",
                            vmin=0, vmax=e_max, extent=extent)
        ax_e1.set_title(f"|GT - {model_name}|", fontsize=12)
        ax_e1.set_xlabel("Lateral (mm)")
        ax_e1.set_ylabel("Depth (mm)")

        ax_e2 = axes[1, 1]
        ax_e2.imshow(err_init, aspect="equal", cmap="hot",
                     vmin=0, vmax=e_max, extent=extent)
        ax_e2.set_title("|GT - Init|", fontsize=12)
        ax_e2.set_xlabel("Lateral (mm)")
        ax_e2.set_ylabel("Depth (mm)")
        fig.colorbar(im_e, ax=[ax_e1, ax_e2], label="Envelope Error",
                     shrink=0.7, pad=0.02)

        ax_lat = axes[1, 2]
        mid_z = len(z_mm) // 2
        ax_lat.plot(x_mm, db_gt[mid_z], label="GT", lw=1.0, alpha=0.8)
        ax_lat.plot(x_mm, db_pred[mid_z], label=model_name, lw=1.0, alpha=0.8)
        ax_lat.plot(x_mm, db_init[mid_z], label="Init", lw=0.8, alpha=0.5, ls="--")
        ax_lat.set_title(f"Lateral Profile (z={z_mm[mid_z]:.1f} mm)", fontsize=12)
        ax_lat.set_xlabel("Lateral (mm)")
        ax_lat.set_ylabel("dB")
        ax_lat.legend(fontsize=9)
        ax_lat.grid(True, alpha=0.3)
    else:
        axes[1, 0].axis("off")

    fig.suptitle(f"DAS B-mode  |  {model_name}  |  {dynamic_range:.0f} dB",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_dir, "das_bmode_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  das_bmode_comparison.png saved")


# ======================== 可视化 ========================

def _plot_bmode(gt, pred, init, metrics_list, save_dir, dataset,
                dynamic_range=60.0, model_name="Recon"):
    """全量 B-mode 图像对比 — RF 差值 + 缩窄动态范围 + 局部放大."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    def _to_env(arr):
        signals = arr.squeeze()
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        return np.array([envelope_np(s) for s in signals])

    env_gt = _to_env(gt)
    env_pred = _to_env(pred)
    env_init = _to_env(init)

    bgt = to_db(env_gt, dynamic_range)
    bpred = to_db(env_pred, dynamic_range)
    binit = to_db(env_init, dynamic_range)

    n_lines, n_samples = bgt.shape
    fs = getattr(dataset, 'fs', None)
    c = 1540.0
    if fs:
        depth_mm = np.arange(n_samples) / fs * c / 2 * 1e3
        extent = [0, depth_mm[-1], n_lines - 0.5, -0.5]
        xlabel = "Depth (mm)"
    else:
        extent = [0, n_samples, n_lines - 0.5, -0.5]
        xlabel = "Sample"

    snrs = np.array([m["SNR_dB"] for m in metrics_list])
    avg_snr = np.mean(snrs)
    avg_nmse = np.mean([m["NMSE"] for m in metrics_list])

    gt_rf = gt.squeeze()
    pred_rf = pred.squeeze()
    init_rf = init.squeeze()
    if gt_rf.ndim == 1:
        gt_rf, pred_rf, init_rf = gt_rf[np.newaxis, :], pred_rf[np.newaxis, :], init_rf[np.newaxis, :]

    err_rf_recon = np.abs(gt_rf - pred_rf)
    err_rf_init = np.abs(gt_rf - init_rf)

    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    vmin, vmax = -dynamic_range, 0
    titles_top = ["Ground Truth", model_name, "Init (A\u2020y)"]
    data_top = [bgt, bpred, binit]
    for col, (d, t) in enumerate(zip(data_top, titles_top)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(d, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(t, fontsize=13, fontweight="bold" if col == 1 else "normal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Scan Line")

    rf_vmax = max(np.percentile(err_rf_init, 99), 1e-6)
    ax_r1 = fig.add_subplot(gs[1, 0])
    im_rf = ax_r1.imshow(err_rf_recon, aspect="auto", cmap="magma", vmin=0, vmax=rf_vmax, extent=extent)
    ax_r1.set_title(f"|GT - {model_name}|  RF Error", fontsize=13)
    ax_r1.set_xlabel(xlabel); ax_r1.set_ylabel("Scan Line")

    ax_r2 = fig.add_subplot(gs[1, 1])
    ax_r2.imshow(err_rf_init, aspect="auto", cmap="magma", vmin=0, vmax=rf_vmax, extent=extent)
    ax_r2.set_title("|GT - Init|  RF Error", fontsize=13)
    ax_r2.set_xlabel(xlabel); ax_r2.set_ylabel("Scan Line")
    fig.colorbar(im_rf, ax=[ax_r1, ax_r2], label="RF Amplitude Error", shrink=0.8, pad=0.02)

    ax_snr = fig.add_subplot(gs[1, 2])
    ax_snr.barh(range(len(snrs)), snrs, color="steelblue", alpha=0.7, height=0.8)
    ax_snr.axvline(avg_snr, color="red", ls="--", lw=1.5, label=f"Mean={avg_snr:.2f} dB")
    ax_snr.set_xlabel("SNR (dB)")
    ax_snr.set_ylabel("Sample Index")
    ax_snr.set_title("Per-sample SNR", fontsize=13)
    ax_snr.legend(fontsize=9)
    ax_snr.invert_yaxis()
    ax_snr.grid(True, alpha=0.3, axis="x")

    narrow_dr = 20.0
    narrow_center = -15.0
    narrow_vmin = narrow_center - narrow_dr / 2
    narrow_vmax = narrow_center + narrow_dr / 2
    ax_n1 = fig.add_subplot(gs[2, 0])
    ax_n1.imshow(bgt, aspect="auto", cmap="gray", vmin=narrow_vmin, vmax=narrow_vmax, extent=extent)
    ax_n1.set_title(f"GT (narrow: {narrow_vmin:.0f}~{narrow_vmax:.0f} dB)", fontsize=12)
    ax_n1.set_xlabel(xlabel); ax_n1.set_ylabel("Scan Line")

    ax_n2 = fig.add_subplot(gs[2, 1])
    im_n = ax_n2.imshow(bpred, aspect="auto", cmap="gray", vmin=narrow_vmin, vmax=narrow_vmax, extent=extent)
    ax_n2.set_title(f"{model_name} (narrow: {narrow_vmin:.0f}~{narrow_vmax:.0f} dB)", fontsize=12)
    ax_n2.set_xlabel(xlabel); ax_n2.set_ylabel("Scan Line")
    fig.colorbar(im_n, ax=[ax_n1, ax_n2], label="dB", shrink=0.8, pad=0.02)

    ax_prof = fig.add_subplot(gs[2, 2])
    mid = n_lines // 2
    ax_prof.plot(bgt[mid], label="GT", alpha=0.8, lw=0.8)
    ax_prof.plot(bpred[mid], label=model_name, alpha=0.8, lw=0.8)
    ax_prof.plot(binit[mid], label="Init", alpha=0.5, lw=0.6, ls="--")
    ax_prof.set_title(f"Envelope Profile (Line {mid})", fontsize=13)
    ax_prof.set_xlabel(xlabel)
    ax_prof.set_ylabel("dB")
    ax_prof.legend(fontsize=9)
    ax_prof.grid(True, alpha=0.3)

    fig.suptitle(
        f"B-mode Reconstruction  |  {model_name}  |  {n_lines} lines \u00d7 {n_samples} samples  |  "
        f"Avg SNR: {avg_snr:.2f} dB  |  Avg NMSE: {avg_nmse:.4f}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    plt.savefig(os.path.join(save_dir, "bmode_reconstruction.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  bmode_reconstruction.png saved ({n_lines}\u00d7{n_samples})")


def _plot_reconstruction_detail(gt, pred, init, metrics_list, save_dir,
                                dynamic_range=60.0, model_name="Recon"):
    """Best / Median / Worst sample reconstruction detail."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = np.array([m["SNR_dB"] for m in metrics_list])
    ranking = np.argsort(snrs)
    picks = {
        "Best":   ranking[-1],
        "Median": ranking[len(ranking) // 2],
        "Worst":  ranking[0],
    }

    for tag, idx in picks.items():
        g = gt[idx].squeeze()
        p = pred[idx].squeeze()
        ini = init[idx].squeeze()
        snr_val = snrs[idx]
        nmse_val = metrics_list[idx]["NMSE"]
        t = np.arange(len(g))

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        axes[0].plot(t, g, label="Ground Truth", alpha=0.7, lw=0.5)
        axes[0].plot(t, p, label=model_name, alpha=0.7, lw=0.5)
        axes[0].plot(t, ini, label="A\u2020y (init)", alpha=0.3, lw=0.4, ls="--")
        axes[0].set_title(f"{tag} Sample #{idx}  |  SNR={snr_val:.2f} dB  |  NMSE={nmse_val:.4f}",
                          fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.2)

        env_g = to_db(envelope_np(g), dynamic_range)
        env_p = to_db(envelope_np(p), dynamic_range)
        env_i = to_db(envelope_np(ini), dynamic_range)
        axes[1].plot(t, env_g, label="GT Envelope", alpha=0.7, lw=0.8)
        axes[1].plot(t, env_p, label=f"{model_name} Envelope", alpha=0.7, lw=0.8)
        axes[1].plot(t, env_i, label="Init Envelope", alpha=0.4, lw=0.6, ls="--")
        axes[1].set_ylabel("dB")
        axes[1].set_title("Envelope (dB)")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.2)

        err_recon = np.abs(g - p)
        err_init = np.abs(g - ini)
        axes[2].plot(t, err_init, label="|GT - Init|", alpha=0.5, lw=0.5, color="gray")
        axes[2].plot(t, err_recon, label=f"|GT - {model_name}|", alpha=0.7, lw=0.6, color="red")
        axes[2].set_ylabel("Absolute Error")
        axes[2].set_xlabel("Sample")
        axes[2].set_title("Reconstruction Error")
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.2)

        plt.tight_layout()
        fname = f"recon_detail_{tag.lower()}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {fname} saved ({tag}: Sample #{idx}, SNR={snr_val:.2f} dB)")


# ======================== 主评估流程 ========================

def evaluate_1d(load_model_fn, model_name, args, extra_viz_fn=None):
    """1D 模型通用评估流程.

    Parameters
    ----------
    load_model_fn : callable(ckpt_path, device) -> (model, meta)
    model_name : str  (用于图表标题, 如 "HASA-ADMM", "HASA-FISTA")
    args : argparse.Namespace
    extra_viz_fn : callable(model, dataset, eval_idx, device, out_dir) or None
        额外可视化 (如 ADMM 的 plot_layer_convergence)
    """
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = args.exp_dir
    ckpt_path = os.path.join(exp_dir, args.ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    out_dir = os.path.join(exp_dir, getattr(args, "eval_subdir", "eval_results"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n加载模型: {ckpt_path}")
    model, meta = load_model_fn(ckpt_path, device)
    args_dict = meta.get("args", {})
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  model={model_name}, params={num_params:,}")

    npz_path = args.npz or args_dict.get("npz", "../dataset_fdbf_energy_mu_8_9_15.npz")
    cs_ratio = args.cs_ratio or args_dict.get("cs_ratio", 8)
    print(f"  加载数据: {npz_path} (cs_ratio={cs_ratio})")
    dataset = UltrasoundDataset(npz_path, cs_ratio=cs_ratio, device="cpu").to(device)

    n_total = len(dataset)
    if args.eval_all:
        eval_idx = torch.arange(n_total, device=device)
        print(f"  评估全部 {n_total} 个样本")
    else:
        if "val_idx" in meta:
            eval_idx = meta["val_idx"].to(device)
        else:
            _, eval_idx = split_indices(
                num_samples=n_total,
                val_ratio=args_dict.get("val_ratio", 0.1),
                seed=args_dict.get("seed", 42),
                split_mode=args_dict.get("split_mode", "group"),
                group_id=dataset.group_id,
            )
            eval_idx = eval_idx.to(device)
        print(f"  评估验证集 {len(eval_idx)} 个样本 (总 {n_total})")

    print("\n开始推理...")
    results = run_inference(model, dataset, eval_idx, batch_size=args.batch_size)
    pred, gt, init = results["pred"], results["gt"], results["init"]
    print(f"  pred shape: {pred.shape}")

    print("\n计算指标...")
    metrics_list = []
    for i in range(pred.shape[0]):
        m = compute_sample_metrics(gt[i].squeeze(), pred[i].squeeze())
        metrics_list.append(m)

    agg = summarize_metrics(metrics_list)
    print(f"  Avg SNR:   {agg['SNR_dB_mean']:.2f} \u00b1 {agg['SNR_dB_std']:.2f} dB")
    print(f"  Avg NMSE:  {agg['NMSE_mean']:.6f} \u00b1 {agg['NMSE_std']:.6f}")
    print(f"  Avg PSNR:  {agg['PSNR_dB_mean']:.2f} \u00b1 {agg['PSNR_dB_std']:.2f} dB")
    print(f"  Avg SSIM1D:{agg['SSIM_1D_mean']:.4f} \u00b1 {agg['SSIM_1D_std']:.4f}")

    np.savez_compressed(
        os.path.join(out_dir, "eval_results.npz"),
        pred=pred, gt=gt, init=init,
        metrics=np.array(metrics_list),
    )
    print(f"  eval_results.npz saved")

    log_path = os.path.join(exp_dir, "train_log.txt")
    history = parse_train_log(log_path)

    print("\n生成可视化...")
    plot_training_curves(history, out_dir, model_name=model_name)

    show_idx = list(range(min(4, pred.shape[0])))
    plot_signal_comparison(gt, pred, init, show_idx, out_dir, args.dynamic_range,
                           model_name=model_name)
    plot_error_distribution(metrics_list, out_dir)
    plot_envelope_comparison_2d(gt, pred, init, out_dir, args.dynamic_range,
                                model_name=model_name)
    _plot_reconstruction_detail(gt, pred, init, metrics_list, out_dir,
                                args.dynamic_range, model_name)
    _plot_bmode(gt, pred, init, metrics_list, out_dir, dataset,
                args.dynamic_range, model_name)

    if extra_viz_fn is not None:
        try:
            extra_viz_fn(model, dataset, eval_idx, device, out_dir)
        except Exception as e:
            print(f"  [警告] 额外可视化失败: {e}")

    if not getattr(args, "no_das", False):
        try:
            _compute_and_plot_das(dataset, pred, init, args, out_dir, model_name)
        except Exception as e:
            print(f"  [警告] DAS 可视化失败: {e}")

    print("\n生成评估报告...")
    generate_evaluation_report(metrics_list, meta, out_dir, history)

    summary = {
        "exp_dir": exp_dir,
        "ckpt": args.ckpt_name,
        "model_type": model_name,
        "n_eval": len(metrics_list),
        **agg,
        "args": args_dict,
    }
    save_eval_summary(out_dir, summary)
    print(f"  eval_summary.json saved")

    print(f"\n评估完成! 结果保存至: {out_dir}")


# ======================== 共享 argparse ========================

def build_eval_parser_1d(description="1D 模型评估与可视化"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="实验目录 (包含 best_model.pth 和 train_log.txt)")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pth")
    parser.add_argument("--npz", type=str, default=None,
                        help="npz 数据路径 (默认从 checkpoint 读取)")
    parser.add_argument("--cs_ratio", type=int, default=None)
    parser.add_argument("--eval_all", action="store_true", default=False,
                        help="评估全量数据 (默认仅验证集)")
    parser.add_argument("--eval_subdir", type=str, default="eval_results",
                        help="评估结果输出子目录 (相对 exp_dir)，默认 eval_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dynamic_range", type=float, default=60.0)
    parser.add_argument("--no_das", action="store_true", default=False,
                        help="跳过 DAS B-mode 重建 (节省计算时间)")
    parser.add_argument("--gpu", type=int, default=0)
    return parser


# ================================================================
#                          2D 评估
# ================================================================

def _env_2d(rf_2d: np.ndarray, dynamic_range: float = 60.0) -> np.ndarray:
    """(H, W) RF -> (H, W) dB envelope."""
    env = np.stack([envelope_np(rf_2d[r]) for r in range(rf_2d.shape[0])])
    return to_db(env, dynamic_range)


def _plot_bmode_comparison_2d(gt, pred, init, save_path, dynamic_range=60.0,
                              sample_idx=0, model_name="Recon"):
    """GT / Recon / Init 三幅 B-mode 图对比 + 误差图."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bgt = _env_2d(gt, dynamic_range)
    bpred = _env_2d(pred, dynamic_range)
    binit = _env_2d(init, dynamic_range)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    vmin, vmax = -dynamic_range, 0
    for ax, img, title in zip(
        axes[0], [bgt, bpred, binit],
        ["Ground Truth", model_name, "Init (A\u2020y)"],
    ):
        im = ax.imshow(img, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Sample (t)")
        ax.set_ylabel("Element")
    fig.colorbar(im, ax=axes[0], label="dB", shrink=0.8)

    err_pred = np.abs(bgt - bpred)
    err_init = np.abs(bgt - binit)
    emax = max(err_pred.max(), err_init.max(), 1e-6)
    for ax, err, title in zip(
        axes[1, :2], [err_pred, err_init],
        [f"|GT - {model_name}| Error", "|GT - Init| Error"],
    ):
        im2 = ax.imshow(err, aspect="auto", cmap="hot", vmin=0, vmax=emax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Sample (t)")
        ax.set_ylabel("Element")
    fig.colorbar(im2, ax=axes[1, :2], label="dB Error", shrink=0.8)

    mid = gt.shape[0] // 2
    t = np.arange(gt.shape[1])
    axes[1, 2].plot(t, bgt[mid], label="GT", alpha=0.8, lw=0.8)
    axes[1, 2].plot(t, bpred[mid], label=model_name, alpha=0.8, lw=0.8)
    axes[1, 2].plot(t, binit[mid], label="Init", alpha=0.5, lw=0.6, ls="--")
    axes[1, 2].set_title(f"Envelope Profile (Element {mid})", fontsize=13)
    axes[1, 2].set_xlabel("Sample (t)")
    axes[1, 2].set_ylabel("dB")
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)

    snr_recon = 10 * np.log10(np.sum(gt**2) / (np.sum((gt - pred)**2) + 1e-10))
    snr_init = 10 * np.log10(np.sum(gt**2) / (np.sum((gt - init)**2) + 1e-10))
    fig.suptitle(
        f"Sample #{sample_idx}  |  {model_name} SNR: {snr_recon:.2f} dB  |  Init SNR: {snr_init:.2f} dB",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_rf_lines_2d(gt, pred, init, save_path, model_name="Recon", n_lines=4):
    """选取若干阵元的 RF 时域波形对比."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = gt.shape[0]
    indices = np.linspace(0, H - 1, n_lines, dtype=int)
    fig, axes = plt.subplots(n_lines, 2, figsize=(16, 3.5 * n_lines))
    if n_lines == 1:
        axes = axes[np.newaxis, :]

    for row, ei in enumerate(indices):
        g, p, ini = gt[ei], pred[ei], init[ei]
        t = np.arange(len(g))

        axes[row, 0].plot(t, g, label="GT", alpha=0.7, lw=0.6)
        axes[row, 0].plot(t, p, label=model_name, alpha=0.7, lw=0.6)
        axes[row, 0].plot(t, ini, label="Init", alpha=0.4, lw=0.5, ls="--")
        axes[row, 0].set_title(f"Element {ei} \u2014 RF Signal")
        axes[row, 0].legend(fontsize=7)
        axes[row, 0].grid(True, alpha=0.2)

        env_g = to_db(envelope_np(g), 60)
        env_p = to_db(envelope_np(p), 60)
        env_i = to_db(envelope_np(ini), 60)
        axes[row, 1].plot(t, env_g, label="GT Env", alpha=0.7, lw=0.8)
        axes[row, 1].plot(t, env_p, label=f"{model_name} Env", alpha=0.7, lw=0.8)
        axes[row, 1].plot(t, env_i, label="Init Env", alpha=0.4, lw=0.6, ls="--")
        axes[row, 1].set_title(f"Element {ei} \u2014 Envelope (dB)")
        axes[row, 1].legend(fontsize=7)
        axes[row, 1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_metrics_summary_2d(snrs, nmses, save_path):
    """SNR / NMSE 分布直方图."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(snrs, bins=max(5, len(snrs) // 2), edgecolor="k", alpha=0.75)
    axes[0].axvline(np.mean(snrs), color="r", ls="--", label=f"Mean={np.mean(snrs):.2f} dB")
    axes[0].set(title="SNR Distribution (dB)", xlabel="SNR (dB)", ylabel="Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(nmses, bins=max(5, len(nmses) // 2), edgecolor="k", alpha=0.75, color="orange")
    axes[1].axvline(np.mean(nmses), color="r", ls="--", label=f"Mean={np.mean(nmses):.4f}")
    axes[1].set(title="NMSE Distribution", xlabel="NMSE", ylabel="Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _parse_config_txt(exp_dir):
    """从 config.txt 解析训练参数字典."""
    config = {}
    config_path = os.path.join(exp_dir, "config.txt")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    k, v = k.strip(), v.strip()
                    for cast in (int, float):
                        try:
                            v = cast(v)
                            break
                        except ValueError:
                            pass
                    if v == "True":
                        v = True
                    elif v == "False":
                        v = False
                    config[k] = v
    return config


def _resolve_npz_path(npz_raw, exp_dir):
    """将 config 中记录的 npz 路径解析为绝对路径.

    优先级: 原始路径 > 相对于 exp_dir > 相对于 data/ 上级目录.
    """
    if os.path.isfile(npz_raw):
        return npz_raw
    candidate = os.path.join(exp_dir, npz_raw)
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", npz_raw))
    if os.path.isfile(candidate):
        return candidate
    return npz_raw


@torch.no_grad()
def _reconstruct_full_frames_2d(model, ds, device):
    """逐帧全幅推理 (patch_h=None), 返回 (pred, gt, init) numpy arrays.

    每个 shape: (n_frames, H, W).
    """
    op = ds.op
    preds, gts, inits = [], [], []
    for fi in range(len(ds)):
        idx_t = torch.tensor([fi], device=device)
        x_input, y_target, y_k = ds.get_batch(idx_t)
        y_sub = y_k if y_k is not None else op.A(x_input)
        x_hat = model(y_sub, op)
        x_init = op.At(y_sub)
        preds.append(x_hat[0, 0].cpu().numpy())
        gts.append(y_target[0, 0].cpu().numpy())
        inits.append(x_init[0, 0].cpu().numpy())
    return np.stack(preds), np.stack(gts), np.stack(inits)


def evaluate_2d(load_model_fn, model_name, args):
    """2D 模型通用评估流程.

    Parameters
    ----------
    load_model_fn : callable(config, ckpt, device) -> model
        根据 config 和 checkpoint 实例化模型并加载权重.
    model_name : str
    args : argparse.Namespace
        支持 --eval_all, --eval_subdir, --no_das, --npz 等.
    """
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir

    config = _parse_config_txt(exp_dir)
    print(f"Config: {config}")

    ckpt_path = os.path.join(exp_dir, args.ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    model = load_model_fn(config, ckpt, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}, params={num_params:,}")

    # npz: 命令行 --npz 优先, 否则从 config
    if getattr(args, "npz", None):
        npz_list = [args.npz] if isinstance(args.npz, str) else list(args.npz)
    else:
        npz_list = config.get("npz", "picmus_simu_reso_frames.npz")
        if isinstance(npz_list, str):
            npz_list = npz_list.strip("[]'\" ").split(",")
            npz_list = [s.strip().strip("'\"") for s in npz_list]
    npz_list = [_resolve_npz_path(p, exp_dir) for p in npz_list]

    cs_ratio = config.get("cs_ratio", 8)
    if isinstance(cs_ratio, str):
        cs_ratio = int(cs_ratio)

    eval_all = getattr(args, "eval_all", False)
    no_das = getattr(args, "no_das", False)
    want_das = eval_all and not no_das

    patch_h = config.get("patch_h", None)
    if str(patch_h).lower() in ("none", "null"):
        patch_h = None
    else:
        patch_h = int(patch_h) if patch_h is not None else None
    patch_stride = config.get("patch_stride", None)
    if str(patch_stride).lower() in ("none", "null"):
        patch_stride = None
    else:
        patch_stride = int(patch_stride) if patch_stride is not None else None

    # DAS 需要完整帧 (每帧 = 全部阵元), patch 模式下无法直接做 DAS
    if want_das and patch_h is not None:
        print(f"  [DAS] 训练用 patch_h={patch_h}, 但 DAS 需要完整帧 → 强制 patch_h=None")
        patch_h = None
        patch_stride = None

    # 输出目录
    eval_subdir = getattr(args, "eval_subdir", "visualizations")
    out_dir = os.path.join(exp_dir, eval_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # training curves
    log_path = os.path.join(exp_dir, "train_log.txt")
    if os.path.isfile(log_path):
        history = parse_train_log(log_path)
        if history["epoch"]:
            plot_training_curves(history, out_dir, model_name=model_name)
            print(f"  Training curves saved")

    all_snrs, all_nmses = [], []
    sample_counter = 0

    for di, npz_path in enumerate(npz_list):
        print(f"\n加载数据: {npz_path}")
        ds = UltrasoundFrameDataset(
            npz_path, cs_ratio=cs_ratio, patch_h=patch_h,
            patch_stride=patch_stride, device="cpu",
        )
        ds.to(device)
        eval_unit = "frame" if patch_h is None else f"patch({patch_h})"
        print(f"  Dataset [{di}]: N={ds.N}, H_full={ds.H_full}, "
              f"n_frames={ds.n_frames}, samples={len(ds)}, unit={eval_unit}")
        op = ds.op

        n_vis = min(args.n_vis, len(ds))
        vis_indices = set(np.linspace(0, len(ds) - 1, n_vis, dtype=int))

        frame_preds, frame_gts, frame_inits = [], [], []

        for pi in range(len(ds)):
            idx_t = torch.tensor([pi], device=device)
            x_input, y_target, y_k = ds.get_batch(idx_t)
            y_sub = y_k if y_k is not None else op.A(x_input)

            with torch.no_grad():
                x_hat = model(y_sub, op)
                x_init = op.At(y_sub)

            snr = calc_snr(y_target, x_hat).item()
            nmse = calc_nmse(y_target, x_hat).item()
            init_snr = calc_snr(y_target, x_init).item()
            all_snrs.append(snr)
            all_nmses.append(nmse)

            gt_np = y_target[0, 0].cpu().numpy()
            pred_np = x_hat[0, 0].cpu().numpy()
            init_np = x_init[0, 0].cpu().numpy()

            if eval_all:
                frame_preds.append(pred_np)
                frame_gts.append(gt_np)
                frame_inits.append(init_np)

            if pi in vis_indices:
                _plot_bmode_comparison_2d(
                    gt_np, pred_np, init_np,
                    os.path.join(out_dir, f"bmode_ds{di}_p{pi}.png"),
                    dynamic_range=args.dynamic_range,
                    sample_idx=pi, model_name=model_name,
                )
                _plot_rf_lines_2d(
                    gt_np, pred_np, init_np,
                    os.path.join(out_dir, f"rf_lines_ds{di}_p{pi}.png"),
                    model_name=model_name, n_lines=4,
                )
                print(f"  Frame {pi}: SNR={snr:.2f} dB (init={init_snr:.2f}), NMSE={nmse:.4f}")

            sample_counter += 1

        # DAS: 只在 eval_all + 全帧模式下可用
        if eval_all and not getattr(args, "no_das", False):
            pred_3d = np.stack(frame_preds)   # (n_frames, H, W)
            gt_3d = np.stack(frame_gts)
            init_3d = np.stack(frame_inits)

            np.savez_compressed(
                os.path.join(out_dir, f"eval_frames_ds{di}.npz"),
                pred=pred_3d, gt=gt_3d, init=init_3d,
            )
            print(f"  eval_frames_ds{di}.npz saved ({pred_3d.shape})")

            try:
                _compute_and_plot_das_2d(
                    npz_path, pred_3d, gt_3d, init_3d,
                    out_dir, args.dynamic_range, model_name,
                )
            except Exception as e:
                print(f"  [警告] DAS 可视化失败: {e}")

    all_snrs = np.array(all_snrs)
    all_nmses = np.array(all_nmses)
    _plot_metrics_summary_2d(all_snrs, all_nmses,
                             os.path.join(out_dir, "metrics_summary.png"))

    granularity = "per-frame" if patch_h is None else f"per-patch(h={patch_h})"

    print(f"\n{'='*60}")
    print(f"  Model: {model_name} | Params: {num_params:,}")
    print(f"  Evaluated: {sample_counter} samples ({granularity})")
    print(f"  SNR  ({granularity}): {all_snrs.mean():.2f} +/- {all_snrs.std():.2f} dB")
    print(f"  NMSE ({granularity}): {all_nmses.mean():.4f} +/- {all_nmses.std():.4f}")
    print(f"  Results saved to: {out_dir}")
    print(f"{'='*60}")

    summary_path = os.path.join(out_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Params: {num_params:,}\n")
        f.write(f"Samples: {sample_counter} ({granularity})\n")
        f.write(f"eval_all: {eval_all}\n")
        f.write(f"patch_h: {patch_h}\n")
        f.write(f"SNR ({granularity}): {all_snrs.mean():.2f} +/- {all_snrs.std():.2f} dB\n")
        f.write(f"NMSE ({granularity}): {all_nmses.mean():.4f} +/- {all_nmses.std():.4f}\n")
        for i, (s, n) in enumerate(zip(all_snrs, all_nmses)):
            f.write(f"  sample_{i}: SNR={s:.2f} dB, NMSE={n:.4f}\n")

    print(f"\n评估完成! 结果保存至: {out_dir}")


def build_eval_parser_2d(description="2D 模型评估与可视化"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="实验目录 (包含 best_model.pth)")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pth")
    parser.add_argument("--npz", type=str, default=None,
                        help="npz 数据路径 (默认从 config 读取)")
    parser.add_argument("--eval_all", action="store_true", default=False,
                        help="全帧推理 (强制 patch_h=None) + DAS 成像")
    parser.add_argument("--eval_subdir", type=str, default="visualizations",
                        help="评估结果输出子目录 (相对 exp_dir)")
    parser.add_argument("--no_das", action="store_true", default=False,
                        help="跳过 DAS B-mode 重建")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dynamic_range", type=float, default=60.0,
                        help="B-mode 动态范围 (dB)")
    parser.add_argument("--n_vis", type=int, default=5,
                        help="每个数据集可视化样本数")
    return parser
