"""Classical NESTA 1D 评估

无需训练 / checkpoint — 直接用命令行参数实例化经典 NESTA 算法并评估.

用法示例:
    python evaluate_NESTA_Classical.py \
        --npz picmus_simu_reso.npz --cs_ratio 8 \
        --n_iters 50 --n_restarts 5 --eta 1e-4 \
        --transform identity --eval_all
"""

import os
import argparse
from datetime import datetime

import numpy as np
import torch

from NESTA_Classical import ClassicalNESTA
from admm_data import UltrasoundDataset, split_indices
from admm_metrics import compute_sample_metrics, summarize_metrics, envelope_np, to_db
from admm_visualization import (
    plot_signal_comparison,
    plot_error_distribution,
    plot_envelope_comparison_2d,
)


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


def _plot_bmode(gt, pred, init, metrics_list, save_dir, dataset,
                dynamic_range=60.0, model_name="Classical NESTA"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    def _to_env(arr):
        signals = arr.squeeze()
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        return np.array([envelope_np(s) for s in signals])

    env_gt, env_pred, env_init = _to_env(gt), _to_env(pred), _to_env(init)
    bgt = to_db(env_gt, dynamic_range)
    bpred = to_db(env_pred, dynamic_range)
    binit = to_db(env_init, dynamic_range)
    n_lines, n_samples = bgt.shape

    fs = getattr(dataset, "fs", None)
    if fs:
        depth_mm = np.arange(n_samples) / fs * 1540.0 / 2 * 1e3
        extent = [0, depth_mm[-1], n_lines - 0.5, -0.5]
        xlabel = "Depth (mm)"
    else:
        extent = [0, n_samples, n_lines - 0.5, -0.5]
        xlabel = "Sample"

    snrs = np.array([m["SNR_dB"] for m in metrics_list])
    avg_snr, avg_nmse = np.mean(snrs), np.mean([m["NMSE"] for m in metrics_list])

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    vmin, vmax = -dynamic_range, 0

    for col, (d, t) in enumerate(
        zip([bgt, bpred, binit], ["Ground Truth", model_name, "Init (A\u2020y)"])
    ):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(d, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(t, fontsize=13, fontweight="bold" if col == 1 else "normal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Scan Line")

    ax_snr = fig.add_subplot(gs[1, 0])
    ax_snr.barh(range(len(snrs)), snrs, color="steelblue", alpha=0.7, height=0.8)
    ax_snr.axvline(avg_snr, color="red", ls="--", lw=1.5, label=f"Mean={avg_snr:.2f} dB")
    ax_snr.set_xlabel("SNR (dB)")
    ax_snr.set_title("Per-sample SNR")
    ax_snr.legend(fontsize=9)
    ax_snr.invert_yaxis()
    ax_snr.grid(True, alpha=0.3, axis="x")

    mid = n_lines // 2
    ax_prof = fig.add_subplot(gs[1, 1:])
    ax_prof.plot(bgt[mid], label="GT", alpha=0.8, lw=0.8)
    ax_prof.plot(bpred[mid], label=model_name, alpha=0.8, lw=0.8)
    ax_prof.plot(binit[mid], label="Init", alpha=0.5, lw=0.6, ls="--")
    ax_prof.set_title(f"Envelope Profile (Line {mid})")
    ax_prof.set_xlabel(xlabel)
    ax_prof.set_ylabel("dB")
    ax_prof.legend(fontsize=9)
    ax_prof.grid(True, alpha=0.3)

    fig.suptitle(
        f"Classical NESTA  |  {n_lines} lines \u00d7 {n_samples} samples  |  "
        f"Avg SNR: {avg_snr:.2f} dB  |  Avg NMSE: {avg_nmse:.4f}",
        fontsize=14, fontweight="bold", y=0.99,
    )
    plt.savefig(os.path.join(save_dir, "bmode_reconstruction.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_reconstruction_detail(gt, pred, init, metrics_list, save_dir,
                                dynamic_range=60.0, model_name="Classical NESTA"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = np.array([m["SNR_dB"] for m in metrics_list])
    ranking = np.argsort(snrs)
    picks = {"Best": ranking[-1], "Median": ranking[len(ranking) // 2], "Worst": ranking[0]}

    for tag, idx in picks.items():
        g, p, ini = gt[idx].squeeze(), pred[idx].squeeze(), init[idx].squeeze()
        snr_val, nmse_val = snrs[idx], metrics_list[idx]["NMSE"]
        t = np.arange(len(g))
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        axes[0].plot(t, g, label="Ground Truth", alpha=0.7, lw=0.5)
        axes[0].plot(t, p, label=model_name, alpha=0.7, lw=0.5)
        axes[0].plot(t, ini, label="A\u2020y (init)", alpha=0.3, lw=0.4, ls="--")
        axes[0].set_title(f"{tag} #{idx} | SNR={snr_val:.2f} dB | NMSE={nmse_val:.4f}",
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

        axes[2].plot(t, np.abs(g - ini), label="|GT - Init|", alpha=0.5, lw=0.5, color="gray")
        axes[2].plot(t, np.abs(g - p), label=f"|GT - {model_name}|", alpha=0.7, lw=0.6, color="red")
        axes[2].set_ylabel("Absolute Error")
        axes[2].set_xlabel("Sample")
        axes[2].set_title("Reconstruction Error")
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"recon_detail_{tag.lower()}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Classical NESTA 1D 评估 (无需训练)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", type=str, required=True, help="npz 数据路径")
    parser.add_argument("--cs_ratio", type=int, default=8, help="压缩采样比")

    g = parser.add_argument_group("NESTA 超参数")
    g.add_argument("--n_iters", type=int, default=60,
                   help="每次 restart 的内部迭代次数")
    g.add_argument("--mu", type=float, default=1e-3,
                   help="平滑参数 µ (仅 plain NESTA, n_restarts=0 时使用)")
    g.add_argument("--eta", type=float, default=1e-4,
                   help="噪声容忍度 η (约束 ||y - Ax|| ≤ η)")
    g.add_argument("--n_restarts", type=int, default=0,
                   help="Restart 次数")
    g.add_argument("--restart_decay", type=float, default=0.25,
                   help="Restart 收缩率 r")
    g.add_argument("--zeta", type=float, default=1e-9,
                   help="Restart 目标误差下限")
    g.add_argument("--transform", type=str, default="identity",
                   choices=["identity", "tv", "dct"],
                   help="稀疏化变换 W: identity / tv (全变分) / dct")

    g2 = parser.add_argument_group("评估选项")
    g2.add_argument("--eval_all", action="store_true",
                    help="评估全部数据 (默认仅验证集)")
    g2.add_argument("--val_ratio", type=float, default=0.1)
    g2.add_argument("--seed", type=int, default=42)
    g2.add_argument("--batch_size", type=int, default=32)
    g2.add_argument("--dynamic_range", type=float, default=60.0)
    g2.add_argument("--save_dir", type=str, default="model")
    g2.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ClassicalNESTA(
        n_iters=args.n_iters,
        mu=args.mu,
        eta=args.eta,
        n_restarts=args.n_restarts,
        restart_decay=args.restart_decay,
        zeta=args.zeta,
        transform=args.transform,
    ).to(device)
    model.eval()

    total_iters = args.n_iters * max(args.n_restarts, 1)
    print(f"\nClassical NESTA:")
    print(f"  n_iters={args.n_iters}, n_restarts={args.n_restarts} → 总迭代 {total_iters}")
    print(f"  µ={args.mu}, η={args.eta}, restart_decay={args.restart_decay}")
    print(f"  transform={args.transform}")
    print(f"  可学习参数: 0")

    print(f"\n加载数据: {args.npz} (cs_ratio={args.cs_ratio})")
    dataset = UltrasoundDataset(args.npz, cs_ratio=args.cs_ratio, device="cpu").to(device)

    n_total = len(dataset)
    if args.eval_all:
        eval_idx = torch.arange(n_total, device=device)
        print(f"  评估全部 {n_total} 个样本")
    else:
        _, eval_idx = split_indices(
            num_samples=n_total, val_ratio=args.val_ratio,
            seed=args.seed, split_mode="group", group_id=dataset.group_id,
        )
        eval_idx = eval_idx.to(device)
        print(f"  评估验证集 {len(eval_idx)} 个样本 (总 {n_total})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"ClassicalNESTA_ratio{args.cs_ratio}_iter{args.n_iters}"
        f"_R{args.n_restarts}_eta{args.eta}_{args.transform}_{timestamp}"
    )
    out_dir = os.path.join(args.save_dir, exp_name, "eval_results")
    os.makedirs(out_dir, exist_ok=True)

    print("\n开始推理...")
    results = run_inference(model, dataset, eval_idx, batch_size=args.batch_size)
    pred, gt, init = results["pred"], results["gt"], results["init"]
    print(f"  pred shape: {pred.shape}")

    print("\n计算指标...")
    metrics_list = [
        compute_sample_metrics(gt[i].squeeze(), pred[i].squeeze())
        for i in range(pred.shape[0])
    ]
    agg = summarize_metrics(metrics_list)
    print(f"  Avg SNR:    {agg['SNR_dB_mean']:.2f} \u00b1 {agg['SNR_dB_std']:.2f} dB")
    print(f"  Avg NMSE:   {agg['NMSE_mean']:.6f} \u00b1 {agg['NMSE_std']:.6f}")
    print(f"  Avg PSNR:   {agg['PSNR_dB_mean']:.2f} \u00b1 {agg['PSNR_dB_std']:.2f} dB")
    print(f"  Avg SSIM1D: {agg['SSIM_1D_mean']:.4f} \u00b1 {agg['SSIM_1D_std']:.4f}")

    np.savez_compressed(
        os.path.join(out_dir, "eval_results.npz"),
        pred=pred, gt=gt, init=init,
        metrics=np.array(metrics_list),
    )

    print("\n生成可视化...")
    model_name = f"NESTA ({args.transform})"
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

    summary_path = os.path.join(out_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Algorithm: Classical NESTA\n")
        f.write(f"Transform: {args.transform}\n")
        f.write(f"n_iters: {args.n_iters}, n_restarts: {args.n_restarts}\n")
        f.write(f"mu: {args.mu}, eta: {args.eta}\n")
        f.write(f"restart_decay: {args.restart_decay}, zeta: {args.zeta}\n")
        f.write(f"cs_ratio: {args.cs_ratio}\n")
        f.write(f"Samples: {pred.shape[0]}\n")
        f.write(f"SNR: {agg['SNR_dB_mean']:.2f} +/- {agg['SNR_dB_std']:.2f} dB\n")
        f.write(f"NMSE: {agg['NMSE_mean']:.6f} +/- {agg['NMSE_std']:.6f}\n")
        f.write(f"PSNR: {agg['PSNR_dB_mean']:.2f} +/- {agg['PSNR_dB_std']:.2f} dB\n")
        f.write(f"SSIM1D: {agg['SSIM_1D_mean']:.4f} +/- {agg['SSIM_1D_std']:.4f}\n")
        for i, m in enumerate(metrics_list):
            f.write(f"  sample_{i}: SNR={m['SNR_dB']:.2f} dB, NMSE={m['NMSE']:.4f}\n")

    print(f"\n{'='*60}")
    print(f"  Classical NESTA ({args.transform})")
    print(f"  Total iterations: {total_iters}")
    print(f"  SNR:  {agg['SNR_dB_mean']:.2f} \u00b1 {agg['SNR_dB_std']:.2f} dB")
    print(f"  NMSE: {agg['NMSE_mean']:.6f}")
    print(f"  结果保存至: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
