import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List

from metrics import envelope_np, to_db, calc_snr
from ops import hilbert_envelope


def parse_train_log(log_path: str) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {
        "epoch": [], "train_loss": [], "train_snr": [],
        "val_loss": [], "val_snr": [], "val_nmse": [],
        "init_snr": [], "lr": [],
        "rho1": [], "rho2": [], "eta": [],
    }
    if not os.path.isfile(log_path):
        return history

    epoch_pat = re.compile(
        r"\[Epoch\s+(\d+)/\d+\].*?"
        r"Train:\s+([\d.]+)\s+\(([\d.]+)dB\).*?"
        r"Val:\s+([\d.]+)\s+\(([\d.]+)dB,\s*init:([\d.]+)\).*?"
        r"NMSE:([\d.]+).*?"
        r"LR:([\d.]+)"
    )
    param_pat = re.compile(r"rho1=([\d.]+)\s+rho2=([\d.]+)\s+eta=([\d.]+)")

    with open(log_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        m = epoch_pat.search(lines[i])
        if m:
            history["epoch"].append(int(m.group(1)))
            history["train_loss"].append(float(m.group(2)))
            history["train_snr"].append(float(m.group(3)))
            history["val_loss"].append(float(m.group(4)))
            history["val_snr"].append(float(m.group(5)))
            history["init_snr"].append(float(m.group(6)))
            history["val_nmse"].append(float(m.group(7)))
            history["lr"].append(float(m.group(8)))
            if i + 1 < len(lines):
                pm = param_pat.search(lines[i + 1])
                if pm:
                    history["rho1"].append(float(pm.group(1)))
                    history["rho2"].append(float(pm.group(2)))
                    history["eta"].append(float(pm.group(3)))
                    i += 1
        i += 1
    return history


def _is_admm(model_name: str) -> bool:
    """根据模型名判断是否为 ADMM 展开."""
    if model_name is None:
        return False
    return "ADMM" in model_name.upper()


def plot_training_curves(history: Dict[str, List[float]], save_dir: str,
                         model_name: str = None):
    epochs = history.get("epoch", [])
    if not epochs:
        print("  [跳过] 无训练日志数据")
        return

    is_admm = _is_admm(model_name)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val")
    axes[0, 0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_snr"], label="Train")
    axes[0, 1].plot(epochs, history["val_snr"], label="Val")
    axes[0, 1].plot(epochs, history["init_snr"], label="Init (A^T y)", ls="--", alpha=0.6)
    axes[0, 1].set(title="SNR (dB)", xlabel="Epoch", ylabel="dB")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history["val_nmse"])
    axes[0, 2].set(title="Val NMSE", xlabel="Epoch", ylabel="NMSE")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["lr"])
    axes[1, 0].set(title="Learning Rate", xlabel="Epoch", ylabel="LR")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if history["rho1"]:
        ep_sub = epochs[:len(history["rho1"])]
        if is_admm:
            axes[1, 1].plot(ep_sub, history["rho1"], label=r"$\rho_1$ (wav)")
            axes[1, 1].plot(ep_sub, history["rho2"], label=r"$\rho_2$ (TV)")
            axes[1, 1].set(title="ADMM Penalty ρ", xlabel="Epoch")
        else:
            axes[1, 1].plot(ep_sub, history["rho1"], label=r"$\rho$ (step)")
            axes[1, 1].set(title="Gradient Step ρ", xlabel="Epoch")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(ep_sub, history["eta"], label=r"$\eta$ (step)")
        if is_admm:
            axes[1, 2].set(title="Step Size η", xlabel="Epoch")
        else:
            axes[1, 2].set(title="Soft-Threshold η", xlabel="Epoch")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_signal_comparison(gt, pred, init, sample_indices, save_dir, dynamic_range=60.0,
                           model_name="Recon"):
    n_show = min(len(sample_indices), 4)
    fig, axes = plt.subplots(n_show, 2, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(sample_indices[:n_show]):
        g = gt[idx].squeeze()
        p = pred[idx].squeeze()
        ini = init[idx].squeeze()
        t = np.arange(len(g))

        axes[row, 0].plot(t, g, label="Ground Truth", alpha=0.7, lw=0.6)
        axes[row, 0].plot(t, p, label=model_name, alpha=0.7, lw=0.6)
        axes[row, 0].plot(t, ini, label="A†y (init)", alpha=0.4, lw=0.5, ls="--")
        axes[row, 0].set(title=f"Sample #{idx}  RF Signal", ylabel="Amplitude")
        axes[row, 0].legend(fontsize=7)
        axes[row, 0].grid(True, alpha=0.2)

        env_g = to_db(envelope_np(g), dynamic_range)
        env_p = to_db(envelope_np(p), dynamic_range)
        env_i = to_db(envelope_np(ini), dynamic_range)
        axes[row, 1].plot(t, env_g, label="GT Env", alpha=0.7, lw=0.8)
        axes[row, 1].plot(t, env_p, label=f"{model_name} Env", alpha=0.7, lw=0.8)
        axes[row, 1].plot(t, env_i, label="Init Env", alpha=0.4, lw=0.6, ls="--")
        axes[row, 1].set(title=f"Envelope (dB, DR={dynamic_range:.0f})", ylabel="dB")
        axes[row, 1].legend(fontsize=7)
        axes[row, 1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "signal_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_distribution(metrics_list, save_dir):
    snrs = [m["SNR_dB"] for m in metrics_list]
    nmses = [m["NMSE"] for m in metrics_list]
    env_corrs = [m["Env_Corr"] for m in metrics_list]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(snrs, bins=20, edgecolor="k", alpha=0.75)
    axes[0].axvline(np.mean(snrs), color="r", ls="--", label=f"Mean={np.mean(snrs):.2f}")
    axes[0].set(title="SNR Distribution (dB)", xlabel="SNR (dB)", ylabel="Count")
    axes[0].legend()

    axes[1].hist(nmses, bins=20, edgecolor="k", alpha=0.75, color="orange")
    axes[1].axvline(np.mean(nmses), color="r", ls="--", label=f"Mean={np.mean(nmses):.6f}")
    axes[1].set(title="NMSE Distribution", xlabel="NMSE", ylabel="Count")
    axes[1].legend()

    axes[2].hist(env_corrs, bins=20, edgecolor="k", alpha=0.75, color="green")
    axes[2].axvline(np.mean(env_corrs), color="r", ls="--", label=f"Mean={np.mean(env_corrs):.4f}")
    axes[2].set(title="Envelope Correlation", xlabel="Corr", ylabel="Count")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_envelope_comparison_2d(gt, pred, init, save_dir, dynamic_range=60.0,
                                model_name="Recon"):
    n_samples = gt.shape[0]
    if n_samples < 2:
        return

    def make_bscan(arr):
        signals = arr.squeeze()
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        env = np.array([envelope_np(s) for s in signals])
        return to_db(env, dynamic_range)

    bscan_gt = make_bscan(gt)
    bscan_pred = make_bscan(pred)
    bscan_init = make_bscan(init)

    H, W = bscan_gt.shape
    aspect_ratio = W / max(H, 1)
    fig_h = max(5, min(10, 18 / 3 / max(aspect_ratio, 0.3)))
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_h))
    for ax, data, title in zip(
        axes, [bscan_gt, bscan_pred, bscan_init],
        ["Ground Truth", model_name, "A†y (init)"],
    ):
        im = ax.imshow(data, aspect="auto", cmap="gray", vmin=-dynamic_range, vmax=0)
        ax.set(title=title, xlabel="Sample", ylabel="Line")
    fig.colorbar(im, ax=axes, label="dB", shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bscan_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_layer_convergence(model, dataset, sample_idx, device, save_dir):
    model.eval()
    op = dataset.op
    idx_t = torch.tensor([sample_idx], device=device)
    x_input, y_target, y_k = dataset.get_batch(idx_t)
    y_sub = y_k if y_k is not None else op.A(x_input)
    x = op.At(y_sub)

    w = model.W.forward(x)
    p = model.D(hilbert_envelope(x))
    u1 = torch.zeros_like(w)
    u2 = torch.zeros_like(p)

    snrs_per_layer, rhos1, rhos2, etas = [calc_snr(y_target, x).item()], [], [], []
    for blk in model.blocks:
        x, w, p, u1, u2, aux = blk(x, w, p, u1, u2, y_sub, op)
        snrs_per_layer.append(calc_snr(y_target, x).item())
        rhos1.append(aux["rho1"].item())
        rhos2.append(aux["rho2"].item())
        etas.append(aux["eta"].item())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(range(len(snrs_per_layer))), snrs_per_layer, "o-", color="steelblue", lw=2)
    axes[0].set(title=f"Layer-wise SNR (Sample #{sample_idx})", xlabel="Layer", ylabel="SNR (dB)")
    axes[0].grid(True, alpha=0.3)

    layer_idx = list(range(1, len(rhos1) + 1))
    axes[1].plot(layer_idx, rhos1, "s-", label=r"$\rho_1$")
    axes[1].plot(layer_idx, rhos2, "^-", label=r"$\rho_2$")
    ax2 = axes[1].twinx()
    ax2.plot(layer_idx, etas, "D-", color="green", label=r"$\eta$")
    axes[1].set(title="Learned Parameters per Layer", xlabel="Layer")
    axes[1].set_ylabel(r"$\rho$")
    ax2.set_ylabel(r"$\eta$", color="green")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "layer_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
