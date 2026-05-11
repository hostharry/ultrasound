"""Shared training utilities for Ultrasound reconstruction models."""

from __future__ import annotations

import os
import time
from typing import Iterable

import numpy as np
import torch

from loss import CombinedLoss
from metrics import calc_nmse, calc_snr


def create_optimizer(model, lr, weight_decay, epochs, warm_restarts=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if warm_restarts > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=warm_restarts, T_mult=2, eta_min=lr * 0.01)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)
    return optimizer, scheduler


def create_criterion(gamma_env, gamma_constraint=0.0, loss_mode="mse",
                     depth_weight="none", depth_weight_alpha=2.0,
                     gamma_msle=0.0, gamma_grad=0.0, gamma_stat=0.0,
                     gamma_kld=0.0, gamma_das=0.0,
                     alpha_db=-60.0, kld_bins=40, kld_eta=0.5,
                     kld_low_db=-60.0, kld_high_db=0.0,
                     stat_var_weight=0.5, stat_win=7,
                     das_loss_mode="log_l1", das_alpha_db=-60.0,
                     das_beta_kld=0.5, das_kld_bins=40, das_kld_eta=0.5,
                     das_kld_low_db=-40.0, das_kld_high_db=40.0):
    use_nmse = loss_mode in ("nmse", "nmse_logenv")
    use_log_env = loss_mode == "nmse_logenv"
    use_slae = loss_mode == "slae"
    return CombinedLoss(
        gamma_env=gamma_env,
        gamma_constraint=gamma_constraint,
        gamma_msle=gamma_msle,
        gamma_grad=gamma_grad,
        gamma_stat=gamma_stat,
        gamma_kld=gamma_kld,
        gamma_das=gamma_das,
        stat_var_weight=stat_var_weight,
        stat_win=stat_win,
        use_nmse=use_nmse,
        use_log_env=use_log_env,
        use_slae=use_slae,
        alpha_db=alpha_db,
        kld_bins=kld_bins,
        kld_eta=kld_eta,
        kld_low_db=kld_low_db,
        kld_high_db=kld_high_db,
        das_loss_mode=das_loss_mode,
        das_alpha_db=das_alpha_db,
        das_beta_kld=das_beta_kld,
        das_kld_bins=das_kld_bins,
        das_kld_eta=das_kld_eta,
        das_kld_low_db=das_kld_low_db,
        das_kld_high_db=das_kld_high_db,
        depth_weight=depth_weight,
        depth_weight_alpha=depth_weight_alpha,
    )


def create_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train_log.txt")

    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(str(msg) + "\n")

    return log


def save_config(save_dir, args):
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")


def train_one_batch(model, y_sub, y_target, op, criterion, optimizer,
                    grad_clip, das_meta=None):
    x_hat, aux_list = model(y_sub, op, return_aux=True)
    loss, loss_dict = criterion(x_hat, y_target, aux_list, das_meta=das_meta)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        snr = calc_snr(y_target, x_hat).mean().item()
    return loss_dict["loss_total"], snr


def validate_one_batch(model, y_sub, y_target, op, criterion, das_meta=None):
    x_hat, aux_list = model(y_sub, op, return_aux=True)
    _, loss_dict = criterion(x_hat, y_target, aux_list, das_meta=das_meta)
    snr = calc_snr(y_target, x_hat).mean().item()
    nmse = calc_nmse(y_target, x_hat).mean().item()
    init_snr = calc_snr(y_target, op.At(y_sub)).mean().item()
    return loss_dict, snr, nmse, init_snr, aux_list


def _aux_scalar(aux, key, default=0.0):
    val = aux.get(key, default) if aux else default
    if torch.is_tensor(val):
        if val.numel() == 0:
            return float(default)
        return val.detach().float().mean().item()
    return float(val)


def format_epoch_log(epoch, total_epochs, train_loss, train_snr,
                     val_loss_dict, val_snr, val_nmse, init_snr,
                     aux_last, lr, elapsed):
    parts = [
        f"RF:{val_loss_dict.get('loss_rf', 0):.5f}",
        f"Env:{val_loss_dict.get('loss_env', 0):.5f}",
    ]
    for name, label in [
        ("loss_msle", "MSLE"),
        ("loss_grad", "Grad"),
        ("loss_stat", "Stat"),
        ("loss_kld", "KLD"),
        ("loss_das", "DAS"),
    ]:
        if val_loss_dict.get(name, 0) > 0:
            parts.append(f"{label}:{val_loss_dict[name]:.5f}")
    parts.append(f"Con:{val_loss_dict.get('loss_constraint', 0):.5f}")

    line1 = (
        f"[Epoch {epoch:4d}/{total_epochs}] "
        f"Train: {train_loss:.6f} ({train_snr:.2f}dB) | "
        f"Val: {val_loss_dict.get('loss_total', 0):.6f} "
        f"({val_snr:.2f}dB, init:{init_snr:.2f}) "
        f"NMSE:{val_nmse:.6f} | {' '.join(parts)} | "
        f"LR:{lr:.6f} | {elapsed:.1f}s"
    )
    line2 = (
        f"         rho1={_aux_scalar(aux_last, 'rho1'):.3f} "
        f"rho2={_aux_scalar(aux_last, 'rho2'):.3f} "
        f"eta={_aux_scalar(aux_last, 'eta'):.4f}"
    )
    if aux_last and "gamma" in aux_last:
        line2 += f" gamma={_aux_scalar(aux_last, 'gamma'):.3f}"
    if aux_last and "alpha" in aux_last:
        line2 += f" alpha={_aux_scalar(aux_last, 'alpha'):.3f}"
    return line1, line2


def save_best(save_dir, epoch, model, optimizer, scheduler, val_snr, val_loss,
              args, extra=None):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_snr": val_snr,
        "best_val_snr": val_snr,
        "best_epoch": epoch,
        "val_loss": val_loss,
        "args": vars(args),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, os.path.join(save_dir, "best_model.pth"))


def save_checkpoint(save_dir, epoch, model, optimizer, scheduler, extra=None):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth"))


def save_final(save_dir, epochs, model, best_val_snr, best_epoch, args,
               extra=None):
    ckpt = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "best_val_snr": best_val_snr,
        "best_epoch": best_epoch,
        "args": vars(args),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, os.path.join(save_dir, "final_model.pth"))


_TRAINING_KEYS = frozenset({
    "epochs", "batch_size", "val_batch_size", "lr", "weight_decay",
    "warm_restarts", "grad_clip", "gpu", "save_dir", "resume",
    "log_interval", "save_interval", "loss_mode", "gamma_env",
    "gamma_constraint", "depth_weight", "depth_weight_alpha",
    "gamma_msle", "gamma_grad", "gamma_stat", "gamma_kld", "gamma_das",
    "val_ratio", "split_mode", "seed", "npz", "amp", "compile",
})


def override_args_from_checkpoint(args, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    if not saved:
        print("Warning: checkpoint 中没有保存 args, 请确保手动传入的架构参数与原始训练一致")
        return
    restored = []
    for key, val in saved.items():
        if key not in _TRAINING_KEYS and hasattr(args, key):
            old_val = getattr(args, key)
            if old_val != val:
                setattr(args, key, val)
                restored.append(f"  {key}: {old_val} -> {val}")
    if restored:
        print("从 checkpoint 恢复架构参数:")
        for line in restored:
            print(line)
    else:
        print("架构参数与 checkpoint 一致")


def resume_training(ckpt_path, model, optimizer, scheduler, device=None):
    ckpt = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_snr = ckpt.get("best_val_snr", ckpt.get("val_snr", -float("inf")))
    best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", 0))
    return start_epoch, best_val_snr, best_epoch


def add_common_train_args(parser, defaults_2d=False):
    parser.add_argument("--cs_ratio", type=int, default=8, help="压缩比")
    parser.add_argument("--val_ratio", type=float, default=0.15 if defaults_2d else 0.1)
    parser.add_argument("--split_mode", type=str, default="group",
                        choices=["group", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=4 if defaults_2d else 9)

    parser.add_argument("--gamma_env", type=float, default=0.1)
    parser.add_argument("--loss_mode", type=str, default="mse",
                        choices=["mse", "nmse", "nmse_logenv", "slae"])
    parser.add_argument("--depth_weight", type=str, default="none",
                        choices=["none", "linear", "exp"])
    parser.add_argument("--depth_weight_alpha", type=float, default=2.0)
    parser.add_argument("--gamma_msle", type=float, default=0.0)
    parser.add_argument("--gamma_kld", type=float, default=0.0)
    parser.add_argument("--alpha_db", type=float, default=-60.0)
    parser.add_argument("--kld_bins", type=int, default=40)
    parser.add_argument("--kld_eta", type=float, default=0.5)
    parser.add_argument("--kld_low_db", type=float, default=-60.0)
    parser.add_argument("--kld_high_db", type=float, default=0.0)

    parser.add_argument("--gamma_grad", type=float, default=0.0)
    parser.add_argument("--gamma_stat", type=float, default=0.0)
    parser.add_argument("--stat_var_weight", type=float, default=0.5)
    parser.add_argument("--stat_win", type=int, default=7)

    parser.add_argument("--gamma_das", type=float, default=0.0)
    parser.add_argument("--das_loss_mode", type=str, default="log_l1",
                        choices=["log_l1", "kld_mslae"])
    parser.add_argument("--das_alpha_db", type=float, default=-60.0)
    parser.add_argument("--das_beta_kld", type=float, default=0.5)
    parser.add_argument("--das_kld_bins", type=int, default=40)
    parser.add_argument("--das_kld_eta", type=float, default=0.5)
    parser.add_argument("--das_kld_low_db", type=float, default=-40.0)
    parser.add_argument("--das_kld_high_db", type=float, default=40.0)
    parser.add_argument("--das_chunk_size", type=int, default=10000)
    parser.add_argument("--das_checkpoint", action="store_true")
    parser.add_argument("--settings_dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=300 if defaults_2d else 200)
    parser.add_argument("--batch_size", type=int, default=4 if defaults_2d else 16)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4 if defaults_2d else 1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warm_restarts", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=5 if defaults_2d else 10)
    parser.add_argument("--save_interval", type=int, default=50)


def derive_dataset_tag(npz_paths) -> str:
    if isinstance(npz_paths, (str, os.PathLike)):
        paths: Iterable[object] = [npz_paths]
    else:
        paths = list(npz_paths)
    stems = [os.path.splitext(os.path.basename(os.fspath(path)))[0]
             for path in paths]
    if not stems:
        return "dataset"
    if len(stems) == 1:
        return stems[0]
    return f"{stems[0]}_plus{len(stems) - 1}"


def _criterion_from_args(args):
    return create_criterion(
        getattr(args, "gamma_env", 0.1),
        gamma_constraint=getattr(args, "gamma_constraint", 0.0),
        loss_mode=getattr(args, "loss_mode", "mse"),
        depth_weight=getattr(args, "depth_weight", "none"),
        depth_weight_alpha=getattr(args, "depth_weight_alpha", 2.0),
        gamma_msle=getattr(args, "gamma_msle", 0.0),
        gamma_grad=getattr(args, "gamma_grad", 0.0),
        gamma_stat=getattr(args, "gamma_stat", 0.0),
        gamma_kld=getattr(args, "gamma_kld", 0.0),
        gamma_das=getattr(args, "gamma_das", 0.0),
        alpha_db=getattr(args, "alpha_db", -60.0),
        kld_bins=getattr(args, "kld_bins", 40),
        kld_eta=getattr(args, "kld_eta", 0.5),
        kld_low_db=getattr(args, "kld_low_db", -60.0),
        kld_high_db=getattr(args, "kld_high_db", 0.0),
        stat_var_weight=getattr(args, "stat_var_weight", 0.5),
        stat_win=getattr(args, "stat_win", 7),
        das_loss_mode=getattr(args, "das_loss_mode", "log_l1"),
        das_alpha_db=getattr(args, "das_alpha_db", -60.0),
        das_beta_kld=getattr(args, "das_beta_kld", 0.5),
        das_kld_bins=getattr(args, "das_kld_bins", 40),
        das_kld_eta=getattr(args, "das_kld_eta", 0.5),
        das_kld_low_db=getattr(args, "das_kld_low_db", -40.0),
        das_kld_high_db=getattr(args, "das_kld_high_db", 40.0),
    )


def run_train_1d(args, model, model_type, exp_name, arch_log_lines=None):
    from data import UltrasoundDataset, split_indices

    device = next(model.parameters()).device
    print(f"Device: {device}")
    dataset = UltrasoundDataset(args.npz, cs_ratio=args.cs_ratio, device="cpu").to(device)
    train_idx, val_idx = split_indices(
        len(dataset), args.val_ratio, args.seed, args.split_mode, dataset.group_id)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer, scheduler = create_optimizer(
        model, args.lr, args.weight_decay, args.epochs, args.warm_restarts)
    criterion = _criterion_from_args(args)
    start_epoch, best_val_snr, best_epoch = 1, -float("inf"), 0
    if args.resume:
        save_dir = os.path.dirname(args.resume)
        log = create_logger(save_dir)
        start_epoch, best_val_snr, best_epoch = resume_training(
            args.resume, model, optimizer, scheduler, device)
    else:
        save_dir = os.path.join(args.save_dir, exp_name)
        log = create_logger(save_dir)
        save_config(save_dir, args)
        log(f"{model_type} Training | {exp_name}")
        log(f"  samples={len(dataset)}, train={len(train_idx)}, val={len(val_idx)}")
        log(f"  layers={args.layers}, params={num_params:,}")
        if arch_log_lines:
            for line in arch_log_lines:
                log(f"  {line}")

    op = dataset.op
    extra_base = {"model_type": model_type}
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = train_idx[torch.randperm(len(train_idx), device=train_idx.device)]
        n_batches = (len(perm) + args.batch_size - 1) // args.batch_size
        epoch_loss = epoch_snr = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size:min((b + 1) * args.batch_size, len(perm))]
            x_input, y_target, y_k = dataset.get_batch(idx, device=device)
            y_sub = y_k if y_k is not None else op.A(x_input)
            lv, sv = train_one_batch(
                model, y_sub, y_target, op, criterion, optimizer, args.grad_clip)
            epoch_loss += lv
            epoch_snr += sv
        epoch_loss /= max(n_batches, 1)
        epoch_snr /= max(n_batches, 1)
        scheduler.step()

        val_ld, val_snr, val_nmse, init_snr, aux_last = _validate_1d(
            model, dataset, op, criterion, val_idx, args.batch_size, device)
        elapsed = time.time() - t0
        if epoch % args.log_interval == 0 or epoch == 1:
            l1, l2 = format_epoch_log(
                epoch, args.epochs, epoch_loss, epoch_snr, val_ld, val_snr,
                val_nmse, init_snr, aux_last, optimizer.param_groups[0]["lr"],
                elapsed)
            log(l1)
            log(l2)
        if val_snr > best_val_snr:
            best_val_snr, best_epoch = val_snr, epoch
            save_best(save_dir, epoch, model, optimizer, scheduler, val_snr,
                      val_ld["loss_total"], args,
                      extra={**extra_base, "train_idx": train_idx.cpu(),
                             "val_idx": val_idx.cpu()})
        if epoch % args.save_interval == 0:
            save_checkpoint(save_dir, epoch, model, optimizer, scheduler,
                            extra=extra_base)

    log(f"\n最佳验证 SNR: {best_val_snr:.2f} dB @ Epoch {best_epoch}")
    log(f"模型保存至: {save_dir}")
    save_final(save_dir, args.epochs, model, best_val_snr, best_epoch, args,
               extra={**extra_base, "train_idx": train_idx.cpu(),
                      "val_idx": val_idx.cpu()})


def _validate_1d(model, dataset, op, criterion, val_idx, batch_size, device):
    model.eval()
    val_loss_acc = {}
    val_snr_acc = val_nmse_acc = init_snr_acc = 0.0
    aux_list = None
    with torch.no_grad():
        n_val = len(val_idx)
        for s in range(0, n_val, batch_size):
            e = min(s + batch_size, n_val)
            idx = val_idx[s:e]
            x_input, y_target, y_k = dataset.get_batch(idx, device=device)
            y_sub = y_k if y_k is not None else op.A(x_input)
            vld, vs, vn, ins, al = validate_one_batch(
                model, y_sub, y_target, op, criterion)
            w = (e - s) / max(n_val, 1)
            for k, v in vld.items():
                val_loss_acc[k] = val_loss_acc.get(k, 0.0) + v * w
            val_snr_acc += vs * w
            val_nmse_acc += vn * w
            init_snr_acc += ins * w
            aux_list = al
    return val_loss_acc, val_snr_acc, val_nmse_acc, init_snr_acc, (aux_list[-1] if aux_list else {})


def _probe_xz(probe_geometry):
    geom = np.asarray(probe_geometry)
    if geom.ndim != 2:
        raise ValueError("probe_geometry must be 2D")
    if geom.shape[0] >= 3:
        return geom[0], geom[2]
    if geom.shape[1] >= 3:
        return geom[:, 0], geom[:, 2]
    if geom.shape[0] >= 2:
        return geom[0], geom[1]
    if geom.shape[1] >= 2:
        return geom[:, 0], geom[:, 1]
    return geom.reshape(-1), np.zeros(geom.size, dtype=geom.dtype)


def _make_das_forward(ds, args, device):
    if getattr(args, "gamma_das", 0.0) <= 0:
        return None
    if getattr(ds, "_compress_mode", "") == "post_das":
        from das_torch import PostDASEnvelope
        return PostDASEnvelope().to(device)
    if not getattr(ds, "has_das_meta", False):
        return None
    from das_torch import DASForwardSingleAngle
    probe_x, probe_z = _probe_xz(ds.probe_geometry)
    return DASForwardSingleAngle(
        probe_x=probe_x,
        probe_z=probe_z,
        x_axis=ds.scan_x_axis,
        z_axis=ds.scan_z_axis,
        initial_time=ds.initial_time,
        fs=ds.fs,
        c=ds.c,
        chunk_size=getattr(args, "das_chunk_size", 10000),
        use_checkpoint=getattr(args, "das_checkpoint", False),
    ).to(device)


def _load_datasets(npz_list, cs_ratio, patch_h, patch_stride):
    from data import UltrasoundFrameDataset
    return [
        UltrasoundFrameDataset(
            path, cs_ratio=cs_ratio,
            patch_h=patch_h, patch_stride=patch_stride, device="cpu")
        for path in npz_list
    ]


def run_train_2d(args, model, model_type, exp_name, arch_log_lines=None):
    from data import MultiDatasetSampler

    device = next(model.parameters()).device
    print(f"Device: {device}")
    if getattr(args, "compile", False):
        model = torch.compile(model)

    npz_list = args.npz if isinstance(args.npz, list) else [args.npz]
    datasets = _load_datasets(npz_list, args.cs_ratio, args.patch_h, args.patch_stride)
    sampler = MultiDatasetSampler(
        datasets, val_ratio=args.val_ratio, seed=args.seed,
        split_mode=args.split_mode).to(device)
    das_modules = {id(ds): _make_das_forward(ds, args, device) for ds in datasets}

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer, scheduler = create_optimizer(
        model, args.lr, args.weight_decay, args.epochs, args.warm_restarts)
    criterion = _criterion_from_args(args)
    start_epoch, best_val_snr, best_epoch = 1, -float("inf"), 0

    if args.resume:
        save_dir = os.path.dirname(args.resume)
        log = create_logger(save_dir)
        start_epoch, best_val_snr, best_epoch = resume_training(
            args.resume, model, optimizer, scheduler, device)
    else:
        save_dir = os.path.join(args.save_dir, exp_name)
        log = create_logger(save_dir)
        save_config(save_dir, args)
        log(f"{model_type} 2D Training | {exp_name}")
        log(sampler.summary())
        log(f"  datasets={len(datasets)}, train={sampler.n_train}, val={sampler.n_val}")
        log(f"  layers={args.layers}, params={num_params:,}")
        if arch_log_lines:
            for line in arch_log_lines:
                log(f"  {line}")

    val_idx_per_ds = [v.detach().cpu().numpy().astype("int64")
                      for v in sampler.val_indices]
    train_idx_per_ds = [t.detach().cpu().numpy().astype("int64")
                        for t in sampler.train_indices]
    extra_base = {
        "mode": "2d",
        "model_type": model_type,
        "val_idx_per_ds": val_idx_per_ds,
        "train_idx_per_ds": train_idx_per_ds,
        "split_meta": {
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "split_mode": str(args.split_mode),
            "npz_paths": list(npz_list),
        },
    }
    val_bs = args.val_batch_size or args.batch_size
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = epoch_snr = 0.0
        n_batches = 0
        for ds, op, idx in sampler.iter_train_batches(args.batch_size, epoch_seed=epoch):
            x_input, y_target, y_k = ds.get_batch(idx, device=device)
            y_sub = y_k if y_k is not None else op.A(x_input)
            das_meta = _batch_das_meta(ds, idx, das_modules.get(id(ds)), device)
            lv, sv = train_one_batch(
                model, y_sub, y_target, op, criterion, optimizer,
                args.grad_clip, das_meta=das_meta)
            epoch_loss += lv
            epoch_snr += sv
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        epoch_snr /= max(n_batches, 1)
        scheduler.step()

        val_ld, val_snr, val_nmse, init_snr, aux_last = _validate_2d(
            model, sampler, criterion, val_bs, das_modules, device)
        elapsed = time.time() - t0
        if epoch % args.log_interval == 0 or epoch == 1:
            l1, l2 = format_epoch_log(
                epoch, args.epochs, epoch_loss, epoch_snr, val_ld, val_snr,
                val_nmse, init_snr, aux_last, optimizer.param_groups[0]["lr"],
                elapsed)
            log(l1)
            log(l2)

        if val_snr > best_val_snr:
            best_val_snr, best_epoch = val_snr, epoch
            save_best(save_dir, epoch, model, optimizer, scheduler, val_snr,
                      val_ld["loss_total"], args, extra=extra_base)
        if epoch % args.save_interval == 0:
            save_checkpoint(save_dir, epoch, model, optimizer, scheduler,
                            extra=extra_base)

    log(f"\n最佳验证 SNR: {best_val_snr:.2f} dB @ Epoch {best_epoch}")
    log(f"模型保存至: {save_dir}")
    save_final(save_dir, args.epochs, model, best_val_snr, best_epoch, args,
               extra=extra_base)


def _batch_das_meta(ds, idx, das_forward, device):
    if das_forward is None:
        return None
    angles = ds.get_frame_angles(idx, device=device)
    if angles is None:
        return None
    return {"das_forward": das_forward, "angles": angles}


def _validate_2d(model, sampler, criterion, batch_size, das_modules, device):
    model.eval()
    val_loss_sum = {}
    val_snr_s = val_nmse_s = init_snr_s = 0.0
    vb = 0
    aux_last = {}
    with torch.no_grad():
        for ds, op, idx in sampler.iter_val_batches(batch_size):
            x_input, y_target, y_k = ds.get_batch(idx, device=device)
            y_sub = y_k if y_k is not None else op.A(x_input)
            das_meta = _batch_das_meta(ds, idx, das_modules.get(id(ds)), device)
            vld, vs, vn, ins, al = validate_one_batch(
                model, y_sub, y_target, op, criterion, das_meta=das_meta)
            for k, v in vld.items():
                val_loss_sum[k] = val_loss_sum.get(k, 0.0) + v
            val_snr_s += vs
            val_nmse_s += vn
            init_snr_s += ins
            vb += 1
            aux_last = al[-1] if al else {}
    for k in val_loss_sum:
        val_loss_sum[k] /= max(vb, 1)
    return (
        val_loss_sum,
        val_snr_s / max(vb, 1),
        val_nmse_s / max(vb, 1),
        init_snr_s / max(vb, 1),
        aux_last,
    )


__all__ = [
    "create_optimizer",
    "create_criterion",
    "create_logger",
    "save_config",
    "train_one_batch",
    "validate_one_batch",
    "format_epoch_log",
    "save_best",
    "save_checkpoint",
    "save_final",
    "override_args_from_checkpoint",
    "resume_training",
    "add_common_train_args",
    "derive_dataset_tag",
    "run_train_1d",
    "run_train_2d",
]
