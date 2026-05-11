"""HUNet-1D 训练脚本

使用 Homotopy Unfolding + Swin-1D 窗口注意力 + DFFM 跨阶段融合。
数据、损失、日志格式与其他训练脚本一致.
"""

import os
import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Utils"))

import torch

from HUNet_1D import HUNet1D
from data import UltrasoundDataset, split_indices
from train_common import (
    create_optimizer, create_criterion, create_logger, save_config,
    train_one_batch, validate_one_batch, format_epoch_log,
    save_best, save_checkpoint, save_final, add_common_train_args,
    resume_training, override_args_from_checkpoint,
)


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n加载数据: {args.npz}")
    dataset = UltrasoundDataset(args.npz, cs_ratio=args.cs_ratio, device="cpu").to(device)

    num_samples = len(dataset)
    train_idx, val_idx = split_indices(
        num_samples=num_samples, val_ratio=args.val_ratio,
        seed=args.seed, split_mode=args.split_mode, group_id=dataset.group_id,
    )
    num_train, num_val = len(train_idx), len(val_idx)
    print(f"  样本数: {num_samples}, 训练: {num_train}, 验证: {num_val}")
    print(f"  信号长度 N: {dataset.N}, 频域观测 K: {dataset.op.K}")

    model = HUNet1D(
        num_stages=args.num_stages,
        depth=args.depth,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        swin_depth=args.swin_depth,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  阶段数: {args.num_stages}, 参数量: {num_params:,}")
    print(f"  depth={args.depth}, embed_dim={args.embed_dim}, nhead={args.nhead}, "
          f"window_size={args.window_size}, mlp_ratio={args.mlp_ratio}, "
          f"swin_depth={args.swin_depth}")

    optimizer, scheduler = create_optimizer(
        model, args.lr, args.weight_decay, args.epochs, args.warm_restarts)
    criterion = create_criterion(
        args.gamma_env, loss_mode=args.loss_mode,
        depth_weight=args.depth_weight, depth_weight_alpha=args.depth_weight_alpha,
        gamma_msle=args.gamma_msle,
    )

    start_epoch = 1
    best_val_snr = -float("inf")
    best_epoch = 0

    if args.resume:
        save_dir = os.path.dirname(args.resume)
        log = create_logger(save_dir)
        start_epoch, best_val_snr, best_epoch = resume_training(
            args.resume, model, optimizer, scheduler, device)
        log(f"\n--- 从 epoch {start_epoch} 恢复训练 (目标 {args.epochs} epochs) ---")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = (f"HUNet1D_ratio{args.cs_ratio}"
                    f"_S{args.num_stages}_d{args.embed_dim}"
                    f"_w{args.window_size}_{timestamp}")
        save_dir = os.path.join(args.save_dir, exp_name)
        log = create_logger(save_dir)
        save_config(save_dir, args)
        log(f"HUNet-1D Training | {exp_name}")
        log(f"  stages={args.num_stages}, depth={args.depth}, params={num_params:,}")
        log(f"  embed_dim={args.embed_dim}, nhead={args.nhead}, "
            f"window_size={args.window_size}, mlp_ratio={args.mlp_ratio}, "
            f"swin_depth={args.swin_depth}")
        log(f"  samples={num_samples}, train={num_train}, val={num_val}")
        log("")

    op = dataset.op

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()

        train_perm = train_idx[torch.randperm(num_train, device=train_idx.device)]
        n_batches = (num_train + args.batch_size - 1) // args.batch_size
        epoch_loss, epoch_snr = 0.0, 0.0

        for b in range(n_batches):
            idx = train_perm[b * args.batch_size:
                             min((b + 1) * args.batch_size, num_train)]
            x_input, y_target, y_k = dataset.get_batch(idx)
            y_sub = y_k if y_k is not None else op.A(x_input)

            loss_val, snr_val = train_one_batch(
                model, y_sub, y_target, op, criterion, optimizer, args.grad_clip)
            epoch_loss += loss_val
            epoch_snr += snr_val

        epoch_loss /= n_batches
        epoch_snr /= n_batches
        scheduler.step()

        model.eval()
        with torch.no_grad():
            x_input, y_target, y_k = dataset.get_batch(val_idx)
            y_sub = y_k if y_k is not None else op.A(x_input)
            val_ld, val_snr, val_nmse, init_snr, aux_list = validate_one_batch(
                model, y_sub, y_target, op, criterion)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        if epoch % args.log_interval == 0 or epoch == 1:
            l1, l2 = format_epoch_log(
                epoch, args.epochs, epoch_loss, epoch_snr,
                val_ld, val_snr, val_nmse, init_snr, aux_list[-1], lr, elapsed)
            log(l1)
            log(l2)

        if val_snr > best_val_snr:
            best_val_snr, best_epoch = val_snr, epoch
            save_best(save_dir, epoch, model, optimizer, scheduler,
                      val_snr, val_ld["loss_total"], args,
                      extra={"model_type": "hunet_1d",
                             "train_idx": train_idx.cpu(),
                             "val_idx": val_idx.cpu()})

        if epoch % args.save_interval == 0:
            save_checkpoint(save_dir, epoch, model, optimizer, scheduler,
                            extra={"model_type": "hunet_1d"})

    log(f"\n最佳验证 SNR: {best_val_snr:.2f} dB @ Epoch {best_epoch}")
    log(f"模型保存至: {save_dir}")
    save_final(save_dir, args.epochs, model, best_val_snr, best_epoch, args,
               extra={"model_type": "hunet_1d",
                      "train_idx": train_idx.cpu(),
                      "val_idx": val_idx.cpu()})


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="HUNet-1D 训练",
        add_help=add_help,
    )
    parser.add_argument("--npz", type=str,
                        default="../dataset_fdbf_energy_mu_8_9_15.npz",
                        help="npz 数据路径")
    add_common_train_args(parser, defaults_2d=False)

    g = parser.add_argument_group("HUNet-1D 架构参数")
    g.add_argument("--num_stages", type=int, default=7,
                   help="Homotopy 展开阶段数")
    g.add_argument("--depth", type=int, default=4,
                   help="每阶段 encoder/decoder 层数 (含下采样级别)")
    g.add_argument("--embed_dim", type=int, default=32,
                   help="特征维度")
    g.add_argument("--nhead", type=int, default=4,
                   help="注意力头数")
    g.add_argument("--window_size", type=int, default=32,
                   help="Swin 窗口大小")
    g.add_argument("--mlp_ratio", type=float, default=2.0,
                   help="FFN 扩展比")
    g.add_argument("--swin_depth", type=int, default=2,
                   help="每级 Swin block 数量")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
