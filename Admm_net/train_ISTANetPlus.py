"""ISTA-Net+ 1D 训练脚本

参考: ISTA-Net-PyTorch/Train_CS_ISTA_Net_plus.py
默认超参与原始论文对齐: lr=1e-4, gamma_sym=0.01, layers=9, n_channels=32
"""

import os
import time
import argparse
from datetime import datetime

import torch

from ISTANetPlus_Baseline import ISTANetPlus, ISTANetPlusLoss
from admm_data import UltrasoundDataset, split_indices
from admm_metrics import calc_snr, calc_nmse
from train_common import (
    create_optimizer, create_logger, save_config,
    format_epoch_log, save_best, save_checkpoint, save_final,
    resume_training, override_args_from_checkpoint,
)


def train_one_batch(model, y_sub, y_target, op, criterion, optimizer, grad_clip):
    x_hat, aux_list = model(y_sub, op, return_aux=True)
    loss, loss_dict = criterion(x_hat, y_target, aux_list)

    optimizer.zero_grad()
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        snr = calc_snr(y_target, x_hat).mean().item()
    return loss_dict["loss_total"], snr


def validate_one_batch(model, y_sub, y_target, op, criterion):
    x_hat, aux_list = model(y_sub, op, return_aux=True)
    _, loss_dict = criterion(x_hat, y_target, aux_list)
    snr = calc_snr(y_target, x_hat).mean().item()
    nmse = calc_nmse(y_target, x_hat).mean().item()
    init_snr = calc_snr(y_target, op.At(y_sub)).mean().item()
    return loss_dict, snr, nmse, init_snr, aux_list


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

    model = ISTANetPlus(
        layer_num=args.layers,
        n_channels=args.n_channels,
        kernel_size=args.kernel_size,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  层数: {args.layers}, n_channels={args.n_channels}, "
          f"kernel_size={args.kernel_size}, 参数量: {num_params:,}")

    optimizer, scheduler = create_optimizer(
        model, args.lr, args.weight_decay, args.epochs, args.warm_restarts)

    use_nmse = args.loss_mode in ("nmse", "nmse_logenv")
    criterion = ISTANetPlusLoss(gamma_sym=args.gamma_sym, use_nmse=use_nmse)
    print(f"  损失: ISTANetPlusLoss(gamma_sym={args.gamma_sym}, "
          f"use_nmse={use_nmse})")

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
        exp_name = (f"ISTANetPlus_ratio{args.cs_ratio}_L{args.layers}"
                    f"_ch{args.n_channels}_k{args.kernel_size}_{timestamp}")
        save_dir = os.path.join(args.save_dir, exp_name)
        log = create_logger(save_dir)
        save_config(save_dir, args)
        log(f"ISTA-Net+ 1D Training | {exp_name}")
        log(f"  layers={args.layers}, n_channels={args.n_channels}, "
            f"kernel_size={args.kernel_size}, params={num_params:,}")
        log(f"  gamma_sym={args.gamma_sym}, lr={args.lr}")
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
            log(l1); log(l2)

        if val_snr > best_val_snr:
            best_val_snr, best_epoch = val_snr, epoch
            save_best(save_dir, epoch, model, optimizer, scheduler,
                      val_snr, val_ld["loss_total"], args,
                      extra={"model_type": "ista_net_plus",
                             "train_idx": train_idx.cpu(),
                             "val_idx": val_idx.cpu()})

        if epoch % args.save_interval == 0:
            save_checkpoint(save_dir, epoch, model, optimizer, scheduler,
                            extra={"model_type": "ista_net_plus"})

    log(f"\n最佳验证 SNR: {best_val_snr:.2f} dB @ Epoch {best_epoch}")
    log(f"模型保存至: {save_dir}")
    save_final(save_dir, args.epochs, model, best_val_snr, best_epoch, args,
               extra={"model_type": "ista_net_plus",
                      "train_idx": train_idx.cpu(),
                      "val_idx": val_idx.cpu()})


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="ISTA-Net+ 1D 训练", add_help=add_help)
    parser.add_argument("--npz", type=str,
                        default="picmus_simu_reso.npz", help="npz 数据路径")

    parser.add_argument("--cs_ratio", type=int, default=8, help="压缩比")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--split_mode", type=str, default="group",
                        choices=["group", "random"], help="划分方式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--layers", type=int, default=9, help="展开层数")
    parser.add_argument("--n_channels", type=int, default=32, help="中间卷积通道数")
    parser.add_argument("--kernel_size", type=int, default=3, help="卷积核大小")

    parser.add_argument("--gamma_sym", type=float, default=0.01,
                        help="对称损失权重 γ (原始论文 0.01)")
    parser.add_argument("--loss_mode", type=str, default="mse",
                        choices=["mse", "nmse", "nmse_logenv"],
                        help="重建损失模式")

    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率 (原始论文 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--warm_restarts", type=int, default=0,
                        help="CosineAnnealingWarmRestarts T_0; 0=不使用")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")

    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument("--save_dir", type=str, default="model", help="保存目录")
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint 路径")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=50, help="保存间隔")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
