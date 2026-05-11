"""HUNet-1D DepthAdaptive 训练脚本.

在 HUNet-1D 基础上引入:
  - 自适应阈值映射 (feature-gated + depth curve)
  - 深度加权损失的推荐默认配置
"""

import os
import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Utils"))

import torch

from HUNet_1D_DepthAdaptive import HUNet1DDepthAdaptive
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

    model = HUNet1DDepthAdaptive(
        num_stages=args.num_stages,
        depth=args.depth,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        swin_depth=args.swin_depth,
        adaptive_thr=args.adaptive_thr,
        thr_hidden_ratio=args.thr_hidden_ratio,
        thr_scale_min=args.thr_scale_min,
        thr_scale_max=args.thr_scale_max,
        thr_init_depth_slope=args.thr_init_depth_slope,
        gamma_init=args.gamma_init,
        gamma_decay=args.gamma_decay,
        gamma_floor=args.gamma_floor,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  阶段数: {args.num_stages}, 参数量: {num_params:,}")
    print(f"  depth={args.depth}, embed_dim={args.embed_dim}, nhead={args.nhead}, "
          f"window_size={args.window_size}, mlp_ratio={args.mlp_ratio}, "
          f"swin_depth={args.swin_depth}")
    print(f"  adaptive_thr={args.adaptive_thr}, thr_hidden_ratio={args.thr_hidden_ratio}, "
          f"thr_scale=[{args.thr_scale_min}, {args.thr_scale_max}], "
          f"thr_init_depth_slope={args.thr_init_depth_slope}")
    print(f"  gamma_init={args.gamma_init}, gamma_decay={args.gamma_decay}, "
          f"gamma_floor={args.gamma_floor}")

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
        exp_name = (f"HUNet1D_DA_ratio{args.cs_ratio}"
                    f"_S{args.num_stages}_d{args.embed_dim}"
                    f"_w{args.window_size}_{timestamp}")
        save_dir = os.path.join(args.save_dir, exp_name)
        log = create_logger(save_dir)
        save_config(save_dir, args)
        log(f"HUNet-1D-DepthAdaptive Training | {exp_name}")
        log(f"  stages={args.num_stages}, depth={args.depth}, params={num_params:,}")
        log(f"  embed_dim={args.embed_dim}, nhead={args.nhead}, "
            f"window_size={args.window_size}, mlp_ratio={args.mlp_ratio}, "
            f"swin_depth={args.swin_depth}")
        log(f"  adaptive_thr={args.adaptive_thr}, thr_hidden_ratio={args.thr_hidden_ratio}, "
            f"thr_scale=[{args.thr_scale_min}, {args.thr_scale_max}], "
            f"thr_init_depth_slope={args.thr_init_depth_slope}")
        log(f"  gamma_init={args.gamma_init}, gamma_decay={args.gamma_decay}, "
            f"gamma_floor={args.gamma_floor}")
        log(f"  loss_mode={args.loss_mode}, depth_weight={args.depth_weight}, "
            f"depth_weight_alpha={args.depth_weight_alpha}")
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
                      extra={"model_type": "hunet_1d_depth_adaptive",
                             "train_idx": train_idx.cpu(),
                             "val_idx": val_idx.cpu()})

        if epoch % args.save_interval == 0:
            save_checkpoint(save_dir, epoch, model, optimizer, scheduler,
                            extra={"model_type": "hunet_1d_depth_adaptive"})

    log(f"\n最佳验证 SNR: {best_val_snr:.2f} dB @ Epoch {best_epoch}")
    log(f"模型保存至: {save_dir}")
    save_final(save_dir, args.epochs, model, best_val_snr, best_epoch, args,
               extra={"model_type": "hunet_1d_depth_adaptive",
                      "train_idx": train_idx.cpu(),
                      "val_idx": val_idx.cpu()})


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="HUNet-1D DepthAdaptive 训练",
        add_help=add_help,
    )
    parser.add_argument("--npz", type=str,
                        default="../data/picmus_simu_reso.npz",
                        help="npz 数据路径")
    add_common_train_args(parser, defaults_2d=False)

    # 推荐默认损失配置: nmse + depth weighting
    rec = HUNet1DDepthAdaptive.recommended_loss_cfg()
    parser.set_defaults(
        loss_mode=rec["loss_mode"],
        depth_weight=rec["depth_weight"],
        depth_weight_alpha=rec["depth_weight_alpha"],
        adaptive_thr=True,
    )

    g = parser.add_argument_group("HUNet-1D DepthAdaptive 架构参数")
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

    g.add_argument("--adaptive_thr", dest="adaptive_thr", action="store_true",
                   help="启用自适应阈值映射")
    g.add_argument("--disable_adaptive_thr", dest="adaptive_thr", action="store_false",
                   help="关闭自适应阈值映射 (回退固定阈值)")
    g.add_argument("--thr_hidden_ratio", type=float, default=0.5,
                   help="阈值映射 MLP 隐层比例")
    g.add_argument("--thr_scale_min", type=float, default=0.5,
                   help="阈值缩放最小值")
    g.add_argument("--thr_scale_max", type=float, default=1.5,
                   help="阈值缩放最大值")
    g.add_argument("--thr_init_depth_slope", type=float, default=-0.3,
                   help="深度曲线初始斜率, <0 表示深层阈值更小")
    g.add_argument("--gamma_init", type=float, default=0.03,
                   help="第1阶段基准阈值初始化")
    g.add_argument("--gamma_decay", type=float, default=0.85,
                   help="阶段间阈值衰减系数")
    g.add_argument("--gamma_floor", type=float, default=1e-4,
                   help="阈值下界, 防止塌到0")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
