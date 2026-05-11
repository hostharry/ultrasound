"""HASA-ADMM-Net 2D 训练脚本."""

import argparse
from datetime import datetime

import torch

from HASA_ADMM_Net_2D import HASA_ADMM_Net_2D, HASAWeight2D
from train_common import (
    add_common_train_args, override_args_from_checkpoint, run_train_2d,
)


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = HASA_ADMM_Net_2D(
        layer_num=args.layers,
        hasa_ctor=lambda: HASAWeight2D(
            hidden_ch=args.hasa_hidden,
            num_layers=args.num_hasa_layers,
            inner_ks=args.hasa_kernel,
        ),
        W_mode="A",
        share_W=args.share_W,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (f"HASA_ADMM_2D_ratio{args.cs_ratio}_L{args.layers}"
                f"_ph{args.patch_h}_{timestamp}")

    run_train_2d(
        args, model,
        model_type="hasa_admm_2d",
        exp_name=exp_name,
        arch_log_lines=[
            f"hasa: hidden={args.hasa_hidden}, layers={args.num_hasa_layers}, "
            f"kernel={args.hasa_kernel}",
            f"share_W={args.share_W}, patch_h={args.patch_h}",
        ],
    )


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(description="HASA-ADMM-Net 2D 训练", add_help=add_help)
    parser.add_argument("--npz", type=str, nargs="+",
                        default=["picmus_simu_reso_frames.npz"], help="帧级 npz 数据路径 (支持多个)")
    add_common_train_args(parser, defaults_2d=True)

    parser.add_argument("--hasa_hidden", type=int, default=16, help="HASA 隐藏通道数")
    parser.add_argument("--num_hasa_layers", type=int, default=2, help="HASA 卷积层数")
    parser.add_argument("--hasa_kernel", type=int, default=5,
                        help="HASA 内层卷积核大小 (第1层固定3, 后续层用此值)")
    parser.add_argument("--share_W", action="store_true", default=True, help="跨层共享 W")
    parser.add_argument("--no_share_W", dest="share_W", action="store_false")
    parser.add_argument("--gamma_constraint", type=float, default=0.01, help="约束一致性损失权重")

    parser.add_argument("--patch_h", type=int, default=None, help="patch 高度 (阵元数方向)")
    parser.add_argument("--patch_stride", type=int, default=None, help="patch 步长, None=patch_h//2")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
