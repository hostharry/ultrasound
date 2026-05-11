"""HASA-ADMM-Net 1D 训练脚本"""

import argparse
from datetime import datetime

import torch

from HASA_ADMM_Net import HASA_ADMM_Net, HASAWeight1D
from train_common import (
    add_common_train_args, override_args_from_checkpoint, run_train_1d,
)


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = HASA_ADMM_Net(
        layer_num=args.layers,
        hasa_ctor=lambda: HASAWeight1D(hidden_ch=args.hasa_hidden,
                                       num_layers=args.num_hasa_layers,
                                       inner_ks=args.hasa_kernel),
        W_mode=args.W_mode, W_num_filters=args.W_filters,
        W_kernel_size=args.W_kernel, share_W=args.share_W,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"HASA_ADMM_ratio{args.cs_ratio}_L{args.layers}_W{args.W_mode}_{timestamp}"

    run_train_1d(
        args, model,
        model_type="hasa_admm",
        exp_name=exp_name,
        arch_log_lines=[
            f"W_mode={args.W_mode}, share_W={args.share_W}",
            f"hasa: hidden={args.hasa_hidden}, layers={args.num_hasa_layers}, "
            f"kernel={args.hasa_kernel}",
        ],
    )


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(description="HASA-ADMM-Net 1D 训练", add_help=add_help)
    parser.add_argument("--npz", type=str, default="picmus_simu_reso.npz", help="npz 数据路径")
    add_common_train_args(parser, defaults_2d=False)

    parser.add_argument("--hasa_hidden", type=int, default=16, help="HASA 隐藏通道数")
    parser.add_argument("--num_hasa_layers", type=int, default=2, help="HASA 卷积层数")
    parser.add_argument("--hasa_kernel", type=int, default=5,
                        help="HASA 内层卷积核大小 (第1层固定3, 后续层用此值)")
    parser.add_argument("--share_W", action="store_true", default=True, help="跨层共享 W")
    parser.add_argument("--no_share_W", dest="share_W", action="store_false")
    parser.add_argument("--gamma_constraint", type=float, default=0.01, help="约束一致性损失权重")

    parser.add_argument("--W_mode", type=str, default="A", choices=["A", "B"],
                        help="稀疏算子模式: A=固定Haar, B=可学习")
    parser.add_argument("--W_filters", type=int, default=2, help="Mode B 滤波器数")
    parser.add_argument("--W_kernel", type=int, default=8, help="Mode B 核大小")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
