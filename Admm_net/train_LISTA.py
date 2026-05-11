"""LISTA 1D 训练脚本"""

import argparse
from datetime import datetime

import torch

from LISTA_Baseline import LISTA_Net
from train_common import (
    add_common_train_args, override_args_from_checkpoint, run_train_1d,
)


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = LISTA_Net(
        layer_num=args.layers,
        kernel_size=args.kernel_size,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"LISTA_ratio{args.cs_ratio}_L{args.layers}_k{args.kernel_size}_{timestamp}"

    run_train_1d(
        args, model,
        model_type="lista",
        exp_name=exp_name,
        arch_log_lines=[f"kernel_size={args.kernel_size}"],
    )


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(description="LISTA 1D 训练", add_help=add_help)
    parser.add_argument("--npz", type=str,
                        default="picmus_simu_reso.npz", help="npz 数据路径")
    add_common_train_args(parser, defaults_2d=False)

    g = parser.add_argument_group("LISTA 架构参数")
    g.add_argument("--kernel_size", type=int, default=5,
                   help="W_e / W_t 卷积核大小 (论文使用 5)")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
