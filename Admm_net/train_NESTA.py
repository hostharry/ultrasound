"""Deep-Unfolded NESTA 1D 训练脚本"""

import argparse
from datetime import datetime

import torch

from NESTA_Baseline import NESTA_Net
from train_common import (
    add_common_train_args, override_args_from_checkpoint, run_train_1d,
)


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = NESTA_Net(layer_num=args.layers).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"NESTA_ratio{args.cs_ratio}_L{args.layers}_{timestamp}"

    run_train_1d(args, model, model_type="nesta", exp_name=exp_name)


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="Deep-Unfolded NESTA 1D 训练", add_help=add_help)
    parser.add_argument("--npz", type=str,
                        default="picmus_simu_reso.npz", help="npz 数据路径")
    add_common_train_args(parser, defaults_2d=False)
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
