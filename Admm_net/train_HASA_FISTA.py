"""HASA-FISTA-Net 1D 训练脚本

与 HASA-ADMM-Net 的对比实验: 同样使用 HASA 自适应权重,
但求解器从 ADMM 变量分裂换为 FISTA 双分支 prox + momentum.
"""

import argparse
from datetime import datetime

import torch

from FISTA_Baseline import HASA_FISTA_Net_1D, HASAWeightFISTA1D
from train_common import (
    add_common_train_args, override_args_from_checkpoint, run_train_1d,
)


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = HASA_FISTA_Net_1D(
        layer_num=args.layers,
        hasa_ctor=lambda: HASAWeightFISTA1D(
            hidden_ch=args.hasa_hidden,
            num_layers=args.num_hasa_layers,
            inner_ks=args.hasa_kernel,
        ),
        feat_ch=args.fista_feat_ch,
        prox_k=args.fista_prox_k,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"HASA_FISTA_ratio{args.cs_ratio}_L{args.layers}_{timestamp}"

    run_train_1d(
        args, model,
        model_type="hasa_fista",
        exp_name=exp_name,
        arch_log_lines=[
            f"fista_feat_ch={args.fista_feat_ch}, fista_prox_k={args.fista_prox_k}",
            f"hasa: hidden={args.hasa_hidden}, layers={args.num_hasa_layers}, "
            f"kernel={args.hasa_kernel}",
        ],
    )


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="HASA-FISTA-Net 1D 训练 (对比 HASA-ADMM)",
        add_help=add_help,
    )
    parser.add_argument("--npz", type=str,
                        default="dataset_fdbf_energy_mu_8_9_15.npz",
                        help="npz 数据路径")
    add_common_train_args(parser, defaults_2d=False)

    parser.add_argument("--hasa_hidden", type=int, default=16, help="HASA 隐藏通道数")
    parser.add_argument("--num_hasa_layers", type=int, default=2, help="HASA 卷积层数")
    parser.add_argument("--hasa_kernel", type=int, default=5,
                        help="HASA 内层卷积核大小 (第1层固定3, 后续层用此值)")

    parser.add_argument("--fista_feat_ch", type=int, default=64,
                        help="FISTA prox 双分支特征通道数")
    parser.add_argument("--fista_prox_k", type=int, default=3,
                        help="FISTA prox 卷积核大小")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
