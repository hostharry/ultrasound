"""HASA-FISTA-Transformer-Net 1D 训练脚本

将 FISTA 双分支 prox 的卷积算子替换为 Transformer。
"""

import argparse
from datetime import datetime

import torch

from FISTA_Transformer import (
    HASA_FISTA_Transformer_Net_1D,
    HASAWeightTransformer1D,
)
from train_common import (
    add_common_train_args, override_args_from_checkpoint, run_train_1d,
)


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = HASA_FISTA_Transformer_Net_1D(
        layer_num=args.layers,
        hasa_ctor=lambda: HASAWeightTransformer1D(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_transformer_layers,
        ),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_transformer_layers,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (f"HASA_FISTA_Transformer_ratio{args.cs_ratio}"
                f"_L{args.layers}_d{args.d_model}_{timestamp}")

    run_train_1d(
        args, model,
        model_type="hasa_fista_transformer",
        exp_name=exp_name,
        arch_log_lines=[
            f"d_model={args.d_model}, nhead={args.nhead}, "
            f"num_transformer_layers={args.num_transformer_layers}",
        ],
    )


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="HASA-FISTA-Transformer-Net 1D 训练",
        add_help=add_help,
    )
    parser.add_argument("--npz", type=str,
                        default="../dataset_fdbf_energy_mu_8_9_15.npz",
                        help="npz 数据路径")
    add_common_train_args(parser, defaults_2d=False)

    g = parser.add_argument_group("Transformer 架构参数")
    g.add_argument("--d_model", type=int, default=32,
                   help="Transformer 特征维度")
    g.add_argument("--nhead", type=int, default=4,
                   help="Multi-head attention 头数")
    g.add_argument("--num_transformer_layers", type=int, default=2,
                   help="每个 encoder/decoder 的 Transformer block 数")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
