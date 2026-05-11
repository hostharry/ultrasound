"""FISTA-Transformer-DWT 1D 训练脚本"""

import argparse
import sys
import os
from datetime import datetime

import torch

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "Utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from FISTA_Transformer_DWT import FISTA_Transformer_DWT_Net, HASAWeightTransformer1D
from train_common import add_common_train_args, override_args_from_checkpoint, run_train_1d, derive_dataset_tag


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = FISTA_Transformer_DWT_Net(
        layer_num=args.layers,
        hasa_ctor=lambda: HASAWeightTransformer1D(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_transformer_layers,
        ),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_transformer_layers,
        J=args.dwt_levels,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_tag = derive_dataset_tag(args.npz)
    exp_name = f"FISTA_DWT_{ds_tag}_{timestamp}"

    run_train_1d(
        args, model,
        model_type="fista_transformer_dwt",
        exp_name=exp_name,
        arch_log_lines=[
            f"d_model={args.d_model}, nhead={args.nhead}, "
            f"num_transformer_layers={args.num_transformer_layers}, "
            f"dwt_levels={args.dwt_levels}",
        ],
    )


def build_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="FISTA-Transformer-DWT 1D 训练", add_help=add_help)
    parser.add_argument("--npz", type=str,
                        default="../data/picmus_simu_reso.npz",
                        help="npz 数据路径")
    add_common_train_args(parser, defaults_2d=False)

    g = parser.add_argument_group("Transformer-DWT 架构参数")
    g.add_argument("--d_model", type=int, default=16)
    g.add_argument("--nhead", type=int, default=4)
    g.add_argument("--num_transformer_layers", type=int, default=1,
                   help="每个 encoder/decoder 的 Transformer block 数")
    g.add_argument("--dwt_levels", type=int, default=3,
                   help="Haar DWT 分解级数 (J), 需要 L 能被 2^J 整除")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
