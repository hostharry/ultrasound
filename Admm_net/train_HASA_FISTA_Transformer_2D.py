"""HASA-FISTA-Transformer-Net 2D 训练脚本

将 FISTA 双分支 prox 的卷积算子替换为 Transformer (patch-based)。
"""

import argparse
from datetime import datetime

import torch

from FISTA_Transformer import (
    HASA_FISTA_Transformer_Net_2D,
    HASAWeightTransformer2D,
)
from train_common import (
    add_common_train_args, override_args_from_checkpoint, run_train_2d,
)


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = HASA_FISTA_Transformer_Net_2D(
        layer_num=args.layers,
        hasa_ctor=lambda: HASAWeightTransformer2D(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_transformer_layers,
            patch_size=args.patch_size,
            win_size=args.window_size,
        ),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_transformer_layers,
        patch_size=args.patch_size,
        win_size=args.window_size,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (f"HASA_FISTA_SwinT_2D_ratio{args.cs_ratio}"
                f"_L{args.layers}_d{args.d_model}_ps{args.patch_size}"
                f"_ws{args.window_size}_ph{args.patch_h}_{timestamp}")

    run_train_2d(
        args, model,
        model_type="hasa_fista_transformer_2d",
        exp_name=exp_name,
        arch_log_lines=[
            f"d_model={args.d_model}, nhead={args.nhead}, "
            f"num_transformer_layers={args.num_transformer_layers}, "
            f"patch_size={args.patch_size}, win_size={args.window_size}",
            f"patch_h={args.patch_h}",
        ],
    )


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="HASA-FISTA-Transformer-Net 2D 训练",
        add_help=add_help,
    )
    parser.add_argument("--npz", type=str, nargs="+",
                        default=["picmus_simu_reso_frames.npz"],
                        help="帧级 npz 数据路径 (支持多个)")
    add_common_train_args(parser, defaults_2d=True)
    parser.add_argument("--patch_h", type=int, default=None,
                        help="patch 高度 (阵元数方向), None=完整帧")
    parser.add_argument("--patch_stride", type=int, default=None,
                        help="patch 步长, None=patch_h//2")

    g = parser.add_argument_group("Transformer 架构参数")
    g.add_argument("--d_model", type=int, default=64,
                   help="Transformer 特征维度")
    g.add_argument("--nhead", type=int, default=4,
                   help="Multi-head attention 头数")
    g.add_argument("--num_transformer_layers", type=int, default=2,
                   help="每个 encoder/decoder 的 Transformer block 数")
    g.add_argument("--patch_size", type=int, default=2,
                   help="Transformer patch embedding 的 patch 大小")
    g.add_argument("--window_size", type=int, default=8,
                   help="Swin window attention 的窗口大小")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
