"""FISTA-DWT-LiteV3-2D 训练脚本 (帧/patch).

V3 = Lite-2D 独立子带 WAV enc/dec + Pre-Norm GroupNorm.
其它接口与 train_lite_v2_2d.py 完全一致, 仅模型类换成 LiteV3 + 多 1 个
--num_groups CLI 参数.
"""

import argparse
import sys
import os
from datetime import datetime

import torch

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "Utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from FISTA_DWT_LiteV3_2D import (
    FISTA_DWT_LiteV3_2D_Net, MultiScaleHASA2D, MiniUNetHASA2D,
)
from train_common import (
    add_common_train_args,
    derive_dataset_tag,
    override_args_from_checkpoint,
    run_train_2d,
)


def _make_hasa_ctor(args):
    if args.hasa_type == "unet":
        return lambda: MiniUNetHASA2D(base_ch=args.hasa_base_ch)
    return lambda: MultiScaleHASA2D(
        hidden_ch=args.hasa_hidden,
        num_layers=args.num_hasa_layers,
        inner_ks=args.hasa_kernel,
        context_ks=args.hasa_context_ks,
        context_dilation=args.hasa_context_dilation,
    )


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = FISTA_DWT_LiteV3_2D_Net(
        layer_num=args.layers,
        hasa_ctor=_make_hasa_ctor(args),
        d_model=args.d_model,
        num_conv_blocks=args.num_conv_blocks,
        conv_ks=args.conv_ks,
        J=args.dwt_levels,
        prox_tau=args.prox_tau,
        num_groups=args.num_groups,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_tag = derive_dataset_tag(args.npz)
    exp_name = f"LiteV3_2D_{ds_tag}_{timestamp}"

    run_train_2d(
        args, model,
        model_type="fista_dwt_litev3_2d",
        exp_name=exp_name,
        arch_log_lines=[
            f"d_model={args.d_model}, num_conv_blocks={args.num_conv_blocks}, "
            f"conv_ks={args.conv_ks}, dwt_levels={args.dwt_levels}",
            f"hasa: type={args.hasa_type}, hidden={args.hasa_hidden}, "
            f"num_layers={args.num_hasa_layers}, kernel={args.hasa_kernel}, "
            f"base_ch={args.hasa_base_ch}",
            f"patch_h={args.patch_h}, patch_stride={args.patch_stride}",
            f"prox_tau={args.prox_tau}",
            f"V3: independent WAV enc/dec per DWT band + "
            f"Pre-Norm GN (num_groups={args.num_groups})",
        ],
    )


def build_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="FISTA-DWT-LiteV3-2D 训练 (V2 + Pre-Norm GroupNorm)",
        add_help=add_help,
    )
    parser.add_argument(
        "--npz", type=str, nargs="+",
        default=["../data/picmus_simu_cont_frames.npz"],
        help="帧级 npz 路径 (可多个)",
    )
    add_common_train_args(parser, defaults_2d=True)

    g = parser.add_argument_group("LiteV3-2D 架构")
    g.add_argument("--d_model", type=int, default=32, help="TV/DWT 分支特征维")
    g.add_argument("--num_conv_blocks", type=int, default=2,
                   help="每分支 ConvResBlock 数")
    g.add_argument("--conv_ks", type=int, default=5, help="ConvRes 卷积核")
    g.add_argument("--num_groups", type=int, default=8,
                   help="ConvResBlock 内 GroupNorm 的 group 数 "
                        "(d_model=32 时建议 8, 即每组 4 channel)")
    g.add_argument("--dwt_levels", type=int, default=1, choices=[1],
                   help="DWT 层数 (当前仅支持 1)")
    g.add_argument("--prox_tau", type=float, default=0.005,
                   help="平滑 soft-threshold 的 tau")
    g.add_argument("--hasa_type", type=str, default="conv", choices=["conv", "unet"],
                   help="HASA 类型: conv=MultiScaleHASA2D, unet=MiniUNetHASA2D")
    g.add_argument("--hasa_base_ch", type=int, default=16,
                   help="U-Net HASA 的基础通道数")
    g.add_argument("--hasa_hidden", type=int, default=16, help="Conv HASA 隐藏通道")
    g.add_argument("--num_hasa_layers", type=int, default=2, help="Conv HASA 卷积层数")
    g.add_argument("--hasa_kernel", type=int, default=5,
                   help="Conv HASA 内层卷积核")
    g.add_argument("--hasa_context_ks", type=int, default=3,
                   help="Conv HASA context 分支卷积核")
    g.add_argument("--hasa_context_dilation", type=int, default=3,
                   help="Conv HASA context 分支 dilation")
    g.add_argument("--patch_h", type=int, default=32,
                   help="patch 高度 (阵元方向)")
    g.add_argument("--patch_stride", type=int, default=16,
                   help="patch 步长")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
