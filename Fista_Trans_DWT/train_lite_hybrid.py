"""FISTA-DWT-Lite-Hybrid 1D 训练脚本"""

import argparse
import sys
import os
from datetime import datetime

import torch

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "Utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from FISTA_DWT_Lite_Hybrid import FISTA_DWT_Lite_Hybrid_Net, HASAWeightTransformer1D
from train_common import add_common_train_args, override_args_from_checkpoint, run_train_1d, derive_dataset_tag


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = FISTA_DWT_Lite_Hybrid_Net(
        layer_num=args.layers,
        hasa_ctor=lambda: HASAWeightTransformer1D(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_transformer_layers,
        ),
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.num_transformer_layers,
        num_conv_blocks=args.num_conv_blocks,
        conv_ks=args.conv_ks,
        J=args.dwt_levels,
        mixer_nhead=args.mixer_nhead,
        mixer_mlp_ratio=args.mixer_mlp_ratio,
        mixer_gamma_init=args.mixer_gamma_init,
        detail_thr_gain=args.detail_thr_gain,
    ).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_tag = derive_dataset_tag(args.npz)
    exp_name = f"LiteHybrid_{ds_tag}_{timestamp}"

    run_train_1d(
        args, model,
        model_type="fista_dwt_lite_hybrid",
        exp_name=exp_name,
        arch_log_lines=[
            f"d_model={args.d_model}, nhead={args.nhead}, "
            f"hasa_layers={args.num_transformer_layers}",
            f"num_conv_blocks={args.num_conv_blocks}, conv_ks={args.conv_ks}, "
            f"dwt_levels={args.dwt_levels}",
            f"mixer_nhead={args.mixer_nhead}, mixer_mlp_ratio={args.mixer_mlp_ratio}, "
            f"mixer_gamma_init={args.mixer_gamma_init}, detail_thr_gain={args.detail_thr_gain}",
        ],
    )


def build_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="FISTA-DWT-Lite-Hybrid 1D 训练", add_help=add_help)
    parser.add_argument("--npz", type=str,
                        default="../data/picmus_simu_reso.npz",
                        help="npz 数据路径")
    add_common_train_args(parser, defaults_2d=False)

    g = parser.add_argument_group("DWT-Lite-Hybrid 架构参数")
    g.add_argument("--d_model", type=int, default=32)
    g.add_argument("--nhead", type=int, default=4,
                   help="HASA Transformer 注意力头数")
    g.add_argument("--num_transformer_layers", type=int, default=1,
                   help="HASA Transformer block 数")
    g.add_argument("--num_conv_blocks", type=int, default=2,
                   help="每个 prox encoder/decoder 的 ConvResBlock 数")
    g.add_argument("--conv_ks", type=int, default=5,
                   help="ConvResBlock 卷积核大小")
    g.add_argument("--dwt_levels", type=int, default=3,
                   help="Haar DWT 分解级数 (J)")
    g.add_argument("--mixer_nhead", type=int, default=4,
                   help="CrossSubbandMixer 注意力头数")
    g.add_argument("--mixer_mlp_ratio", type=float, default=2.0,
                   help="CrossSubbandMixer FFN 扩展比")
    g.add_argument("--mixer_gamma_init", type=float, default=0.01,
                   help="CrossSubband mixer 初始门控值 (记录的是 tanh(gamma))")
    g.add_argument("--detail_thr_gain", type=float, default=1.0,
                   help="DWT detail 子带 shrink 强度倍率, <1 更软, >1 更强")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume:
        override_args_from_checkpoint(args, args.resume)
    train(args)


if __name__ == "__main__":
    main()
