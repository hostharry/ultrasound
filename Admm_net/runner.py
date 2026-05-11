"""HASA-ADMM-Net 统一入口

用法:
  # 1D 流程 (逐线)
  python runner.py prepare --datasets simu_reso --cs_ratios 4 8 15
  python runner.py train --npz picmus_simu_reso.npz --cs_ratio 8
  python runner.py eval --exp_dir model/xxx

  # 2D 流程 (帧级)
  python runner.py prepare --mode frame --datasets simu_reso --cs_ratios 4 8
  python runner.py train2d --npz picmus_simu_reso_frames.npz --cs_ratio 8
"""

import argparse
from train_common import add_common_train_args


def _apply_train_preset(args):
    if args.preset == "fast":
        args.epochs = 40
        args.batch_size = max(args.batch_size, 16)
        args.lr = 1e-3
        args.layers = 6
    elif args.preset == "paper":
        args.epochs = 250
        args.batch_size = min(args.batch_size, 16)
        args.lr = 5e-4
        args.layers = 12


def _apply_eval_preset(args):
    if args.preset == "quick":
        args.eval_all = False
        args.batch_size = max(args.batch_size, 64)
    elif args.preset == "full":
        args.eval_all = True
        args.batch_size = min(args.batch_size, 32)


def main():
    parser = argparse.ArgumentParser(description="HASA-ADMM-Net 统一任务入口")
    sub = parser.add_subparsers(dest="command", required=True)

    # ========== prepare ==========
    p_prepare = sub.add_parser("prepare", help="预处理 PICMUS 数据")
    p_prepare.add_argument("--picmus_root", type=str, default=None)
    p_prepare.add_argument("--datasets", type=str, nargs="+", default=["simu_reso", "simu_cont"],
                           choices=["simu_reso", "simu_cont", "expe_reso", "expe_cont"])
    p_prepare.add_argument("--cs_ratios", type=int, nargs="+", default=[4, 8, 15])
    p_prepare.add_argument("--max_samples", type=int, default=2000)
    p_prepare.add_argument("--angle_stride", type=int, default=3)
    p_prepare.add_argument("--output_dir", type=str, default=".")
    p_prepare.add_argument("--merge", action="store_true", default=False)
    p_prepare.add_argument("--mode", type=str, default="line", choices=["line", "frame"])
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.set_defaults(_task="prepare")

    # ========== train (1D) ==========
    p_train = sub.add_parser("train", help="训练 HASA-ADMM-Net (1D)")
    p_train.add_argument("--npz", type=str, default="picmus_simu_reso.npz", help="npz 数据路径")
    add_common_train_args(p_train, defaults_2d=False)
    p_train.add_argument("--W_mode", type=str, default="A", choices=["A", "B"], help="稀疏算子模式")
    p_train.add_argument("--W_filters", type=int, default=2, help="Mode B 滤波器数")
    p_train.add_argument("--W_kernel", type=int, default=8, help="Mode B 核大小")
    p_train.add_argument("--preset", type=str, default="base",
                         choices=["base", "fast", "paper"], help="训练预设")
    p_train.set_defaults(_task="train")

    # ========== train2d ==========
    p_train2d = sub.add_parser("train2d", help="训练 2D HASA-ADMM-Net (帧级)")
    p_train2d.add_argument("--npz", type=str, nargs="+",
                           default=["picmus_simu_reso_frames.npz"], help="帧级 npz")
    add_common_train_args(p_train2d, defaults_2d=True)
    p_train2d.add_argument("--patch_h", type=int, default=None, help="patch 高度")
    p_train2d.add_argument("--patch_stride", type=int, default=None)
    p_train2d.set_defaults(_task="train2d")

    # ========== train_fista2d ==========
    p_fista2d = sub.add_parser("train_fista2d", help="训练 2D HASA-FISTA-Net (对比实验)")
    p_fista2d.add_argument("--npz", type=str, nargs="+",
                           default=["picmus_simu_reso_frames.npz"], help="帧级 npz")
    add_common_train_args(p_fista2d, defaults_2d=True)
    p_fista2d.add_argument("--patch_h", type=int, default=None, help="patch 高度")
    p_fista2d.add_argument("--patch_stride", type=int, default=None)
    p_fista2d.add_argument("--fista_feat_ch", type=int, default=64,
                           help="FISTA prox 特征通道数")
    p_fista2d.add_argument("--fista_prox_k", type=int, default=3,
                           help="FISTA prox 卷积核大小")
    p_fista2d.set_defaults(_task="train_fista2d")

    # ========== eval ==========
    p_eval = sub.add_parser("eval", help="评估 HASA-ADMM-Net")
    p_eval.add_argument("--exp_dir", type=str, required=True, help="实验目录")
    p_eval.add_argument("--ckpt_name", type=str, default="best_model.pth")
    p_eval.add_argument("--npz", type=str, default=None)
    p_eval.add_argument("--cs_ratio", type=int, default=None)
    p_eval.add_argument("--eval_all", action="store_true", default=False)
    p_eval.add_argument("--batch_size", type=int, default=32)
    p_eval.add_argument("--dynamic_range", type=float, default=60.0)
    p_eval.add_argument("--gpu", type=int, default=0)
    p_eval.add_argument("--preset", type=str, default="base",
                         choices=["base", "quick", "full"])
    p_eval.set_defaults(_task="eval")

    ns = parser.parse_args()

    if ns._task == "prepare":
        from prepare_picmus_data import run_prepare, PICMUS_ROOT
        if ns.picmus_root is None:
            ns.picmus_root = PICMUS_ROOT
        run_prepare(ns)
    elif ns._task == "train":
        from train_HASA_ADMM import train
        _apply_train_preset(ns)
        train(ns)
    elif ns._task == "train2d":
        from train_HASA_ADMM_2D import train as train2d
        train2d(ns)
    elif ns._task == "train_fista2d":
        from train_HASA_FISTA_2D import train as train_fista2d
        train_fista2d(ns)
    elif ns._task == "eval":
        from evaluate_HASA_ADMM import evaluate
        _apply_eval_preset(ns)
        evaluate(ns)


if __name__ == "__main__":
    main()
