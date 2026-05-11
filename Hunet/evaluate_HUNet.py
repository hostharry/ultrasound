"""HUNet-1D 评估脚本"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Utils"))

import torch

from HUNet_1D import HUNet1D
from evaluate_common import evaluate_1d, build_eval_parser_1d


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ckpt.get("args", {})

    model = HUNet1D(
        num_stages=args_dict.get("num_stages", 7),
        depth=args_dict.get("depth", 4),
        embed_dim=args_dict.get("embed_dim", 32),
        nhead=args_dict.get("nhead", 4),
        window_size=args_dict.get("window_size", 32),
        mlp_ratio=args_dict.get("mlp_ratio", 2.0),
        swin_depth=args_dict.get("swin_depth", 2),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta


if __name__ == "__main__":
    args = build_eval_parser_1d("HUNet-1D 评估").parse_args()
    evaluate_1d(load_model, "HUNet-1D", args)
