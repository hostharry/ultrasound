"""ISTA-Net+ 1D 评估"""

import torch

from ISTANetPlus_Baseline import ISTANetPlus
from evaluate_common import evaluate_1d, build_eval_parser_1d


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ckpt.get("args", {})
    model = ISTANetPlus(
        layer_num=args_dict.get("layers", 9),
        n_channels=args_dict.get("n_channels", 32),
        kernel_size=args_dict.get("kernel_size", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta


if __name__ == "__main__":
    args = build_eval_parser_1d("ISTA-Net+ 1D 评估").parse_args()
    evaluate_1d(load_model, "ISTA-Net+", args)
