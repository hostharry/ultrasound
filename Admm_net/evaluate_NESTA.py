"""Deep-Unfolded NESTA 1D 评估"""

import torch

from NESTA_Baseline import NESTA_Net
from evaluate_common import evaluate_1d, build_eval_parser_1d


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ckpt.get("args", {})
    model = NESTA_Net(
        layer_num=args_dict.get("layers", 15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta


if __name__ == "__main__":
    args = build_eval_parser_1d("Deep-Unfolded NESTA 1D 评估").parse_args()
    evaluate_1d(load_model, "NESTA", args)
