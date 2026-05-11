"""FISTA-Transformer-DWT 1D 评估"""

import sys
import os

import torch

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "Utils")
_ADMM_DIR = os.path.join(os.path.dirname(__file__), "..", "Admm_net")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
if _ADMM_DIR not in sys.path:
    sys.path.append(_ADMM_DIR)

from FISTA_Transformer_DWT import FISTA_Transformer_DWT_Net, HASAWeightTransformer1D
from evaluate_common import evaluate_1d, build_eval_parser_1d


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})

    d_model = a.get("d_model", 16)
    nhead = a.get("nhead", 4)
    num_layers = a.get("num_transformer_layers", 1)
    J = a.get("dwt_levels", 3)

    model = FISTA_Transformer_DWT_Net(
        layer_num=a.get("layers", 4),
        hasa_ctor=lambda: HASAWeightTransformer1D(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
        ),
        d_model=d_model, nhead=nhead, num_layers=num_layers, J=J,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta


if __name__ == "__main__":
    args = build_eval_parser_1d("FISTA-Transformer-DWT 1D 评估").parse_args()
    evaluate_1d(load_model, "FISTA-Trans-DWT", args)
