"""HASA-FISTA-Net 1D 评估"""

import torch

from FISTA_Baseline import HASA_FISTA_Net_1D, HASAWeightFISTA1D
from evaluate_common import evaluate_1d, build_eval_parser_1d


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ckpt.get("args", {})
    model = HASA_FISTA_Net_1D(
        layer_num=args_dict.get("layers", 9),
        hasa_ctor=lambda: HASAWeightFISTA1D(
            hidden_ch=args_dict.get("hasa_hidden", 16),
            num_layers=args_dict.get("num_hasa_layers", 2),
            inner_ks=args_dict.get("hasa_kernel", 5),
        ),
        feat_ch=args_dict.get("fista_feat_ch", 64),
        prox_k=args_dict.get("fista_prox_k", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta


if __name__ == "__main__":
    args = build_eval_parser_1d("HASA-FISTA-Net 1D 评估").parse_args()
    evaluate_1d(load_model, "HASA-FISTA", args)
