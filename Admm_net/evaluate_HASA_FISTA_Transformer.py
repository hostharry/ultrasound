"""HASA-FISTA-Transformer-Net 1D 评估"""

import torch

from FISTA_Transformer import HASA_FISTA_Transformer_Net_1D, HASAWeightTransformer1D
from evaluate_common import evaluate_1d, build_eval_parser_1d


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ckpt.get("args", {})
    model = HASA_FISTA_Transformer_Net_1D(
        layer_num=args_dict.get("layers", 9),
        hasa_ctor=lambda: HASAWeightTransformer1D(
            d_model=args_dict.get("d_model", 32),
            nhead=args_dict.get("nhead", 4),
            num_layers=args_dict.get("num_transformer_layers", 2),
        ),
        d_model=args_dict.get("d_model", 32),
        nhead=args_dict.get("nhead", 4),
        num_layers=args_dict.get("num_transformer_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta


if __name__ == "__main__":
    args = build_eval_parser_1d("HASA-FISTA-Transformer 1D 评估").parse_args()
    evaluate_1d(load_model, "HASA-FISTA-Transformer", args)
