"""HASA-FISTA-Transformer-Net 2D (Swin) 评估"""

from FISTA_Transformer import HASA_FISTA_Transformer_Net_2D, HASAWeightTransformer2D
from evaluate_common import evaluate_2d, build_eval_parser_2d


def load_model(args_dict, ckpt, device):
    model = HASA_FISTA_Transformer_Net_2D(
        layer_num=args_dict.get("layers", 9),
        hasa_ctor=lambda: HASAWeightTransformer2D(
            d_model=args_dict.get("d_model", 64),
            nhead=args_dict.get("nhead", 4),
            num_layers=args_dict.get("num_transformer_layers", 2),
            patch_size=args_dict.get("patch_size", 2),
            win_size=args_dict.get("window_size", 8),
        ),
        d_model=args_dict.get("d_model", 64),
        nhead=args_dict.get("nhead", 4),
        num_layers=args_dict.get("num_transformer_layers", 2),
        patch_size=args_dict.get("patch_size", 2),
        win_size=args_dict.get("window_size", 8),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


if __name__ == "__main__":
    args = build_eval_parser_2d("HASA-FISTA-Transformer 2D 评估").parse_args()
    evaluate_2d(load_model, "HASA-FISTA-Transformer-2D", args)
