"""HASA-FISTA-Net 2D 评估"""

from FISTA_Baseline_2D import HASA_FISTA_Net_2D, HASAWeightFISTA2D
from evaluate_common import evaluate_2d, build_eval_parser_2d


def load_model(args_dict, ckpt, device):
    model = HASA_FISTA_Net_2D(
        layer_num=args_dict.get("layers", 9),
        hasa_ctor=lambda: HASAWeightFISTA2D(
            hidden_ch=args_dict.get("hasa_hidden", 16),
            num_layers=args_dict.get("num_hasa_layers", 2),
            inner_ks=args_dict.get("hasa_kernel", 5),
        ),
        feat_ch=args_dict.get("fista_feat_ch", 64),
        prox_k=args_dict.get("fista_prox_k", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


if __name__ == "__main__":
    args = build_eval_parser_2d("HASA-FISTA-Net 2D 评估").parse_args()
    evaluate_2d(load_model, "HASA-FISTA-2D", args)
