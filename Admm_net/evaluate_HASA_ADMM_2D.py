"""HASA-ADMM-Net 2D 评估"""

from HASA_ADMM_Net_2D import HASA_ADMM_Net_2D, HASAWeight2D
from evaluate_common import evaluate_2d, build_eval_parser_2d


def load_model(args_dict, ckpt, device):
    model = HASA_ADMM_Net_2D(
        layer_num=args_dict.get("layers", 9),
        hasa_ctor=lambda: HASAWeight2D(
            hidden_ch=args_dict.get("hasa_hidden", 16),
            num_layers=args_dict.get("num_hasa_layers", 2),
            inner_ks=args_dict.get("hasa_kernel", 5),
        ),
        W_mode="A",
        share_W=args_dict.get("share_W", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


if __name__ == "__main__":
    args = build_eval_parser_2d("HASA-ADMM-Net 2D 评估").parse_args()
    evaluate_2d(load_model, "HASA-ADMM-2D", args)
