"""HASA-ADMM-Net 1D 评估"""

import torch

from HASA_ADMM_Net import HASA_ADMM_Net, HASAWeight1D
from admm_visualization import plot_layer_convergence
from evaluate_common import evaluate_1d, build_eval_parser_1d


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ckpt.get("args", {})
    model = HASA_ADMM_Net(
        layer_num=args_dict.get("layers", 9),
        hasa_ctor=lambda: HASAWeight1D(hidden_ch=args_dict.get("hasa_hidden", 16)),
        W_mode=args_dict.get("W_mode", "A"),
        W_num_filters=args_dict.get("W_filters", 2),
        W_kernel_size=args_dict.get("W_kernel", 8),
        share_W=args_dict.get("share_W", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta


def _extra_viz(model, dataset, eval_idx, device, out_dir):
    plot_layer_convergence(
        model=model, dataset=dataset,
        sample_idx=int(eval_idx[0].item()), device=device, save_dir=out_dir,
    )


if __name__ == "__main__":
    args = build_eval_parser_1d("HASA-ADMM-Net 1D 评估").parse_args()
    evaluate_1d(load_model, "HASA-ADMM", args, extra_viz_fn=_extra_viz)
