"""HUNet-1D DepthAdaptive 评估脚本."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Utils"))

import torch

from HUNet_1D_DepthAdaptive import HUNet1DDepthAdaptive
from evaluate_common import evaluate_1d, build_eval_parser_1d


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


def _upgrade_legacy_state_dict(state_dict, gamma_floor: float):
    """Backward-compat:
    old ckpt uses stages.*.gamma
    new model uses stages.*.gamma_raw
    """
    upgraded = dict(state_dict)
    for key in list(state_dict.keys()):
        if key.endswith(".gamma"):
            raw_key = key[:-6] + ".gamma_raw"
            if raw_key not in upgraded:
                gamma = state_dict[key].abs()
                delta = (gamma - gamma_floor).clamp_min(1e-6)
                upgraded[raw_key] = _inverse_softplus(delta)
            upgraded.pop(key, None)
    return upgraded


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ckpt.get("args", {})
    gamma_floor = args_dict.get("gamma_floor", 1e-4)

    model = HUNet1DDepthAdaptive(
        num_stages=args_dict.get("num_stages", 7),
        depth=args_dict.get("depth", 4),
        embed_dim=args_dict.get("embed_dim", 32),
        nhead=args_dict.get("nhead", 4),
        window_size=args_dict.get("window_size", 32),
        mlp_ratio=args_dict.get("mlp_ratio", 2.0),
        swin_depth=args_dict.get("swin_depth", 2),
        adaptive_thr=args_dict.get("adaptive_thr", True),
        thr_hidden_ratio=args_dict.get("thr_hidden_ratio", 0.5),
        thr_scale_min=args_dict.get("thr_scale_min", 0.5),
        thr_scale_max=args_dict.get("thr_scale_max", 1.5),
        thr_init_depth_slope=args_dict.get("thr_init_depth_slope", -0.3),
        gamma_init=args_dict.get("gamma_init", 0.03),
        gamma_decay=args_dict.get("gamma_decay", 0.85),
        gamma_floor=gamma_floor,
    ).to(device)

    state_dict = _upgrade_legacy_state_dict(ckpt["model_state_dict"], gamma_floor=gamma_floor)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, meta


if __name__ == "__main__":
    args = build_eval_parser_1d("HUNet-1D DepthAdaptive 评估").parse_args()
    evaluate_1d(load_model, "HUNet-1D-DepthAdaptive", args)
