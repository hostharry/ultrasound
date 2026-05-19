"""FISTA-DWT-LiteV3-2D 评估 (V2 + Pre-Norm GroupNorm)."""

import sys
import os

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "Utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from FISTA_DWT_LiteV3_2D import (
    FISTA_DWT_LiteV3_2D_Net, MultiScaleHASA2D, MiniUNetHASA2D,
)
from evaluate_common import evaluate_2d, build_eval_parser_2d


def _cfg_int(cfg, key, default):
    v = cfg.get(key, default)
    if v is None or str(v).lower() in ("none", "null"):
        return default
    return int(v)


def _cfg_float(cfg, key, default):
    v = cfg.get(key, default)
    if v is None or str(v).lower() in ("none", "null"):
        return default
    return float(v)


def load_model(config, ckpt, device):
    """config 来自实验目录 config.txt."""
    layers = _cfg_int(config, "layers", 4)
    d_model = _cfg_int(config, "d_model", 32)
    num_cb = _cfg_int(config, "num_conv_blocks", 2)
    conv_ks = _cfg_int(config, "conv_ks", 5)
    J = _cfg_int(config, "dwt_levels", 1)
    prox_tau = _cfg_float(config, "prox_tau", 0.005)
    num_groups = _cfg_int(config, "num_groups", 8)

    hasa_type = str(config.get("hasa_type", "conv")).strip()

    if hasa_type == "unet":
        base_ch = _cfg_int(config, "hasa_base_ch", 16)
        hasa_ctor = lambda: MiniUNetHASA2D(base_ch=base_ch)
    else:
        hasa_h = _cfg_int(config, "hasa_hidden", 16)
        hasa_nl = _cfg_int(config, "num_hasa_layers", 2)
        hasa_k = _cfg_int(config, "hasa_kernel", 5)
        ctx_ks = _cfg_int(config, "hasa_context_ks", 3)
        ctx_dil = _cfg_int(config, "hasa_context_dilation", 3)
        hasa_ctor = lambda: MultiScaleHASA2D(
            hidden_ch=hasa_h, num_layers=hasa_nl, inner_ks=hasa_k,
            context_ks=ctx_ks, context_dilation=ctx_dil,
        )

    model = FISTA_DWT_LiteV3_2D_Net(
        layer_num=layers,
        hasa_ctor=hasa_ctor,
        d_model=d_model,
        num_conv_blocks=num_cb,
        conv_ks=conv_ks,
        J=J,
        prox_tau=prox_tau,
        num_groups=num_groups,
    ).to(device)

    from train_common import load_model_state_dict
    load_model_state_dict(model, ckpt["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    args = build_eval_parser_2d("FISTA-DWT-LiteV3-2D 评估").parse_args()
    evaluate_2d(load_model, "FISTA-DWT-LiteV3-2D", args)
