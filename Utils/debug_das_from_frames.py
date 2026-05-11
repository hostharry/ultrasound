"""用当前评估链路复现 DAS 合成，便于核查 GT / Recon / Init 图像。

两种输入模式:
1. 直接给 `frames_npz`:
   - frames_npz: 评估阶段保存的 `eval_frames_ds*.npz`，包含 pred/gt/init
   - data_npz:   原始帧级数据 npz，提供 angles / probe_geometry / scan_x_axis 等 DAS 元数据
2. 直接给实验目录并现场推理:
   - exp_dir:      实验目录 (包含 best_model.pth / config.txt)
   - eval_script:  对应模型的 evaluate_xxx_2d.py，需暴露 `load_model`
   - data_npz:     可选；默认从 config.txt 读取

示例:
    python Utils/debug_das_from_frames.py \
      --frames_npz "model/.../eval_results_das/eval_frames_ds0.npz" \
      --data_npz "../data_epfl/some_frames.npz" \
      --settings_dir "../epfl/settings" \
      --out_dir "tmp_das_check"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

from bmode_eval import (
    compute_bmode_metrics,
    env_to_db,
    load_epfl_settings,
    plot_bmode_comparison,
    regroup_frames,
)
from das import das_pw_rf
from data import UltrasoundFrameDataset
from evaluate_common import _parse_config_txt, _resolve_npz_path


def _load_eval_module(eval_script: str):
    eval_script = os.path.abspath(eval_script)
    eval_dir = os.path.dirname(eval_script)
    utils_dir = os.path.dirname(__file__)

    for path in (eval_dir, utils_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    spec = importlib.util.spec_from_file_location("debug_eval_module", eval_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 eval_script: {eval_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "load_model"):
        raise AttributeError(f"{eval_script} 中未找到 load_model(config, ckpt, device)")
    return module


def _resolve_debug_npz_path(npz_value, exp_dir: str) -> str:
    """兼容 config.txt 中单路径或列表字符串形式的 npz 配置."""
    if isinstance(npz_value, (list, tuple)):
        if not npz_value:
            raise ValueError("config.txt 中的 npz 列表为空")
        npz_raw = npz_value[0]
    else:
        npz_raw = npz_value
        if isinstance(npz_raw, str):
            raw = npz_raw.strip()
            if raw.startswith("[") and raw.endswith("]"):
                items = raw.strip("[]'\" ").split(",")
                items = [s.strip().strip("'\"") for s in items if s.strip()]
                if not items:
                    raise ValueError("config.txt 中的 npz 列表为空")
                npz_raw = items[0]
    return _resolve_npz_path(npz_raw, exp_dir)


def _reconstruct_frames_from_exp(exp_dir: str, eval_script: str, data_npz: str | None,
                                 ckpt_name: str, gpu: int):
    eval_module = _load_eval_module(eval_script)
    config = _parse_config_txt(exp_dir)
    npz_path = data_npz or config.get("npz")
    if npz_path is None:
        raise ValueError("未提供 --data_npz，且 config.txt 中缺少 npz")
    npz_path = _resolve_debug_npz_path(npz_path, exp_dir)

    cs_ratio = config.get("cs_ratio", 8)
    if isinstance(cs_ratio, str):
        cs_ratio = int(cs_ratio)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(exp_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = eval_module.load_model(config, ckpt, device)
    model.eval()

    ds = UltrasoundFrameDataset(npz_path, cs_ratio=cs_ratio, patch_h=None, patch_stride=None, device="cpu")
    ds.to(device)

    preds, gts, inits = [], [], []
    for fi in range(len(ds)):
        idx_t = torch.tensor([fi])
        x_input, y_target, y_k = ds.get_batch(idx_t, device=device)
        y_sub = y_k if y_k is not None else ds.op.A(x_input)
        with torch.no_grad():
            x_hat = model(y_sub, ds.op)
            x_init = ds.op.At(y_sub)
        preds.append(x_hat[0, 0].cpu().numpy())
        gts.append(y_target[0, 0].cpu().numpy())
        inits.append(x_init[0, 0].cpu().numpy())

    return npz_path, np.stack(preds), np.stack(gts), np.stack(inits)


def _load_das_meta(data_npz: str, settings_dir: str | None):
    data = np.load(data_npz, allow_pickle=True)

    if "scan_x_axis" in data and "angles" in data:
        angles = np.asarray(data["angles"], dtype=np.float64)
        sel = data["selected_angles"] if "selected_angles" in data else None
        if sel is not None:
            sel = np.asarray(sel, dtype=int)
            if sel.size == 0:
                sel = None
        probe_geom = np.asarray(data["probe_geometry"], dtype=np.float64)
        t0 = float(data["initial_time"])
        fs_val = float(data["fs"])
        c_val = float(data["c"])
        x_axis = np.asarray(data["scan_x_axis"], dtype=np.float64)
        z_axis = np.asarray(data["scan_z_axis"], dtype=np.float64)
        ang = angles[sel] if sel is not None else angles
    elif settings_dir and os.path.isdir(settings_dir):
        epfl = load_epfl_settings(settings_dir)
        probe_geom = epfl["probe_geometry"]
        t0 = epfl["initial_time"]
        fs_val = epfl["fs"]
        c_val = epfl["c"]
        x_axis = epfl["x_axis"]
        z_axis = epfl["z_axis"]
        ang = epfl["angles"]
    else:
        raise ValueError("data_npz 中缺少 scan 网格，且未提供有效的 settings_dir")

    return {
        "data": data,
        "ang": ang,
        "probe_geom": probe_geom,
        "t0": t0,
        "fs_val": fs_val,
        "c_val": c_val,
        "x_axis": x_axis,
        "z_axis": z_axis,
    }


def _beamform_triplet(gt_cube, pred_cube, init_cube, meta, dynamic_range: float):
    das_gt = das_pw_rf(
        gt_cube.astype(np.float64), meta["ang"], meta["probe_geom"],
        meta["t0"], meta["fs_val"], meta["c_val"], meta["x_axis"], meta["z_axis"],
        verbose=False,
    )
    das_pred = das_pw_rf(
        pred_cube.astype(np.float64), meta["ang"], meta["probe_geom"],
        meta["t0"], meta["fs_val"], meta["c_val"], meta["x_axis"], meta["z_axis"],
        verbose=False,
    )
    das_init = das_pw_rf(
        init_cube.astype(np.float64), meta["ang"], meta["probe_geom"],
        meta["t0"], meta["fs_val"], meta["c_val"], meta["x_axis"], meta["z_axis"],
        verbose=False,
    )

    db_gt = env_to_db(das_gt, dynamic_range)
    db_pred = env_to_db(das_pred, dynamic_range)
    db_init = env_to_db(das_init, dynamic_range)

    x_mm = meta["x_axis"] * 1e3
    z_mm = meta["z_axis"] * 1e3
    metrics = compute_bmode_metrics(
        gt_env=das_gt,
        pred_env=das_pred,
        x_axis_mm=x_mm,
        z_axis_mm=z_mm,
        dynamic_range=dynamic_range,
    )
    return das_gt, das_pred, das_init, db_gt, db_pred, db_init, metrics


def _save_outputs(save_prefix: str, x_axis, z_axis, das_gt, das_pred, das_init,
                  db_gt, db_pred, db_init, dynamic_range: float, metrics: dict):
    x_mm = x_axis * 1e3
    z_mm = z_axis * 1e3

    plot_bmode_comparison(
        db_gt, db_pred, db_init, x_mm, z_mm,
        save_prefix + "_comparison.png",
        dynamic_range=dynamic_range,
        model_name="Recon",
        metrics=metrics,
    )

    np.savez_compressed(
        save_prefix + "_das.npz",
        das_gt=das_gt,
        das_pred=das_pred,
        das_init=das_init,
        db_gt=db_gt,
        db_pred=db_pred,
        db_init=db_init,
        x_axis=x_axis,
        z_axis=z_axis,
    )


def main():
    parser = argparse.ArgumentParser(description="复现当前评估链路的 DAS 合成")
    parser.add_argument("--frames_npz", type=str, default=None,
                        help="评估保存的 eval_frames_ds*.npz")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="实验目录；不提供 frames_npz 时使用")
    parser.add_argument("--eval_script", type=str, default=None,
                        help="对应模型的 evaluate_xxx_2d.py；不提供 frames_npz 时使用")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pth")
    parser.add_argument("--data_npz", type=str, default=None,
                        help="原始帧级数据 npz，需包含 DAS 元数据或配合 settings_dir")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--settings_dir", type=str, default=None,
                        help="当 data_npz 缺少 scan 网格时，从这里补 DAS 参数")
    parser.add_argument("--dynamic_range", type=float, default=60.0)
    parser.add_argument("--acquisition", type=int, default=None,
                        help="多 acquisition 数据时只重建指定 acquisition")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.frames_npz:
        frames = np.load(args.frames_npz, allow_pickle=True)
        pred_frames = np.asarray(frames["pred"])
        gt_frames = np.asarray(frames["gt"])
        init_frames = np.asarray(frames["init"])
        source_npz = args.data_npz
        if source_npz is None:
            raise ValueError("使用 --frames_npz 时，必须同时提供 --data_npz")
        print(f"frames_npz: {args.frames_npz}")
    else:
        if not args.exp_dir or not args.eval_script:
            raise ValueError("请提供 --frames_npz，或提供 --exp_dir + --eval_script")
        source_npz, pred_frames, gt_frames, init_frames = _reconstruct_frames_from_exp(
            args.exp_dir, args.eval_script, args.data_npz, args.ckpt_name, args.gpu,
        )
        np.savez_compressed(
            os.path.join(args.out_dir, "eval_frames_debug.npz"),
            pred=pred_frames, gt=gt_frames, init=init_frames,
        )
        print(f"reconstructed frames saved: {os.path.join(args.out_dir, 'eval_frames_debug.npz')}")

    meta = _load_das_meta(source_npz, args.settings_dir)
    data = meta["data"]

    print(f"data_npz:   {source_npz}")
    print(f"pred/gt/init shape: {pred_frames.shape}")
    print(f"angles used for DAS: {len(meta['ang'])}")

    has_multi_acq = ("acquisition_id" in data and "frame_angle_idx" in data)
    if has_multi_acq:
        acq_id = np.asarray(data["acquisition_id"])
        angle_idx = np.asarray(data["frame_angle_idx"])
        n_angles = int(data["n_angles_per_acq"]) if "n_angles_per_acq" in data else len(meta["ang"])

        acq_gt = regroup_frames(gt_frames, acq_id, angle_idx, n_angles)
        acq_pred = regroup_frames(pred_frames, acq_id, angle_idx, n_angles)
        acq_init = regroup_frames(init_frames, acq_id, angle_idx, n_angles)

        acq_list = sorted(acq_gt.keys())
        if args.acquisition is not None:
            acq_list = [args.acquisition]

        for aid in acq_list:
            if aid not in acq_pred or aid not in acq_init:
                print(f"[跳过] acquisition {aid} 不完整")
                continue
            print(f"\n=== Acquisition {aid} ===")
            das_gt, das_pred, das_init, db_gt, db_pred, db_init, metrics = _beamform_triplet(
                acq_gt[aid], acq_pred[aid], acq_init[aid], meta, args.dynamic_range,
            )
            prefix = os.path.join(args.out_dir, f"acq{aid}")
            _save_outputs(prefix, meta["x_axis"], meta["z_axis"],
                          das_gt, das_pred, das_init,
                          db_gt, db_pred, db_init,
                          args.dynamic_range, metrics)
            print(f"saved: {prefix}_comparison.png")
            print(f"PSNR={metrics['bmode_PSNR_dB']:.2f} dB  SSIM={metrics['bmode_SSIM']:.4f}")
    else:
        das_gt, das_pred, das_init, db_gt, db_pred, db_init, metrics = _beamform_triplet(
            gt_frames, pred_frames, init_frames, meta, args.dynamic_range,
        )
        prefix = os.path.join(args.out_dir, "full")
        _save_outputs(prefix, meta["x_axis"], meta["z_axis"],
                      das_gt, das_pred, das_init,
                      db_gt, db_pred, db_init,
                      args.dynamic_range, metrics)
        print(f"saved: {prefix}_comparison.png")
        print(f"PSNR={metrics['bmode_PSNR_dB']:.2f} dB  SSIM={metrics['bmode_SSIM']:.4f}")


if __name__ == "__main__":
    main()
