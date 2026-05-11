#!/usr/bin/env python3
"""
npz_info.py
Inspect .npz files and print dataset metadata and optional stats.

Usage:
  python npz_info.py /path/to/file.npz
  python npz_info.py /path/to/file.npz --stats --preview 8
  python npz_info.py /path/to/file.npz --allow-pickle
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, Tuple

import numpy as np


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"


def is_numeric(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number)


def get_basic_info(arr: np.ndarray) -> Dict[str, Any]:
    return {
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "size": int(arr.size),
        "nbytes": int(arr.nbytes),
    }


def compute_stats(arr: np.ndarray) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if np.iscomplexobj(arr):
        mag = np.abs(arr)
        stats["complex"] = True
        stats["min_abs"] = float(np.nanmin(mag))
        stats["max_abs"] = float(np.nanmax(mag))
        stats["mean_abs"] = float(np.nanmean(mag))
        stats["std_abs"] = float(np.nanstd(mag))
        stats["nan_count"] = int(np.isnan(mag).sum())
        return stats

    if arr.dtype == np.bool_:
        stats["true_count"] = int(np.count_nonzero(arr))
        stats["false_count"] = int(arr.size - stats["true_count"])
        return stats

    if is_numeric(arr):
        stats["min"] = float(np.nanmin(arr))
        stats["max"] = float(np.nanmax(arr))
        stats["mean"] = float(np.nanmean(arr))
        stats["std"] = float(np.nanstd(arr))
        stats["nan_count"] = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
        return stats

    return stats


def preview_array(arr: np.ndarray, n: int) -> str:
    flat = arr.ravel()
    n = min(n, flat.size)
    if n == 0:
        return "[]"
    preview = flat[:n]
    return np.array2string(preview, threshold=n, max_line_width=120)


def print_info(
    key: str,
    arr: np.ndarray,
    stats: bool,
    preview: int,
    max_stats_elems: int,
) -> None:
    info = get_basic_info(arr)
    print(f"- {key}: dtype={info['dtype']} shape={info['shape']} size={info['size']} bytes={format_bytes(info['nbytes'])}")

    if stats:
        if arr.size <= max_stats_elems:
            s = compute_stats(arr)
            if s:
                for k, v in s.items():
                    print(f"  {k}: {v}")
        else:
            print(f"  stats: skipped (size {arr.size} > max_stats_elems {max_stats_elems})")

    if preview > 0:
        print(f"  preview: {preview_array(arr, preview)}")


def parse_indices(spec: str, max_len: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(max_len))
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_s, end_s = part.split(":", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else max_len
            indices.extend(range(start, end))
        else:
            indices.append(int(part))
    # keep order, remove out-of-range
    return [i for i in indices if 0 <= i < max_len]


def save_signal(path: str, signal: np.ndarray, fmt: str) -> None:
    if fmt == "npy":
        np.save(path, signal)
    elif fmt == "csv":
        np.savetxt(path, signal, delimiter=",")
    elif fmt == "txt":
        np.savetxt(path, signal)
    else:
        raise ValueError(f"Unsupported signal format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect .npz file contents")
    parser.add_argument("path", type=str, help="Path to .npz file")
    parser.add_argument("--allow-pickle", action="store_true", help="Allow pickle when loading")
    parser.add_argument("--stats", action="store_true", help="Compute stats for numeric arrays")
    parser.add_argument("--preview", type=int, default=0, help="Preview first N elements")
    parser.add_argument("--max-stats-elems", type=int, default=5_000_000, help="Max elements to compute stats")
    parser.add_argument("--signal-key", type=str, default="", help="Key to read signals from (e.g., X, Y, X9)")
    parser.add_argument("--signal-indices", type=str, default="all",
                        help="Indices to export: all | 0:10 | 0,5,7 | :100")
    parser.add_argument("--signal-out-dir", type=str, default="",
                        help="Directory to save each signal (one file per index)")
    parser.add_argument("--signal-format", type=str, default="npy", choices=["npy", "csv", "txt"],
                        help="Output format when saving signals")
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise SystemExit(f"File not found: {args.path}")

    data = np.load(args.path, allow_pickle=args.allow_pickle)
    keys = sorted(list(data.keys()))
    print(f"NPZ: {args.path}")
    print(f"Keys: {len(keys)}")

    for key in keys:
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print_info(
                key=key,
                arr=arr,
                stats=args.stats,
                preview=args.preview,
                max_stats_elems=args.max_stats_elems,
            )
        else:
            print(f"- {key}: type={type(arr)}")

    if args.signal_key:
        if args.signal_key not in data:
            raise SystemExit(f"signal_key not found: {args.signal_key}")
        sig_arr = data[args.signal_key]
        if not isinstance(sig_arr, np.ndarray):
            raise SystemExit("signal_key is not a numpy array")
        if sig_arr.ndim == 1:
            sig_arr = sig_arr.reshape(1, -1)
        if sig_arr.ndim < 2:
            raise SystemExit("signal_key must be 1D or 2D (samples x length)")

        total = sig_arr.shape[0]
        indices = parse_indices(args.signal_indices, total)
        print(f"Signals: key={args.signal_key}, total={total}, selected={len(indices)}")

        if args.signal_out_dir:
            os.makedirs(args.signal_out_dir, exist_ok=True)
            for idx in indices:
                signal = sig_arr[idx]
                out_name = f"{args.signal_key}_{idx:06d}.{args.signal_format}"
                out_path = os.path.join(args.signal_out_dir, out_name)
                save_signal(out_path, signal, args.signal_format)
            print(f"Saved {len(indices)} signals to {args.signal_out_dir}")
        else:
            for idx in indices:
                signal = sig_arr[idx]
                print(f"[{idx}] len={signal.shape[-1]} data={preview_array(signal, max(args.preview, 10))}")


if __name__ == "__main__":
    main()
