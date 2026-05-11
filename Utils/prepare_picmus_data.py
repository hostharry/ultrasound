"""
PICMUS HDF5 → 训练用 npz 预处理脚本
=====================================

将 PICMUS Challenge 的 HDF5 RF 数据转换为 HASA-ADMM-Net 训练所需的 npz 格式。

PICMUS RF data 结构:
    data/real : (n_angles, n_elements, n_time_samples)   float32
    sampling_frequency, modulation_frequency, sound_speed, ...

转换逻辑:
    1. 将 (n_angles × n_elements) 展开为独立 1D RF 信号
    2. 对每条信号在 rfft 频域做随机下采样, 模拟压缩感知
    3. 生成 mask, Y_k, X (zero-filled IFFT) 等字段
    4. 保存为与原有 dataset_fdbf_energy_mu_*.npz 完全兼容的格式

用法:
    python prepare_picmus_data.py                          # 使用默认参数
    python prepare_picmus_data.py --datasets simu_reso     # 仅处理仿真分辨率
    python prepare_picmus_data.py --max_samples 2000 --cs_ratios 4 8 15
"""

import os
import argparse
import numpy as np
import h5py
from typing import List, Tuple, Optional


# ======================== PICMUS 路径映射 ========================

PICMUS_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "archive_to_download"
)

DATASET_MAP = {
    "simu_reso": "database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
    "simu_cont": "database/simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
    "expe_reso": "database/experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
    "expe_cont": "database/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
}

SCAN_MAP = {
    "simu_reso": "database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5",
    "simu_cont": "database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5",
    "expe_reso": "database/experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5",
    "expe_cont": "database/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5",
}


# ======================== 读取 HDF5 ========================

def load_picmus_rf(hdf5_path: str) -> dict:
    """读取 PICMUS RF HDF5 文件, 返回 metadata + RF data"""
    with h5py.File(hdf5_path, "r") as f:
        ds = f["US/US_DATASET0000"]
        rf_real = ds["data/real"][:]           # (n_angles, n_elements, n_samples)
        fs = float(ds["sampling_frequency"][0])
        fc = float(ds["modulation_frequency"][0])
        c  = float(ds["sound_speed"][0])
        angles = ds["angles"][:]
        probe_geom = ds["probe_geometry"][:]
        initial_time = float(ds["initial_time"][0])

    n_angles, n_elements, n_samples = rf_real.shape
    print(f"  RF shape: ({n_angles}, {n_elements}, {n_samples})")
    print(f"  fs={fs/1e6:.2f} MHz, fc={fc/1e6:.2f} MHz, c={c:.0f} m/s")
    print(f"  angles: {n_angles} in [{np.rad2deg(angles.min()):.1f}°, {np.rad2deg(angles.max()):.1f}°]")

    return {
        "rf_real": rf_real,
        "fs": fs,
        "fc": fc,
        "c": c,
        "angles": angles,
        "probe_geometry": probe_geom,
        "initial_time": initial_time,
        "n_angles": n_angles,
        "n_elements": n_elements,
        "n_samples": n_samples,
    }


def load_picmus_scan(scan_hdf5_path: str) -> dict:
    """读取 PICMUS scan HDF5, 返回 DAS 成像所需的 x_axis / z_axis."""
    with h5py.File(scan_hdf5_path, "r") as f:
        scan = f["US/US_DATASET0000"]
        x_axis = scan["x_axis"][:].flatten().astype(np.float64)
        z_axis = scan["z_axis"][:].flatten().astype(np.float64)
    print(f"  scan: x_axis {len(x_axis)} pts [{x_axis.min()*1e3:.2f}, {x_axis.max()*1e3:.2f}] mm, "
          f"z_axis {len(z_axis)} pts [{z_axis.min()*1e3:.2f}, {z_axis.max()*1e3:.2f}] mm")
    return {"x_axis": x_axis, "z_axis": z_axis}


# ======================== 信号采集策略 ========================

def extract_signals(
    rf_data: dict,
    max_samples: Optional[int] = None,
    angle_stride: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 PICMUS RF 数据中提取 1D 信号.

    策略: 对选定角度, 取每个阵元的时域信号作为一个独立样本.
    返回: (N_samples, signal_length), float32
    """
    rf = rf_data["rf_real"]  # (n_angles, n_elements, n_time)

    selected_angles = np.arange(0, rf.shape[0], angle_stride)
    signals = rf[selected_angles].reshape(-1, rf.shape[2])  # (n_sel*n_elem, n_time)
    group_id = np.repeat(selected_angles, rf.shape[1])      # 按角度分组

    if max_samples is not None and max_samples > 0 and signals.shape[0] > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(signals.shape[0], max_samples, replace=False)
        idx.sort()
        signals = signals[idx]
        group_id = group_id[idx]

    signals = signals.astype(np.float32)

    sig_max = np.abs(signals).max()
    if sig_max > 0:
        signals /= sig_max

    print(f"  提取信号: {signals.shape[0]} 条, 长度 {signals.shape[1]}")
    return signals, group_id.astype(np.int32)


# ======================== 频域压缩感知 ========================

def create_cs_mask(
    N: int, cs_ratio: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 rfft 频域 (长度 N//2+1) 上创建随机下采样 mask.
    保留 DC 分量并随机选取 K = (N//2+1) // cs_ratio 个频率分量.

    返回:
        mask: (N//2+1,) uint8 二值掩码
        mu:   选中的频率索引 (sorted)
    """
    n_freq = N // 2 + 1
    K = max(2, n_freq // cs_ratio)

    rng = np.random.RandomState(seed + cs_ratio)
    candidates = np.arange(1, n_freq)
    chosen = rng.choice(candidates, K - 1, replace=False)
    chosen = np.sort(np.concatenate([[0], chosen]))

    mask = np.zeros(n_freq, dtype=np.uint8)
    mask[chosen] = 1
    return mask, chosen.astype(np.int32)


def apply_cs(
    Y: np.ndarray, mask: np.ndarray, mu: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对信号批量做频域压缩采样.

    Args:
        Y: (B, N) 时域信号
        mask: (N//2+1,) 二值掩码
        mu: 选中的频率索引

    Returns:
        X: (B, N) zero-filled IFFT 重建 (降质信号)
        Y_k: (B, K) 选中频率的复数观测
    """
    Y_freq = np.fft.rfft(Y, axis=-1)      # (B, N//2+1)
    Y_k = Y_freq[:, mu]                    # (B, K)

    Y_masked = Y_freq * mask[np.newaxis, :]
    X = np.fft.irfft(Y_masked, n=Y.shape[-1], axis=-1).astype(np.float32)

    return X, Y_k.astype(np.complex64)


# ======================== 帧级处理 (2D) ========================

def process_dataset_frames(
    dataset_key: str,
    picmus_root: str,
    cs_ratios: List[int],
    angle_stride: int,
    output_dir: str,
    seed: int = 42,
):
    """以帧为单位处理 PICMUS 数据 (用于 2D ADMM 网络)

    每个帧 = 固定角度下所有阵元的 RF 信号, shape (n_elements, n_samples).
    保留阵元间的空间关系以利用横向相关性.
    """
    hdf5_rel = DATASET_MAP[dataset_key]
    hdf5_path = os.path.join(picmus_root, hdf5_rel)

    if not os.path.isfile(hdf5_path):
        print(f"[跳过] 文件不存在: {hdf5_path}")
        return None

    print(f"\n{'='*60}")
    print(f"帧级处理: {dataset_key}")
    print(f"  文件: {hdf5_path}")

    rf_data = load_picmus_rf(hdf5_path)
    rf = rf_data["rf_real"]  # (n_angles, n_elements, n_samples)

    selected_angles = np.arange(0, rf.shape[0], angle_stride)
    frames = rf[selected_angles].astype(np.float32)  # (n_sel, H, W)

    sig_max = np.abs(frames).max()
    if sig_max > 0:
        frames /= sig_max

    n_frames, H, W = frames.shape
    print(f"  帧数: {n_frames}, 阵元: {H}, 信号长度: {W}")

    scan_rel = SCAN_MAP.get(dataset_key)
    scan_data = {}
    if scan_rel:
        scan_path = os.path.join(picmus_root, scan_rel)
        if os.path.isfile(scan_path):
            scan_data = load_picmus_scan(scan_path)

    save_dict = {
        "Y_frames": frames,
        "fs": np.float32(rf_data["fs"]),
        "fc": np.float32(rf_data["fc"]),
        "c": np.float32(rf_data["c"]),
        "n_elements": np.int32(H),
        "n_frames": np.int32(n_frames),
        "selected_angles": selected_angles.astype(np.int32),
        "group_id": np.arange(n_frames, dtype=np.int32),
        "angles": rf_data["angles"].astype(np.float32),
        "probe_geometry": rf_data["probe_geometry"].astype(np.float32),
        "initial_time": np.float32(rf_data["initial_time"]),
    }
    if "x_axis" in scan_data:
        save_dict["scan_x_axis"] = scan_data["x_axis"]
        save_dict["scan_z_axis"] = scan_data["z_axis"]

    for ratio in cs_ratios:
        mask, mu = create_cs_mask(W, ratio, seed=seed)

        Y_freq = np.fft.rfft(frames, axis=-1)          # (n_frames, H, n_freq)
        Y_k = Y_freq[:, :, mu]                          # (n_frames, H, K)
        Y_masked = Y_freq * mask[np.newaxis, np.newaxis, :]
        X = np.fft.irfft(Y_masked, n=W, axis=-1).astype(np.float32)

        save_dict[f"X{ratio}_frames"] = X
        save_dict[f"Y{ratio}_k_frames"] = Y_k.astype(np.complex64)
        save_dict[f"mask{ratio}"] = mask
        save_dict[f"mu{ratio}"] = mu

        snr_per = []
        for fi in range(n_frames):
            sp = np.sum(frames[fi] ** 2)
            np_ = np.sum((frames[fi] - X[fi]) ** 2)
            snr_per.append(10 * np.log10(sp / (np_ + 1e-10)))
        snr_arr = np.array(snr_per)
        print(f"  cs_ratio={ratio}: K={len(mu)}/{W//2+1}, "
              f"init SNR = {np.mean(snr_arr):.2f} ± {np.std(snr_arr):.2f} dB")

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"picmus_{dataset_key}_frames.npz"
    out_path = os.path.join(output_dir, out_name)
    np.savez_compressed(out_path, **save_dict)
    print(f"  保存至: {out_path}")
    return out_path


def merge_frame_datasets(
    npz_paths: List[str],
    cs_ratios: List[int],
    output_dir: str,
):
    """将多个帧级 npz 按信号长度分组合并

    不同 PICMUS 数据集可能有不同信号长度 (e.g. 1527, 1891, 3328),
    相同长度的才能合并 (共享频域 mask).
    """
    by_length: dict = {}
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        W = int(d["Y_frames"].shape[-1])
        by_length.setdefault(W, []).append(p)

    for W, paths in by_length.items():
        if len(paths) < 2:
            print(f"\n信号长度 {W}: 仅 1 个文件, 跳过合并")
            continue

        all_Y, all_group = [], []
        all_cs = {r: {"X": [], "Yk": []} for r in cs_ratios}
        ref_mask = {}
        group_offset = 0

        for p in paths:
            d = np.load(p, allow_pickle=True)
            frames = d["Y_frames"]
            all_Y.append(frames)
            gid = d["group_id"] if "group_id" in d else np.arange(frames.shape[0], dtype=np.int32)
            all_group.append(gid + group_offset)
            group_offset += int(np.max(gid)) + 1

            for r in cs_ratios:
                xk, yk = f"X{r}_frames", f"Y{r}_k_frames"
                if xk in d and yk in d:
                    all_cs[r]["X"].append(d[xk])
                    all_cs[r]["Yk"].append(d[yk])
                    if r not in ref_mask:
                        ref_mask[r] = d[f"mask{r}"]

        Y_merged = np.concatenate(all_Y, axis=0)

        das_keys = ["rf_data_3d", "angles", "probe_geometry", "initial_time",
                     "scan_x_axis", "scan_z_axis", "selected_angles"]
        das_info = {}
        for p2 in paths:
            d2 = np.load(p2, allow_pickle=True)
            for k in das_keys:
                if k in d2 and k not in das_info:
                    das_info[k] = d2[k]

        save_dict = {
            "Y_frames": Y_merged,
            "fs": np.float32(d["fs"]),
            "fc": np.float32(d["fc"]),
            "c": np.float32(d["c"]),
            "n_elements": np.int32(Y_merged.shape[1]),
            "n_frames": np.int32(Y_merged.shape[0]),
            "group_id": np.concatenate(all_group, axis=0),
            **das_info,
        }

        for r in cs_ratios:
            if all_cs[r]["X"]:
                save_dict[f"X{r}_frames"] = np.concatenate(all_cs[r]["X"], axis=0)
                save_dict[f"Y{r}_k_frames"] = np.concatenate(all_cs[r]["Yk"], axis=0)
                save_dict[f"mask{r}"] = ref_mask[r]
                save_dict[f"mu{r}"] = np.nonzero(ref_mask[r])[0].astype(np.int32)

        out_path = os.path.join(output_dir, f"picmus_merged_W{W}_frames.npz")
        np.savez_compressed(out_path, **save_dict)
        src_names = [os.path.basename(p) for p in paths]
        print(f"\n合并 (W={W}): {' + '.join(src_names)}")
        print(f"  → {out_path}")
        print(f"  总帧数: {Y_merged.shape[0]}, 阵元: {Y_merged.shape[1]}")


# ======================== 逐线处理 (1D, 原有) ========================

def process_dataset(
    dataset_key: str,
    picmus_root: str,
    cs_ratios: List[int],
    max_samples: Optional[int],
    angle_stride: int,
    output_dir: str,
    seed: int = 42,
):
    """处理单个 PICMUS 数据集并保存 npz"""
    hdf5_rel = DATASET_MAP[dataset_key]
    hdf5_path = os.path.join(picmus_root, hdf5_rel)

    if not os.path.isfile(hdf5_path):
        print(f"[跳过] 文件不存在: {hdf5_path}")
        return None

    print(f"\n{'='*60}")
    print(f"处理: {dataset_key}")
    print(f"  文件: {hdf5_path}")

    rf_data = load_picmus_rf(hdf5_path)
    Y, group_id = extract_signals(rf_data, max_samples=max_samples,
                                  angle_stride=angle_stride, seed=seed)

    scan_rel = SCAN_MAP.get(dataset_key)
    scan_data = {}
    if scan_rel:
        scan_path = os.path.join(picmus_root, scan_rel)
        if os.path.isfile(scan_path):
            scan_data = load_picmus_scan(scan_path)

    selected_angles = np.arange(0, rf_data["n_angles"], angle_stride)

    rf_3d = rf_data["rf_real"].astype(np.float32)
    rf_3d_max = np.abs(rf_3d).max()
    if rf_3d_max > 0:
        rf_3d = rf_3d / rf_3d_max

    N = Y.shape[1]
    save_dict = {
        "Y": Y,
        "fs": np.float32(rf_data["fs"]),
        "fc": np.float32(rf_data["fc"]),
        "c": np.float32(rf_data["c"]),
        "N1": np.int32(rf_data["n_elements"]),
        "N2": np.int32(rf_data["n_angles"]),
        "theta": np.float32(0.0),
        "group_id": group_id,
        "source_id": np.full((Y.shape[0],), fill_value=list(DATASET_MAP.keys()).index(dataset_key), dtype=np.int32),
        # DAS 重建所需的完整信息
        "rf_data_3d": rf_3d,           # (n_angles, n_elements, n_samples)
        "angles": rf_data["angles"].astype(np.float32),
        "probe_geometry": rf_data["probe_geometry"].astype(np.float32),  # (3, n_elements)
        "initial_time": np.float32(rf_data["initial_time"]),
        "selected_angles": selected_angles.astype(np.int32),
    }
    if "x_axis" in scan_data:
        save_dict["scan_x_axis"] = scan_data["x_axis"]
        save_dict["scan_z_axis"] = scan_data["z_axis"]

    for ratio in cs_ratios:
        mask, mu = create_cs_mask(N, ratio, seed=seed)
        X, Y_k = apply_cs(Y, mask, mu)

        save_dict[f"X{ratio}"] = X
        save_dict[f"mu{ratio}"] = mu
        save_dict[f"Y{ratio}_k"] = Y_k
        save_dict[f"mask{ratio}"] = mask

        snr_init = 10 * np.log10(
            np.sum(Y ** 2, axis=-1) / (np.sum((Y - X) ** 2, axis=-1) + 1e-10)
        )
        print(f"  cs_ratio={ratio}: K={len(mu)}/{N//2+1}, "
              f"init SNR = {np.mean(snr_init):.2f} ± {np.std(snr_init):.2f} dB")

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"picmus_{dataset_key}.npz"
    out_path = os.path.join(output_dir, out_name)
    np.savez_compressed(out_path, **save_dict)
    print(f"  保存至: {out_path}")
    print(f"  Y: {Y.shape}, 每条长度: {N}")
    return out_path


def merge_datasets(
    npz_paths: List[str],
    cs_ratios: List[int],
    output_path: str,
):
    """将多个 npz 合并为一个统一训练集"""
    all_Y = []
    all_cs = {r: {"X": [], "Y_k": []} for r in cs_ratios}
    all_group = []
    all_source = []
    ref_fs, ref_fc, ref_c = None, None, None
    ref_mask = {}

    group_offset = 0
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        all_Y.append(d["Y"])
        group_cur = d["group_id"] if "group_id" in d else np.arange(len(d["Y"]), dtype=np.int32)
        src_cur = d["source_id"] if "source_id" in d else np.zeros((len(d["Y"]),), dtype=np.int32)
        all_group.append(group_cur + group_offset)
        all_source.append(src_cur)
        group_offset += int(np.max(group_cur)) + 1
        if ref_fs is None:
            ref_fs = float(d["fs"])
            ref_fc = float(d["fc"])
            ref_c  = float(d["c"])
        for r in cs_ratios:
            all_cs[r]["X"].append(d[f"X{r}"])
            all_cs[r]["Y_k"].append(d[f"Y{r}_k"])
            if r not in ref_mask:
                ref_mask[r] = d[f"mask{r}"]

    Y_merged = np.concatenate(all_Y, axis=0)

    das_keys = ["rf_data_3d", "angles", "probe_geometry", "initial_time",
                 "scan_x_axis", "scan_z_axis", "selected_angles"]
    das_info = {}
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        for k in das_keys:
            if k in d and k not in das_info:
                das_info[k] = d[k]

    save_dict = {
        "Y": Y_merged,
        "fs": np.float32(ref_fs),
        "fc": np.float32(ref_fc),
        "c": np.float32(ref_c),
        "theta": np.float32(0.0),
        "N1": np.int32(128),
        "N2": np.int32(75),
        "group_id": np.concatenate(all_group, axis=0),
        "source_id": np.concatenate(all_source, axis=0),
        **das_info,
    }

    for r in cs_ratios:
        X_all = all_cs[r]["X"]
        if all(x.shape[1] == X_all[0].shape[1] for x in X_all):
            save_dict[f"X{r}"] = np.concatenate(X_all, axis=0)
            save_dict[f"Y{r}_k"] = np.concatenate(all_cs[r]["Y_k"], axis=0)
            save_dict[f"mask{r}"] = ref_mask[r]

            mu = np.nonzero(ref_mask[r])[0].astype(np.int32)
            save_dict[f"mu{r}"] = mu

    np.savez_compressed(output_path, **save_dict)
    print(f"\n合并数据集已保存: {output_path}")
    print(f"  总样本数: {Y_merged.shape[0]}, 信号长度: {Y_merged.shape[1]}")


# ======================== CLI ========================

def run_prepare(args):

    abs_root = os.path.abspath(args.picmus_root)
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(f"PICMUS 根目录不存在: {abs_root}")

    mode = getattr(args, "mode", "line")
    print(f"PICMUS root: {abs_root}")
    print(f"Mode:        {mode}")
    print(f"Datasets:    {args.datasets}")
    print(f"CS ratios:   {args.cs_ratios}")

    if mode == "frame":
        saved_paths = []
        for ds_key in args.datasets:
            p = process_dataset_frames(
                dataset_key=ds_key,
                picmus_root=abs_root,
                cs_ratios=args.cs_ratios,
                angle_stride=args.angle_stride,
                output_dir=args.output_dir,
                seed=args.seed,
            )
            if p is not None:
                saved_paths.append(p)
        if args.merge and len(saved_paths) > 1:
            merge_frame_datasets(saved_paths, args.cs_ratios, args.output_dir)
        print("\n帧级预处理完成!")
        return

    print(f"Max samples: {args.max_samples}")

    saved_paths = []
    for ds_key in args.datasets:
        p = process_dataset(
            dataset_key=ds_key,
            picmus_root=abs_root,
            cs_ratios=args.cs_ratios,
            max_samples=args.max_samples,
            angle_stride=args.angle_stride,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        if p is not None:
            saved_paths.append(p)

    if args.merge and len(saved_paths) > 1:
        merge_path = os.path.join(args.output_dir, "picmus_merged.npz")
        merge_datasets(saved_paths, args.cs_ratios, merge_path)

    print("\n预处理完成!")


def build_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="PICMUS HDF5 → npz 预处理 (兼容 HASA-ADMM-Net 训练格式)",
        add_help=add_help)
    parser.add_argument("--picmus_root", type=str, default=PICMUS_ROOT, help="archive_to_download 根目录")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=list(DATASET_MAP.keys()),
                        choices=list(DATASET_MAP.keys()),
                        help="要处理的数据集 (默认全部 4 个子集)")
    parser.add_argument("--cs_ratios", type=int, nargs="+", default=[4, 8, 15],
                        help="压缩比列表 (默认 4 8 15)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="每个数据集最大采样数 (0=全量不截断)")
    parser.add_argument("--angle_stride", type=int, default=1,
                        help="角度采样步长 (1=全部角度)")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    parser.add_argument("--merge", action="store_true", default=False, help="将多个数据集合并为一个 npz (仅限信号长度相同时)")
    parser.add_argument("--mode", type=str, default="line", choices=["line", "frame"],
                        help="输出模式: line=逐线1D (默认), frame=帧级2D")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser


def main():
    args = build_parser().parse_args()
    run_prepare(args)


if __name__ == "__main__":
    main()
