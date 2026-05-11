"""
CUBDL HDF5 → 训练用 npz 预处理脚本
====================================

将 CUBDL Challenge 的 HDF5 channel_data 转换为 2D 帧级训练 NPZ，
支持论文 "Deep Unfolded Recovery of Sub-Nyquist Sampled Ultrasound Images"
中的联合空域+时域 sub-Nyquist 采样方案。

HDF5 结构 (JHU024-034 等):
    channel_data : (n_tx, n_elements, n_samples)  float32
    angles, element_positions, sampling_frequency, modulation_frequency, ...

输出 NPZ 格式 (兼容 UltrasoundFrameDataset):
    Y_frames, X{r}_frames, Y{r}_k_frames, mask{r}, mu{r},
    spatial_mask{r}, spatial_mu{r}, group_id, fs, fc, c, ...

用法:
    python prepare_cubdl_data.py --hdf5 JHU024.hdf5 JHU025.hdf5 ...
    python prepare_cubdl_data.py --hdf5 JHU024.hdf5 --temporal_ratio 8
    python prepare_cubdl_data.py --hdf5 *.hdf5 --spatial_ratio 4 --temporal_ratio 2 --merge
"""

import os
import sys
import argparse
import shutil
import tempfile
import zipfile
import numpy as np
import h5py
from typing import List, Tuple, Optional


# ======================== 分形阵列 ========================

def fractal_array(generator: List[int], order: int, n_elements: int) -> np.ndarray:
    """生成分形阵列的活跃阵元索引 (论文 Section II-C)。

    Parameters
    ----------
    generator : 生成元集合, e.g. [0, 1]
    order : 分形阶数
    n_elements : 总阵元数 (超出部分裁掉)
    """
    L = 2 * max(generator) + 1
    W = {0}
    for r in range(order):
        W_next = set()
        for n in generator:
            shift = n * (L ** r)
            for w in W:
                idx = w + shift
                if idx < n_elements:
                    W_next.add(idx)
        W = W_next
    return np.array(sorted(W), dtype=np.int32)


def _best_fractal_order(n_elements: int, n_active_target: int) -> int:
    """找到使活跃阵元数最接近 n_active_target 的分形阶数。"""
    best_order, best_diff = 1, n_elements
    for order in range(1, 12):
        arr = fractal_array([0, 1], order, n_elements)
        diff = abs(len(arr) - n_active_target)
        if diff < best_diff:
            best_diff = diff
            best_order = order
        if len(arr) >= n_elements:
            break
    return best_order


# ======================== 掩膜生成 ========================

def create_spatial_mask(
    n_elements: int,
    spatial_ratio: int,
    method: str = "fractal",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """创建空域掩膜。

    Returns
    -------
    mask : (n_elements,) uint8
    mu   : 活跃阵元索引
    """
    if spatial_ratio <= 1:
        mask = np.ones(n_elements, dtype=np.uint8)
        mu = np.arange(n_elements, dtype=np.int32)
        return mask, mu

    n_active = max(2, n_elements // spatial_ratio)

    if method == "fractal":
        order = _best_fractal_order(n_elements, n_active)
        mu = fractal_array([0, 1], order, n_elements)
        if len(mu) > n_active:
            mu = mu[:n_active]
    else:
        rng = np.random.RandomState(seed)
        mu = np.sort(rng.choice(n_elements, n_active, replace=False))

    mask = np.zeros(n_elements, dtype=np.uint8)
    mask[mu] = 1
    return mask, mu.astype(np.int32)


def create_freq_mask(
    n_samples: int,
    temporal_ratio: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """创建时域频率掩膜 (rfft 域)。

    始终保留 DC 分量，随机选取其余频率。

    Returns
    -------
    mask : (n_freq,) uint8, n_freq = n_samples//2+1
    mu   : 选中频率索引
    """
    n_freq = n_samples // 2 + 1

    if temporal_ratio <= 1:
        mask = np.ones(n_freq, dtype=np.uint8)
        mu = np.arange(n_freq, dtype=np.int32)
        return mask, mu

    K = max(2, n_freq // temporal_ratio)
    rng = np.random.RandomState(seed + temporal_ratio)
    candidates = np.arange(1, n_freq)
    chosen = rng.choice(candidates, K - 1, replace=False)
    mu = np.sort(np.concatenate([[0], chosen])).astype(np.int32)

    mask = np.zeros(n_freq, dtype=np.uint8)
    mask[mu] = 1
    return mask, mu


# ======================== 加载 HDF5 ========================

def _read_scalar(f, key: str, fallback: float = 0.0) -> float:
    """从 HDF5 安全读取标量，兼容 (1,) 和 (1,1) 形状。"""
    if key not in f:
        return fallback
    v = f[key][()]
    return float(np.asarray(v).flat[0])


def load_cubdl_file(hdf5_path: str) -> dict:
    """读取 CUBDL HDF5 文件，返回 metadata + channel_data。

    兼容 JHU / EUT / INS / MYO / OSL / UFL 等不同机构的字段差异。
    """
    with h5py.File(hdf5_path, "r") as f:
        channel_data = f["channel_data"][:].astype(np.float32)

        angles = None
        if "angles" in f:
            angles = f["angles"][:].flatten().astype(np.float32)
        elif "transmit_direction" in f:
            td = f["transmit_direction"][:]
            if td.ndim >= 2:
                angles = td[:, 0].flatten().astype(np.float32)
            else:
                angles = td.flatten().astype(np.float32)

        if channel_data.ndim == 2:
            n_tx_hint = int(angles.size) if angles is not None and angles.size > 0 else 1
            if n_tx_hint > 1 and channel_data.shape[0] % n_tx_hint == 0:
                n_elem = channel_data.shape[0] // n_tx_hint
                channel_data = channel_data.reshape(n_tx_hint, n_elem, channel_data.shape[1])
            elif n_tx_hint > 1 and channel_data.shape[1] % n_tx_hint == 0:
                n_elem = channel_data.shape[0]
                n_samples = channel_data.shape[1] // n_tx_hint
                channel_data = channel_data.reshape(n_elem, n_tx_hint, n_samples).transpose(1, 0, 2)
            else:
                channel_data = channel_data[None, ...]
        elif channel_data.ndim != 3:
            raise ValueError(
                f"Unsupported channel_data shape {channel_data.shape}, expected 2D or 3D"
            )

        fs_key = "channel_data_sampling_frequency" if "channel_data_sampling_frequency" in f else "sampling_frequency"
        fs = _read_scalar(f, fs_key, 0.0)
        fc = _read_scalar(f, "modulation_frequency", 0.0)
        c = _read_scalar(f, "sound_speed", 1540.0)
        pitch = _read_scalar(f, "pitch", 0.0)

        n_tx = channel_data.shape[0]

        if angles is None:
            angles = np.zeros(n_tx, dtype=np.float32)
        elif angles.size == 1 and n_tx > 1:
            angles = np.full(n_tx, float(angles[0]), dtype=np.float32)
        else:
            angles = angles[:n_tx].astype(np.float32)

        if "element_positions" in f:
            ep = f["element_positions"][:].astype(np.float32)
            if ep.ndim == 2 and ep.shape[0] == 3:
                elem_pos = ep[0, :]
            elif ep.ndim == 1:
                elem_pos = ep
            else:
                elem_pos = ep.flatten()[:channel_data.shape[1]]
        else:
            elem_pos = np.arange(channel_data.shape[1], dtype=np.float32)

        if "time_zero" in f:
            time_zero = f["time_zero"][:].flatten().astype(np.float32)
        elif "start_time" in f:
            time_zero = f["start_time"][:].flatten().astype(np.float32)
        elif "channel_data_t0" in f:
            t0 = _read_scalar(f, "channel_data_t0", 0.0)
            time_zero = np.full(n_tx, t0, dtype=np.float32)
        else:
            time_zero = np.zeros(n_tx, dtype=np.float32)

        if time_zero.size == 1 and n_tx > 1:
            time_zero = np.full(n_tx, float(time_zero[0]), dtype=np.float32)
        else:
            time_zero = time_zero[:n_tx].astype(np.float32)

        beamformed = None
        if "beamformed_data" in f:
            beamformed = f["beamformed_data"][:].astype(np.float32)

    n_tx, n_elem, n_samples = channel_data.shape
    print(f"  channel_data: ({n_tx}, {n_elem}, {n_samples})")
    print(f"  fs={fs/1e6:.2f} MHz, fc={fc/1e6:.2f} MHz, c={c:.0f} m/s")

    return {
        "channel_data": channel_data,
        "fs": fs, "fc": fc, "c": c,
        "angles": angles,
        "element_positions": elem_pos,
        "time_zero": time_zero,
        "pitch": pitch,
        "beamformed_data": beamformed,
        "n_tx": n_tx, "n_elements": n_elem, "n_samples": n_samples,
    }


# ======================== 联合 sub-Nyquist 采样 ========================

def apply_joint_subnyquist(
    frames: np.ndarray,
    spatial_mask: np.ndarray,
    spatial_mu: np.ndarray,
    freq_mask: np.ndarray,
    freq_mu: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """对 2D 帧施加联合空域+时域 sub-Nyquist 退化。

    Parameters
    ----------
    frames : (n_frames, H, W) float32   原始 channel_data
    spatial_mask : (H,) uint8           空域掩膜
    spatial_mu : 活跃阵元索引
    freq_mask : (n_freq,) uint8         频域掩膜
    freq_mu : 选中频率索引

    Returns
    -------
    X_frames : (n_frames, H, W) float32   退化后的帧
    Yk_frames : (n_frames, H, K) complex64 频域观测
    """
    n_frames, H, W = frames.shape
    n_freq = W // 2 + 1
    K = len(freq_mu)

    X_frames = np.zeros_like(frames)
    Yk_frames = np.zeros((n_frames, H, K), dtype=np.complex64)

    for fi in range(n_frames):
        for ei in spatial_mu:
            row = frames[fi, ei, :]
            row_freq = np.fft.rfft(row)
            Yk_frames[fi, ei, :] = row_freq[freq_mu].astype(np.complex64)
            row_masked = row_freq * freq_mask
            X_frames[fi, ei, :] = np.fft.irfft(row_masked, n=W).astype(np.float32)

    return X_frames, Yk_frames


# ======================== 处理单个 HDF5 ========================

def process_cubdl_file(
    hdf5_path: str,
    output_dir: str,
    spatial_ratio: int = 1,
    temporal_ratio: int = 8,
    spatial_method: str = "fractal",
    seed: int = 42,
) -> Optional[str]:
    """处理单个 CUBDL HDF5 文件，输出一个 NPZ。"""
    basename = os.path.splitext(os.path.basename(hdf5_path))[0]
    cs_ratio = spatial_ratio * temporal_ratio

    print(f"\n{'='*60}")
    print(f"处理: {basename}")
    print(f"  文件: {hdf5_path}")
    print(f"  spatial_ratio={spatial_ratio}, temporal_ratio={temporal_ratio}, cs_ratio={cs_ratio}")

    info = load_cubdl_file(hdf5_path)
    frames = info["channel_data"]  # (n_tx, n_elem, n_samples)

    sig_max = np.abs(frames).max()
    if sig_max > 0:
        frames /= sig_max

    n_tx, n_elem, n_samples = frames.shape

    spatial_mask, spatial_mu = create_spatial_mask(
        n_elem, spatial_ratio, method=spatial_method, seed=seed)
    freq_mask, freq_mu = create_freq_mask(n_samples, temporal_ratio, seed=seed)

    print(f"  空域: {len(spatial_mu)}/{n_elem} 活跃阵元")
    print(f"  时域: {len(freq_mu)}/{n_samples//2+1} 频率分量")

    X_frames, Yk_frames = apply_joint_subnyquist(
        frames, spatial_mask, spatial_mu, freq_mask, freq_mu)

    snr_per = []
    for fi in range(n_tx):
        sp = np.sum(frames[fi] ** 2)
        np_ = np.sum((frames[fi] - X_frames[fi]) ** 2)
        snr_per.append(10 * np.log10(sp / (np_ + 1e-10)))
    snr_arr = np.array(snr_per)
    print(f"  init SNR = {np.mean(snr_arr):.2f} +/- {np.std(snr_arr):.2f} dB")

    save_dict = {
        "Y_frames": frames,
        f"X{cs_ratio}_frames": X_frames,
        f"Y{cs_ratio}_k_frames": Yk_frames,
        f"mask{cs_ratio}": freq_mask,
        f"mu{cs_ratio}": freq_mu,
        f"spatial_mask{cs_ratio}": spatial_mask,
        f"spatial_mu{cs_ratio}": spatial_mu,
        "group_id": np.zeros(n_tx, dtype=np.int32),
        "fs": np.float32(info["fs"]),
        "fc": np.float32(info["fc"]),
        "c": np.float32(info["c"]),
        "n_elements": np.int32(n_elem),
        "n_frames": np.int32(n_tx),
        "cs_ratio": np.int32(cs_ratio),
        "spatial_ratio": np.int32(spatial_ratio),
        "temporal_ratio": np.int32(temporal_ratio),
        "angles": info["angles"],
        "element_positions": info["element_positions"],
        "time_zero": info["time_zero"],
        "pitch": np.float32(info["pitch"]),
    }

    if info["beamformed_data"] is not None:
        save_dict["beamformed_data"] = info["beamformed_data"]

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"cubdl_{basename}_st{cs_ratio}.npz"
    out_path = os.path.join(output_dir, out_name)
    np.savez_compressed(out_path, **save_dict)
    print(f"  保存至: {out_path}")
    return out_path


# ======================== 合并多个 NPZ ========================

def merge_npz_files(
    npz_paths: List[str],
    output_path: str,
):
    """将多个同形状的 NPZ 合并为一个，group_id 自动顺延。

    使用 memmap 避免一次性读入全部数据导致 OOM。
    """
    print(f"\n合并 {len(npz_paths)} 个文件 -> {output_path}")

    first = np.load(npz_paths[0], allow_pickle=True)
    n0 = int(first["Y_frames"].shape[0])

    concat_keys = [k for k in first.files
                   if np.ndim(first[k]) > 0 and first[k].shape[0] == n0]
    carry_keys = [k for k in first.files if k not in concat_keys]

    total = 0
    shapes, dtypes = {}, {}
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        n = int(d["Y_frames"].shape[0])
        total += n
        for k in concat_keys:
            arr = d[k]
            if k not in shapes:
                shapes[k] = (total,) + arr.shape[1:]
                dtypes[k] = arr.dtype
            else:
                shapes[k] = (total,) + shapes[k][1:]

    tmpdir = tempfile.mkdtemp(prefix="merge_cubdl_")
    try:
        mmap_paths = {k: os.path.join(tmpdir, f"{k}.npy") for k in concat_keys}
        mmaps = {
            k: np.lib.format.open_memmap(
                mmap_paths[k], mode="w+", dtype=dtypes[k], shape=shapes[k])
            for k in concat_keys
        }

        offset = 0
        gid_offset = 0
        for p in npz_paths:
            d = np.load(p, allow_pickle=True)
            n = int(d["Y_frames"].shape[0])
            for k in concat_keys:
                if k == "group_id":
                    mmaps[k][offset:offset+n] = d[k] + gid_offset
                else:
                    mmaps[k][offset:offset+n] = d[k]
            if "group_id" in d.files:
                gid_offset += int(np.max(d["group_id"])) + 1
            else:
                gid_offset += 1
            offset += n
            print(f"  + {os.path.basename(p)}: {n} 帧, gid_offset -> {gid_offset}")

        for k in carry_keys:
            np.save(os.path.join(tmpdir, f"{k}.npy"), first[k])

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for k in concat_keys + carry_keys:
                zf.write(os.path.join(tmpdir, f"{k}.npy"), arcname=f"{k}.npy")

        print(f"  总帧数: {total}")
        print(f"  保存至: {output_path}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ======================== CLI ========================

def build_parser():
    parser = argparse.ArgumentParser(
        description="CUBDL HDF5 -> NPZ 预处理 (联合 sub-Nyquist 采样)")
    parser.add_argument("--hdf5", type=str, nargs="+", required=True,
                        help="输入 HDF5 文件路径 (可多个)")
    parser.add_argument("--spatial_ratio", type=int, default=1,
                        help="空域压缩比 (1=不做空域降采样)")
    parser.add_argument("--temporal_ratio", type=int, default=8,
                        help="时域频率压缩比")
    parser.add_argument("--spatial_method", type=str, default="fractal",
                        choices=["fractal", "random"],
                        help="空域采样方式")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="输出目录")
    parser.add_argument("--merge", action="store_true", default=False,
                        help="合并所有输出 NPZ 为一个")
    parser.add_argument("--merge_name", type=str, default=None,
                        help="合并后的文件名 (默认自动生成)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser


def main():
    args = build_parser().parse_args()
    cs_ratio = args.spatial_ratio * args.temporal_ratio

    saved_paths = []
    for hdf5_path in args.hdf5:
        p = process_cubdl_file(
            hdf5_path=hdf5_path,
            output_dir=args.output_dir,
            spatial_ratio=args.spatial_ratio,
            temporal_ratio=args.temporal_ratio,
            spatial_method=args.spatial_method,
            seed=args.seed,
        )
        if p is not None:
            saved_paths.append(p)

    if args.merge and len(saved_paths) > 1:
        if args.merge_name:
            merge_out = os.path.join(args.output_dir, args.merge_name)
        else:
            merge_out = os.path.join(args.output_dir, f"cubdl_merged_st{cs_ratio}.npz")
        merge_npz_files(saved_paths, merge_out)

    print("\n预处理完成!")


if __name__ == "__main__":
    main()
