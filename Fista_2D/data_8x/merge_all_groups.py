"""
将 data_8x 目录中所有单文件 npz 按 (H, W) 形状分组合并。

输出:
  cubdl_EUT_st8_merged.npz          (80, 1664)  150 frames
  cubdl_INS-1536_st8_merged.npz     (128, 1536) 225 frames
  cubdl_JHU024-034_st8_merged.npz   (128, 1558) 815 frames  (已存在, 跳过)
  cubdl_INS-OSL-1920_st8_merged.npz (128, 1920) 375 frames
  cubdl_MYO_st8_merged.npz          (128, 2688) 375 frames
  cubdl_UFL_st8_merged.npz          (128, 4480) 300 frames

总计: 2240 帧, 6 个合并文件
"""

import os
import glob
import hashlib
import re
import shutil
import tempfile
import zipfile
import numpy as np
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SIGNATURE_KEYS = (
    "fs", "fc", "c", "n_elements", "pitch", "element_positions",
    "mask8", "mu8", "spatial_mask8", "spatial_mu8",
    "cs_ratio", "spatial_ratio", "temporal_ratio",
)


def freeze_value(value):
    """Convert arrays/scalars into a hashable signature fragment."""
    arr = np.asarray(value)
    return (str(arr.dtype), tuple(arr.shape), hashlib.sha1(arr.tobytes()).hexdigest())


def build_merge_signature(npz_path):
    """Files can be merged only when frame shape and key metadata match."""
    d = np.load(npz_path, allow_pickle=True)
    try:
        sig = [tuple(d["Y_frames"].shape[1:])]
        for key in SIGNATURE_KEYS:
            if key in d.files:
                sig.append((key, freeze_value(d[key])))
            else:
                sig.append((key, None))
        return tuple(sig)
    finally:
        d.close()


def split_compatible_groups(npz_paths):
    """Split files into metadata-compatible subgroups."""
    groups = defaultdict(list)
    for path in npz_paths:
        groups[build_merge_signature(path)].append(path)
    return list(groups.values())


def compress_numbers(nums):
    """Format [24, 25, 26, 30] as 024-026_030."""
    nums = sorted(set(nums))
    if not nums:
        return ""

    chunks = []
    start = prev = nums[0]
    for num in nums[1:]:
        if num == prev + 1:
            prev = num
            continue
        chunks.append(f"{start:03d}-{prev:03d}" if start != prev else f"{start:03d}")
        start = prev = num
    chunks.append(f"{start:03d}-{prev:03d}" if start != prev else f"{start:03d}")
    return "_".join(chunks)


def build_group_label(npz_paths):
    """Derive a stable filename label from source files."""
    parts = defaultdict(list)
    raw_tags = []

    for path in npz_paths:
        tag = os.path.basename(path).replace("cubdl_", "").replace("_st8.npz", "")
        m = re.match(r"([A-Za-z-]+?)(\d+)$", tag)
        if m:
            parts[m.group(1)].append(int(m.group(2)))
        else:
            raw_tags.append(tag)

    labels = []
    for prefix in sorted(parts):
        labels.append(f"{prefix}{compress_numbers(parts[prefix])}")
    labels.extend(sorted(raw_tags))
    return "-".join(labels)


def build_output_name(npz_paths, shape, use_legacy_name):
    """Use legacy names only when a shape has exactly one compatible merge group."""
    H, W = shape
    if use_legacy_name:
        legacy_name = GROUP_NAMES.get((H, W))
        if legacy_name:
            return legacy_name

    label = build_group_label(npz_paths)
    return f"cubdl_{label}_{H}x{W}_st8_merged.npz"


GROUP_NAMES = {
    (80, 1664):  "cubdl_EUT_st8_merged.npz",
    (128, 1536): "cubdl_INS-1536_st8_merged.npz",
    (128, 1558): "cubdl_JHU024-034_st8_merged.npz",
    (128, 1920): "cubdl_INS-OSL-1920_st8_merged.npz",
    (128, 2688): "cubdl_MYO_st8_merged.npz",
    (128, 4480): "cubdl_UFL_st8_merged.npz",
}


def merge_group(npz_paths, output_path):
    """Merge npz files that share the same (H, W) frame shape."""
    print(f"\n{'='*60}")
    print(f"合并 {len(npz_paths)} 个文件 -> {os.path.basename(output_path)}")

    ref = np.load(npz_paths[0], allow_pickle=True)
    ref_n = ref["Y_frames"].shape[0]

    concat_keys = []
    for k in ref.files:
        arr = ref[k]
        if k == "beamformed_data":
            continue
        if arr.ndim > 0 and arr.shape[0] == ref_n:
            concat_keys.append(k)

    carry_keys = [k for k in ref.files if k not in concat_keys and k != "beamformed_data"]

    total_frames = 0
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        total_frames += d["Y_frames"].shape[0]
        d.close()

    print(f"  concat keys: {concat_keys}")
    print(f"  carry  keys: {carry_keys}")
    print(f"  总帧数: {total_frames}")

    tmpdir = tempfile.mkdtemp(prefix="merge_cubdl_")
    try:
        mmaps = {}
        for k in concat_keys:
            tail_shape = ref[k].shape[1:]
            full_shape = (total_frames,) + tail_shape
            path = os.path.join(tmpdir, f"{k}.npy")
            mmaps[k] = np.lib.format.open_memmap(
                path, mode="w+", dtype=ref[k].dtype, shape=full_shape
            )

        offset = 0
        gid_offset = 0
        for p in npz_paths:
            d = np.load(p, allow_pickle=True)
            n = d["Y_frames"].shape[0]
            for k in concat_keys:
                if k == "group_id":
                    mmaps[k][offset : offset + n] = d[k] + gid_offset
                else:
                    mmaps[k][offset : offset + n] = d[k]
            if "group_id" in d.files:
                gid_offset += int(np.max(d["group_id"])) + 1
            else:
                gid_offset += 1
            offset += n
            print(f"  + {os.path.basename(p)}: {n} 帧, gid_offset -> {gid_offset}")
            d.close()

        for mm in mmaps.values():
            mm.flush()

        for k in carry_keys:
            value = total_frames if k == "n_frames" else ref[k]
            np.save(os.path.join(tmpdir, f"{k}.npy"), value)

        ref.close()

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for k in concat_keys + carry_keys:
                zf.write(os.path.join(tmpdir, f"{k}.npy"), arcname=f"{k}.npy")

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  保存至: {output_path} ({size_mb:.1f} MB)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "cubdl_*_st8.npz")))
    individual = [f for f in all_files if "merged" not in os.path.basename(f)]

    shape_groups = defaultdict(list)
    for f in individual:
        d = np.load(f, allow_pickle=True)
        H, W = d["Y_frames"].shape[1], d["Y_frames"].shape[2]
        shape_groups[(H, W)].append(f)
        d.close()

    merged_paths = []

    for (H, W), files in sorted(shape_groups.items()):
        compatible_groups = split_compatible_groups(files)
        use_legacy_name = len(compatible_groups) == 1

        if len(compatible_groups) > 1:
            print(f"\n[提示] ({H}, {W}) 检测到 {len(compatible_groups)} 个元数据不兼容的子组, 将分别处理")

        for group_files in compatible_groups:
            group_files = sorted(group_files)
            name = build_output_name(group_files, (H, W), use_legacy_name)
            out_path = os.path.join(DATA_DIR, name)

            if os.path.exists(out_path):
                print(f"\n[跳过] {name} 已存在 ({os.path.getsize(out_path)/1024/1024:.1f} MB)")
                merged_paths.append(out_path)
                continue

            if len(group_files) == 1:
                print(
                    f"\n[跳过] ({H}, {W}) 仅 1 个兼容文件, 无需合并: "
                    f"{os.path.basename(group_files[0])}"
                )
                merged_paths.append(group_files[0])
                continue

            merge_group(group_files, out_path)
            merged_paths.append(out_path)

    print(f"\n{'='*60}")
    print("合并完成! 可用数据集文件:")
    print(f"{'='*60}")
    total_frames = 0
    for p in merged_paths:
        d = np.load(p, allow_pickle=True)
        Y = d["Y_frames"]
        n, H, W = Y.shape
        total_frames += n
        print(f"  {os.path.basename(p):<45} ({H:>3}, {W:>4}) x {n:>4} frames")
        d.close()

    print(f"\n  总计: {total_frames} 帧, {len(merged_paths)} 个文件")

    print(f"\n{'='*60}")
    print("训练命令示例:")
    print(f"{'='*60}")
    npz_args = " \\\n    ".join(p for p in merged_paths)
    print(f"""
python train_lite_2d.py \\
    --npz \\
    {npz_args} \\
    --cs_ratio 8 \\
    --save_dir ../Fista_2D \\
    --patch_h 64 --layers 4 --d_model 32
""")


if __name__ == "__main__":
    main()
