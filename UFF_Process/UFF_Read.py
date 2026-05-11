import csv
import re

import h5py
import numpy as np

path = "/home/user/毕业设计/Ultrasound/Alpinion_L3-8_FI_hyperechoic_scatterers.uff"   # 改成你的路径
output_csv = "/home/user/毕业设计/Ultrasound/UFF_Process/uff_metadata.csv"
output_pos_csv = "/home/user/毕业设计/Ultrasound/UFF_Process/uff_positions.csv"


def extract_sequence(name: str) -> str:
    match = re.search(r"sequence_\d+", name)
    return match.group(0) if match else ""


def classify_path(name: str) -> str:
    if "/apodization/apex/" in name or name.endswith("/apodization/apex"):
        return "apodization/apex"
    if "/apodization/scan/" in name or name.endswith("/apodization/scan"):
        return "apodization/scan"
    if "/apodization/" in name or name.endswith("/apodization"):
        return "apodization"
    if "/probe/" in name or name.endswith("/probe"):
        return "probe"
    if "/source/" in name or name.endswith("/source"):
        return "source"
    if "/sound_speed" in name:
        return "sound_speed"
    return "other"


def is_position_dataset(name: str) -> bool:
    last = name.rsplit("/", 1)[-1].lower()
    pos_keys = {
        "x", "y", "z",
        "azimuth", "elevation", "distance",
        "geometry", "position", "pos",
        "pitch", "element_width", "element_height",
    }
    if last in pos_keys:
        return True
    if "position" in last or "coord" in last:
        return True
    return False


def summarize_dataset(dset: h5py.Dataset) -> str:
    if not np.issubdtype(dset.dtype, np.number):
        return ""
    arr = np.array(dset)
    if arr.size <= 16:
        flat = arr.reshape(-1)
        return f"values={flat.tolist()}"
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    mean_v = float(np.nanmean(arr))
    return f"min={min_v}, max={max_v}, mean={mean_v}"


rows = []
pos_rows = []
with h5py.File(path, "r") as f:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("DATASET:", name, obj.shape, obj.dtype)
            rows.append({
                "path": name,
                "kind": "DATASET",
                "shape": str(tuple(obj.shape)),
                "dtype": str(obj.dtype),
                "sequence": extract_sequence(name),
                "category": classify_path(name),
            })
            if is_position_dataset(name):
                pos_rows.append({
                    "group_path": name.rsplit("/", 1)[0],
                    "dataset_path": name,
                    "shape": str(tuple(obj.shape)),
                    "dtype": str(obj.dtype),
                    "sequence": extract_sequence(name),
                    "category": classify_path(name),
                    "summary": summarize_dataset(obj),
                })
        else:
            print("GROUP  :", name)
            rows.append({
                "path": name,
                "kind": "GROUP",
                "shape": "",
                "dtype": "",
                "sequence": extract_sequence(name),
                "category": classify_path(name),
            })
    f.visititems(walk)

with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["path", "kind", "shape", "dtype", "sequence", "category"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved CSV to: {output_csv}")

with open(output_pos_csv, "w", newline="") as csvfile:
    fieldnames = ["group_path", "dataset_path", "shape", "dtype", "sequence", "category", "summary"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(pos_rows)

print(f"Saved position CSV to: {output_pos_csv}")
