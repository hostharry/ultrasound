import h5py
import numpy as np

path = "./Alpinion_L3-8_FI_hyperechoic_scatterers.uff"

items = []
with h5py.File(path, "r") as f:
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            n = int(np.prod(obj.shape)) if obj.shape is not None else 0
            items.append((n, name, obj.shape, str(obj.dtype)))
    f.visititems(collect)

items.sort(reverse=True, key=lambda x: x[0])
for n, name, shape, dtype in items[:30]:
    print(f"{n:12d}  {dtype:>10s}  {shape!s:>15s}  {name}")