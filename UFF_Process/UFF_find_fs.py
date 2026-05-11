import h5py

path = "./Alpinion_L3-8_FI_hyperechoic_scatterers.uff"
keys = ["sampling", "fs", "frequency", "sample"]

with h5py.File(path, "r") as f:
    def find_keys(name, obj):
        if isinstance(obj, h5py.Dataset):
            low = name.lower()
            if any(k in low for k in keys):
                print("HIT:", name, obj.shape, obj.dtype)
    f.visititems(find_keys)

with h5py.File(path, "r") as f:
    fs = float(f["channel_data/sampling_frequency"][()][0, 0])          # Hz
    fc = float(f["channel_data/pulse/center_frequency"][()][0, 0])      # Hz
    fm = float(f["channel_data/modulation_frequency"][()][0, 0])        # Hz（可选）

print("sampling_frequency fs =", fs, "Hz")
print("center_frequency   fc =", fc, "Hz")
print("modulation_freq    fm =", fm, "Hz")
print("fs (MHz) =", fs/1e6, "MHz")
print("fc (MHz) =", fc/1e6, "MHz")
