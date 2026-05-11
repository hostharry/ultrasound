import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

npz_path = "/home/user/毕业设计/Ultrasound/dataset_fdbf_energy_mu_8_9_15.npz"
data = np.load(npz_path)

Y  = data["Y"]    # (L, N)  GT
X8 = data["X8"]   # (L, N)  8x input
X15= data["X15"]  # (L, N) 15x input
fs = float(data["fs"])
c  = float(data["c"]) if "c" in data else 1540.0

def bmode(img_rf, dr=60):
    env = np.abs(hilbert(img_rf, axis=1))   # 沿时间轴做Hilbert（这里axis=1对应N）
    env /= (env.max() + 1e-12)
    db = 20*np.log10(env + 1e-12)
    db = np.clip(db, -dr, 0)
    return db

def show(db_img, title):
    # 深度轴（mm）
    z_mm = (c * (np.arange(db_img.shape[1])/fs) / 2) * 1e3
    plt.figure(figsize=(6,4))
    plt.imshow(db_img.T, cmap="gray", aspect="auto", origin="upper",
               extent=[0, db_img.shape[0]-1, z_mm[-1], z_mm[0]])
    plt.xlabel("line index")
    plt.ylabel("depth (mm)")
    plt.title(title)
    plt.colorbar(label="dB")
    plt.show()

show(bmode(Y),  "B-mode | GT (Y)")
show(bmode(X8), "B-mode | input X8 (band-limited)")
show(bmode(X15),"B-mode | input X15 (band-limited)")
