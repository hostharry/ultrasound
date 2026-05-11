import numpy as np

def rfft_freqs(N, fs):
    return np.fft.rfftfreq(N, d=1.0/fs)

def check_downsample_correct(npz_path, keyX="X8", keyMu="mu8", n_check=5, eps=1e-6):
    d = np.load(npz_path, allow_pickle=True)
    Y  = d["Y"]          # (L,N)
    X  = d[keyX]         # (L,N)
    mu = d[keyMu].astype(np.int64)
    fs = float(d["fs"])
    fc = float(d["fc"])
    L, N = Y.shape

    freqs = rfft_freqs(N, fs)

    print(f"[info] L={L}, N={N}, fs={fs/1e6:.2f} MHz, fc={fc/1e6:.2f} MHz")
    print(f"[info] {keyMu}: K={len(mu)}, f-range=({freqs[mu[0]]/1e6:.2f}~{freqs[mu[-1]]/1e6:.2f}) MHz")

    # 随机抽几条线检查
    idx = np.random.choice(L, size=min(n_check, L), replace=False)

    errs_mu = []
    leak = []
    energy_keep = []

    for i in idx:
        y = Y[i]
        x = X[i]
        Yf = np.fft.rfft(y)
        Xf = np.fft.rfft(x)

        # 1) mu 上的误差
        e_mu = np.max(np.abs(Xf[mu] - Yf[mu])) / (np.max(np.abs(Yf[mu])) + eps)
        errs_mu.append(e_mu)

        # 2) mu 外泄漏（应接近 0）
        mask = np.ones_like(Xf, dtype=bool)
        mask[mu] = False
        leak_ratio = (np.linalg.norm(Xf[mask]) / (np.linalg.norm(Xf[mu]) + eps))
        leak.append(leak_ratio)

        # 3) 频域能量保留率（用 Y 的能量）
        r = (np.sum(np.abs(Yf[mu])**2) / (np.sum(np.abs(Yf)**2) + eps))
        energy_keep.append(r)

    print(f"[check] max rel error on mu (lower is better): mean={np.mean(errs_mu):.3e}, max={np.max(errs_mu):.3e}")
    print(f"[check] leakage ratio off-mu (should be tiny): mean={np.mean(leak):.3e}, max={np.max(leak):.3e}")
    print(f"[check] energy kept in mu (higher means less info loss): mean={np.mean(energy_keep):.3f}")

    return {
        "err_mu_mean": float(np.mean(errs_mu)),
        "leak_mean": float(np.mean(leak)),
        "energy_keep_mean": float(np.mean(energy_keep)),
    }

# ---- 用法示例 ----
check_downsample_correct("dataset_fdbf_energy_mu_8_9_15.npz", "X8", "mu8")
check_downsample_correct("dataset_fdbf_energy_mu_8_9_15.npz", "X15", "mu15")
