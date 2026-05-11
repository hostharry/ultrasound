import h5py
import numpy as np
import torch

# -----------------------------
# 1) geometry -> delta_m (meters)
# -----------------------------
def infer_delta_m_from_geometry(geom: np.ndarray) -> np.ndarray:
    """
    geom: (7, C) in your file. Pick the row with largest span as x-coordinate.
    Return delta_m wrt reference element m0 (center element).
    """
    geom = np.asarray(geom)
    spans = geom.max(axis=1) - geom.min(axis=1)
    row = int(np.argmax(spans))
    x = geom[row].astype(np.float64)

    # unit guess: if looks like mm, convert to meters
    if np.nanmax(np.abs(x)) > 1.0:
        x *= 1e-3

    m0 = len(x) // 2
    delta = x - x[m0]
    return delta.astype(np.float64)

# -----------------------------
# 2) pick mu (rfft bins around fc)
# -----------------------------
def estimate_avg_rfft_mag_from_uff(dset, fs_hz: float, num_lines=32, device="cuda", seed=0):
    """
    dset: h5py dataset, shape (L, M, N)
    return:
      freqs: (N//2+1,) Hz
      mag:   (N//2+1,) averaged magnitude (cpu numpy)
    """
    L, M, N = dset.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(L, size=min(num_lines, L), replace=False)

    acc = None
    for i in idx:
        rf = dset[i, ...].astype(np.float32)            # (M,N) on CPU
        rf_t = torch.from_numpy(rf).to(device)          # (M,N) on GPU
        R = torch.fft.rfft(rf_t, dim=-1)                # (M,N//2+1)
        m = R.abs().mean(dim=0)                         # (N//2+1)
        acc = m if acc is None else (acc + m)

    mag = (acc / float(len(idx))).detach().cpu().numpy()
    freqs = np.fft.rfftfreq(N, d=1.0 / fs_hz)          # Hz
    return freqs, mag


def pick_best_contiguous_mu(freqs_hz: np.ndarray, mag: np.ndarray, K: int,
                            fmin_hz=None, fmax_hz=None, avoid_dc=True):
    """
    在 [fmin,fmax] 内选一个长度为 K 的连续频带，让能量和最大。
    return mu: int64 indices in [0..N//2]
    """
    Nf = len(freqs_hz)
    K = int(K)
    if K <= 1 or K > Nf:
        raise ValueError(f"K={K} invalid for rfft bins {Nf}")

    # default search range（经验：围绕 fc 的一大段，不要太窄）
    if fmin_hz is None:
        fmin_hz = 0.0
    if fmax_hz is None:
        fmax_hz = freqs_hz[-1]

    # 可行的窗口起点 s：要求窗口两端都在频段内
    s_min = np.searchsorted(freqs_hz, fmin_hz, side="left")
    s_max = np.searchsorted(freqs_hz, fmax_hz, side="right") - K
    s_min = max(s_min, 0)
    s_max = min(s_max, Nf - K)

    if avoid_dc:
        s_min = max(s_min, 1)  # 跳过 DC=0 bin

    if s_min > s_max:
        raise ValueError("No valid window in given [fmin,fmax]. Try widen the band.")

    # 滑动窗口能量：用卷积/累加实现 O(N)
    w = np.ones(K, dtype=np.float64)
    score = np.convolve(mag.astype(np.float64), w, mode="valid")  # len = Nf-K+1
    s = int(np.argmax(score[s_min:s_max + 1]) + s_min)

    mu = np.arange(s, s + K, dtype=np.int64)
    if avoid_dc:
        mu = mu[mu != 0]
    return mu

def pick_mu_rfft_band(N: int, fs_hz: float, fc_hz: float, K: int, avoid_dc=True):
    """
    Choose K rfft bins centered at fc (positive spectrum).
    Return mu as int64 indices in [0..N//2].
    """
    f = np.fft.rfftfreq(N, d=1.0 / fs_hz)  # 0..fs/2
    k0 = int(np.argmin(np.abs(f - fc_hz)))
    half = K // 2
    lo = k0 - half
    hi = lo + K - 1

    lo = max(0, lo)
    hi = min(len(f) - 1, hi)
    lo = max(0, hi - (K - 1))

    mu = np.arange(lo, hi + 1, dtype=np.int64)
    if avoid_dc:
        mu = mu[mu != 0]
    return mu

# -----------------------------
# 3) tau_m(t,theta) from paper eq (1)
# -----------------------------
def compute_tau(delta_m: torch.Tensor, t: torch.Tensor, theta: float, c: float):
    """
    delta_m: (M,) meters
    t: (N,) seconds
    theta: radians
    return tau: (M,N) seconds
    """
    dm = delta_m[:, None]
    tt = t[None, :]
    sin_th = torch.sin(torch.tensor(theta, device=t.device, dtype=t.dtype))
    inside = tt**2 - 4.0*(dm/c)*tt*sin_th + 4.0*(dm/c)**2
    inside = torch.clamp(inside, min=0.0)
    tau = 0.5 * (tt + torch.sqrt(inside))
    return tau

# -----------------------------
# 4) precompute Q_{k,m}[n] numerically on GPU
# -----------------------------
def precompute_Q(delta_m_m: np.ndarray, fs: float, N: int, mu_union: np.ndarray,
                 N1=10, N2=10, theta=0.0, c=1540.0, chunk_k=16, device="cuda"):
    """
    Q shape: (M, K, Nn) complex64
    mu_union: rfft bins (>=0). We'll treat k as those nonnegative indices.
    """
    delta_m = torch.tensor(delta_m_m, device=device, dtype=torch.float32)
    t = torch.arange(N, device=device, dtype=torch.float32) / fs
    T = float(N / fs)
    dt = 1.0 / fs

    tau = compute_tau(delta_m, t, theta=theta, c=c)  # (M,N)

    n_list = torch.arange(-N1, N2 + 1, device=device, dtype=torch.int64)
    mu_k = torch.tensor(mu_union.astype(np.int64), device=device, dtype=torch.int64)
    M = delta_m.numel()
    K = mu_k.numel()
    Nn = n_list.numel()

    Q = torch.zeros((M, K, Nn), device=device, dtype=torch.complex64)
    two_pi_over_T = (2.0 * np.pi / T)

    # chunk over K to control memory
    for s in range(0, K, chunk_k):
        k_chunk = mu_k[s:s+chunk_k].to(torch.float32)  # (Kc,)
        # exp(-j 2π k t / T): (Kc,N)
        e_kt = torch.exp(-1j * (two_pi_over_T * k_chunk[:, None] * t[None, :])).to(torch.complex64)

        for ni, n in enumerate(n_list):
            kn = (k_chunk - float(n.item()))  # (Kc,)
            # exp(+j 2π (k-n) tau / T): (M,Kc,N)
            e_kn_tau = torch.exp(1j * (two_pi_over_T * kn[None, :, None] * tau[:, None, :])).to(torch.complex64)
            integrand = e_kn_tau * e_kt[None, :, :]  # (M,Kc,N)
            Q_val = (integrand.sum(dim=-1) * dt) / T  # (M,Kc)
            Q[:, s:s+chunk_k, ni] = Q_val

    return Q, n_list.cpu().numpy()

# -----------------------------
# 5) FDBF: c[k] = mean_m sum_n c_m[k-n] * Q[k,m][n]   (paper eq (7))
# -----------------------------
def fdbf_coeffs_from_Q(rf_ch: torch.Tensor, Q: torch.Tensor,
                       mu_union: np.ndarray, n_list: np.ndarray):
    """
    rf_ch: (M,N) float32
    Q: (M,K,Nn) complex64 for K=len(mu_union)
    Return c_mu: (K,) complex64 (normalized FS-like, ~1/N factor)
    """
    device = rf_ch.device
    M, N = rf_ch.shape
    K = len(mu_union)
    Nn = len(n_list)

    # cm[k] (full FFT, normalized by N to mimic Fourier-series coefficient)
    Cm = torch.fft.fft(rf_ch.to(torch.float32), dim=-1) / float(N)  # (M,N), complex64

    mu_k = torch.tensor(mu_union.astype(np.int64), device=device, dtype=torch.int64)  # (K,)
    n_t = torch.tensor(n_list.astype(np.int64), device=device, dtype=torch.int64)     # (Nn,)

    # build indices kk = (k - n) mod N  -> shape (K,Nn)
    kk = (mu_k[:, None] - n_t[None, :]) % N  # (K,Nn)
    kk = kk[None, :, :].expand(M, -1, -1)    # (M,K,Nn)

    # gather Cm along last dim: need (M,K,N) expanded
    Cm_exp = Cm[:, None, :].expand(-1, K, -1)            # (M,K,N)
    Cm_sel = torch.take_along_dim(Cm_exp, kk, dim=-1)    # (M,K,Nn)

    # apply Q and sum over n
    c_hat_mk = (Cm_sel * Q).sum(dim=-1)  # (M,K)
    c_k = c_hat_mk.mean(dim=0)           # (K,)
    return c_k.to(torch.complex64)

# -----------------------------
# 6) time-domain DAS using SAME tau model (paper eq (1)(2)) for GT
# -----------------------------
def das_time_domain_paper(rf_ch: torch.Tensor, delta_m_m: np.ndarray,
                          fs: float, theta=0.0, c=1540.0):
    """
    y(t) = mean_m phi_m( tau_m(t,theta) )  with linear interpolation.
    """
    device = rf_ch.device
    M, N = rf_ch.shape
    delta_m = torch.tensor(delta_m_m, device=device, dtype=torch.float32)
    t = torch.arange(N, device=device, dtype=torch.float32) / fs
    tau = compute_tau(delta_m, t, theta=theta, c=c)  # (M,N)

    idx = tau * fs  # (M,N) in samples
    i0 = torch.floor(idx).to(torch.int64)
    a = (idx - i0.to(idx.dtype)).to(torch.float32)

    i0c = torch.clamp(i0, 0, N-1)
    i1c = torch.clamp(i0 + 1, 0, N-1)

    # gather rf at i0/i1
    rf0 = torch.gather(rf_ch, 1, i0c)
    rf1 = torch.gather(rf_ch, 1, i1c)

    aligned = (1.0 - a) * rf0 + a * rf1
    y = aligned.mean(dim=0)
    return y.to(torch.float32)

def K_like_paper(N: int, rate: int):
    # paper reference: Nref=1920, K8=230, K15=130
    if rate == 8:
        return int(round(N * (230.0 / 1920.0)))
    if rate == 15:
        return int(round(N * (130.0 / 1920.0)))
    # for 9x (not given in paper): use N/rate
    return int(round(N / float(rate)))

# -----------------------------
# 7) build dataset (GPU)
# -----------------------------
def build_dataset_fdbf_gpu(uff_path: str, out_npz: str,
                           reductions=(8, 9, 15),
                           N1=10, N2=10, theta=0.0, c=1540.0,
                           mu_mode="band", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    with h5py.File(uff_path, "r") as f:
        dset = f["channel_data/data"]              # (L,M,N)
        geom = f["channel_data/probe/geometry"][...]
        fs = float(np.squeeze(f["channel_data/sampling_frequency"][...]))
        fc = float(np.squeeze(f["channel_data/pulse/center_frequency"][...]))

        L, M, N = dset.shape

        delta_m = infer_delta_m_from_geometry(geom)  # (M,)

        # ---- 先估计平均频谱（GPU上做几条线的 rfft）----
        freqs, mag = estimate_avg_rfft_mag_from_uff(
            dset, fs_hz=fs, num_lines=32, device=device, seed=0
        )

        fmin = 0.5 * fc
        fmax = 1.5 * fc
        mu_dict = {}
        for r in reductions:
            K = K_like_paper(N, r)
            mu = pick_best_contiguous_mu(freqs, mag, K, fmin_hz=fmin, fmax_hz=fmax, avoid_dc=True)
            mu_dict[r] = mu

        # union mu
        mu_union = np.unique(np.concatenate([mu_dict[r] for r in reductions])).astype(np.int64)
        mu_union.sort()

        # precompute Q once (for mu_union)
        Q, n_list = precompute_Q(delta_m, fs, N, mu_union, N1=N1, N2=N2, theta=theta, c=c, device=device)

        # map mu -> position in mu_union
        pos = {int(k): i for i, k in enumerate(mu_union.tolist())}

        # allocate outputs (CPU arrays)
        Y = np.zeros((L, N), np.float32)
        X = {r: np.zeros((L, N), np.float32) for r in reductions}

        # main loop: stream line by line
        for i in range(L):
            rf = dset[i, ...].astype(np.float32)          # (M,N)
            rf_t = torch.from_numpy(rf).to(device)

            # GT: DAS (paper model)
            y = das_time_domain_paper(rf_t, delta_m, fs, theta=theta, c=c)
            Y[i] = y.detach().cpu().numpy()

            # FDBF coeffs on mu_union
            c_mu_union = fdbf_coeffs_from_Q(rf_t, Q, mu_union, n_list)  # (Kunion,)
            c_mu_union_cpu = c_mu_union.detach().cpu()

            # build each reduction input by placing only mu bins then irfft
            for r in reductions:
                mu = mu_dict[r]
                idx = torch.tensor([pos[int(k)] for k in mu], dtype=torch.int64)
                c_sub = c_mu_union_cpu[idx]  # (|mu|,) complex64 on CPU

                # rfft spectrum (0..N//2), normalized coeffs => irfft()*N
                spec = torch.zeros((N // 2 + 1,), dtype=torch.complex64)
                spec[torch.from_numpy(mu)] = c_sub.to(torch.complex64)
                x = torch.fft.irfft(spec, n=N) * float(N)
                X[r][i] = x.to(torch.float32).numpy()

            if (i + 1) % 16 == 0:
                print(f"[{i+1}/{L}] done")

    # save npz
    save_dict = {
        "Y": Y,
        "fs": np.float32(fs),
        "fc": np.float32(fc),
        "c":  np.float32(c),
        "theta": np.float32(theta),
        "N1": np.int32(N1),
        "N2": np.int32(N2),
    }
    for r in reductions:
        save_dict[f"X{r}"] = X[r]
        save_dict[f"mu{r}"] = mu_dict[r].astype(np.int32)

    np.savez_compressed(out_npz, **save_dict)
    print("saved:", out_npz, " | shapes:", {k: v.shape for k, v in save_dict.items() if hasattr(v, "shape")})

build_dataset_fdbf_gpu(
    uff_path="Alpinion_L3-8_FI_hyperechoic_scatterers.uff",
    out_npz="dataset_fdbf_energy_mu_8_9_15.npz",
    reductions=(8, 9, 15),
    N1=10, N2=10, theta=0.0, c=1540.0,
    device="cuda"
)