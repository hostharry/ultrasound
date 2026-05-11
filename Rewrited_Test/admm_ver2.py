import numpy as np
import pywt
from scipy.sparse.linalg import cg, LinearOperator
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter

# -----------------------------
# A: 超声测量算子（partial rFFT on mu）
# 与你的 downsample 逻辑一致：
#   Y_k = rfft(y)[mu]              （本数据集里保存的 Y{r}_k 就是这个）
#   x_time = irfft(spec, n=N)*N
# -----------------------------
def A_forward(x, mu):
    """
    x: (L,N) real
    return: (L,K) complex, K=len(mu)
    """
    N = x.shape[1]
    X = np.fft.rfft(x, axis=1)        # (L, N//2+1)
    return X[:, mu]

def A_adjoint(b, mu, N):
    """
    b: (L,K) complex
    return: (L,N) real
    """
    L, K = b.shape
    full = np.zeros((L, N//2 + 1), dtype=np.complex64)
    full[:, mu] = b
    x = np.fft.irfft(full, n=N, axis=1) 
    return x.astype(np.float32)

def make_mask_full(mu, n_rfft):
    mask = np.zeros((n_rfft,), dtype=np.float32)
    mask[mu] = 1.0
    return mask

def AtA_apply(x, mask_full):
    """
    x: (L,N) real
    mask_full: (N//2+1,) float, 0/1
    return: (L,N) real
    """
    N = x.shape[1]
    X = np.fft.rfft(x, axis=1) 
    X = X * mask_full[None, :]
    out = np.fft.irfft(X, n=N, axis=1) 
    return out.astype(np.float32)

# -----------------------------
# 2D 梯度/散度（Neumann 边界，保证 adjoint）
# x shape: (L,N)  (line, depth)
# -----------------------------
def grad2d_forward(x):
    L, N = x.shape
    gx = np.zeros_like(x, dtype=np.float32)  # depth 方向
    gy = np.zeros_like(x, dtype=np.float32)  # line 方向
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    return gx, gy

def div2d_adjoint(px, py):
    # enforce boundary components that are not in range to be zero
    px = px.copy()
    py = py.copy()
    px[:, -1] = 0.0
    py[-1, :] = 0.0

    div = np.zeros_like(px, dtype=np.float32)
    div[:, 0] = px[:, 0]
    div[:, 1:] = px[:, 1:] - px[:, :-1]
    div[0, :] += py[0, :]
    div[1:, :] += py[1:, :] - py[:-1, :]
    return -div

def isotropic_shrink(gx, gy, thresh, eps=1e-12):
    """
    每个像素对 (gx,gy) 做向量软阈值（isotropic TV）
    thresh: (L,N) 或标量
    """
    mag = np.sqrt(gx * gx + gy * gy) + eps
    scale = np.maximum(0.0, 1.0 - (thresh / mag))
    return gx * scale, gy * scale

def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

# -----------------------------
# 2D Wavelet 分析/合成（periodization 近似正交，尺寸更稳定）
# -----------------------------
def wav_analysis2d(x, wavelet="db4", level=3, mode="periodization"):
    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level, mode=mode)
    arr, slices = pywt.coeffs_to_array(coeffs)
    return arr.astype(np.float32), slices

def wav_synthesis2d(arr, slices, shape, wavelet="db4", level=3, mode="periodization"):
    coeffs = pywt.array_to_coeffs(arr, slices, output_format="wavedec2")
    x = pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)
    # 防止边界模式导致的尺寸偏差
    return x[:shape[0], :shape[1]].astype(np.float32)

# -----------------------------
# B/C: 超声权重（从 log-envelope 上算）
# - W_tv：边缘大 -> 权重小（TV弱，保边）
# - wav_scale：熵大 -> 阈值小（L1弱，保纹理）
#   这里用 “块熵” 快速近似（比滑窗熵快很多）
# -----------------------------
def robust_norm01(x, p_lo=1.0, p_hi=99.0, eps=1e-6):
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    y = (x - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0)

def block_entropy_map(img01, block=(16, 64), bins=32):
    """
    img01: (L,N) in [0,1]
    返回同尺寸的块熵图（每块一个熵值）
    """
    L, N = img01.shape
    bh, bw = block
    Hn = (L + bh - 1) // bh
    Wn = (N + bw - 1) // bw

    q = np.floor(img01 * (bins - 1)).astype(np.int32)

    ent_blocks = np.zeros((Hn, Wn), dtype=np.float32)
    for i in range(Hn):
        for j in range(Wn):
            r0, r1 = i * bh, min((i + 1) * bh, L)
            c0, c1 = j * bw, min((j + 1) * bw, N)
            patch = q[r0:r1, c0:c1].ravel()
            hist = np.bincount(patch, minlength=bins).astype(np.float32)
            p = hist / (hist.sum() + 1e-12)
            p = p[p > 0]
            ent = -np.sum(p * np.log(p + 1e-12))
            ent_blocks[i, j] = ent

    # upsample by repeat
    ent = np.repeat(np.repeat(ent_blocks, bh, axis=0), bw, axis=1)
    return ent[:L, :N]

def compute_weights_ultrasound(x, tv_k=8.0, tv_baseline=0.05,
                              ent_k=6.0, ent_baseline=0.05,
                              smooth_sigma=1.0,
                              ent_block=(16, 64), ent_bins=32):
    """
    x: (L,N) time-domain beamformed lines
    return:
      W_tv: (L,N) positive, mean~1
      wav_scale: scalar in (baseline, ~baseline+1), entropy大 -> 更小
    """
    # 包络（沿 depth/time 轴）
    env = np.abs(hilbert(x, axis=1)).astype(np.float32)
    log_env = np.log1p(env).astype(np.float32)
    if smooth_sigma > 0:
        log_env_s = gaussian_filter(log_env, sigma=smooth_sigma)
    else:
        log_env_s = log_env

    # --- TV 权重（边缘大 -> 权重小）---
    gx, gy = grad2d_forward(log_env_s)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    g01 = robust_norm01(grad_mag)
    W_tv = np.exp(-tv_k * g01) + tv_baseline
    W_tv = W_tv / (W_tv.mean() + 1e-6)

    # --- 熵（纹理大 -> 阈值小）---
    img01 = robust_norm01(log_env_s)
    ent = block_entropy_map(img01, block=ent_block, bins=ent_bins)
    ent01 = robust_norm01(ent)
    W_ent = np.exp(-ent_k * ent01) + ent_baseline   # 熵大 -> W小
    wav_scale = float(W_ent.mean())                 # 用标量阈值（稳定、对齐“方向”）

    return W_tv.astype(np.float32), wav_scale

# -----------------------------
# ADMM 主函数（联合重建 256×3474）
# -----------------------------
def admm_ultrasound_joint(
    b, mu, x0=None, max_iter=60, cg_iter=30,
    lambda_tv=0.02, lambda_wav=0.005,
    rho1=0.5, rho2=0.5,
    wavelet="db4", level=3,
    weight_update_interval=5,
    verbose=True
):
    """
    b: (L,K) complex64  -> 用 Y8_k / Y9_k / Y15_k
    mu: (K,) int        -> 用 mu8 / mu9 / mu15
    x0: (L,N) float32   -> 用 X8 / X9 / X15 或 None
    """
    L, K = b.shape
    # N 从 rFFT 长度推回
    # 你的 N=3474 -> n_rfft=1738
    # 这里用 x0 取 N；若没有则用 A* b 取 N（需要用户提供 N）
    if x0 is None:
        raise ValueError("x0 不能为空（建议直接用 npz['X8'] 之类），因为需要 N=3474。")
    x = x0.astype(np.float32)
    N = x.shape[1]
    n_rfft = N // 2 + 1

    mask_full = make_mask_full(mu, n_rfft)

    # A^T b（物理一致的反投影）
    Atb = A_adjoint(b.astype(np.complex64), mu, N)  # (L,N)

    # 初始化权重
    W_tv, wav_scale = compute_weights_ultrasound(x)

    # 初始化 wavelet 结构（slices）
    coeff_arr, coeff_slices = wav_analysis2d(x, wavelet=wavelet, level=level)
    z2 = coeff_arr.copy()
    u2 = np.zeros_like(z2, dtype=np.float32)

    # TV 分裂变量
    gx, gy = grad2d_forward(x)
    z1x, z1y = gx.copy(), gy.copy()
    u1x = np.zeros_like(z1x, dtype=np.float32)
    u1y = np.zeros_like(z1y, dtype=np.float32)

    hist = {"obj": [], "data": [], "tv": [], "wav": [], "r1": [], "r2": []}

    # CG 线性算子： (AtA + rho1*DtD + rho2*I)
    def matvec(v_flat):
        v = v_flat.reshape(L, N).astype(np.float32)
        out = AtA_apply(v, mask_full)
        vx, vy = grad2d_forward(v)
        out += rho1 * div2d_adjoint(vx, vy)
        out += rho2 * v
        return out.ravel()

    Aop = LinearOperator((L * N, L * N), matvec=matvec, dtype=np.float32)

    for it in range(max_iter):
        # 周期更新权重（HASA 思路）
        if (it % weight_update_interval == 0) and (it > 0):
            W_tv, wav_scale = compute_weights_ultrasound(x)

        # ---- x-update (CG) ----
        # rhs = Atb + rho1*div(z1-u1) + rho2*Psi^T(z2-u2)
        div_term = div2d_adjoint(z1x - u1x, z1y - u1y)
        wav_back = wav_synthesis2d(z2 - u2, coeff_slices, (L, N), wavelet=wavelet, level=level)
        rhs = Atb + rho1 * div_term + rho2 * wav_back

        x_flat, info = cg(Aop, rhs.ravel(), x0=x.ravel(), maxiter=cg_iter)
        x = x_flat.reshape(L, N).astype(np.float32)

        # ---- z1-update (isotropic TV) ----
        gx, gy = grad2d_forward(x)
        v1x = gx + u1x
        v1y = gy + u1y
        thresh_tv = (lambda_tv * W_tv) / rho1
        z1x_new, z1y_new = isotropic_shrink(v1x, v1y, thresh_tv)
        # dual
        u1x += gx - z1x_new
        u1y += gy - z1y_new
        z1x, z1y = z1x_new, z1y_new

        # ---- z2-update (wavelet soft-threshold, entropy-inverted scalar) ----
        coeff_arr, _ = wav_analysis2d(x, wavelet=wavelet, level=level)
        v2 = coeff_arr + u2
        thresh_wav = (lambda_wav * wav_scale) / rho2
        z2_new = soft_threshold(v2, thresh_wav)
        u2 += coeff_arr - z2_new
        z2 = z2_new

        # ---- 记录指标 ----
        Ax = A_forward(x, mu)
        data = 0.5 * np.sum(np.abs(Ax - b) ** 2)

        gx2, gy2 = grad2d_forward(x)
        tv = lambda_tv * np.sum(W_tv * np.sqrt(gx2 * gx2 + gy2 * gy2))

        wav = (lambda_wav * wav_scale) * np.sum(np.abs(coeff_arr))

        obj = float(data + tv + wav)

        r1 = float(np.sqrt(np.sum((gx - z1x) ** 2) + np.sum((gy - z1y) ** 2)))
        r2 = float(np.sqrt(np.sum((coeff_arr - z2) ** 2)))

        hist["obj"].append(obj)
        hist["data"].append(float(data))
        hist["tv"].append(float(tv))
        hist["wav"].append(float(wav))
        hist["r1"].append(r1)
        hist["r2"].append(r2)

        if verbose and (it % 10 == 0 or it == max_iter - 1):
            print(f"[it {it:03d}] obj={obj:.4e}  data={data:.3e}  tv={tv:.3e}  wav={wav:.3e}  wav_scale={wav_scale:.3f}")

    return x, hist


# -----------------------------
# 一些评估函数（可选）
# -----------------------------
def mse(a, b):
    return float(np.mean((a - b) ** 2))

def psnr(a, b, data_range=None, eps=1e-12):
    if data_range is None:
        data_range = float(np.max(b) - np.min(b))
        if data_range < eps:
            data_range = float(np.max(np.abs(b)) + eps)
    m = mse(a, b)
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(m + eps)

def snr(a, b, eps=1e-12):
    # 以 b 为信号，(a-b) 为噪声
    sig = np.mean(b ** 2)
    noi = np.mean((a - b) ** 2)
    return 10.0 * np.log10((sig + eps) / (noi + eps))


if __name__ == "__main__":

    # 例：加载你的 npz（把路径换成你的实际文件）
    npz = np.load("/home/user/毕业设计/Ultrasound/dataset_fdbf_energy_mu_8_9_15.npz", allow_pickle=True)

    x0 = npz["X8"]                     # (256,3474) subsampled/aliased time-domain input
    yk = npz["Y8_k"]                   # (256,416) complex64, defined as rfft(Y)[mu8]
    mu = npz["mu8"].astype(np.int64)   # (416,) rfft bin indices
    gt = npz["Y"]                      # (256,3474) time-domain DAS target

    # 一致性 sanity check（关键）：数据集定义应满足  A(Y)=Y8_k
    chk_gt = A_forward(gt, mu)
    rel_err_gt = np.linalg.norm(chk_gt - yk) / (np.linalg.norm(yk) + 1e-12)
    per = (np.linalg.norm(chk_gt - yk, axis=1) / (np.linalg.norm(yk, axis=1) + 1e-12)).astype(np.float64)
    print("sanity(data): ||A(Y)-Y8_k|| / ||Y8_k|| =", rel_err_gt)
    print("sanity(data) per-line: mean/median/max =", float(per.mean()), float(np.median(per)), float(per.max()))

    # 诊断：X8 不是 Y 的观测（所以 A(X8) 一般不会等于 Y8_k）
    chk_x0 = A_forward(x0, mu)
    rel_err_x0 = np.linalg.norm(chk_x0 - yk) / (np.linalg.norm(yk) + 1e-12)
    print("diagnostic: ||A(X8)-Y8_k|| / ||Y8_k|| =", rel_err_x0)

    x_rec, hist = admm_ultrasound_joint(
        b=yk, mu=mu, x0=x0,
        max_iter=500, cg_iter=30,
        lambda_tv=0.02, lambda_wav=0.005,
        rho1=0.5, rho2=0.5,
        weight_update_interval=5,
        verbose=True
    )

    print("PSNR:", psnr(x_rec, gt))
    print("SNR :", snr(x_rec, gt))
