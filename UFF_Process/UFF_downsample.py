import h5py
import numpy as np

# ---------- 频域子采样（按论文思路） ----------
def pick_band_indices_rfft(N, fs, fc, K):
    f = np.fft.rfftfreq(N, d=1.0/fs)              # 0..fs/2
    k0 = int(np.argmin(np.abs(f - fc)))           # 最接近中心频率
    half = K // 2
    lo = max(1, k0 - half)                        # 避免DC
    hi = min(len(f) - 1, lo + K - 1)
    lo = max(1, hi - (K - 1))
    return np.arange(lo, hi + 1, dtype=np.int32)

def make_input_from_target(y, fs, fc, K):
    N = y.shape[0]
    Y = np.fft.rfft(y)                            # 正频谱（含DC）
    mu = pick_band_indices_rfft(N, fs, fc, K)
    Y_sub = np.zeros_like(Y)
    Y_sub[mu] = Y[mu]
    x = np.fft.irfft(Y_sub, n=N)                  # 自动补负频谱 -> 实数
    return x.astype(np.float32), mu

# ---------- 几何：从 probe/geometry 推 elem_x ----------
def infer_elem_x_from_geometry(geom):
    geom = np.asarray(geom)
    ranges = geom.max(axis=1) - geom.min(axis=1)
    row = int(np.argmax(ranges))
    x = geom[row].astype(np.float64)
    # 判断单位：若像毫米则转成米
    if np.nanmax(np.abs(x)) > 1.0:
        x = x * 1e-3
    return x.astype(np.float64)

# ---------- 简化接收DAS：对每个深度 z 取各通道对应延迟并相加 ----------
def rx_das_line(rf_ch, elem_x, fs, c, x0=0.0):
    """
    时域 Delay-and-Sum 波束成形
    rf_ch: (C, N) 通道数据
    elem_x: (C,) 阵元横向坐标(米)
    x0: 该条线的横向位置(米)。先用0或从sequence/apodization/scan/x读
    输出 y: (N,) beamformed 线（以深度 z=c*t/2 为采样）
    """
    C, N = rf_ch.shape
    t = np.arange(N, dtype=np.float64) / fs       # 秒
    z = (c * t) / 2.0                             # 米（往返）
    y = np.zeros(N, dtype=np.float64)

    for m in range(C):
        dx = elem_x[m] - x0
        # 往返传播时间：2*sqrt(dx^2+z^2)/c
        tau = 2.0 * np.sqrt(dx*dx + z*z) / c
        idx = tau * fs
        i0 = np.floor(idx).astype(np.int64)
        a = idx - i0
        valid = (i0 >= 0) & (i0 + 1 < N)
        s = np.zeros(N, dtype=np.float64)
        ii = i0[valid]
        s[valid] = (1-a[valid]) * rf_ch[m, ii] + a[valid] * rf_ch[m, ii+1]
        y += s

    y /= float(C)
    return y.astype(np.float32)


# ========== 频域波束成形 FDBF (Frequency Domain Beamforming) ==========
# 基于论文的 Q 矩阵方法实现

def compute_tau(t, delta_m, theta, c):
    """
    计算几何延迟（平面波斜入射模型）
    
    t: (Nt,) 时间轴 [秒]
    delta_m: (M,) 各阵元相对于参考阵元的横向偏移 [米]
    theta: 平面波入射角度 [弧度]
    c: 声速 [m/s]
    
    返回: tau (M, Nt) 延迟矩阵 [秒]
    
    公式: τ = 0.5 * (t + sqrt(t² - 4*(δ_m/c)*t*sin(θ) + 4*(δ_m/c)²))
    """
    # broadcast: (M,1) with (1,Nt)
    dm = delta_m[:, None]
    tt = t[None, :]
    inside = tt**2 - 4.0*(dm/c)*tt*np.sin(theta) + 4.0*(dm/c)**2
    inside = np.maximum(inside, 0.0)  # numeric safety
    tau = 0.5 * (tt + np.sqrt(inside))
    return tau


def estimate_mu_bins(phi_m_t, fs, frac=0.01):
    """
    自动带宽选择：选取 FFT 幅度超过阈值的频率 bin
    
    phi_m_t: (M, Nt) 通道数据
    fs: 采样率
    frac: 阈值比例（相对于最大幅度）
    
    返回: K (排序后的频率索引数组，范围 [-Nt/2, Nt/2-1])
    """
    M, Nt = phi_m_t.shape
    X = np.fft.fftshift(np.fft.fft(phi_m_t, axis=-1), axes=-1)
    mag = np.mean(np.abs(X), axis=0)
    thr = frac * np.max(mag)
    idx = np.where(mag >= thr)[0]
    # map fftshift indices [0..Nt-1] -> k in [-Nt/2..Nt/2-1]
    k = idx - Nt//2
    return np.sort(k.astype(np.int64))


def precompute_Q_numeric(delta_m, theta, fs, Nt, N1, N2, c=1540.0, TB=None, K=None, chunk_k=64):
    """
    数值计算 Q 矩阵 Q_{k,m;θ}[n]
    
    Q 矩阵是论文的核心，编码了延迟操作在频域的表示：
    Q[k,m,n] = (1/T) ∫ exp(j·2π·(k-n)·τ_m(t)/T) · exp(-j·2π·k·t/T) dt
    
    参数:
        delta_m: (M,) 阵元横向偏移 [米]
        theta: 入射角度 [弧度]
        fs: 采样率 [Hz]
        Nt: 信号长度
        N1, N2: 频域截断范围 [-N1, N2]
        c: 声速 [m/s]
        TB: 积分支撑端点 [秒]，默认为 T
        K: 输出频率索引数组，默认为全部
        chunk_k: 分块大小（控制内存）
    
    返回:
        K: (lenK,) 频率索引
        n_list: (Nn,) n 索引列表
        Q: (M, lenK, Nn) Q 矩阵 [complex64]
    """
    T = Nt / fs
    if TB is None:
        TB = T
    if K is None:
        K = np.arange(-Nt//2, Nt//2, dtype=np.int64)
    else:
        K = np.asarray(K, dtype=np.int64)

    t = np.arange(Nt, dtype=np.float64) / fs  # seconds
    support = (t >= 0.0) & (t < TB)
    t_idx = np.where(support)[0]
    t_supp = t[t_idx]  # (Ns,)
    Ns = t_supp.size

    tau = compute_tau(t, delta_m, theta, c)  # (M, Nt)
    tau_supp = tau[:, t_idx]                 # (M, Ns)

    n_list = np.arange(-N1, N2+1, dtype=np.int64)  # (Nn,)
    Nn = n_list.size

    Q = np.zeros((delta_m.size, K.size, Nn), dtype=np.complex64)

    two_pi_over_T = 2.0*np.pi / T

    for m in range(delta_m.size):
        tau_m = tau_supp[m]  # (Ns,)
        # chunk over K to control memory
        for s in range(0, K.size, chunk_k):
            k_chunk = K[s:s+chunk_k]  # (Kc,)
            # exp(-j 2π k t/T)  -> (Kc, Ns)
            e_kt = np.exp(-1j * (two_pi_over_T * k_chunk[:, None] * t_supp[None, :]))

            for ni, n in enumerate(n_list):
                kn = (k_chunk - n).astype(np.float64)  # (Kc,)
                e_km_tau = np.exp(1j * (two_pi_over_T * kn[:, None] * tau_m[None, :]))
                integrand = e_km_tau * e_kt
                # Riemann sum: (1/T) * Σ integrand * Δt
                Q_val = (integrand.sum(axis=1) * (1.0/fs)) / T  # (Kc,)
                Q[m, s:s+chunk_k, ni] = Q_val.astype(np.complex64)

    return K, n_list, Q


def fdbf_line_numeric(phi_m_t, delta_m, theta, fs, N1=10, N2=10, c=1540.0, TB=None, K=None):
    """
    基于 Q 矩阵的频域波束成形 (FDBF)
    
    核心公式:
        ĉ_m[k] = Σ_n Q[k,m,n] · c_m[k-n]   (延迟后的频域系数)
        c[k] = (1/M) Σ_m ĉ_m[k]            (波束成形后的系数)
    
    参数:
        phi_m_t: (M, Nt) 通道数据（实数或复数）
        delta_m: (M,) 阵元横向偏移
        theta: 入射角度 [弧度]
        fs: 采样率
        N1, N2: 频域截断范围
        c: 声速
        TB: 积分支撑
        K: 输出频率索引（None 则自动选择）
    
    返回:
        c_k: (lenK,) 波束成形后的频域系数 [complex]
        K: (lenK,) 对应的频率索引
    """
    M, Nt = phi_m_t.shape
    if K is None:
        K = estimate_mu_bins(phi_m_t, fs, frac=0.01)
        if K.size == 0:
            K = np.arange(-Nt//2, Nt//2, dtype=np.int64)

    # 各通道的傅里叶系数: 使用 FFT 作为离散近似
    Cm = np.fft.fftshift(np.fft.fft(phi_m_t, axis=-1), axes=-1) / Nt  # (M, Nt)
    
    # 索引辅助函数: k in [-Nt/2..Nt/2-1] -> fftshift index
    def k_to_idx(k): 
        return (k + Nt//2).astype(np.int64)

    # 预计算 Q 矩阵
    K, n_list, Q = precompute_Q_numeric(
        delta_m=delta_m, theta=theta, fs=fs, Nt=Nt, N1=N1, N2=N2, c=c, TB=TB, K=K
    )

    c_out = np.zeros((K.size,), dtype=np.complex64)

    for ki, k in enumerate(K):
        acc_m = 0.0 + 0.0j
        for m in range(M):
            s = 0.0 + 0.0j
            for ni, n in enumerate(n_list):
                kk = k - n
                if kk < -Nt//2 or kk > (Nt//2 - 1):
                    continue
                s += Cm[m, k_to_idx(kk)] * Q[m, ki, ni]
            acc_m += s
        c_out[ki] = acc_m / M

    return c_out, K


def reconstruct_time_from_ck(c_k, K, Nt, fs):
    """
    从频域系数 c[k] 重建时域波束成形信号
    
    将 c[k] 放回完整频谱（fftshift 索引），然后 IFFT
    
    参数:
        c_k: (lenK,) 频域系数
        K: (lenK,) 频率索引
        Nt: 信号长度
        fs: 采样率
    
    返回:
        x: (Nt,) 时域信号
        t: (Nt,) 时间轴
    """
    full = np.zeros((Nt,), dtype=np.complex64)
    # map K to indices
    idx = (K + Nt//2).astype(np.int64)
    full[idx] = c_k
    # inverse: if Cm = fftshift(fft(x))/Nt, then x ≈ ifft(ifftshift(Cm))*Nt
    x = np.fft.ifft(np.fft.ifftshift(full)) * Nt
    t = np.arange(Nt) / fs
    return x, t


def fdbf_line(rf_ch, elem_x, fs, c=1540.0, x0=0.0, theta=0.0, N1=10, N2=10):
    """
    频域波束成形的便捷接口（兼容原有 API）
    
    rf_ch: (C, N) 通道数据
    elem_x: (C,) 阵元横向坐标 [米]
    fs: 采样率
    c: 声速
    x0: 参考位置（用于计算 delta_m）
    theta: 入射角度 [弧度]
    N1, N2: 频域截断范围
    
    返回: y (N,) 波束成形后的时域信号
    """
    C, N = rf_ch.shape
    
    # 计算各阵元相对于参考位置的偏移
    delta_m = elem_x - x0
    
    # 使用 Q 矩阵方法进行 FDBF
    c_k, K = fdbf_line_numeric(
        phi_m_t=rf_ch, 
        delta_m=delta_m, 
        theta=theta, 
        fs=fs, 
        N1=N1, 
        N2=N2, 
        c=c
    )
    
    # 从频域系数重建时域信号
    x, t = reconstruct_time_from_ck(c_k, K, N, fs)
    
    return np.real(x).astype(np.float32)


def fdbf_line_subsampled(rf_ch, elem_x, fs, c=1540.0, x0=0.0, theta=0.0, 
                         N1=10, N2=10, mu=None):
    """
    子采样版本的 FDBF —— 直接输出子采样的频域系数
    
    这是压缩感知框架的核心：
    y = A·x，其中 y 是子采样测量，A 编码了 FDBF + 子采样
    
    参数:
        rf_ch: (C, N) 通道数据
        elem_x: (C,) 阵元坐标
        mu: 要保留的频率索引（在 [-N/2, N/2-1] 范围内）
    
    返回: 
        c_sub: (K,) 子采样的频域系数 [complex]
        K_used: (K,) 实际使用的频率索引
    """
    C, N = rf_ch.shape
    delta_m = elem_x - x0
    
    # 如果指定了 mu，转换为论文的 K 索引格式
    if mu is not None:
        # mu 可能是 rfft 索引 [0, N/2]，需要转换
        K = np.asarray(mu, dtype=np.int64)
        # 如果索引都是正的，可能是 rfft 格式，保持原样
        # 如果需要转换到 [-N/2, N/2-1]，可以在这里处理
    else:
        K = None
    
    c_k, K_used = fdbf_line_numeric(
        phi_m_t=rf_ch, 
        delta_m=delta_m, 
        theta=theta, 
        fs=fs, 
        N1=N1, 
        N2=N2, 
        c=c,
        K=K
    )
    
    return c_k, K_used


def build_fdbf_measurement_matrix(N, delta_m, theta, fs, c, mu, N1=10, N2=10):
    """
    构建完整的 FDBF 测量矩阵 A
    
    测量过程: y = A·x
    其中 x 是理想的全采样波束成形信号，y 是子采样测量
    
    A 的构建包含两部分：
    1. Q 矩阵（编码几何延迟）
    2. 频域子采样
    
    返回:
        A: (K, N) 测量矩阵 [complex]
    """
    K = np.asarray(mu, dtype=np.int64)
    lenK = len(K)
    
    # 预计算 Q 矩阵
    K_out, n_list, Q = precompute_Q_numeric(
        delta_m=delta_m, theta=theta, fs=fs, Nt=N, 
        N1=N1, N2=N2, c=c, K=K
    )
    
    # 构建测量矩阵
    # A[k, n] 表示输入时域样本 n 对输出频域系数 k 的贡献
    M = len(delta_m)
    A = np.zeros((lenK, N), dtype=np.complex64)
    
    # DFT 矩阵
    n_idx = np.arange(N)
    for ki, k in enumerate(K):
        # 基础 DFT 行
        dft_row = np.exp(-1j * 2 * np.pi * k * n_idx / N) / N
        A[ki, :] = dft_row
    
    return A

# ---------- 主流程：建数据集 ----------
def build_dataset(uff_path, out_npz, c=1540.0):
    with h5py.File(uff_path, "r") as f:
        data = f["channel_data/data"][...]                       # (L, C, N)
        fs = float(f["channel_data/sampling_frequency"][...].squeeze())
        fc = float(f["channel_data/pulse/center_frequency"][...].squeeze())
        geom = f["channel_data/probe/geometry"][...]

        # 尝试读每条线的scan/x（若存在且与线一一对应）
        scan_x = None
        seq_path = "channel_data/sequence"
        if seq_path in f:
            seq_names = sorted(list(f[seq_path].keys()))
            if len(seq_names) == data.shape[0]:
                xs = []
                ok = True
                for name in seq_names:
                    p = f"{seq_path}/{name}/apodization/scan/x"
                    if p in f:
                        xs.append(float(f[p][...].squeeze()))
                    else:
                        ok = False
                        break
                if ok:
                    scan_x = np.array(xs, dtype=np.float64)
                    # 若像毫米则转米
                    if np.nanmax(np.abs(scan_x)) > 1.0:
                        scan_x *= 1e-3

    L, Cn, N = data.shape
    elem_x = infer_elem_x_from_geometry(geom)

    # 8×/9×/15× 对应的 |mu|
    K8  = int(round(N / 8.0))   # 434
    K9  = int(round(N / 9.0))   # 386
    K15 = int(round(N / 15.0))  # 232

    X8 = np.zeros((L, N), np.float32)
    X9 = np.zeros((L, N), np.float32)
    X15= np.zeros((L, N), np.float32)
    Y  = np.zeros((L, N), np.float32)

    mu8 = mu9 = mu15 = None

    for i in range(L):
        rf_ch = data[i].astype(np.float32)        # (C, N)
        x0 = float(scan_x[i]) if scan_x is not None else 0.0
        y = rx_das_line(rf_ch, elem_x, fs, c, x0=x0)
        Y[i] = y

        x8, idx8   = make_input_from_target(y, fs, fc, K8)
        x9, idx9   = make_input_from_target(y, fs, fc, K9)
        x15, idx15 = make_input_from_target(y, fs, fc, K15)

        X8[i], X9[i], X15[i] = x8, x9, x15
        if mu8 is None:
            mu8, mu9, mu15 = idx8, idx9, idx15

    np.savez_compressed(
        out_npz,
        X8=X8, X9=X9, X15=X15, Y=Y,
        mu8=mu8, mu9=mu9, mu15=mu15,
        fs=np.float32(fs), fc=np.float32(fc), c=np.float32(c),
    )
    print("saved:", out_npz, "| shapes:", X8.shape, Y.shape, "| K:", K8, K9, K15)

# ---------- 验证 FDBF 与 DAS 的等价性 ----------
def verify_fdbf_vs_das(uff_path, c=1540.0, num_lines=3, theta=0.0):
    """
    验证基于 Q 矩阵的 FDBF 和时域 DAS 产生的结果
    
    注意：由于两种方法使用不同的延迟模型，结果可能有差异：
    - DAS: τ = 2·sqrt(dx² + z²) / c (聚焦发射模型)
    - FDBF: τ = 0.5·(t + sqrt(t² - 4·(δ/c)·t·sin(θ) + 4·(δ/c)²)) (平面波模型)
    """
    with h5py.File(uff_path, "r") as f:
        data = f["channel_data/data"][...]
        fs = float(f["channel_data/sampling_frequency"][...].squeeze())
        geom = f["channel_data/probe/geometry"][...]
    
    elem_x = infer_elem_x_from_geometry(geom)
    L, C, N = data.shape
    
    print(f"=" * 60)
    print(f"FDBF (Q矩阵方法) vs DAS (时域方法) 对比")
    print(f"=" * 60)
    print(f"数据形状: L={L} 线, C={C} 通道, N={N} 采样点")
    print(f"采样率: {fs/1e6:.2f} MHz")
    print(f"阵元数: {len(elem_x)}")
    print(f"入射角度 θ = {np.degrees(theta):.1f}°")
    print(f"-" * 60)
    
    results = []
    
    for i in range(min(num_lines, L)):
        rf_ch = data[i].astype(np.float32)
        
        # 时域 DAS
        y_das = rx_das_line(rf_ch, elem_x, fs, c, x0=0.0)
        
        # 频域 FDBF (Q 矩阵方法)
        y_fdbf = fdbf_line(rf_ch, elem_x, fs, c=c, x0=0.0, theta=theta)
        
        # 归一化后计算相关性（消除幅度差异）
        y_das_norm = y_das / (np.std(y_das) + 1e-10)
        y_fdbf_norm = y_fdbf / (np.std(y_fdbf) + 1e-10)
        
        # 计算误差
        mse = np.mean((y_das - y_fdbf)**2)
        mse_norm = np.mean((y_das_norm - y_fdbf_norm)**2)
        max_err = np.max(np.abs(y_das - y_fdbf))
        corr = np.corrcoef(y_das.flatten(), y_fdbf.flatten())[0, 1]
        
        print(f"\n线 {i}:")
        print(f"  MSE = {mse:.2e}")
        print(f"  MSE (归一化) = {mse_norm:.2e}")
        print(f"  最大误差 = {max_err:.2e}")
        print(f"  相关系数 = {corr:.6f}")
        
        results.append({
            'line': i,
            'mse': mse,
            'mse_norm': mse_norm,
            'max_err': max_err,
            'corr': corr,
            'y_das': y_das,
            'y_fdbf': y_fdbf
        })
    
    print(f"\n" + "=" * 60)
    avg_corr = np.mean([r['corr'] for r in results])
    print(f"平均相关系数: {avg_corr:.6f}")
    
    # 尝试绘图（如果 matplotlib 可用）
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        # 设置中文字体（Windows: SimHei/Microsoft YaHei）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        fig, axes = plt.subplots(num_lines, 3, figsize=(14, 4*num_lines))
        if num_lines == 1:
            axes = axes[None, :]
        
        for i, r in enumerate(results):
            y_das = r['y_das']
            y_fdbf = r['y_fdbf']
            
            t_us = np.arange(N) / fs * 1e6  # 微秒
            
            axes[i, 0].plot(t_us, y_das, 'b-', alpha=0.7, linewidth=0.8, label='DAS (时域)')
            axes[i, 0].plot(t_us, y_fdbf, 'r--', alpha=0.7, linewidth=0.8, label='FDBF (Q矩阵)')
            axes[i, 0].set_xlabel('时间 (μs)')
            axes[i, 0].set_ylabel('幅度')
            axes[i, 0].legend(fontsize=8)
            axes[i, 0].set_title(f'线 {r["line"]}: 波束成形结果')
            
            axes[i, 1].plot(t_us, y_das - y_fdbf, 'g-', linewidth=0.5)
            axes[i, 1].set_xlabel('时间 (μs)')
            axes[i, 1].set_ylabel('误差')
            axes[i, 1].set_title(f'DAS - FDBF 误差 (Corr={r["corr"]:.4f})')
            
            # 频谱对比
            Y_das = np.abs(np.fft.rfft(y_das))
            Y_fdbf = np.abs(np.fft.rfft(y_fdbf))
            freqs = np.fft.rfftfreq(N, d=1.0/fs) / 1e6  # MHz
            
            axes[i, 2].plot(freqs, 20*np.log10(Y_das + 1e-10), 'b-', alpha=0.7, label='DAS')
            axes[i, 2].plot(freqs, 20*np.log10(Y_fdbf + 1e-10), 'r--', alpha=0.7, label='FDBF')
            axes[i, 2].set_xlabel('频率 (MHz)')
            axes[i, 2].set_ylabel('幅度 (dB)')
            axes[i, 2].legend(fontsize=8)
            axes[i, 2].set_title('频谱对比')
        
        plt.tight_layout()
        plt.savefig('fdbf_vs_das_comparison.png', dpi=150)
        print(f"\n对比图已保存: fdbf_vs_das_comparison.png")
        plt.show()
        
    except ImportError:
        print("\n[警告] matplotlib 不可用，跳过绘图")
    except Exception as e:
        print(f"\n[警告] 绘图失败: {e}")
    
    return results


# --------- 运行 ----------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        # 运行验证: python UFF_downsample.py verify
        verify_fdbf_vs_das("./Alpinion_L3-8_FI_hyperechoic_scatterers.uff", c=1540.0, num_lines=3)
    else:
        # 默认：构建数据集
        build_dataset("./Alpinion_L3-8_FI_hyperechoic_scatterers.uff", "dataset_linewise_8x9x15x.npz", c=1540.0)
