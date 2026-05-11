"""
双正则化 ADMM 求解器：1D 超声压缩感知重建
==========================================

针对 dataset_fdbf_energy_mu_8_9_15.npz 数据集

优化问题：
    argmin(x) 0.5*||Ax-y||^2 + lambda_tv*||W_tv*D(x)||_1 + lambda_wav*||W_wav*Psi(x)||_1

其中：
    - A: 频域子采样测量算子
    - y: 子采样测量值
    - D: 1D 差分算子
    - Psi: 1D 小波变换
    - W_tv: 基于局部梯度的 TV 权重
    - W_wav: 基于局部熵的小波权重
"""

import numpy as np
import os
import time
import pywt
from numba import jit
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    'npz_path': '../dataset_fdbf_energy_mu_8_9_15.npz',
    'cs_ratio': 9,              # 8, 9, 或 15
    
    # ADMM 参数
    'lambda_tv': 0.001,         # TV 正则化强度
    'lambda_wav': 0.001,        # 小波正则化强度
    'rho1': 0.1,                # TV 惩罚参数
    'rho2': 0.1,                # 小波惩罚参数
    'max_iter': 50,             # 最大迭代次数
    'tol': 1e-5,                # 收敛容差
    
    # 小波参数
    'wavelet': 'db4',
    'wavelet_level': 4,
    
    # 权重图参数
    'entropy_window_size': 32,
    'weight_baseline': 0.1,
    
    # 测试参数
    'num_test_lines': 50,       # -1 表示全部
    'output_dir': 'outputs_admm_1d',
}


# ============================================================================
# 数据加载
# ============================================================================

def load_ultrasound_dataset(npz_path, cs_ratio=9):
    print(f"Loading dataset from {npz_path} ...")
    data = np.load(npz_path)
    
    if cs_ratio == 8:
        X = data['X8'].astype(np.float64)
        mu = data['mu8']
    elif cs_ratio == 9:
        X = data['X9'].astype(np.float64)
        mu = data['mu9']
    elif cs_ratio == 15:
        X = data['X15'].astype(np.float64)
        mu = data['mu15']
    else:
        raise ValueError(f"Unsupported cs_ratio: {cs_ratio}")
    
    Y = data['Y'].astype(np.float64)
    fs = float(data['fs'])
    fc = float(data['fc'])
    c = float(data['c']) if 'c' in data else 1540.0
    
    L, N = Y.shape
    print(f"  Loaded: L={L} lines, N={N} samples, cs_ratio={cs_ratio}x")
    
    return X, Y, mu, fs, fc, c


# ============================================================================
# 频域测量算子
# ============================================================================

class FrequencyMaskOperator:
    """频域子采样测量算子"""
    def __init__(self, N, mu):
        self.N = N
        self.mu = mu
        self.M = len(mu)
        self.rfft_len = N // 2 + 1
        self.mask = np.zeros(self.rfft_len, dtype=np.float64)
        self.mask[mu] = 1.0
    
    def forward(self, x):
        """A: x -> y_sub"""
        X_freq = np.fft.rfft(x)
        return X_freq[self.mu]
    
    def adjoint(self, y_sub):
        """A^T: y_sub -> x"""
        X_freq = np.zeros(self.rfft_len, dtype=np.complex128)
        X_freq[self.mu] = y_sub
        return np.fft.irfft(X_freq, n=self.N)
    
    def AtA(self, x):
        """A^T A x"""
        X_freq = np.fft.rfft(x)
        X_masked = X_freq * self.mask
        return np.fft.irfft(X_masked, n=self.N)


# ============================================================================
# 1D 差分算子
# ============================================================================

def diff_forward(x):
    """1D 前向差分: D*x"""
    return np.diff(x, append=x[0])  # 周期边界


def diff_adjoint(d):
    """1D 差分的伴随算子: D^T*d = -div(d)"""
    # d[i] = x[i+1] - x[i], 伴随: -d[i] + d[i-1]
    return -d + np.roll(d, 1)


def DtD(x):
    """D^T D x"""
    return diff_adjoint(diff_forward(x))


# ============================================================================
# 1D 小波变换
# ============================================================================

def wavelet_forward_1d(x, wavelet='db4', level=4):
    """1D 小波分解"""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # 展平为一维数组
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return coeff_arr, coeff_slices


def wavelet_inverse_1d(coeff_arr, coeff_slices, wavelet='db4'):
    """1D 小波重构"""
    coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec')
    return pywt.waverec(coeffs, wavelet)


def PsiTPsi(x, wavelet='db4', level=4):
    """Psi^T Psi x = x (正交小波)"""
    coeff_arr, coeff_slices = wavelet_forward_1d(x, wavelet, level)
    return wavelet_inverse_1d(coeff_arr, coeff_slices, wavelet)[:len(x)]


# ============================================================================
# 软阈值
# ============================================================================

def soft_threshold(x, threshold):
    """软阈值算子"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


# ============================================================================
# 1D 权重图
# ============================================================================

@jit(nopython=True)
def local_gradient_1d(x, window_size=32):
    """计算 1D 局部梯度幅度"""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)
    half = window_size // 2
    
    for i in range(n):
        i_min = max(0, i - half)
        i_max = min(n - 1, i + half)
        local_grad = 0.0
        count = 0
        for j in range(i_min, i_max):
            local_grad += abs(x[j + 1] - x[j])
            count += 1
        result[i] = local_grad / max(1, count)
    
    return result


@jit(nopython=True)
def local_entropy_1d(x, window_size=32, n_bins=64):
    """计算 1D 局部熵"""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)
    half = window_size // 2
    
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        return result
    
    x_norm = ((x - x_min) / (x_max - x_min) * (n_bins - 1)).astype(np.int32)
    x_norm = np.clip(x_norm, 0, n_bins - 1)
    
    for i in range(n):
        i_min = max(0, i - half)
        i_max = min(n, i + half + 1)
        
        hist = np.zeros(n_bins, dtype=np.int32)
        for j in range(i_min, i_max):
            hist[x_norm[j]] += 1
        
        total = i_max - i_min
        entropy = 0.0
        for k in range(n_bins):
            if hist[k] > 0:
                p = hist[k] / total
                entropy -= p * np.log2(p)
        
        result[i] = entropy
    
    return result


def create_tv_weight_1d(x, window_size=32, baseline=0.1):
    """基于局部梯度创建 TV 权重"""
    grad = local_gradient_1d(x, window_size)
    W = grad / (grad.max() + 1e-10)
    W = W + baseline
    W = W / W.mean()  # 归一化均值为 1
    return W


def create_wav_weight_1d(x, window_size=32, baseline=0.1):
    """基于局部熵创建小波权重"""
    ent = local_entropy_1d(x, window_size)
    W = ent / (ent.max() + 1e-10)
    W = W + baseline
    W = W / W.mean()
    return W


# ============================================================================
# ADMM 求解器
# ============================================================================

def admm_dual_regularizer_1d(y_sub, op, N, lambda_tv=0.001, lambda_wav=0.001,
                             rho1=0.1, rho2=0.1, max_iter=50, tol=1e-5,
                             wavelet='db4', level=4, W_tv=None, W_wav=None,
                             verbose=False):
    """
    1D 双正则化 ADMM 求解器
    
    优化问题:
        argmin(x) 0.5*||Ax-y||^2 + lambda_tv*||W_tv*Dx||_1 + lambda_wav*||W_wav*Psi(x)||_1
    
    ADMM 分解:
        x-update: (A^T A + rho1*D^T D + rho2*Psi^T Psi) x = A^T y + rho1*D^T(z1-u1) + rho2*Psi^T(z2-u2)
        z1-update: z1 = soft_threshold(Dx + u1, lambda_tv*W_tv/rho1)
        z2-update: z2 = soft_threshold(Psi(x) + u2, lambda_wav*W_wav/rho2)
        u1-update: u1 = u1 + Dx - z1
        u2-update: u2 = u2 + Psi(x) - z2
    """
    # 初始化
    x = op.adjoint(y_sub)
    
    # 获取小波系数结构
    coeff_arr_init, coeff_slices = wavelet_forward_1d(x, wavelet, level)
    n_coeffs = len(coeff_arr_init)
    
    # 辅助变量
    z1 = np.zeros(N, dtype=np.float64)       # D*x
    z2 = np.zeros(n_coeffs, dtype=np.float64)  # Psi(x)
    
    # 对偶变量
    u1 = np.zeros(N, dtype=np.float64)
    u2 = np.zeros(n_coeffs, dtype=np.float64)
    
    # 默认权重
    if W_tv is None:
        W_tv = np.ones(N, dtype=np.float64)
    if W_wav is None:
        W_wav = np.ones(n_coeffs, dtype=np.float64)
    
    # 小波域权重（简化：使用均匀权重或插值）
    if len(W_wav) != n_coeffs:
        # 简单插值到小波系数长度
        W_wav_coeff = np.interp(np.linspace(0, 1, n_coeffs),
                                np.linspace(0, 1, len(W_wav)), W_wav)
    else:
        W_wav_coeff = W_wav
    
    obj_history = []
    
    for it in range(max_iter):
        # === x-update: 使用 FFT 快速求解 ===
        # 对于频域子采样 + 周期边界差分，可以在频域高效求解
        # (A^T A + rho1*D^T D + rho2*I) x = rhs
        # 由于 Psi^T Psi = I（正交小波），简化为上式
        
        # 右侧
        rhs = op.adjoint(y_sub)
        rhs += rho1 * diff_adjoint(z1 - u1)
        
        wav_term = wavelet_inverse_1d(z2 - u2, coeff_slices, wavelet)
        rhs += rho2 * wav_term[:N]
        
        # 使用迭代法求解（简化：固定点迭代或直接近似）
        # 这里使用简化的固定点迭代
        for _ in range(10):
            x_new = rhs - rho1 * DtD(x) + rho1 * x
            x_new = x_new - op.AtA(x) + op.AtA(x_new) / (1 + rho1 + rho2)
            # 简化：直接使用
            x = (rhs + rho1 * x - rho1 * DtD(x)) / (1 + rho1 + rho2)
        
        # 更精确的方法：直接在频域求解
        # F(D^T D) = |1 - e^{-j2πk/N}|^2 = 4*sin^2(πk/N)
        # 但由于 A^T A 是部分频点的选择，这里使用迭代
        
        # 简化实现：使用 CG 或直接迭代
        # 这里使用 ISTA 风格的近似
        for inner in range(5):
            # 数据项梯度
            grad_data = op.adjoint(op.forward(x) - y_sub)
            # TV 项
            grad_tv = rho1 * diff_adjoint(diff_forward(x) - z1 + u1)
            # 小波项
            psi_x, _ = wavelet_forward_1d(x, wavelet, level)
            grad_wav = rho2 * wavelet_inverse_1d(psi_x - z2 + u2, coeff_slices, wavelet)[:N]
            
            # 更新
            x = x - 0.5 * (grad_data + grad_tv + grad_wav)
        
        # === z1-update: TV 软阈值 ===
        Dx = diff_forward(x)
        v1 = Dx + u1
        threshold_tv = (lambda_tv * W_tv) / rho1
        z1 = soft_threshold(v1, threshold_tv)
        
        # === z2-update: 小波软阈值 ===
        psi_x, _ = wavelet_forward_1d(x, wavelet, level)
        v2 = psi_x + u2
        threshold_wav = (lambda_wav * W_wav_coeff) / rho2
        z2 = soft_threshold(v2, threshold_wav)
        
        # === 对偶变量更新 ===
        u1 = u1 + Dx - z1
        u2 = u2 + psi_x - z2
        
        # 计算目标函数
        residual = op.forward(x) - y_sub
        data_term = 0.5 * np.sum(np.abs(residual) ** 2)
        tv_term = lambda_tv * np.sum(W_tv * np.abs(Dx))
        wav_term = lambda_wav * np.sum(W_wav_coeff * np.abs(psi_x))
        objective = data_term + tv_term + wav_term
        obj_history.append(objective)
        
        if verbose and it % 10 == 0:
            print(f"  Iter {it:3d}: Obj = {objective:.6f}, "
                  f"Data = {data_term:.6f}, TV = {tv_term:.6f}, Wav = {wav_term:.6f}")
        
        # 检查收敛
        if it > 0 and abs(obj_history[-1] - obj_history[-2]) / (obj_history[-2] + 1e-12) < tol:
            if verbose:
                print(f"  Converged at iteration {it}")
            break
    
    return x, obj_history


# ============================================================================
# 指标计算
# ============================================================================

def calc_snr(y_true, y_pred, eps=1e-12):
    signal_power = np.sum(y_true ** 2)
    noise_power = np.sum((y_true - y_pred) ** 2)
    return 10.0 * np.log10((signal_power + eps) / (noise_power + eps))


def calc_psnr(y_true, y_pred, eps=1e-12):
    mse = np.mean((y_true - y_pred) ** 2)
    max_val = np.max(np.abs(y_true))
    return 20.0 * np.log10((max_val + eps) / (np.sqrt(mse) + eps))


def calc_nmse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2)


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("双正则化 ADMM 求解器 - 1D 超声压缩感知重建")
    print("=" * 70)
    
    config = CONFIG
    
    # 打印配置
    print(f"\n配置参数:")
    print(f"  数据集: {config['npz_path']}")
    print(f"  压缩比: {config['cs_ratio']}x")
    print(f"  ADMM 迭代: {config['max_iter']}")
    print(f"  lambda_tv: {config['lambda_tv']}, lambda_wav: {config['lambda_wav']}")
    print(f"  rho1: {config['rho1']}, rho2: {config['rho2']}")
    
    # 1. 加载数据
    print("\n[1] 加载数据集...")
    X, Y, mu, fs, fc, c = load_ultrasound_dataset(config['npz_path'], config['cs_ratio'])
    L, N = Y.shape
    
    # 2. 创建测量算子
    print("\n[2] 创建频域测量算子...")
    op = FrequencyMaskOperator(N, mu)
    print(f"  信号长度: {N}, 保留频点: {len(mu)}")
    
    # 3. 预编译 JIT
    print("\n[3] 预编译 JIT 函数...")
    _ = local_gradient_1d(Y[0], window_size=32)
    _ = local_entropy_1d(Y[0], window_size=32)
    print("  JIT 编译完成")
    
    # 4. 测试
    num_test = config['num_test_lines'] if config['num_test_lines'] > 0 else L
    num_test = min(num_test, L)
    
    print(f"\n[4] 测试 {num_test} 条线...")
    
    snr_init = []
    snr_admm = []
    psnr_init = []
    psnr_admm = []
    time_list = []
    
    for i in range(num_test):
        y_gt = Y[i]
        x_init = X[i]
        y_sub = op.forward(y_gt)
        
        # Initial
        snr_init.append(calc_snr(y_gt, x_init))
        psnr_init.append(calc_psnr(y_gt, x_init))
        
        # 创建权重图（基于零填充初始估计）
        W_tv = create_tv_weight_1d(x_init, config['entropy_window_size'], config['weight_baseline'])
        W_wav = create_wav_weight_1d(x_init, config['entropy_window_size'], config['weight_baseline'])
        
        # ADMM 重建
        t0 = time.time()
        x_admm, obj_hist = admm_dual_regularizer_1d(
            y_sub, op, N,
            lambda_tv=config['lambda_tv'],
            lambda_wav=config['lambda_wav'],
            rho1=config['rho1'],
            rho2=config['rho2'],
            max_iter=config['max_iter'],
            tol=config['tol'],
            wavelet=config['wavelet'],
            level=config['wavelet_level'],
            W_tv=W_tv,
            W_wav=W_wav,
            verbose=False
        )
        t1 = time.time()
        
        snr_admm.append(calc_snr(y_gt, x_admm))
        psnr_admm.append(calc_psnr(y_gt, x_admm))
        time_list.append(t1 - t0)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{num_test}] Init SNR: {snr_init[-1]:.2f} dB | "
                  f"ADMM SNR: {snr_admm[-1]:.2f} dB | Time: {time_list[-1]:.2f}s")
    
    # 转为数组
    snr_init = np.array(snr_init)
    snr_admm = np.array(snr_admm)
    psnr_init = np.array(psnr_init)
    psnr_admm = np.array(psnr_admm)
    time_list = np.array(time_list)
    
    # 5. 汇总结果
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    
    print(f"\nInitial (零填充):")
    print(f"  SNR:  {snr_init.mean():.2f} ± {snr_init.std():.2f} dB")
    print(f"  PSNR: {psnr_init.mean():.2f} ± {psnr_init.std():.2f} dB")
    
    print(f"\nADMM 双正则化:")
    print(f"  SNR:  {snr_admm.mean():.2f} ± {snr_admm.std():.2f} dB")
    print(f"  PSNR: {psnr_admm.mean():.2f} ± {psnr_admm.std():.2f} dB")
    print(f"  Time: {time_list.mean():.2f} ± {time_list.std():.2f} s/line")
    
    print(f"\nSNR 提升: +{snr_admm.mean() - snr_init.mean():.2f} dB")
    
    # 6. 保存结果
    os.makedirs(config['output_dir'], exist_ok=True)
    result_file = os.path.join(config['output_dir'], f"results_admm_ratio{config['cs_ratio']}.txt")
    with open(result_file, 'w') as f:
        f.write("双正则化 ADMM 1D 超声重建结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"数据集: {config['npz_path']}\n")
        f.write(f"压缩比: {config['cs_ratio']}x\n")
        f.write(f"测试样本: {num_test}\n")
        f.write(f"ADMM 迭代: {config['max_iter']}\n\n")
        
        f.write(f"Initial:\n")
        f.write(f"  SNR:  {snr_init.mean():.2f} ± {snr_init.std():.2f} dB\n")
        f.write(f"  PSNR: {psnr_init.mean():.2f} ± {psnr_init.std():.2f} dB\n\n")
        
        f.write(f"ADMM 双正则化:\n")
        f.write(f"  SNR:  {snr_admm.mean():.2f} ± {snr_admm.std():.2f} dB\n")
        f.write(f"  PSNR: {psnr_admm.mean():.2f} ± {psnr_admm.std():.2f} dB\n\n")
        
        f.write(f"SNR 提升: +{snr_admm.mean() - snr_init.mean():.2f} dB\n")
    
    print(f"\n结果已保存: {result_file}")
    
    # 7. 可视化
    try:
        import matplotlib.pyplot as plt
        
        # 选一条线可视化
        idx = 0
        y_gt = Y[idx]
        x_init = X[idx]
        y_sub = op.forward(y_gt)
        
        W_tv = create_tv_weight_1d(x_init, config['entropy_window_size'], config['weight_baseline'])
        W_wav = create_wav_weight_1d(x_init, config['entropy_window_size'], config['weight_baseline'])
        
        x_admm, obj_hist = admm_dual_regularizer_1d(
            y_sub, op, N,
            lambda_tv=config['lambda_tv'],
            lambda_wav=config['lambda_wav'],
            rho1=config['rho1'],
            rho2=config['rho2'],
            max_iter=config['max_iter'],
            wavelet=config['wavelet'],
            level=config['wavelet_level'],
            W_tv=W_tv,
            W_wav=W_wav,
            verbose=True
        )
        
        t_us = np.arange(N) / fs * 1e6
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        axes[0].plot(t_us, y_gt, 'b-', linewidth=0.5, alpha=0.8)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'GT (Y) - Line {idx}')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t_us, y_gt, 'b-', linewidth=0.5, alpha=0.4, label='GT')
        axes[1].plot(t_us, x_init, 'r-', linewidth=0.5, alpha=0.8, label='Initial')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title(f'Initial (零填充) | SNR: {calc_snr(y_gt, x_init):.2f} dB')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(t_us, y_gt, 'b-', linewidth=0.5, alpha=0.4, label='GT')
        axes[2].plot(t_us, x_admm, 'g-', linewidth=0.5, alpha=0.8, label='ADMM')
        axes[2].set_xlabel('Time (μs)')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title(f'ADMM 双正则化 | SNR: {calc_snr(y_gt, x_admm):.2f} dB')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(config['output_dir'], f'admm_line{idx}_ratio{config["cs_ratio"]}.png')
        plt.savefig(fig_path, dpi=150)
        print(f"可视化图已保存: {fig_path}")
        plt.show()
        
        # 收敛曲线
        if obj_hist:
            plt.figure(figsize=(8, 4))
            plt.plot(obj_hist, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Objective')
            plt.title('ADMM Convergence')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            conv_path = os.path.join(config['output_dir'], f'convergence_ratio{config["cs_ratio"]}.png')
            plt.savefig(conv_path, dpi=150)
            print(f"收敛曲线已保存: {conv_path}")
            plt.show()
        
    except ImportError:
        print("[warn] matplotlib 不可用，跳过可视化")
    except Exception as e:
        print(f"[warn] 可视化失败: {e}")
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    
    return {
        'snr_init': snr_init,
        'snr_admm': snr_admm,
        'psnr_init': psnr_init,
        'psnr_admm': psnr_admm,
    }


if __name__ == '__main__':
    results = main()
