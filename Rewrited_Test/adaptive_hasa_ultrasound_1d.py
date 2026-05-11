"""
Adaptive-HASA 1D 超声压缩感知重建算法
=====================================

针对 dataset_fdbf_energy_mu_8_9_15.npz 数据集的适配版本

数据结构:
  - Y: (L, N) GT beamformed 信号
  - X8/X9/X15: (L, N) 频域子采样后的输入
  - mu8/mu9/mu15: 保留的频率索引

重建方法:
  1. TV-only: 1D 总变分正则化
  2. Static-HASA: 静态混合 TV + 小波
  3. Adaptive-HASA: 自适应混合正则化

指标:
  - SNR, PSNR, NMSE, 相关系数
"""

import numpy as np
import os
import time
import pywt
from numba import jit
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'npz_path': '../dataset_fdbf_energy_mu_8_9_15.npz',
    'cs_ratio': 9,              # 8, 9, 或 15
    'base_lambda_tv': 0.001,    # TV 正则化参数
    'base_lambda_wav': 0.001,   # 小波正则化参数
    'n_iterations': 50,         # FISTA 迭代次数
    'wavelet': 'db4',
    'wavelet_level': 4,
    'entropy_window_size': 32,  # 1D 局部熵窗口
    'num_test_lines': 50,       # 测试多少条线（-1 表示全部）
    'output_dir': 'outputs_hasa_1d',
}


# ==================== 数据加载 ====================

def load_ultrasound_dataset(npz_path, cs_ratio=9):
    """
    加载超声数据集
    返回: X (子采样输入), Y (GT), mu (频率索引), fs, fc, c
    """
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
    print(f"  fs={fs/1e6:.2f} MHz, fc={fc/1e6:.2f} MHz, c={c:.1f} m/s")
    print(f"  Retained frequency bins: {len(mu)}")
    
    return X, Y, mu, fs, fc, c


# ==================== 频域测量算子 ====================

class FrequencyMaskOperator:
    """
    频域子采样测量算子
    A: x → rFFT → mask → 子采样系数
    A^T: 子采样系数 → 补零 → irFFT → x
    """
    def __init__(self, N, mu):
        self.N = N
        self.mu = mu
        self.rfft_len = N // 2 + 1
        self.mask = np.zeros(self.rfft_len, dtype=np.float64)
        self.mask[mu] = 1.0
    
    def forward(self, x):
        """x -> y (子采样测量)"""
        X_freq = np.fft.rfft(x)
        return X_freq[self.mu]
    
    def adjoint(self, y_sub):
        """y_sub -> x (伴随/转置)"""
        X_freq = np.zeros(self.rfft_len, dtype=np.complex128)
        X_freq[self.mu] = y_sub
        return np.fft.irfft(X_freq, n=self.N)
    
    def forward_adjoint(self, x):
        """A^T A x: 用于梯度计算"""
        X_freq = np.fft.rfft(x)
        X_masked = X_freq * self.mask
        return np.fft.irfft(X_masked, n=self.N)


# ==================== 1D 正则化近端算子 ====================

def tv_prox_1d(x, weight, n_iter=20):
    """
    1D 总变分近端算子 (Chambolle 算法)
    min_z 0.5*||z - x||^2 + weight * TV(z)
    """
    if weight <= 0:
        return x.copy()
    
    n = len(x)
    p = np.zeros(n - 1, dtype=np.float64)
    
    for _ in range(n_iter):
        # 计算散度
        div_p = np.zeros(n, dtype=np.float64)
        div_p[0] = p[0]
        div_p[1:-1] = p[1:] - p[:-1]
        div_p[-1] = -p[-1]
        
        # 梯度
        grad_z = x - weight * div_p
        diff = np.diff(grad_z)
        
        # 更新 p
        p = (p + (1.0 / (4.0 * weight)) * diff) / (1.0 + np.abs(diff) / (4.0 * weight))
    
    # 最终结果
    div_p = np.zeros(n, dtype=np.float64)
    div_p[0] = p[0]
    div_p[1:-1] = p[1:] - p[:-1]
    div_p[-1] = -p[-1]
    
    return x - weight * div_p


def wavelet_soft_threshold_1d(x, threshold, wavelet='db4', level=4):
    """
    1D 小波软阈值近端算子
    """
    if threshold <= 0:
        return x.copy()
    
    coeffs = pywt.wavedec(x, wavelet, level=level)
    
    # 对细节系数软阈值
    coeffs_thresh = [coeffs[0]]  # 保留近似系数
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
    
    reconstructed = pywt.waverec(coeffs_thresh, wavelet)
    
    # 处理长度
    if len(reconstructed) != len(x):
        reconstructed = reconstructed[:len(x)]
    
    return reconstructed


@jit(nopython=True)
def local_gradient_1d(x, window_size=32):
    """计算 1D 局部梯度幅度"""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)
    half = window_size // 2
    
    for i in range(n):
        i_min = max(0, i - half)
        i_max = min(n - 1, i + half)
        
        # 窗口内的梯度
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
    
    # 归一化到 [0, n_bins-1]
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        return result
    
    x_norm = ((x - x_min) / (x_max - x_min) * (n_bins - 1)).astype(np.int32)
    x_norm = np.clip(x_norm, 0, n_bins - 1)
    
    for i in range(n):
        i_min = max(0, i - half)
        i_max = min(n, i + half + 1)
        
        # 直方图
        hist = np.zeros(n_bins, dtype=np.int32)
        for j in range(i_min, i_max):
            hist[x_norm[j]] += 1
        
        # 熵
        total = i_max - i_min
        entropy = 0.0
        for k in range(n_bins):
            if hist[k] > 0:
                p = hist[k] / total
                entropy -= p * np.log2(p)
        
        result[i] = entropy
    
    return result


# ==================== FISTA 重建算法 ====================

def fista_tv_only_1d(y_sub, op, lambda_tv, n_iter=50):
    """
    FISTA 1D 重建 - 仅 TV 正则化
    """
    # 初始化
    x = op.adjoint(y_sub)
    z = x.copy()
    t = 1.0
    
    # Lipschitz 常数估计（对于频域 mask，L ≤ 1）
    L = 1.0
    
    for _ in range(n_iter):
        # 梯度下降
        residual = op.forward(z) - y_sub
        grad = op.adjoint(residual)
        x_new = z - (1.0 / L) * grad
        
        # TV 近端
        x_new = tv_prox_1d(x_new, weight=lambda_tv / L)
        
        # FISTA 动量
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
    
    return x


def fista_static_hasa_1d(y_sub, op, lambda_tv, lambda_wav, n_iter=50,
                         wavelet='db4', level=4):
    """
    FISTA 1D 重建 - 静态混合 TV + 小波 (Static-HASA)
    """
    x = op.adjoint(y_sub)
    z = x.copy()
    t = 1.0
    L = 1.0
    
    for _ in range(n_iter):
        residual = op.forward(z) - y_sub
        grad = op.adjoint(residual)
        x_new = z - (1.0 / L) * grad
        
        # TV 近端
        x_new = tv_prox_1d(x_new, weight=lambda_tv / L)
        
        # 小波近端
        x_new = wavelet_soft_threshold_1d(x_new, threshold=lambda_wav / L,
                                          wavelet=wavelet, level=level)
        
        # FISTA 动量
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
    
    return x


def fista_adaptive_hasa_1d(y_sub, op, base_lambda, n_iter=50,
                           wavelet='db4', level=4, entropy_window=32):
    """
    FISTA 1D 重建 - 自适应混合 TV + 小波 (Adaptive-HASA)
    
    基于局部梯度和熵动态调整正则化权重
    """
    x = op.adjoint(y_sub)
    z = x.copy()
    t = 1.0
    L = 1.0
    
    for _ in range(n_iter):
        # 梯度下降
        residual = op.forward(z) - y_sub
        grad = op.adjoint(residual)
        x_intermediate = z - (1.0 / L) * grad
        
        # 计算自适应权重（基于当前估计 z）
        gradient_map = local_gradient_1d(z, window_size=entropy_window)
        entropy_map = local_entropy_1d(z, window_size=entropy_window)
        
        # 归一化
        g_min, g_max = gradient_map.min(), gradient_map.max()
        if g_max - g_min > 1e-12:
            gradient_norm = (gradient_map - g_min) / (g_max - g_min)
        else:
            gradient_norm = np.zeros_like(gradient_map)
        
        e_min, e_max = entropy_map.min(), entropy_map.max()
        if e_max - e_min > 1e-12:
            entropy_norm = (entropy_map - e_min) / (e_max - e_min)
        else:
            entropy_norm = np.zeros_like(entropy_map)
        
        # 自适应权重：梯度大 → TV 强；熵高 → 小波强
        avg_tv_weight = base_lambda * (0.5 + np.mean(gradient_norm)) / L
        avg_wav_threshold = base_lambda * (0.5 + np.mean(entropy_norm)) / L
        
        # TV 近端
        x_new = tv_prox_1d(x_intermediate, weight=avg_tv_weight)
        
        # 小波近端
        x_new = wavelet_soft_threshold_1d(x_new, threshold=avg_wav_threshold,
                                          wavelet=wavelet, level=level)
        
        # FISTA 动量
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
    
    return x


# ==================== 指标计算 ====================

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


def calc_correlation(y_true, y_pred):
    return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]


# ==================== 主函数 ====================

def main():
    print("=" * 70)
    print("Adaptive-HASA 1D 超声压缩感知重建")
    print("=" * 70)
    
    config = CONFIG
    
    # 打印配置
    print(f"\n配置参数:")
    print(f"  数据集: {config['npz_path']}")
    print(f"  压缩比: {config['cs_ratio']}x")
    print(f"  迭代次数: {config['n_iterations']}")
    print(f"  TV 正则化: {config['base_lambda_tv']}")
    print(f"  小波正则化: {config['base_lambda_wav']}")
    
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
    
    results = {
        'init': {'snr': [], 'psnr': [], 'nmse': [], 'corr': []},
        'tv': {'snr': [], 'psnr': [], 'nmse': [], 'corr': [], 'time': []},
        'static': {'snr': [], 'psnr': [], 'nmse': [], 'corr': [], 'time': []},
        'adaptive': {'snr': [], 'psnr': [], 'nmse': [], 'corr': [], 'time': []},
    }
    
    for i in range(num_test):
        y_gt = Y[i]
        x_init = X[i]
        y_sub = op.forward(y_gt)  # 子采样测量
        
        # 初始（零填充）
        results['init']['snr'].append(calc_snr(y_gt, x_init))
        results['init']['psnr'].append(calc_psnr(y_gt, x_init))
        results['init']['nmse'].append(calc_nmse(y_gt, x_init))
        results['init']['corr'].append(calc_correlation(y_gt, x_init))
        
        # TV-only
        t0 = time.time()
        x_tv = fista_tv_only_1d(y_sub, op, lambda_tv=config['base_lambda_tv'],
                                n_iter=config['n_iterations'])
        t_tv = time.time() - t0
        results['tv']['snr'].append(calc_snr(y_gt, x_tv))
        results['tv']['psnr'].append(calc_psnr(y_gt, x_tv))
        results['tv']['nmse'].append(calc_nmse(y_gt, x_tv))
        results['tv']['corr'].append(calc_correlation(y_gt, x_tv))
        results['tv']['time'].append(t_tv)
        
        # Static-HASA
        t0 = time.time()
        x_static = fista_static_hasa_1d(y_sub, op,
                                        lambda_tv=config['base_lambda_tv'],
                                        lambda_wav=config['base_lambda_wav'],
                                        n_iter=config['n_iterations'],
                                        wavelet=config['wavelet'],
                                        level=config['wavelet_level'])
        t_static = time.time() - t0
        results['static']['snr'].append(calc_snr(y_gt, x_static))
        results['static']['psnr'].append(calc_psnr(y_gt, x_static))
        results['static']['nmse'].append(calc_nmse(y_gt, x_static))
        results['static']['corr'].append(calc_correlation(y_gt, x_static))
        results['static']['time'].append(t_static)
        
        # Adaptive-HASA
        t0 = time.time()
        x_adaptive = fista_adaptive_hasa_1d(y_sub, op,
                                            base_lambda=config['base_lambda_tv'],
                                            n_iter=config['n_iterations'],
                                            wavelet=config['wavelet'],
                                            level=config['wavelet_level'],
                                            entropy_window=config['entropy_window_size'])
        t_adaptive = time.time() - t0
        results['adaptive']['snr'].append(calc_snr(y_gt, x_adaptive))
        results['adaptive']['psnr'].append(calc_psnr(y_gt, x_adaptive))
        results['adaptive']['nmse'].append(calc_nmse(y_gt, x_adaptive))
        results['adaptive']['corr'].append(calc_correlation(y_gt, x_adaptive))
        results['adaptive']['time'].append(t_adaptive)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{num_test}] "
                  f"Init SNR: {results['init']['snr'][-1]:.2f} dB | "
                  f"TV: {results['tv']['snr'][-1]:.2f} dB | "
                  f"Static: {results['static']['snr'][-1]:.2f} dB | "
                  f"Adaptive: {results['adaptive']['snr'][-1]:.2f} dB")
    
    # 5. 汇总结果
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    
    def summarize(name, d):
        snr = np.array(d['snr'])
        psnr = np.array(d['psnr'])
        nmse = np.array(d['nmse'])
        corr = np.array(d['corr'])
        print(f"\n{name}:")
        print(f"  SNR:  {snr.mean():.2f} ± {snr.std():.2f} dB")
        print(f"  PSNR: {psnr.mean():.2f} ± {psnr.std():.2f} dB")
        print(f"  NMSE: {nmse.mean():.6f} ± {nmse.std():.6f}")
        print(f"  Corr: {corr.mean():.4f} ± {corr.std():.4f}")
        if 'time' in d and d['time']:
            t = np.array(d['time'])
            print(f"  Time: {t.mean()*1000:.2f} ± {t.std()*1000:.2f} ms/line")
    
    summarize("Initial (zero-filled)", results['init'])
    summarize("TV-only", results['tv'])
    summarize("Static-HASA", results['static'])
    summarize("Adaptive-HASA", results['adaptive'])
    
    # 6. SNR 提升
    init_snr = np.mean(results['init']['snr'])
    tv_snr = np.mean(results['tv']['snr'])
    static_snr = np.mean(results['static']['snr'])
    adaptive_snr = np.mean(results['adaptive']['snr'])
    
    print("\n" + "-" * 70)
    print("SNR 提升 (相对于 Initial):")
    print(f"  TV-only:       +{tv_snr - init_snr:.2f} dB")
    print(f"  Static-HASA:   +{static_snr - init_snr:.2f} dB")
    print(f"  Adaptive-HASA: +{adaptive_snr - init_snr:.2f} dB")
    print("-" * 70)
    
    # 7. 保存结果
    os.makedirs(config['output_dir'], exist_ok=True)
    result_file = os.path.join(config['output_dir'], f"results_ratio{config['cs_ratio']}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Adaptive-HASA 1D 超声压缩感知重建结果\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"数据集: {config['npz_path']}\n")
        f.write(f"压缩比: {config['cs_ratio']}x\n")
        f.write(f"测试线数: {num_test}\n")
        f.write(f"迭代次数: {config['n_iterations']}\n\n")
        
        for name, d in [("Initial", results['init']),
                        ("TV-only", results['tv']),
                        ("Static-HASA", results['static']),
                        ("Adaptive-HASA", results['adaptive'])]:
            snr = np.array(d['snr'])
            psnr = np.array(d['psnr'])
            f.write(f"{name}:\n")
            f.write(f"  SNR:  {snr.mean():.2f} ± {snr.std():.2f} dB\n")
            f.write(f"  PSNR: {psnr.mean():.2f} ± {psnr.std():.2f} dB\n\n")
        
        f.write(f"SNR 提升:\n")
        f.write(f"  TV-only:       +{tv_snr - init_snr:.2f} dB\n")
        f.write(f"  Static-HASA:   +{static_snr - init_snr:.2f} dB\n")
        f.write(f"  Adaptive-HASA: +{adaptive_snr - init_snr:.2f} dB\n")
    
    print(f"\n结果已保存: {result_file}")
    
    # 8. 可视化（如果 matplotlib 可用）
    try:
        import matplotlib.pyplot as plt
        
        # 选一条线可视化
        idx = 0
        y_gt = Y[idx]
        x_init = X[idx]
        y_sub = op.forward(y_gt)
        
        x_tv = fista_tv_only_1d(y_sub, op, lambda_tv=config['base_lambda_tv'],
                                n_iter=config['n_iterations'])
        x_static = fista_static_hasa_1d(y_sub, op,
                                        lambda_tv=config['base_lambda_tv'],
                                        lambda_wav=config['base_lambda_wav'],
                                        n_iter=config['n_iterations'])
        x_adaptive = fista_adaptive_hasa_1d(y_sub, op,
                                            base_lambda=config['base_lambda_tv'],
                                            n_iter=config['n_iterations'])
        
        t_us = np.arange(N) / fs * 1e6
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        axes[0].plot(t_us, y_gt, 'b-', linewidth=0.5, alpha=0.8)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'GT (Y) - Line {idx}')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t_us, y_gt, 'b-', linewidth=0.5, alpha=0.4, label='GT')
        axes[1].plot(t_us, x_tv, 'r-', linewidth=0.5, alpha=0.8, label='TV-only')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title(f'TV-only | SNR: {calc_snr(y_gt, x_tv):.2f} dB')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(t_us, y_gt, 'b-', linewidth=0.5, alpha=0.4, label='GT')
        axes[2].plot(t_us, x_static, 'g-', linewidth=0.5, alpha=0.8, label='Static-HASA')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title(f'Static-HASA | SNR: {calc_snr(y_gt, x_static):.2f} dB')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(t_us, y_gt, 'b-', linewidth=0.5, alpha=0.4, label='GT')
        axes[3].plot(t_us, x_adaptive, 'm-', linewidth=0.5, alpha=0.8, label='Adaptive-HASA')
        axes[3].set_xlabel('Time (μs)')
        axes[3].set_ylabel('Amplitude')
        axes[3].set_title(f'Adaptive-HASA | SNR: {calc_snr(y_gt, x_adaptive):.2f} dB')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(config['output_dir'], f'comparison_line{idx}_ratio{config["cs_ratio"]}.png')
        plt.savefig(fig_path, dpi=150)
        print(f"可视化图已保存: {fig_path}")
        plt.show()
        
    except ImportError:
        print("[warn] matplotlib 不可用，跳过可视化")
    except Exception as e:
        print(f"[warn] 可视化失败: {e}")
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()
