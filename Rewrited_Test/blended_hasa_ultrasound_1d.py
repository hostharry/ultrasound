"""
Blended-HASA 1D 超声压缩感知重建
================================

针对 dataset_fdbf_energy_mu_8_9_15.npz 数据集

Blended-HASA 方法：
    λ_final = λ_static + β * λ_adaptive

其中 β 控制静态和自适应正则化的混合比例。

分析计划：
    1. 实现 Blended-HASA 方法（1D 版本）
    2. 测试不同 β 值（0.0, 0.25, 0.5, 0.75, 1.0）
    3. 与 Static-HASA 和 Adaptive-HASA 基准比较
    4. 找到最优 β 值
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
    
    # Blended-HASA 参数
    'beta_values': [0.0, 0.25, 0.5, 0.75, 1.0],  # β 混合系数
    'lambda_static': 0.001,     # 静态正则化参数
    'lambda_adaptive_base': 0.001,  # 自适应正则化基础参数
    
    # 重建参数
    'n_iterations': 50,
    'wavelet': 'db4',
    'wavelet_level': 4,
    'entropy_window_size': 32,
    
    # 测试参数
    'num_test_lines': 50,       # -1 表示全部
    'output_dir': 'outputs_blended_1d',
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
    def __init__(self, N, mu):
        self.N = N
        self.mu = mu
        self.M = len(mu)
        self.rfft_len = N // 2 + 1
        self.mask = np.zeros(self.rfft_len, dtype=np.float64)
        self.mask[mu] = 1.0
    
    def forward(self, x):
        X_freq = np.fft.rfft(x)
        return X_freq[self.mu]
    
    def adjoint(self, y_sub):
        X_freq = np.zeros(self.rfft_len, dtype=np.complex128)
        X_freq[self.mu] = y_sub
        return np.fft.irfft(X_freq, n=self.N)
    
    def AtA(self, x):
        X_freq = np.fft.rfft(x)
        X_masked = X_freq * self.mask
        return np.fft.irfft(X_masked, n=self.N)


# ============================================================================
# 1D 正则化近端算子
# ============================================================================

def tv_prox_1d(x, weight, n_iter=20):
    """1D TV 近端算子"""
    if weight <= 0:
        return x.copy()
    
    n = len(x)
    p = np.zeros(n - 1, dtype=np.float64)
    
    for _ in range(n_iter):
        div_p = np.zeros(n, dtype=np.float64)
        div_p[0] = p[0]
        div_p[1:-1] = p[1:] - p[:-1]
        div_p[-1] = -p[-1]
        
        grad_z = x - weight * div_p
        diff = np.diff(grad_z)
        p = (p + (1.0 / (4.0 * weight)) * diff) / (1.0 + np.abs(diff) / (4.0 * weight))
    
    div_p = np.zeros(n, dtype=np.float64)
    div_p[0] = p[0]
    div_p[1:-1] = p[1:] - p[:-1]
    div_p[-1] = -p[-1]
    
    return x - weight * div_p


def wavelet_soft_threshold_1d(x, threshold, wavelet='db4', level=4):
    """1D 小波软阈值"""
    if threshold <= 0:
        return x.copy()
    
    coeffs = pywt.wavedec(x, wavelet, level=level)
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
    
    reconstructed = pywt.waverec(coeffs_thresh, wavelet)
    if len(reconstructed) != len(x):
        reconstructed = reconstructed[:len(x)]
    
    return reconstructed


# ============================================================================
# 1D 自适应权重计算
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


def compute_adaptive_weights_1d(x, lambda_base=0.001, window_size=32):
    """
    计算 1D 自适应权重
    
    返回:
        (lambda_tv_adaptive, lambda_wav_adaptive)
    """
    # 计算梯度和熵
    gradient_map = local_gradient_1d(x, window_size)
    entropy_map = local_entropy_1d(x, window_size)
    
    # 归一化
    g_min, g_max = gradient_map.min(), gradient_map.max()
    if g_max - g_min > 1e-12:
        grad_norm = (gradient_map - g_min) / (g_max - g_min)
    else:
        grad_norm = np.zeros_like(gradient_map)
    
    e_min, e_max = entropy_map.min(), entropy_map.max()
    if e_max - e_min > 1e-12:
        entropy_norm = (entropy_map - e_min) / (e_max - e_min)
    else:
        entropy_norm = np.zeros_like(entropy_map)
    
    # 自适应权重：
    # TV 权重与梯度成正比（梯度大的区域需要更强的 TV 正则化）
    # 小波权重与熵成正比（熵高的区域需要更强的小波正则化）
    lambda_tv_adaptive = lambda_base * (0.5 + np.mean(grad_norm))
    lambda_wav_adaptive = lambda_base * (0.5 + np.mean(entropy_norm))
    
    return lambda_tv_adaptive, lambda_wav_adaptive


# ============================================================================
# Blended-HASA 重建
# ============================================================================

def blended_hasa_1d(y_sub, op, lambda_static=0.001, lambda_adaptive_base=0.001,
                   beta=0.5, n_iter=50, wavelet='db4', level=4, 
                   entropy_window=32):
    """
    Blended-HASA 1D 重建
    
    λ_final = λ_static + β * λ_adaptive
    
    参数:
        y_sub: 子采样测量值
        op: 频域测量算子
        lambda_static: 静态正则化参数
        lambda_adaptive_base: 自适应正则化基础参数
        beta: 混合系数 (0=纯静态, 1=纯自适应)
        n_iter: 迭代次数
        wavelet: 小波基
        level: 小波分解层数
        entropy_window: 局部熵窗口大小
    
    返回:
        重建信号
    """
    # 初始化：反投影
    x = op.adjoint(y_sub)
    z = x.copy()
    t = 1.0
    L = 1.0  # Lipschitz 常数
    
    for it in range(n_iter):
        # 计算自适应权重（基于当前估计 z）
        lambda_tv_adaptive, lambda_wav_adaptive = compute_adaptive_weights_1d(
            z, lambda_adaptive_base, entropy_window
        )
        
        # 混合权重
        lambda_tv = lambda_static + beta * lambda_tv_adaptive
        lambda_wav = lambda_static + beta * lambda_wav_adaptive
        
        # 梯度下降步骤
        residual = op.forward(z) - y_sub
        grad = op.adjoint(residual)
        x_new = z - (1.0 / L) * grad
        
        # TV 近端算子
        x_new = tv_prox_1d(x_new, weight=lambda_tv / L)
        
        # 小波近端算子
        x_new = wavelet_soft_threshold_1d(x_new, threshold=lambda_wav / L,
                                          wavelet=wavelet, level=level)
        
        # FISTA 动量更新
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
    
    return x


# ============================================================================
# 基准方法
# ============================================================================

def static_hasa_1d(y_sub, op, lambda_tv=0.001, lambda_wav=0.001,
                  n_iter=50, wavelet='db4', level=4):
    """Static-HASA: 固定权重"""
    x = op.adjoint(y_sub)
    z = x.copy()
    t = 1.0
    L = 1.0
    
    for _ in range(n_iter):
        residual = op.forward(z) - y_sub
        grad = op.adjoint(residual)
        x_new = z - (1.0 / L) * grad
        
        x_new = tv_prox_1d(x_new, weight=lambda_tv / L)
        x_new = wavelet_soft_threshold_1d(x_new, threshold=lambda_wav / L,
                                          wavelet=wavelet, level=level)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new
    
    return x


def adaptive_hasa_1d(y_sub, op, base_lambda=0.001, n_iter=50,
                    wavelet='db4', level=4, entropy_window=32):
    """Adaptive-HASA: 纯自适应权重"""
    return blended_hasa_1d(y_sub, op, 
                           lambda_static=0.0, 
                           lambda_adaptive_base=base_lambda,
                           beta=1.0, 
                           n_iter=n_iter,
                           wavelet=wavelet, 
                           level=level,
                           entropy_window=entropy_window)


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
    print("Blended-HASA 1D 超声压缩感知重建")
    print("=" * 70)
    
    config = CONFIG
    
    # 打印配置
    print(f"\n配置参数:")
    print(f"  数据集: {config['npz_path']}")
    print(f"  压缩比: {config['cs_ratio']}x")
    print(f"  β 值: {config['beta_values']}")
    print(f"  迭代次数: {config['n_iterations']}")
    
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
    
    # 结果存储
    results = {
        'init': {'snr': [], 'psnr': []},
        'static': {'snr': [], 'psnr': []},
        'adaptive': {'snr': [], 'psnr': []},
    }
    for beta in config['beta_values']:
        results[f'blended_{beta}'] = {'snr': [], 'psnr': []}
    
    for i in range(num_test):
        y_gt = Y[i]
        x_init = X[i]
        y_sub = op.forward(y_gt)
        
        # Initial
        results['init']['snr'].append(calc_snr(y_gt, x_init))
        results['init']['psnr'].append(calc_psnr(y_gt, x_init))
        
        # Static-HASA
        x_static = static_hasa_1d(y_sub, op,
                                  lambda_tv=config['lambda_static'],
                                  lambda_wav=config['lambda_static'],
                                  n_iter=config['n_iterations'],
                                  wavelet=config['wavelet'],
                                  level=config['wavelet_level'])
        results['static']['snr'].append(calc_snr(y_gt, x_static))
        results['static']['psnr'].append(calc_psnr(y_gt, x_static))
        
        # Adaptive-HASA
        x_adaptive = adaptive_hasa_1d(y_sub, op,
                                      base_lambda=config['lambda_adaptive_base'],
                                      n_iter=config['n_iterations'],
                                      wavelet=config['wavelet'],
                                      level=config['wavelet_level'],
                                      entropy_window=config['entropy_window_size'])
        results['adaptive']['snr'].append(calc_snr(y_gt, x_adaptive))
        results['adaptive']['psnr'].append(calc_psnr(y_gt, x_adaptive))
        
        # Blended-HASA for each β
        for beta in config['beta_values']:
            x_blended = blended_hasa_1d(y_sub, op,
                                        lambda_static=config['lambda_static'],
                                        lambda_adaptive_base=config['lambda_adaptive_base'],
                                        beta=beta,
                                        n_iter=config['n_iterations'],
                                        wavelet=config['wavelet'],
                                        level=config['wavelet_level'],
                                        entropy_window=config['entropy_window_size'])
            results[f'blended_{beta}']['snr'].append(calc_snr(y_gt, x_blended))
            results[f'blended_{beta}']['psnr'].append(calc_psnr(y_gt, x_blended))
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{num_test}] Static: {results['static']['snr'][-1]:.2f} | "
                  f"Adaptive: {results['adaptive']['snr'][-1]:.2f} | "
                  f"Blended(0.5): {results['blended_0.5']['snr'][-1]:.2f} dB")
    
    # 转为数组
    for key in results:
        results[key]['snr'] = np.array(results[key]['snr'])
        results[key]['psnr'] = np.array(results[key]['psnr'])
    
    # 5. 汇总结果
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    
    def print_stats(name, d):
        snr = d['snr']
        psnr = d['psnr']
        print(f"\n{name}:")
        print(f"  SNR:  {snr.mean():.2f} ± {snr.std():.2f} dB")
        print(f"  PSNR: {psnr.mean():.2f} ± {psnr.std():.2f} dB")
    
    print_stats("Initial (零填充)", results['init'])
    print_stats("Static-HASA", results['static'])
    print_stats("Adaptive-HASA", results['adaptive'])
    
    print("\n" + "-" * 70)
    print("Blended-HASA (不同 β 值):")
    print("-" * 70)
    
    init_snr = results['init']['snr'].mean()
    
    best_beta = None
    best_snr = -np.inf
    
    for beta in config['beta_values']:
        key = f'blended_{beta}'
        snr = results[key]['snr'].mean()
        psnr = results[key]['psnr'].mean()
        improvement = snr - init_snr
        print(f"  β = {beta:.2f}: SNR = {snr:.2f} dB (+{improvement:.2f}), PSNR = {psnr:.2f} dB")
        
        if snr > best_snr:
            best_snr = snr
            best_beta = beta
    
    print(f"\n最优 β = {best_beta} (SNR = {best_snr:.2f} dB)")
    
    # 6. 与基准比较
    print("\n" + "=" * 70)
    print("与基准方法比较")
    print("=" * 70)
    
    static_snr = results['static']['snr'].mean()
    adaptive_snr = results['adaptive']['snr'].mean()
    
    print(f"\nStatic-HASA:   SNR = {static_snr:.2f} dB")
    print(f"Adaptive-HASA: SNR = {adaptive_snr:.2f} dB")
    print(f"Blended-HASA (β={best_beta}): SNR = {best_snr:.2f} dB")
    
    if best_snr > static_snr:
        print(f"\n✓ Blended-HASA 优于 Static-HASA: +{best_snr - static_snr:.2f} dB")
    else:
        print(f"\n✗ Blended-HASA 不优于 Static-HASA")
    
    if best_snr > adaptive_snr:
        print(f"✓ Blended-HASA 优于 Adaptive-HASA: +{best_snr - adaptive_snr:.2f} dB")
    else:
        print(f"✗ Blended-HASA 不优于 Adaptive-HASA")
    
    # 7. 保存结果
    os.makedirs(config['output_dir'], exist_ok=True)
    result_file = os.path.join(config['output_dir'], f"results_blended_ratio{config['cs_ratio']}.txt")
    with open(result_file, 'w') as f:
        f.write("Blended-HASA 1D 超声重建结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"数据集: {config['npz_path']}\n")
        f.write(f"压缩比: {config['cs_ratio']}x\n")
        f.write(f"测试样本: {num_test}\n\n")
        
        f.write("SNR (dB):\n")
        f.write(f"  Initial:       {results['init']['snr'].mean():.2f} ± {results['init']['snr'].std():.2f}\n")
        f.write(f"  Static-HASA:   {static_snr:.2f} ± {results['static']['snr'].std():.2f}\n")
        f.write(f"  Adaptive-HASA: {adaptive_snr:.2f} ± {results['adaptive']['snr'].std():.2f}\n\n")
        
        f.write("Blended-HASA:\n")
        for beta in config['beta_values']:
            key = f'blended_{beta}'
            snr = results[key]['snr'].mean()
            f.write(f"  β = {beta:.2f}: {snr:.2f} ± {results[key]['snr'].std():.2f} dB\n")
        
        f.write(f"\n最优 β = {best_beta}\n")
    
    print(f"\n结果已保存: {result_file}")
    
    # 8. 可视化
    try:
        import matplotlib.pyplot as plt
        
        # β vs SNR 曲线
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        betas = config['beta_values']
        snrs = [results[f'blended_{b}']['snr'].mean() for b in betas]
        snr_stds = [results[f'blended_{b}']['snr'].std() for b in betas]
        
        axes[0].errorbar(betas, snrs, yerr=snr_stds, marker='o', capsize=5, linewidth=2, markersize=8)
        axes[0].axhline(static_snr, color='r', linestyle='--', label='Static-HASA')
        axes[0].axhline(adaptive_snr, color='g', linestyle='--', label='Adaptive-HASA')
        axes[0].set_xlabel('β (blending coefficient)', fontsize=12)
        axes[0].set_ylabel('SNR (dB)', fontsize=12)
        axes[0].set_title('Blended-HASA: SNR vs β', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 箱线图比较
        box_data = [
            results['static']['snr'],
            results['adaptive']['snr'],
            results[f'blended_{best_beta}']['snr']
        ]
        bp = axes[1].boxplot(box_data, labels=['Static', 'Adaptive', f'Blended(β={best_beta})'],
                            patch_artist=True)
        colors = ['lightcoral', 'lightgreen', 'lightskyblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[1].set_ylabel('SNR (dB)', fontsize=12)
        axes[1].set_title('Method Comparison', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(config['output_dir'], f'blended_comparison_ratio{config["cs_ratio"]}.png')
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
