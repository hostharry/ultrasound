"""
自适应权重图空间方差分析
==========================
验证假设：从初始反投影 x₀ = Aᵀy 计算得到的自适应权重图具有高空间方差（归一化标度上 std > 0.1）

实验参数：
- 图像大小: 128×128
- 采样率: 15%
- SNR: 25 dB
- 测量矩阵种子: 42
- 噪声种子: 43
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.transform import resize
from scipy.ndimage import generic_filter
from scipy.stats import entropy as shannon_entropy
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'target_size': 128,
    'sampling_rate': 0.15,
    'target_snr_db': 25.0,
    'measurement_seed': 42,
    'noise_seed': 43,
    'entropy_window_size': 9,
    'base_lambda_tv': 0.01,
    'base_lambda_wav': 0.01,
    'alpha': 1.0,  # 梯度贡献缩放因子
    'beta': 1.0,   # 熵贡献缩放因子
    'threshold': 0.1,  # 假设检验阈值
}

# ==================== 工具函数 ====================

def load_and_preprocess_image(image_path, target_size):
    """加载图像，转换为灰度，调整大小并归一化到[0,1]"""
    img = np.array(Image.open(image_path).convert('L'))
    img_resized = resize(img, (target_size, target_size), anti_aliasing=True)
    img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())
    return img_norm


def create_measurement_matrix(n_pixels, sampling_rate, seed):
    """创建高斯随机测量矩阵"""
    np.random.seed(seed)
    n_measurements = int(sampling_rate * n_pixels)
    A = np.random.randn(n_measurements, n_pixels) / np.sqrt(n_measurements)
    return A


def add_noise_to_measurements(y_clean, target_snr_db, noise_seed):
    """
    向测量值添加高斯噪声以达到目标SNR
    SNR (dB) = 10 * log10(signal_power / noise_power)
    """
    np.random.seed(noise_seed)
    signal_power = np.mean(y_clean ** 2)
    target_snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / target_snr_linear
    noise_std = np.sqrt(noise_power)
    noise = np.random.randn(len(y_clean)) * noise_std
    return y_clean + noise


def compute_gradient_magnitude(image):
    """使用Sobel滤波器计算梯度幅值"""
    return sobel(image)


def compute_local_entropy(image, window_size):
    """
    使用generic_filter计算局部熵
    """
    def local_entropy_func(values):
        hist, _ = np.histogram(values, bins=256, range=(-10, 10), density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0
        return shannon_entropy(hist, base=2)
    
    return generic_filter(image, local_entropy_func, size=window_size, mode='reflect')


def normalize_to_01(arr):
    """归一化数组到[0,1]范围"""
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)


def compute_adaptive_weights(grad_norm, entropy_norm, config):
    """
    计算自适应权重图
    - TV权重图：强调边缘（高梯度区域）
    - 小波权重图：强调纹理（高熵、低梯度区域）
    """
    lambda_tv = config['base_lambda_tv'] * (1 + config['alpha'] * grad_norm)
    lambda_wav = config['base_lambda_wav'] * (1 + config['beta'] * entropy_norm * (1 - grad_norm))
    return lambda_tv, lambda_wav


# ==================== 主处理流程 ====================

def process_image(image_path, A, config):
    """处理单张图像，返回所有中间结果和统计数据"""
    target_size = config['target_size']
    
    # 1. 加载和预处理
    img_norm = load_and_preprocess_image(image_path, target_size)
    
    # 2. 压缩感知测量
    img_flat = img_norm.flatten()
    y_clean = A @ img_flat
    y_noisy = add_noise_to_measurements(y_clean, config['target_snr_db'], config['noise_seed'])
    
    # 3. 反投影重建
    x0_flat = A.T @ y_noisy
    x0 = x0_flat.reshape(target_size, target_size)
    
    # 4. 计算特征图
    grad = compute_gradient_magnitude(x0)
    grad_norm = normalize_to_01(grad)
    
    print("    计算局部熵中...")
    entropy_map = compute_local_entropy(x0, config['entropy_window_size'])
    entropy_norm = normalize_to_01(entropy_map)
    
    # 5. 计算自适应权重图
    lambda_tv, lambda_wav = compute_adaptive_weights(grad_norm, entropy_norm, config)
    lambda_tv_norm = normalize_to_01(lambda_tv)
    lambda_wav_norm = normalize_to_01(lambda_wav)
    
    # 6. 统计分析
    stats = {
        'std_tv_raw': np.std(lambda_tv),
        'std_wav_raw': np.std(lambda_wav),
        'std_tv_norm': np.std(lambda_tv_norm),
        'std_wav_norm': np.std(lambda_wav_norm),
        'mean_tv': np.mean(lambda_tv),
        'mean_wav': np.mean(lambda_wav),
        'cv_tv': np.std(lambda_tv) / np.mean(lambda_tv) * 100,
        'cv_wav': np.std(lambda_wav) / np.mean(lambda_wav) * 100,
        'grad_mean': np.mean(grad_norm),
        'grad_std': np.std(grad_norm),
        'entropy_mean': np.mean(entropy_norm),
        'entropy_std': np.std(entropy_norm),
    }
    
    results = {
        'original': img_norm,
        'x0': x0,
        'grad_norm': grad_norm,
        'entropy_norm': entropy_norm,
        'lambda_tv_norm': lambda_tv_norm,
        'lambda_wav_norm': lambda_wav_norm,
        'stats': stats,
    }
    
    return results


def print_statistics(name, stats, threshold):
    """打印单张图像的统计结果"""
    print(f"\n  {name}:")
    print(f"    TV权重图:     std(原始)={stats['std_tv_raw']:.6f}, std(归一化)={stats['std_tv_norm']:.6f}")
    print(f"    小波权重图:   std(原始)={stats['std_wav_raw']:.6f}, std(归一化)={stats['std_wav_norm']:.6f}")
    print(f"    TV权重图 CV:  {stats['cv_tv']:.2f}%")
    print(f"    小波权重图 CV: {stats['cv_wav']:.2f}%")
    
    tv_pass = stats['std_tv_norm'] > threshold
    wav_pass = stats['std_wav_norm'] > threshold
    print(f"    TV std > {threshold}: {'✓ 通过' if tv_pass else '✗ 未通过'} ({stats['std_tv_norm']:.4f})")
    print(f"    Wavelet std > {threshold}: {'✓ 通过' if wav_pass else '✗ 未通过'} ({stats['std_wav_norm']:.4f})")
    
    return tv_pass and wav_pass


def create_visualization(results_benign, results_malignant, output_path):
    """创建综合可视化图"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # 良性图像（上行）
    axes[0, 0].imshow(results_benign['original'], cmap='gray')
    axes[0, 0].set_title('A. Benign Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results_benign['x0'], cmap='gray')
    axes[0, 1].set_title('Back-projection x0\n(A^T * y)', fontsize=11)
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(results_benign['grad_norm'], cmap='viridis')
    axes[0, 2].set_title('Gradient (norm.)', fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)
    
    std_tv_b = results_benign['stats']['std_tv_norm']
    im2 = axes[0, 3].imshow(results_benign['lambda_tv_norm'], cmap='hot')
    axes[0, 3].set_title(f'TV Weight Map\nstd={std_tv_b:.3f}', fontsize=11)
    axes[0, 3].axis('off')
    plt.colorbar(im2, ax=axes[0, 3], fraction=0.046)
    
    std_wav_b = results_benign['stats']['std_wav_norm']
    im3 = axes[0, 4].imshow(results_benign['lambda_wav_norm'], cmap='hot')
    axes[0, 4].set_title(f'Wavelet Weight Map\nstd={std_wav_b:.3f}', fontsize=11)
    axes[0, 4].axis('off')
    plt.colorbar(im3, ax=axes[0, 4], fraction=0.046)
    
    # 恶性图像（下行）
    axes[1, 0].imshow(results_malignant['original'], cmap='gray')
    axes[1, 0].set_title('B. Malignant Original', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results_malignant['x0'], cmap='gray')
    axes[1, 1].set_title('Back-projection x0\n(A^T * y)', fontsize=11)
    axes[1, 1].axis('off')
    
    im4 = axes[1, 2].imshow(results_malignant['grad_norm'], cmap='viridis')
    axes[1, 2].set_title('Gradient (norm.)', fontsize=11)
    axes[1, 2].axis('off')
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)
    
    std_tv_m = results_malignant['stats']['std_tv_norm']
    im5 = axes[1, 3].imshow(results_malignant['lambda_tv_norm'], cmap='hot')
    axes[1, 3].set_title(f'TV Weight Map\nstd={std_tv_m:.3f}', fontsize=11)
    axes[1, 3].axis('off')
    plt.colorbar(im5, ax=axes[1, 3], fraction=0.046)
    
    std_wav_m = results_malignant['stats']['std_wav_norm']
    im6 = axes[1, 4].imshow(results_malignant['lambda_wav_norm'], cmap='hot')
    axes[1, 4].set_title(f'Wavelet Weight Map\nstd={std_wav_m:.3f}', fontsize=11)
    axes[1, 4].axis('off')
    plt.colorbar(im6, ax=axes[1, 4], fraction=0.046)
    
    plt.suptitle(
        'Adaptive Weight Maps from Initial Back-Projection Show High Spatial Variance (std > 0.1)\n'
        'Compressed Sensing: 15% sampling, 25dB SNR, Seeds: 42 (measurement), 43 (noise)',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化图已保存: {output_path}")
    plt.show()


def create_summary_table(results_benign, results_malignant, threshold):
    """创建汇总表格"""
    stats_b = results_benign['stats']
    stats_m = results_malignant['stats']
    
    summary_data = {
        '图像': ['Benign (1)', 'Benign (1)', 'Malignant (1)', 'Malignant (1)'],
        '权重类型': ['TV', 'Wavelet', 'TV', 'Wavelet'],
        'Std (原始)': [stats_b['std_tv_raw'], stats_b['std_wav_raw'], 
                       stats_m['std_tv_raw'], stats_m['std_wav_raw']],
        'Std (归一化[0,1])': [stats_b['std_tv_norm'], stats_b['std_wav_norm'],
                              stats_m['std_tv_norm'], stats_m['std_wav_norm']],
        'Mean (原始)': [stats_b['mean_tv'], stats_b['mean_wav'],
                        stats_m['mean_tv'], stats_m['mean_wav']],
        'CV (%)': [stats_b['cv_tv'], stats_b['cv_wav'],
                   stats_m['cv_tv'], stats_m['cv_wav']],
        f'通过阈值(>{threshold})': [
            stats_b['std_tv_norm'] > threshold,
            stats_b['std_wav_norm'] > threshold,
            stats_m['std_tv_norm'] > threshold,
            stats_m['std_wav_norm'] > threshold
        ]
    }
    
    return pd.DataFrame(summary_data)


# ==================== 主函数 ====================

def main():
    print("=" * 70)
    print("自适应权重图空间方差分析")
    print("=" * 70)
    
    # 图像路径
    benign_path = 'Dataset_BUSI_with_GT/benign/benign (1).png'
    malignant_path = 'Dataset_BUSI_with_GT/malignant/malignant (1).png'
    
    # 检查文件是否存在
    if not os.path.exists(benign_path):
        print(f"错误: 找不到文件 {benign_path}")
        return
    if not os.path.exists(malignant_path):
        print(f"错误: 找不到文件 {malignant_path}")
        return
    
    # 打印配置
    print(f"\n配置参数:")
    print(f"  图像大小: {CONFIG['target_size']}×{CONFIG['target_size']}")
    print(f"  采样率: {CONFIG['sampling_rate']*100:.0f}%")
    print(f"  SNR: {CONFIG['target_snr_db']} dB")
    print(f"  测量矩阵种子: {CONFIG['measurement_seed']}")
    print(f"  噪声种子: {CONFIG['noise_seed']}")
    print(f"  熵窗口大小: {CONFIG['entropy_window_size']}×{CONFIG['entropy_window_size']}")
    
    # 创建测量矩阵
    n_pixels = CONFIG['target_size'] ** 2
    print(f"\n创建测量矩阵...")
    A = create_measurement_matrix(n_pixels, CONFIG['sampling_rate'], CONFIG['measurement_seed'])
    print(f"  测量矩阵形状: {A.shape}")
    print(f"  测量数量: {A.shape[0]} ({CONFIG['sampling_rate']*100:.0f}% of {n_pixels} pixels)")
    
    # 处理图像
    print(f"\n处理良性图像...")
    results_benign = process_image(benign_path, A, CONFIG)
    
    print(f"\n处理恶性图像...")
    results_malignant = process_image(malignant_path, A, CONFIG)
    
    # 打印统计结果
    print("\n" + "=" * 70)
    print("统计结果")
    print("=" * 70)
    
    threshold = CONFIG['threshold']
    benign_pass = print_statistics("良性图像 (benign (1).png)", results_benign['stats'], threshold)
    malignant_pass = print_statistics("恶性图像 (malignant (1).png)", results_malignant['stats'], threshold)
    
    # 汇总表格
    print("\n" + "=" * 70)
    print("综合汇总表")
    print("=" * 70)
    summary_df = create_summary_table(results_benign, results_malignant, threshold)
    print(summary_df.to_string(index=False))
    
    # 假设检验结论
    print("\n" + "=" * 70)
    print("假设检验")
    print("=" * 70)
    print(f"\n假设: 自适应权重图在归一化[0,1]尺度上的标准差 > {threshold}")
    
    all_pass = benign_pass and malignant_pass
    result_str = "✓ 支持" if all_pass else "✗ 不支持"
    print(f"\n结果: {result_str}")
    
    if all_pass:
        print("\n所有四个权重图的标准差都超过阈值:")
        print(f"  - 良性 TV 权重 std:      {results_benign['stats']['std_tv_norm']:.6f} > {threshold} ✓")
        print(f"  - 良性 小波权重 std:     {results_benign['stats']['std_wav_norm']:.6f} > {threshold} ✓")
        print(f"  - 恶性 TV 权重 std:      {results_malignant['stats']['std_tv_norm']:.6f} > {threshold} ✓")
        print(f"  - 恶性 小波权重 std:     {results_malignant['stats']['std_wav_norm']:.6f} > {threshold} ✓")
        print("\n结论: 初始反投影步骤中存在显著的空间信息。")
        print("      将空间变化的权重平均为标量权重会丢失有意义的空间结构。")
    
    print("=" * 70)
    
    # 创建可视化
    output_path = 'adaptive_weight_maps_variance.png'
    create_visualization(results_benign, results_malignant, output_path)
    
    return results_benign, results_malignant


if __name__ == '__main__':
    results = main()
