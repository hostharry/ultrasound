"""
Adaptive-HASA 50次迭代重建实验
===============================

本脚本实现 Adaptive-HASA 算法在30张研究图像（15良性+15恶性）上的重建实验
使用 50 次迭代，15% 采样率，25dB SNR

实验参数：
- 图像大小: 256×256
- 采样率: 15%
- SNR: 25 dB
- 迭代次数: 50
- 测量矩阵种子: 42
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import sobel
from skimage.filters.rank import entropy as skimage_entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import pywt
from scipy import ndimage
from scipy.ndimage import uniform_filter
from scipy.stats import friedmanchisquare
import pandas as pd
import time
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'target_size': (256, 256),
    'sampling_rate': 0.15,
    'target_snr_db': 25.0,
    'measurement_seed': 42,
    'noise_seed_base': 43,
    'n_iterations': 50,
    'base_lambda': 0.01,
    'window_size': 9,
    'wavelet': 'db4',
    'wavelet_level': 3,
    'n_benign': 15,
    'n_malignant': 15,
}


# ==================== 图像加载 ====================

def load_and_preprocess_image(image_path, mask_path, target_size=(256, 256)):
    """
    加载图像和掩码，调整大小，转换为灰度，归一化到[0,1]
    """
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.BILINEAR)
    img = np.array(img, dtype=np.float64) / 255.0
    
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(target_size, Image.NEAREST)
        mask = np.array(mask, dtype=bool)
    else:
        mask = np.zeros(target_size, dtype=bool)
    
    return img, mask


def load_study_set(benign_dir, malignant_dir, n_benign=15, n_malignant=15, seed=42):
    """加载30张研究图像集"""
    # 获取所有图像（排除掩码）
    benign_images = sorted([f for f in os.listdir(benign_dir) 
                           if f.endswith('.png') and '_mask' not in f])
    malignant_images = sorted([f for f in os.listdir(malignant_dir) 
                              if f.endswith('.png') and '_mask' not in f])
    
    print(f"可用良性图像: {len(benign_images)}")
    print(f"可用恶性图像: {len(malignant_images)}")
    
    # 随机选择
    np.random.seed(seed)
    selected_benign = np.random.choice(benign_images, n_benign, replace=False)
    selected_malignant = np.random.choice(malignant_images, n_malignant, replace=False)
    
    # 加载图像
    study_images = []
    study_masks = []
    study_labels = []
    study_names = []
    
    # 加载良性图像
    for img_name in selected_benign:
        img_path = os.path.join(benign_dir, img_name)
        mask_name = img_name.replace('.png', '_mask.png')
        mask_path = os.path.join(benign_dir, mask_name)
        
        img, mask = load_and_preprocess_image(img_path, mask_path, CONFIG['target_size'])
        study_images.append(img)
        study_masks.append(mask)
        study_labels.append('benign')
        study_names.append(img_name)
    
    # 加载恶性图像
    for img_name in selected_malignant:
        img_path = os.path.join(malignant_dir, img_name)
        mask_name = img_name.replace('.png', '_mask.png')
        mask_path = os.path.join(malignant_dir, mask_name)
        
        img, mask = load_and_preprocess_image(img_path, mask_path, CONFIG['target_size'])
        study_images.append(img)
        study_masks.append(mask)
        study_labels.append('malignant')
        study_names.append(img_name)
    
    print(f"\n已加载 {len(study_images)} 张图像")
    print(f"  良性: {study_labels.count('benign')}")
    print(f"  恶性: {study_labels.count('malignant')}")
    
    return study_images, study_masks, study_labels, study_names


# ==================== 压缩感知测量 ====================

def create_measurement_matrix(n_pixels, sampling_rate, seed=42):
    """创建高斯随机测量矩阵"""
    m = int(n_pixels * sampling_rate)
    np.random.seed(seed)
    A = np.random.randn(m, n_pixels) / np.sqrt(m)
    return A


def generate_cs_measurements(image, A, seed_noise=43):
    """
    生成带高斯噪声的压缩感知测量
    """
    x = image.flatten()
    y_clean = A @ x
    
    # 添加高斯噪声达到目标SNR (25 dB)
    np.random.seed(seed_noise)
    signal_power = np.mean(y_clean ** 2)
    noise_power = signal_power / (10 ** (CONFIG['target_snr_db'] / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(y_clean))
    y = y_clean + noise
    
    return y


# ==================== 自适应权重计算 ====================

def calculate_local_gradient_variance(image, window_size):
    """使用滑动窗口计算局部梯度方差"""
    gy, gx = np.gradient(image)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    
    footprint = np.ones((window_size, window_size))
    local_mean = ndimage.convolve(gradient_mag, footprint / footprint.sum(), mode='reflect')
    local_sq_mean = ndimage.convolve(gradient_mag**2, footprint / footprint.sum(), mode='reflect')
    local_var = np.maximum(local_sq_mean - local_mean**2, 0)
    
    return local_var


def calculate_local_entropy(image, window_size):
    """
    使用 skimage 计算真正的局部熵（基于香农熵）
    
    参数:
        image: 输入图像 [0, 1]
        window_size: 窗口大小（用于计算 disk 半径）
        
    返回:
        归一化的局部熵图 [0, 1]
    """
    # 将窗口大小转换为 disk 半径
    radius = max(1, window_size // 2)
    selem = disk(radius)
    
    # skimage.filters.rank.entropy 需要 uint8 图像
    img_uint8 = img_as_ubyte(np.clip(image, 0, 1))
    
    # 计算局部熵
    entropy_map = skimage_entropy(img_uint8, selem)
    
    # 归一化到 [0, 1]
    if entropy_map.max() > 0:
        entropy_norm = entropy_map.astype(np.float64) / entropy_map.max()
    else:
        entropy_norm = entropy_map.astype(np.float64)
    
    return entropy_norm


def compute_adaptive_weights(image, window_size):
    """计算自适应TV和小波权重图"""
    # 梯度图（用于TV权重）
    gradient_map = sobel(image)
    gradient_norm = (gradient_map - gradient_map.min()) / (gradient_map.max() - gradient_map.min() + 1e-10)
    
    # 熵图（用于小波权重）
    entropy_map = calculate_local_entropy(image, window_size)
    entropy_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-10)
    
    return gradient_norm, entropy_norm


# ==================== 辅助函数 ====================

def estimate_lipschitz_constant(A, max_iter=30):
    """使用幂迭代估计Lipschitz常数"""
    n = A.shape[1]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        x = A.T @ (A @ x)
        x = x / np.linalg.norm(x)
    
    L = np.dot(x, A.T @ (A @ x))
    return L


def wavelet_soft_threshold(image, threshold, wavelet='db4', level=3):
    """小波软阈值去噪"""
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    coeffs_thresh = [coeffs[0]]
    for detail_level in coeffs[1:]:
        coeffs_thresh.append(tuple(
            pywt.threshold(c, threshold, mode='soft') for c in detail_level
        ))
    
    reconstructed = pywt.waverec2(coeffs_thresh, wavelet)
    
    if reconstructed.shape != image.shape:
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
    
    return reconstructed


# ==================== 重建算法 ====================

def fista_adaptive_hasa(y, A, base_lambda, n_iter=50, L=None, 
                        window_size=9, wavelet='db4', level=3):
    """
    Adaptive-HASA FISTA重建算法
    
    在每次迭代中计算基于局部梯度和熵的自适应权重
    """
    if L is None:
        L = estimate_lipschitz_constant(A)
    
    n_pixels = A.shape[1]
    img_size = int(np.sqrt(n_pixels))
    
    # 初始化
    x = (A.T @ y).reshape(img_size, img_size)
    z = x.copy()
    t = 1.0
    
    for iter_num in range(n_iter):
        # 梯度下降步骤
        residual = A @ z.flatten() - y
        grad = (A.T @ residual).reshape(img_size, img_size)
        x_intermediate = z - (1/L) * grad
        
        # 计算自适应权重
        gradient_norm, entropy_norm = compute_adaptive_weights(z, window_size)
        
        # 自适应正则化参数
        lambda_tv_map = base_lambda * (1 + gradient_norm)
        lambda_wav_map = base_lambda * (1 + entropy_norm)
        
        # 使用平均权重
        avg_tv_weight = np.mean(lambda_tv_map) / L
        x_new = denoise_tv_chambolle(x_intermediate, weight=avg_tv_weight)
        
        avg_wav_threshold = np.mean(lambda_wav_map) / L
        x_new = wavelet_soft_threshold(x_new, threshold=avg_wav_threshold, 
                                       wavelet=wavelet, level=level)
        
        # FISTA动量更新
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
    
    return np.clip(x, 0, 1)


def fista_tv_only(y, A, lambda_tv, n_iter=50, L=None):
    """TV-only FISTA重建"""
    if L is None:
        L = estimate_lipschitz_constant(A)
    
    n_pixels = A.shape[1]
    img_size = int(np.sqrt(n_pixels))
    
    x = (A.T @ y).reshape(img_size, img_size)
    z = x.copy()
    t = 1.0
    
    for _ in range(n_iter):
        residual = A @ z.flatten() - y
        grad = (A.T @ residual).reshape(img_size, img_size)
        x_new = z - (1/L) * grad
        
        weight = lambda_tv / L
        x_new = denoise_tv_chambolle(x_new, weight=weight)
        
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
    
    return np.clip(x, 0, 1)


# ==================== 评估函数 ====================

def evaluate_reconstruction(original, reconstructed, mask=None):
    """评估重建质量"""
    # 全图指标
    psnr_full = psnr(original, reconstructed, data_range=1.0)
    ssim_full = ssim(original, reconstructed, data_range=1.0)
    
    # ROI指标（如果有掩码）
    if mask is not None and mask.any():
        # 肿瘤区域
        tumor_orig = original[mask]
        tumor_recon = reconstructed[mask]
        psnr_tumor = psnr(tumor_orig, tumor_recon, data_range=1.0)
        ssim_tumor = ssim(original, reconstructed, data_range=1.0, full=True)[1][mask].mean()
        
        # 背景区域
        bg_mask = ~mask
        bg_orig = original[bg_mask]
        bg_recon = reconstructed[bg_mask]
        psnr_bg = psnr(bg_orig, bg_recon, data_range=1.0)
        ssim_bg = ssim(original, reconstructed, data_range=1.0, full=True)[1][bg_mask].mean()
    else:
        psnr_tumor = psnr_full
        ssim_tumor = ssim_full
        psnr_bg = psnr_full
        ssim_bg = ssim_full
    
    return {
        'psnr_full': psnr_full,
        'ssim_full': ssim_full,
        'psnr_tumor': psnr_tumor,
        'ssim_tumor': ssim_tumor,
        'psnr_bg': psnr_bg,
        'ssim_bg': ssim_bg,
    }


# ==================== 主函数 ====================

def main():
    print("=" * 80)
    print("Adaptive-HASA 50次迭代重建实验")
    print("=" * 80)
    
    config = CONFIG
    
    # 打印配置
    print(f"\n实验参数:")
    print(f"  图像大小: {config['target_size']}")
    print(f"  采样率: {config['sampling_rate']*100:.0f}%")
    print(f"  SNR: {config['target_snr_db']} dB")
    print(f"  迭代次数: {config['n_iterations']}")
    print(f"  正则化参数: {config['base_lambda']}")
    
    # 数据路径
    benign_dir = 'Dataset_BUSI_with_GT/benign/'
    malignant_dir = 'Dataset_BUSI_with_GT/malignant/'
    
    # 检查路径
    if not os.path.exists(benign_dir) or not os.path.exists(malignant_dir):
        print(f"\n错误: 数据集目录不存在!")
        print(f"  请确保 {benign_dir} 和 {malignant_dir} 存在")
        return None
    
    # 1. 加载研究图像集
    print("\n[1] 加载研究图像集...")
    study_images, study_masks, study_labels, study_names = load_study_set(
        benign_dir, malignant_dir, 
        config['n_benign'], config['n_malignant']
    )
    
    # 2. 创建测量矩阵
    print("\n[2] 创建测量矩阵...")
    n_pixels = config['target_size'][0] * config['target_size'][1]
    A = create_measurement_matrix(n_pixels, config['sampling_rate'], config['measurement_seed'])
    print(f"  测量矩阵形状: {A.shape}")
    
    # 3. 估计Lipschitz常数
    print("\n[3] 估计Lipschitz常数...")
    L = estimate_lipschitz_constant(A)
    print(f"  L = {L:.4f}")
    
    # 4. 生成测量数据并重建
    print(f"\n[4] 运行 {config['n_iterations']} 次迭代重建...")
    
    results = []
    n_total = len(study_images)
    
    for i, (img, mask, label, name) in enumerate(zip(study_images, study_masks, study_labels, study_names)):
        print(f"\n  [{i+1}/{n_total}] {name} ({label})")
        
        # 生成测量
        y = generate_cs_measurements(img, A, seed_noise=config['noise_seed_base'] + i)
        
        # Adaptive-HASA重建
        print("    运行 Adaptive-HASA...")
        start = time.time()
        recon_adaptive = fista_adaptive_hasa(
            y, A, 
            base_lambda=config['base_lambda'],
            n_iter=config['n_iterations'],
            L=L,
            window_size=config['window_size'],
            wavelet=config['wavelet'],
            level=config['wavelet_level']
        )
        time_adaptive = time.time() - start
        
        # TV-only重建（用于对比）
        print("    运行 TV-only...")
        start = time.time()
        recon_tv = fista_tv_only(
            y, A,
            lambda_tv=config['base_lambda'],
            n_iter=config['n_iterations'],
            L=L
        )
        time_tv = time.time() - start
        
        # 评估
        metrics_adaptive = evaluate_reconstruction(img, recon_adaptive, mask)
        metrics_tv = evaluate_reconstruction(img, recon_tv, mask)
        
        print(f"    Adaptive-HASA: PSNR={metrics_adaptive['psnr_full']:.2f}dB, "
              f"SSIM={metrics_adaptive['ssim_full']:.4f}, time={time_adaptive:.1f}s")
        print(f"    TV-only:       PSNR={metrics_tv['psnr_full']:.2f}dB, "
              f"SSIM={metrics_tv['ssim_full']:.4f}, time={time_tv:.1f}s")
        
        results.append({
            'name': name,
            'label': label,
            'psnr_adaptive': metrics_adaptive['psnr_full'],
            'ssim_adaptive': metrics_adaptive['ssim_full'],
            'psnr_tumor_adaptive': metrics_adaptive['psnr_tumor'],
            'ssim_tumor_adaptive': metrics_adaptive['ssim_tumor'],
            'psnr_tv': metrics_tv['psnr_full'],
            'ssim_tv': metrics_tv['ssim_full'],
            'time_adaptive': time_adaptive,
            'time_tv': time_tv,
        })
    
    # 5. 汇总结果
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    
    # 总体统计
    print("\n总体平均指标:")
    print(f"  Adaptive-HASA:")
    print(f"    PSNR: {df['psnr_adaptive'].mean():.2f} ± {df['psnr_adaptive'].std():.2f} dB")
    print(f"    SSIM: {df['ssim_adaptive'].mean():.4f} ± {df['ssim_adaptive'].std():.4f}")
    print(f"    平均时间: {df['time_adaptive'].mean():.1f}s")
    print(f"  TV-only:")
    print(f"    PSNR: {df['psnr_tv'].mean():.2f} ± {df['psnr_tv'].std():.2f} dB")
    print(f"    SSIM: {df['ssim_tv'].mean():.4f} ± {df['ssim_tv'].std():.4f}")
    print(f"    平均时间: {df['time_tv'].mean():.1f}s")
    
    # 按类型统计
    for label in ['benign', 'malignant']:
        subset = df[df['label'] == label]
        type_name = "良性" if label == "benign" else "恶性"
        print(f"\n{type_name}图像 (n={len(subset)}):")
        print(f"  Adaptive-HASA PSNR: {subset['psnr_adaptive'].mean():.2f} ± {subset['psnr_adaptive'].std():.2f} dB")
        print(f"  Adaptive-HASA SSIM: {subset['ssim_adaptive'].mean():.4f} ± {subset['ssim_adaptive'].std():.4f}")
        print(f"  TV-only PSNR: {subset['psnr_tv'].mean():.2f} ± {subset['psnr_tv'].std():.2f} dB")
        print(f"  TV-only SSIM: {subset['ssim_tv'].mean():.4f} ± {subset['ssim_tv'].std():.4f}")
    
    # Adaptive-HASA vs TV-only 对比
    print("\n方法对比 (Adaptive-HASA vs TV-only):")
    adaptive_wins_psnr = (df['psnr_adaptive'] > df['psnr_tv']).sum()
    adaptive_wins_ssim = (df['ssim_adaptive'] > df['ssim_tv']).sum()
    print(f"  PSNR: Adaptive-HASA 更优 {adaptive_wins_psnr}/{n_total} 例")
    print(f"  SSIM: Adaptive-HASA 更优 {adaptive_wins_ssim}/{n_total} 例")
    
    # 保存结果
    output_file = 'adaptive_hasa_50iter_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    
    print("=" * 80)
    
    return df


if __name__ == '__main__':
    results = main()
