"""
HASA-ISTA 算法：30 张图像重建实验
================================

本模块实现了 HASA-ISTA (Hybrid Adaptive Sparse Approximation - 
Iterative Soft Thresholding Algorithm) 对 30 张超声图像的重建实验。

实验设置：
    - 30 张图像：15 良性 + 15 恶性
    - 采样率：30%
    - SNR：25 dB
    - 图像尺寸：128×128
    
算法参数：
    - 迭代次数：50
    - 步长 α：0.001
    - 小波正则化参数 λ_w：0.01
    - TV 正则化参数 λ_tv：0.01
    - 权重更新频率：每 5 次迭代

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import os
from PIL import Image
import pywt
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import sobel, generic_filter
from scipy.stats import entropy as scipy_entropy
import matplotlib.pyplot as plt
import time
import glob
import warnings
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 图像参数
    'target_size': (128, 128),
    
    # 数据集参数
    'n_benign': 15,
    'n_malignant': 15,
    'benign_dir': 'Dataset_BUSI_with_GT/benign/',
    'malignant_dir': 'Dataset_BUSI_with_GT/malignant/',
    
    # 压缩感知参数
    'sampling_rate': 0.30,              # 采样率 30%
    'snr_db': 25,                       # 信噪比 25 dB
    'random_seed': 42,
    
    # HASA-ISTA 参数
    'n_iterations': 50,                 # 迭代次数
    'alpha': 0.001,                     # 梯度下降步长
    'lambda_w': 0.01,                   # 小波正则化参数
    'lambda_tv': 0.01,                  # TV 正则化参数
    'update_weights_every': 5,          # 权重更新频率
    
    # 小波参数
    'wavelet': 'db4',
    'wavelet_level': 3,
    
    # 局部统计参数
    'entropy_window_size': 5,
    
    # 输出参数
    'output_dir': 'hasa_ista_reconstructions',
    'save_images': True,
    'verbose': True,
}


# ============================================================================
# 图像加载和预处理
# ============================================================================

def select_images(benign_dir: str, malignant_dir: str, 
                  n_benign: int, n_malignant: int, 
                  seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    选择指定数量的良性和恶性图像
    
    参数:
        benign_dir: 良性图像目录
        malignant_dir: 恶性图像目录
        n_benign: 良性图像数量
        n_malignant: 恶性图像数量
        seed: 随机种子
        
    返回:
        (良性文件列表, 恶性文件列表)
    """
    # 获取所有图像文件（排除 mask）
    benign_files = sorted([f for f in os.listdir(benign_dir) 
                          if f.endswith('.png') and '_mask' not in f])
    malignant_files = sorted([f for f in os.listdir(malignant_dir) 
                             if f.endswith('.png') and '_mask' not in f])
    
    print(f"总良性图像数: {len(benign_files)}")
    print(f"总恶性图像数: {len(malignant_files)}")
    
    # 随机选择
    np.random.seed(seed)
    selected_benign_idx = np.random.choice(len(benign_files), n_benign, replace=False)
    selected_malignant_idx = np.random.choice(len(malignant_files), n_malignant, replace=False)
    
    selected_benign = [benign_files[i] for i in sorted(selected_benign_idx)]
    selected_malignant = [malignant_files[i] for i in sorted(selected_malignant_idx)]
    
    print(f"\n选中 {len(selected_benign)} 张良性图像")
    print(f"选中 {len(selected_malignant)} 张恶性图像")
    print(f"总计: {len(selected_benign) + len(selected_malignant)}")
    
    return selected_benign, selected_malignant


def load_and_preprocess_image(filepath: str, 
                              target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    加载图像，转换为灰度，缩放，归一化到 [0, 1]
    
    参数:
        filepath: 图像路径
        target_size: 目标尺寸
        
    返回:
        归一化的灰度图像
    """
    img = Image.open(filepath).convert('L')
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def load_all_images(benign_dir: str, malignant_dir: str,
                    selected_benign: List[str], selected_malignant: List[str],
                    target_size: Tuple[int, int] = (128, 128)
                    ) -> Tuple[np.ndarray, List[str]]:
    """
    加载所有选中的图像
    
    参数:
        benign_dir: 良性图像目录
        malignant_dir: 恶性图像目录
        selected_benign: 选中的良性文件名列表
        selected_malignant: 选中的恶性文件名列表
        target_size: 目标尺寸
        
    返回:
        (图像数组, 标签列表)
    """
    images = []
    labels = []
    
    # 加载良性图像
    for i, filename in enumerate(selected_benign):
        filepath = os.path.join(benign_dir, filename)
        img = load_and_preprocess_image(filepath, target_size)
        images.append(img)
        labels.append(f'benign_{i+1}')
    
    # 加载恶性图像
    for i, filename in enumerate(selected_malignant):
        filepath = os.path.join(malignant_dir, filename)
        img = load_and_preprocess_image(filepath, target_size)
        images.append(img)
        labels.append(f'malignant_{i+1}')
    
    return np.array(images), labels


# ============================================================================
# 压缩感知测量
# ============================================================================

def generate_cs_measurements(images: np.ndarray, sampling_rate: float = 0.30,
                             snr_db: float = 25, seed: int = 42
                             ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    生成压缩感知测量值
    
    参数:
        images: 图像数组 (N, H, W)
        sampling_rate: 采样率
        snr_db: 信噪比 (dB)
        seed: 随机种子
        
    返回:
        (测量向量列表, 测量矩阵 Φ)
    """
    np.random.seed(seed)
    
    n_images, height, width = images.shape
    n = height * width
    m = int(sampling_rate * n)
    
    print(f"图像尺寸: {height}×{width} = {n} 像素")
    print(f"采样率: {sampling_rate*100}%")
    print(f"测量数量: {m} (共 {n})")
    
    # 生成高斯随机测量矩阵
    Phi = np.random.randn(m, n) / np.sqrt(m)
    
    # 生成每张图像的测量值
    measurement_vectors = []
    
    for img in images:
        x = img.flatten()
        y_clean = Phi @ x
        
        # 添加高斯噪声
        signal_power = np.mean(y_clean ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(m) * np.sqrt(noise_power)
        
        y_noisy = y_clean + noise
        measurement_vectors.append(y_noisy)
    
    print(f"生成 {len(measurement_vectors)} 个测量向量")
    
    return measurement_vectors, Phi


# ============================================================================
# 辅助函数
# ============================================================================

def compute_local_entropy(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    计算局部熵图
    
    参数:
        img: 输入图像
        window_size: 窗口大小
        
    返回:
        局部熵图
    """
    def local_entropy_func(values):
        hist, _ = np.histogram(values, bins=10, range=(0, 1), density=True)
        hist = hist + 1e-10
        return scipy_entropy(hist)
    
    entropy_map = generic_filter(img, local_entropy_func, 
                                 size=window_size, mode='reflect')
    return entropy_map


def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """
    计算梯度幅度图
    
    参数:
        img: 输入图像
        
    返回:
        梯度幅度图
    """
    grad_x = sobel(img, axis=0)
    grad_y = sobel(img, axis=1)
    gradient_map = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_map


def normalize_map(map_array: np.ndarray) -> np.ndarray:
    """
    归一化到 [0, 1]
    
    参数:
        map_array: 输入数组
        
    返回:
        归一化的数组
    """
    map_min = map_array.min()
    map_max = map_array.max()
    if map_max - map_min > 1e-10:
        return (map_array - map_min) / (map_max - map_min)
    else:
        return np.ones_like(map_array)


# ============================================================================
# 小波软阈值
# ============================================================================

def wavelet_soft_threshold(img: np.ndarray, threshold: float,
                           wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """
    在小波域应用软阈值
    
    参数:
        img: 输入图像
        threshold: 阈值（标量或数组）
        wavelet: 小波基
        level: 分解层数
        
    返回:
        阈值处理后的图像
    """
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    
    coeffs_thresh = [coeffs[0]]  # 保持近似系数
    
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        
        # 如果阈值是数组，使用均值
        if isinstance(threshold, np.ndarray):
            thresh_val = np.mean(threshold)
        else:
            thresh_val = threshold
        
        cH_thresh = pywt.threshold(cH, thresh_val, mode='soft')
        cV_thresh = pywt.threshold(cV, thresh_val, mode='soft')
        cD_thresh = pywt.threshold(cD, thresh_val, mode='soft')
        
        coeffs_thresh.append((cH_thresh, cV_thresh, cD_thresh))
    
    img_recon = pywt.waverec2(coeffs_thresh, wavelet)
    
    # 确保输出尺寸匹配
    img_recon = img_recon[:img.shape[0], :img.shape[1]]
    
    return img_recon


# ============================================================================
# HASA-ISTA 算法
# ============================================================================

def hasa_ista_reconstruct(y: np.ndarray, Phi: np.ndarray,
                          alpha: float = 0.001,
                          lambda_w: float = 0.01,
                          lambda_tv: float = 0.01,
                          n_iterations: int = 50,
                          img_shape: Tuple[int, int] = (128, 128),
                          update_weights_every: int = 5,
                          wavelet: str = 'db4',
                          entropy_window: int = 5) -> np.ndarray:
    """
    HASA-ISTA 重建算法
    
    参数:
        y: 测量向量
        Phi: 测量矩阵 (m × n)
        alpha: 梯度下降步长
        lambda_w: 小波正则化参数
        lambda_tv: TV 正则化参数
        n_iterations: 迭代次数
        img_shape: 图像形状 (H, W)
        update_weights_every: 权重更新频率
        wavelet: 小波类型
        entropy_window: 熵计算窗口大小
        
    返回:
        重建图像 (H, W)
    """
    m, n = Phi.shape
    
    # 反投影初始化
    x = Phi.T @ y
    x = x.reshape(img_shape)
    
    # 初始化自适应权重图
    entropy_map = np.ones(img_shape)
    gradient_map = np.ones(img_shape)
    
    for k in range(n_iterations):
        # a. 梯度下降步骤
        x_flat = x.flatten()
        residual = Phi @ x_flat - y
        gradient = Phi.T @ residual
        z = x_flat - alpha * gradient
        z = z.reshape(img_shape)
        
        # b. 更新自适应权重（每 N 次迭代）
        if k % update_weights_every == 0:
            # 计算局部熵图
            entropy_map = compute_local_entropy(np.clip(x, 0, 1), 
                                                window_size=entropy_window)
            entropy_map = normalize_map(entropy_map)
            
            # 计算梯度幅度图
            gradient_map = compute_gradient_magnitude(x)
            gradient_map = normalize_map(gradient_map)
        
        # c. 双近端步骤
        # 自适应小波软阈值
        # 高熵 = 更多结构 = 更强的小波正则化
        lambda_w_adaptive = lambda_w * (1 + entropy_map)
        x_wavelet = wavelet_soft_threshold(z, lambda_w_adaptive, 
                                           wavelet=wavelet, level=3)
        
        # 自适应 TV 去噪
        # 高梯度 = 边缘 = 降低 TV 正则化以保护边缘
        lambda_tv_adaptive = lambda_tv * (1 - gradient_map * 0.5)
        tv_weight_mean = np.mean(lambda_tv_adaptive)
        
        # 应用 TV 去噪
        x = denoise_tv_chambolle(x_wavelet, weight=tv_weight_mean, max_num_iter=10)
        
        # 裁剪到有效范围
        x = np.clip(x, 0, 1)
    
    return x


# ============================================================================
# 评估函数
# ============================================================================

def compute_metrics(original: np.ndarray, 
                    reconstructed: np.ndarray) -> Dict[str, float]:
    """
    计算重建质量指标
    
    参数:
        original: 原始图像
        reconstructed: 重建图像
        
    返回:
        指标字典
    """
    ssim_val = ssim(original, reconstructed, data_range=1.0)
    psnr_val = psnr(original, reconstructed, data_range=1.0)
    mse_val = np.mean((original - reconstructed) ** 2)
    
    return {
        'ssim': ssim_val,
        'psnr': psnr_val,
        'mse': mse_val
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_quality_distribution(ssim_scores: np.ndarray, 
                              labels: List[str],
                              output_path: Optional[str] = None):
    """
    绘制重建质量分布图
    
    参数:
        ssim_scores: SSIM 分数数组
        labels: 标签列表
        output_path: 输出路径
    """
    n_benign = sum(1 for l in labels if 'benign' in l)
    n_malignant = len(labels) - n_benign
    
    benign_ssim = ssim_scores[:n_benign]
    malignant_ssim = ssim_scores[n_benign:]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x_pos = np.arange(len(labels))
    colors = ['steelblue'] * n_benign + ['coral'] * n_malignant
    
    ax.bar(x_pos, ssim_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(ssim_scores.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean SSIM: {ssim_scores.mean():.3f}')
    ax.axhline(benign_ssim.mean(), color='steelblue', linestyle=':', linewidth=2, 
               label=f'Benign: {benign_ssim.mean():.3f}')
    ax.axhline(malignant_ssim.mean(), color='coral', linestyle=':', linewidth=2, 
               label=f'Malignant: {malignant_ssim.mean():.3f}')
    
    ax.set_xlabel('Image Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('SSIM Score', fontsize=12, fontweight='bold')
    ax.set_title('HASA-ISTA Reconstruction Quality (SSIM)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_reconstruction_examples(images: np.ndarray, 
                                reconstructed: np.ndarray,
                                labels: List[str],
                                ssim_scores: np.ndarray,
                                mse_scores: np.ndarray,
                                indices: List[int],
                                output_path: Optional[str] = None):
    """
    绘制重建对比示例
    
    参数:
        images: 原始图像数组
        reconstructed: 重建图像数组
        labels: 标签列表
        ssim_scores: SSIM 分数数组
        mse_scores: MSE 分数数组
        indices: 要显示的图像索引
        output_path: 输出路径
    """
    n_examples = len(indices)
    fig, axes = plt.subplots(n_examples, 3, figsize=(10, 3*n_examples))
    
    for i, idx in enumerate(indices):
        # 原始图像
        axes[i, 0].imshow(images[idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'{labels[idx]}\nOriginal', fontsize=10)
        axes[i, 0].axis('off')
        
        # 重建图像
        axes[i, 1].imshow(reconstructed[idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Reconstructed\nSSIM: {ssim_scores[idx]:.3f}', fontsize=10)
        axes[i, 1].axis('off')
        
        # 差异图
        diff = np.abs(images[idx] - reconstructed[idx])
        im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[i, 2].set_title(f'Abs Diff\nMSE: {mse_scores[idx]:.4f}', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# 主函数
# ============================================================================

def run_hasa_ista_experiment(config: dict = None) -> Dict:
    """
    运行 HASA-ISTA 重建实验
    
    参数:
        config: 配置参数
        
    返回:
        结果字典
    """
    if config is None:
        config = CONFIG
    
    np.random.seed(config['random_seed'])
    
    print("=" * 70)
    print("HASA-ISTA 算法：30 张图像重建实验")
    print("=" * 70)
    
    # 检查数据集
    if not os.path.exists(config['benign_dir']):
        print(f"错误: 数据集路径不存在 - {config['benign_dir']}")
        return None
    
    # Step 1: 选择图像
    print("\n[Step 1] 选择图像...")
    selected_benign, selected_malignant = select_images(
        config['benign_dir'],
        config['malignant_dir'],
        config['n_benign'],
        config['n_malignant'],
        config['random_seed']
    )
    
    # Step 2: 加载图像
    print("\n[Step 2] 加载和预处理图像...")
    images, labels = load_all_images(
        config['benign_dir'],
        config['malignant_dir'],
        selected_benign,
        selected_malignant,
        config['target_size']
    )
    print(f"加载 {len(images)} 张图像，形状: {images.shape}")
    print(f"值范围: [{images.min():.3f}, {images.max():.3f}]")
    
    # Step 3: 生成压缩感知测量
    print("\n[Step 3] 生成压缩感知测量...")
    measurement_vectors, Phi = generate_cs_measurements(
        images,
        sampling_rate=config['sampling_rate'],
        snr_db=config['snr_db'],
        seed=config['random_seed']
    )
    print(f"测量矩阵形状: {Phi.shape}")
    
    # Step 4: 运行 HASA-ISTA 重建
    print("\n[Step 4] 运行 HASA-ISTA 重建...")
    print(f"参数: {config['n_iterations']} 迭代, α={config['alpha']}, "
          f"λ_w={config['lambda_w']}, λ_tv={config['lambda_tv']}")
    print(f"权重更新频率: 每 {config['update_weights_every']} 次迭代")
    print("-" * 70)
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    reconstructed_images = []
    reconstruction_times = []
    
    start_total = time.time()
    
    for i, (y, label) in enumerate(tqdm(zip(measurement_vectors, labels), 
                                        total=len(labels), desc="重建进度")):
        start_time = time.time()
        
        x_recon = hasa_ista_reconstruct(
            y=y,
            Phi=Phi,
            alpha=config['alpha'],
            lambda_w=config['lambda_w'],
            lambda_tv=config['lambda_tv'],
            n_iterations=config['n_iterations'],
            img_shape=config['target_size'],
            update_weights_every=config['update_weights_every'],
            wavelet=config['wavelet'],
            entropy_window=config['entropy_window_size']
        )
        
        elapsed = time.time() - start_time
        reconstruction_times.append(elapsed)
        reconstructed_images.append(x_recon)
        
        # 保存重建图像
        if config['save_images']:
            output_filename = f"{label}_recon_AdaptiveHASA.png"
            output_path = os.path.join(config['output_dir'], output_filename)
            img_to_save = (x_recon * 255).astype(np.uint8)
            Image.fromarray(img_to_save, mode='L').save(output_path)
    
    total_time = time.time() - start_total
    reconstructed_images = np.array(reconstructed_images)
    
    print("-" * 70)
    print(f"重建完成!")
    print(f"总时间: {total_time:.2f}s ({total_time/60:.2f} 分钟)")
    print(f"平均每张: {np.mean(reconstruction_times):.2f}s")
    
    # Step 5: 计算质量指标
    print("\n[Step 5] 计算重建质量指标...")
    
    ssim_scores = []
    psnr_scores = []
    mse_scores = []
    
    for i in range(len(images)):
        metrics = compute_metrics(images[i], reconstructed_images[i])
        ssim_scores.append(metrics['ssim'])
        psnr_scores.append(metrics['psnr'])
        mse_scores.append(metrics['mse'])
    
    ssim_scores = np.array(ssim_scores)
    psnr_scores = np.array(psnr_scores)
    mse_scores = np.array(mse_scores)
    
    # Step 6: 打印汇总
    print("\n" + "=" * 70)
    print("重建质量指标汇总 (30 张图像)")
    print("=" * 70)
    print(f"SSIM - 均值: {ssim_scores.mean():.4f}, 标准差: {ssim_scores.std():.4f}")
    print(f"       范围: [{ssim_scores.min():.4f}, {ssim_scores.max():.4f}]")
    print(f"\nPSNR - 均值: {psnr_scores.mean():.2f} dB, 标准差: {psnr_scores.std():.2f} dB")
    print(f"       范围: [{psnr_scores.min():.2f}, {psnr_scores.max():.2f}] dB")
    print(f"\nMSE  - 均值: {mse_scores.mean():.6f}, 标准差: {mse_scores.std():.6f}")
    
    # 按类别比较
    benign_ssim = ssim_scores[:config['n_benign']]
    malignant_ssim = ssim_scores[config['n_benign']:]
    benign_psnr = psnr_scores[:config['n_benign']]
    malignant_psnr = psnr_scores[config['n_benign']:]
    
    print("\n" + "=" * 70)
    print("良性 vs 恶性比较")
    print("=" * 70)
    print(f"良性 SSIM:    {benign_ssim.mean():.4f} ± {benign_ssim.std():.4f}")
    print(f"恶性 SSIM:    {malignant_ssim.mean():.4f} ± {malignant_ssim.std():.4f}")
    print(f"\n良性 PSNR:    {benign_psnr.mean():.2f} ± {benign_psnr.std():.2f} dB")
    print(f"恶性 PSNR:    {malignant_psnr.mean():.2f} ± {malignant_psnr.std():.2f} dB")
    
    # Step 7: 保存可视化
    print("\n[Step 7] 保存可视化...")
    
    # 质量分布图
    plot_quality_distribution(
        ssim_scores, labels,
        os.path.join(config['output_dir'], 'quality_distribution.png')
    )
    print(f"已保存: quality_distribution.png")
    
    # 重建示例
    example_indices = [0, 7, config['n_benign'], config['n_benign'] + 7]
    plot_reconstruction_examples(
        images, reconstructed_images, labels,
        ssim_scores, mse_scores, example_indices,
        os.path.join(config['output_dir'], 'reconstruction_examples.png')
    )
    print(f"已保存: reconstruction_examples.png")
    
    print("\n" + "=" * 70)
    print("最终汇总")
    print("=" * 70)
    print(f"✓ HASA-ISTA 算法成功处理 {len(reconstructed_images)} 张图像")
    print(f"✓ 总计算时间: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
    print(f"✓ 平均每张: {np.mean(reconstruction_times):.1f}s")
    print(f"✓ 所有重建图像已保存至: {config['output_dir']}/")
    print(f"\n重建质量:")
    print(f"  - 总体 SSIM: {ssim_scores.mean():.4f} ± {ssim_scores.std():.4f}")
    print(f"  - 总体 PSNR: {psnr_scores.mean():.2f} ± {psnr_scores.std():.2f} dB")
    print("=" * 70)
    
    return {
        'images': images,
        'reconstructed': reconstructed_images,
        'labels': labels,
        'ssim_scores': ssim_scores,
        'psnr_scores': psnr_scores,
        'mse_scores': mse_scores,
        'reconstruction_times': reconstruction_times,
        'total_time': total_time
    }


def main():
    """主函数"""
    results = run_hasa_ista_experiment(CONFIG)
    return results


if __name__ == '__main__':
    results = main()
