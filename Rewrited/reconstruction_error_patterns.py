"""
重建误差模式分析
================

本模块分析 TV-only 和 Adaptive-HASA 两种压缩感知重建方法的误差分布，
特别关注肿瘤边界区域的误差模式。

分析内容：
    1. 选择代表性图像（良性和恶性各一张）
    2. 模拟压缩感知（30% 采样率，25 dB SNR）
    3. TV-only 重建（50 迭代，步长 0.001）
    4. Adaptive-HASA 重建（结合 TV 和小波正则化）
    5. 生成带肿瘤边界的差异图
    6. 分析边界区域误差模式

实验设置：
    - 图像尺寸：128×128
    - 采样率：30%
    - SNR：25 dB
    - 迭代次数：50

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from scipy.ndimage import sobel, binary_dilation, binary_erosion
from skimage.restoration import denoise_tv_chambolle
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pywt
import time
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 数据集参数
    'base_dir': 'Dataset_BUSI_with_GT',
    'benign_image': 'benign (50).png',
    'malignant_image': 'malignant (50).png',
    'target_size': (128, 128),
    
    # 压缩感知参数
    'sampling_rate': 0.30,              # 采样率 30%
    'snr_db': 25,                       # 信噪比 25 dB
    'random_seed': 42,
    
    # 重建参数
    'max_iter': 50,                     # 迭代次数
    'step_size': 0.001,                 # 梯度下降步长
    'tv_weight': 0.1,                   # TV 权重
    'wavelet_threshold': 0.05,          # 小波阈值
    'update_weights_every': 5,          # 权重更新频率
    
    # 边界提取参数
    'boundary_dilation': 2,
    
    # 输出参数
    'verbose': True,
    'save_figures': True,
    'output_dir': '.',
}


# ============================================================================
# 图像加载与预处理
# ============================================================================

def load_and_preprocess_image(img_path: str, 
                               target_size: Tuple[int, int] = (128, 128)
                               ) -> np.ndarray:
    """
    加载图像，转换为灰度，缩放，归一化到 [0, 1]
    
    参数:
        img_path: 图像路径
        target_size: 目标尺寸
        
    返回:
        归一化的灰度图像
    """
    img = Image.open(img_path).convert('L')
    img_resized = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float64) / 255.0
    return img_array


def load_and_preprocess_mask(mask_path: str, 
                              target_size: Tuple[int, int] = (128, 128)
                              ) -> np.ndarray:
    """
    加载掩模，缩放，二值化
    
    参数:
        mask_path: 掩模路径
        target_size: 目标尺寸
        
    返回:
        二值掩模
    """
    mask = Image.open(mask_path).convert('L')
    mask_resized = mask.resize(target_size, Image.NEAREST)
    mask_array = np.array(mask_resized, dtype=np.float64)
    mask_binary = (mask_array > 128).astype(np.float64)
    return mask_binary


# ============================================================================
# 压缩感知测量
# ============================================================================

def generate_cs_measurements(image: np.ndarray, sampling_rate: float = 0.3,
                              snr_db: float = 25, seed: int = 42
                              ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    生成压缩感知测量值
    
    参数:
        image: 原始图像
        sampling_rate: 采样率
        snr_db: 信噪比 (dB)
        seed: 随机种子
        
    返回:
        (测量矩阵, 测量向量, 测量数, 像素数)
    """
    np.random.seed(seed)
    
    x = image.flatten()
    n = len(x)
    m = int(n * sampling_rate)
    
    # 高斯测量矩阵
    Phi = np.random.randn(m, n) / np.sqrt(m)
    
    # 生成测量值
    y_clean = Phi @ x
    
    # 添加噪声
    signal_power = np.mean(y_clean ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(m)
    y = y_clean + noise
    
    return Phi, y, m, n


# ============================================================================
# TV-only 重建
# ============================================================================

def tv_reconstruction(Phi: np.ndarray, y: np.ndarray, img_shape: Tuple[int, int],
                      max_iter: int = 50, step_size: float = 0.001, 
                      tv_weight: float = 0.1, verbose: bool = True) -> np.ndarray:
    """
    TV-only 重建（近端梯度下降）
    
    参数:
        Phi: 测量矩阵
        y: 测量向量
        img_shape: 图像形状
        max_iter: 迭代次数
        step_size: 步长
        tv_weight: TV 权重
        verbose: 是否打印进度
        
    返回:
        重建图像
    """
    n = img_shape[0] * img_shape[1]
    
    # 反投影初始化
    x = Phi.T @ y
    
    for i in range(max_iter):
        # 数据保真度梯度
        residual = Phi @ x - y
        gradient = Phi.T @ residual
        
        # 梯度下降
        x = x - step_size * gradient
        
        # TV 近端算子
        x_img = x.reshape(img_shape)
        x_img = denoise_tv_chambolle(x_img, weight=tv_weight)
        x = x_img.flatten()
        
        if verbose and (i + 1) % 10 == 0:
            mse = np.mean(residual ** 2)
            print(f"  迭代 {i+1}/{max_iter}, MSE: {mse:.6f}")
    
    return x.reshape(img_shape)


# ============================================================================
# Adaptive-HASA 重建
# ============================================================================

def compute_adaptive_weights(image: np.ndarray, 
                              window_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算自适应权重（基于局部梯度和熵）
    
    参数:
        image: 输入图像
        window_size: 窗口大小
        
    返回:
        (TV 权重图, 小波权重图)
    """
    # 计算局部梯度幅度
    grad_x = sobel(image, axis=0)
    grad_y = sobel(image, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 归一化
    grad_mag_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    
    # 计算局部熵
    img_uint8 = (image * 255).astype(np.uint8)
    local_entropy = entropy(img_uint8, disk(window_size // 2))
    local_entropy_norm = (local_entropy - local_entropy.min()) / (local_entropy.max() - local_entropy.min() + 1e-8)
    
    # TV 权重：平滑区域更高（低梯度、低熵）
    tv_weight = 1.0 - 0.5 * (grad_mag_norm + local_entropy_norm)
    
    # 小波权重：纹理区域更高（高梯度、高熵）
    wavelet_weight = 0.5 * (grad_mag_norm + local_entropy_norm)
    
    return tv_weight, wavelet_weight


def wavelet_soft_threshold(image: np.ndarray, threshold: float, 
                           wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """
    小波软阈值
    
    参数:
        image: 输入图像
        threshold: 阈值
        wavelet: 小波基
        level: 分解层数
        
    返回:
        阈值处理后的图像
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    coeffs_thresh = [coeffs[0]]
    for detail_level in coeffs[1:]:
        coeffs_thresh.append(tuple([
            pywt.threshold(c, threshold, mode='soft') for c in detail_level
        ]))
    
    img_recon = pywt.waverec2(coeffs_thresh, wavelet)
    
    if img_recon.shape != image.shape:
        img_recon = img_recon[:image.shape[0], :image.shape[1]]
    
    return img_recon


def adaptive_hasa_reconstruction(Phi: np.ndarray, y: np.ndarray, 
                                  img_shape: Tuple[int, int],
                                  max_iter: int = 50, step_size: float = 0.001,
                                  tv_weight_base: float = 0.1, 
                                  wavelet_threshold: float = 0.05,
                                  update_weights_every: int = 5,
                                  verbose: bool = True) -> np.ndarray:
    """
    Adaptive-HASA 重建（结合 TV 和小波正则化）
    
    参数:
        Phi: 测量矩阵
        y: 测量向量
        img_shape: 图像形状
        max_iter: 迭代次数
        step_size: 步长
        tv_weight_base: TV 基础权重
        wavelet_threshold: 小波阈值
        update_weights_every: 权重更新频率
        verbose: 是否打印进度
        
    返回:
        重建图像
    """
    n = img_shape[0] * img_shape[1]
    
    # 反投影初始化
    x = Phi.T @ y
    
    # 初始化自适应权重
    x_img = x.reshape(img_shape)
    tv_weights, wavelet_weights = compute_adaptive_weights(x_img)
    
    for i in range(max_iter):
        # 数据保真度梯度
        residual = Phi @ x - y
        gradient = Phi.T @ residual
        
        # 梯度下降
        x = x - step_size * gradient
        x_img = x.reshape(img_shape)
        
        # TV 去噪
        tv_denoised = denoise_tv_chambolle(x_img, weight=tv_weight_base)
        
        # 小波软阈值
        wavelet_denoised = wavelet_soft_threshold(x_img, threshold=wavelet_threshold)
        
        # 使用自适应权重组合
        x_img = tv_weights * tv_denoised + wavelet_weights * wavelet_denoised
        x = x_img.flatten()
        
        # 定期更新权重
        if (i + 1) % update_weights_every == 0:
            tv_weights, wavelet_weights = compute_adaptive_weights(x_img)
            if verbose:
                mse = np.mean(residual ** 2)
                print(f"  迭代 {i+1}/{max_iter}, MSE: {mse:.6f}, 权重已更新")
        elif verbose and (i + 1) % 10 == 0:
            mse = np.mean(residual ** 2)
            print(f"  迭代 {i+1}/{max_iter}, MSE: {mse:.6f}")
    
    return x.reshape(img_shape)


# ============================================================================
# 边界提取
# ============================================================================

def extract_boundary(mask: np.ndarray, dilation_size: int = 2) -> np.ndarray:
    """
    从掩模提取边界
    
    参数:
        mask: 二值掩模
        dilation_size: 膨胀大小
        
    返回:
        边界掩模
    """
    dilated = binary_dilation(mask, iterations=dilation_size)
    eroded = binary_erosion(mask, iterations=dilation_size)
    boundary = dilated.astype(int) - eroded.astype(int)
    return boundary > 0


# ============================================================================
# 评估函数
# ============================================================================

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    """计算 PSNR"""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_boundary_error_stats(error_map: np.ndarray, 
                                  boundary_mask: np.ndarray) -> Dict[str, float]:
    """
    计算边界区域误差统计
    
    参数:
        error_map: 误差图
        boundary_mask: 边界掩模
        
    返回:
        统计字典
    """
    boundary_errors = error_map[boundary_mask]
    non_boundary_errors = error_map[~boundary_mask]
    
    return {
        'boundary_mean': np.mean(boundary_errors),
        'boundary_std': np.std(boundary_errors),
        'boundary_max': np.max(boundary_errors),
        'non_boundary_mean': np.mean(non_boundary_errors),
        'non_boundary_std': np.std(non_boundary_errors),
        'non_boundary_max': np.max(non_boundary_errors)
    }


# ============================================================================
# 可视化函数
# ============================================================================

def create_image_with_boundary_overlay(image: np.ndarray, boundary: np.ndarray,
                                        color: str = 'red') -> np.ndarray:
    """创建带边界叠加的 RGB 图像"""
    img_rgb = np.stack([image, image, image], axis=2)
    
    if color == 'red':
        img_rgb[boundary, 0] = 1.0
        img_rgb[boundary, 1] = 0.0
        img_rgb[boundary, 2] = 0.0
    elif color == 'yellow':
        img_rgb[boundary, 0] = 1.0
        img_rgb[boundary, 1] = 1.0
        img_rgb[boundary, 2] = 0.0
    elif color == 'cyan':
        img_rgb[boundary, 0] = 0.0
        img_rgb[boundary, 1] = 1.0
        img_rgb[boundary, 2] = 1.0
    
    return img_rgb


def plot_error_comparison(tv_error: np.ndarray, hasa_error: np.ndarray,
                          boundary: np.ndarray, case_name: str,
                          tv_stats: Dict, hasa_stats: Dict,
                          output_path: Optional[str] = None):
    """
    绘制误差对比图
    
    参数:
        tv_error: TV-only 误差图
        hasa_error: Adaptive-HASA 误差图
        boundary: 边界掩模
        case_name: 案例名称
        tv_stats: TV 统计
        hasa_stats: HASA 统计
        output_path: 输出路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # TV-only 误差图
    im1 = axes[0].imshow(tv_error, cmap='hot', vmin=0, vmax=0.3)
    axes[0].contour(boundary, colors='cyan', linewidths=2, levels=[0.5])
    axes[0].set_title(f'{case_name} - TV-only Error Map\n' +
                      f'Overall: {tv_error.mean():.4f}, Boundary: {tv_stats["boundary_mean"]:.4f}',
                      fontsize=11, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Adaptive-HASA 误差图
    im2 = axes[1].imshow(hasa_error, cmap='hot', vmin=0, vmax=0.3)
    axes[1].contour(boundary, colors='cyan', linewidths=2, levels=[0.5])
    axes[1].set_title(f'{case_name} - Adaptive-HASA Error Map\n' +
                      f'Overall: {hasa_error.mean():.4f}, Boundary: {hasa_stats["boundary_mean"]:.4f}',
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_detailed_analysis(gt: np.ndarray, tv_recon: np.ndarray, 
                            hasa_recon: np.ndarray, boundary: np.ndarray,
                            case_name: str, output_path: Optional[str] = None):
    """
    绘制详细分析图
    
    参数:
        gt: 真实图像
        tv_recon: TV 重建
        hasa_recon: HASA 重建
        boundary: 边界
        case_name: 案例名称
        output_path: 输出路径
    """
    tv_error = np.abs(tv_recon - gt)
    hasa_error = np.abs(hasa_recon - gt)
    
    fig, axes = plt.subplots(5, 1, figsize=(8, 20))
    
    # 真实图像
    axes[0].imshow(create_image_with_boundary_overlay(gt, boundary))
    axes[0].set_title(f'{case_name} - Ground Truth with Tumor Boundary', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # TV 重建
    axes[1].imshow(create_image_with_boundary_overlay(tv_recon, boundary))
    axes[1].set_title(f'{case_name} - TV-only Reconstruction', 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # HASA 重建
    axes[2].imshow(create_image_with_boundary_overlay(hasa_recon, boundary))
    axes[2].set_title(f'{case_name} - Adaptive-HASA Reconstruction', 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # TV 误差图
    im1 = axes[3].imshow(tv_error, cmap='hot', vmin=0, vmax=0.3)
    axes[3].contour(boundary, colors='cyan', linewidths=2, levels=[0.5])
    axes[3].set_title(f'{case_name} - TV-only Error (Mean: {tv_error.mean():.4f})', 
                      fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im1, ax=axes[3], fraction=0.046)
    
    # HASA 误差图
    im2 = axes[4].imshow(hasa_error, cmap='hot', vmin=0, vmax=0.3)
    axes[4].contour(boundary, colors='cyan', linewidths=2, levels=[0.5])
    axes[4].set_title(f'{case_name} - Adaptive-HASA Error (Mean: {hasa_error.mean():.4f})', 
                      fontsize=12, fontweight='bold')
    axes[4].axis('off')
    plt.colorbar(im2, ax=axes[4], fraction=0.046)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# 主函数
# ============================================================================

def run_error_analysis(config: dict = None) -> Dict:
    """
    运行重建误差分析
    
    参数:
        config: 配置参数
        
    返回:
        结果字典
    """
    if config is None:
        config = CONFIG
    
    np.random.seed(config['random_seed'])
    
    print("=" * 70)
    print("重建误差模式分析")
    print("=" * 70)
    
    base_dir = Path(config['base_dir'])
    
    # Step 1: 加载图像
    print("\n[Step 1] 加载图像...")
    
    benign_img_path = base_dir / 'benign' / config['benign_image']
    benign_mask_path = base_dir / 'benign' / config['benign_image'].replace('.png', '_mask.png')
    malignant_img_path = base_dir / 'malignant' / config['malignant_image']
    malignant_mask_path = base_dir / 'malignant' / config['malignant_image'].replace('.png', '_mask.png')
    
    if not benign_img_path.exists():
        print(f"错误: 图像不存在 - {benign_img_path}")
        return None
    
    benign_gt = load_and_preprocess_image(str(benign_img_path), config['target_size'])
    benign_mask = load_and_preprocess_mask(str(benign_mask_path), config['target_size'])
    malignant_gt = load_and_preprocess_image(str(malignant_img_path), config['target_size'])
    malignant_mask = load_and_preprocess_mask(str(malignant_mask_path), config['target_size'])
    
    print(f"良性图像: {benign_gt.shape}, 掩模覆盖率: {benign_mask.sum()/benign_mask.size*100:.1f}%")
    print(f"恶性图像: {malignant_gt.shape}, 掩模覆盖率: {malignant_mask.sum()/malignant_mask.size*100:.1f}%")
    
    # Step 2: 生成压缩感知测量
    print(f"\n[Step 2] 生成压缩感知测量 (采样率 {config['sampling_rate']*100}%, SNR {config['snr_db']} dB)...")
    benign_Phi, benign_y, m, n = generate_cs_measurements(benign_gt, config['sampling_rate'], config['snr_db'])
    malignant_Phi, malignant_y, _, _ = generate_cs_measurements(malignant_gt, config['sampling_rate'], config['snr_db'])
    print(f"测量矩阵: {m} × {n}, 压缩比: {n/m:.2f}:1")
    
    # Step 3: TV-only 重建
    print(f"\n[Step 3] TV-only 重建...")
    print("良性图像:")
    start = time.time()
    benign_tv = tv_reconstruction(benign_Phi, benign_y, benign_gt.shape,
                                   config['max_iter'], config['step_size'], 
                                   config['tv_weight'], config['verbose'])
    print(f"  时间: {time.time()-start:.2f}s")
    
    print("恶性图像:")
    start = time.time()
    malignant_tv = tv_reconstruction(malignant_Phi, malignant_y, malignant_gt.shape,
                                      config['max_iter'], config['step_size'], 
                                      config['tv_weight'], config['verbose'])
    print(f"  时间: {time.time()-start:.2f}s")
    
    # Step 4: Adaptive-HASA 重建
    print(f"\n[Step 4] Adaptive-HASA 重建...")
    print("良性图像:")
    start = time.time()
    benign_hasa = adaptive_hasa_reconstruction(benign_Phi, benign_y, benign_gt.shape,
                                                config['max_iter'], config['step_size'],
                                                config['tv_weight'], config['wavelet_threshold'],
                                                config['update_weights_every'], config['verbose'])
    print(f"  时间: {time.time()-start:.2f}s")
    
    print("恶性图像:")
    start = time.time()
    malignant_hasa = adaptive_hasa_reconstruction(malignant_Phi, malignant_y, malignant_gt.shape,
                                                   config['max_iter'], config['step_size'],
                                                   config['tv_weight'], config['wavelet_threshold'],
                                                   config['update_weights_every'], config['verbose'])
    print(f"  时间: {time.time()-start:.2f}s")
    
    # Step 5: 计算误差图和边界
    print(f"\n[Step 5] 计算误差和边界统计...")
    
    benign_tv_error = np.abs(benign_tv - benign_gt)
    benign_hasa_error = np.abs(benign_hasa - benign_gt)
    malignant_tv_error = np.abs(malignant_tv - malignant_gt)
    malignant_hasa_error = np.abs(malignant_hasa - malignant_gt)
    
    benign_boundary = extract_boundary(benign_mask, config['boundary_dilation'])
    malignant_boundary = extract_boundary(malignant_mask, config['boundary_dilation'])
    
    # 计算边界统计
    benign_tv_stats = compute_boundary_error_stats(benign_tv_error, benign_boundary)
    benign_hasa_stats = compute_boundary_error_stats(benign_hasa_error, benign_boundary)
    malignant_tv_stats = compute_boundary_error_stats(malignant_tv_error, malignant_boundary)
    malignant_hasa_stats = compute_boundary_error_stats(malignant_hasa_error, malignant_boundary)
    
    # Step 6: 打印分析结果
    print("\n" + "=" * 70)
    print("良性案例分析")
    print("=" * 70)
    print(f"{'指标':<25} {'TV-only':>15} {'Adaptive-HASA':>15} {'改进':>12}")
    print("-" * 70)
    
    benign_overall_imp = (benign_tv_error.mean() - benign_hasa_error.mean()) / benign_tv_error.mean() * 100
    benign_boundary_imp = (benign_tv_stats['boundary_mean'] - benign_hasa_stats['boundary_mean']) / benign_tv_stats['boundary_mean'] * 100
    
    print(f"{'总体平均误差':<25} {benign_tv_error.mean():>15.6f} {benign_hasa_error.mean():>15.6f} {benign_overall_imp:>11.1f}%")
    print(f"{'边界平均误差':<25} {benign_tv_stats['boundary_mean']:>15.6f} {benign_hasa_stats['boundary_mean']:>15.6f} {benign_boundary_imp:>11.1f}%")
    print(f"{'PSNR':<25} {compute_psnr(benign_gt, benign_tv):>14.2f}dB {compute_psnr(benign_gt, benign_hasa):>14.2f}dB")
    
    print("\n" + "=" * 70)
    print("恶性案例分析")
    print("=" * 70)
    print(f"{'指标':<25} {'TV-only':>15} {'Adaptive-HASA':>15} {'改进':>12}")
    print("-" * 70)
    
    malignant_overall_imp = (malignant_tv_error.mean() - malignant_hasa_error.mean()) / malignant_tv_error.mean() * 100
    malignant_boundary_imp = (malignant_tv_stats['boundary_mean'] - malignant_hasa_stats['boundary_mean']) / malignant_tv_stats['boundary_mean'] * 100
    
    print(f"{'总体平均误差':<25} {malignant_tv_error.mean():>15.6f} {malignant_hasa_error.mean():>15.6f} {malignant_overall_imp:>11.1f}%")
    print(f"{'边界平均误差':<25} {malignant_tv_stats['boundary_mean']:>15.6f} {malignant_hasa_stats['boundary_mean']:>15.6f} {malignant_boundary_imp:>11.1f}%")
    print(f"{'PSNR':<25} {compute_psnr(malignant_gt, malignant_tv):>14.2f}dB {compute_psnr(malignant_gt, malignant_hasa):>14.2f}dB")
    
    # Step 7: 保存可视化
    if config['save_figures']:
        print(f"\n[Step 6] 保存可视化...")
        
        plot_detailed_analysis(benign_gt, benign_tv, benign_hasa, benign_boundary,
                               'Benign', os.path.join(config['output_dir'], 'benign_analysis.png'))
        
        plot_detailed_analysis(malignant_gt, malignant_tv, malignant_hasa, malignant_boundary,
                               'Malignant', os.path.join(config['output_dir'], 'malignant_analysis.png'))
        
        # 对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        im1 = axes[0, 0].imshow(benign_tv_error, cmap='hot', vmin=0, vmax=0.3)
        axes[0, 0].contour(benign_boundary, colors='cyan', linewidths=2)
        axes[0, 0].set_title(f'Benign - TV-only\nBoundary: {benign_tv_stats["boundary_mean"]:.4f}')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        im2 = axes[0, 1].imshow(benign_hasa_error, cmap='hot', vmin=0, vmax=0.3)
        axes[0, 1].contour(benign_boundary, colors='cyan', linewidths=2)
        axes[0, 1].set_title(f'Benign - HASA\nBoundary: {benign_hasa_stats["boundary_mean"]:.4f}')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        im3 = axes[1, 0].imshow(malignant_tv_error, cmap='hot', vmin=0, vmax=0.3)
        axes[1, 0].contour(malignant_boundary, colors='cyan', linewidths=2)
        axes[1, 0].set_title(f'Malignant - TV-only\nBoundary: {malignant_tv_stats["boundary_mean"]:.4f}')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        im4 = axes[1, 1].imshow(malignant_hasa_error, cmap='hot', vmin=0, vmax=0.3)
        axes[1, 1].contour(malignant_boundary, colors='cyan', linewidths=2)
        axes[1, 1].set_title(f'Malignant - HASA\nBoundary: {malignant_hasa_stats["boundary_mean"]:.4f}')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        
        plt.suptitle('Reconstruction Error Comparison (Cyan = Tumor Boundary)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_dir'], 'error_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"已保存: error_comparison.png")
        plt.close()
    
    # 关键发现
    print("\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)
    print(f"1. 良性案例边界误差降低: {benign_boundary_imp:.1f}%")
    print(f"2. 恶性案例边界误差降低: {malignant_boundary_imp:.1f}%")
    print(f"3. 良性 PSNR 提升: {compute_psnr(benign_gt, benign_hasa) - compute_psnr(benign_gt, benign_tv):.2f} dB")
    print(f"4. 恶性 PSNR 提升: {compute_psnr(malignant_gt, malignant_hasa) - compute_psnr(malignant_gt, malignant_tv):.2f} dB")
    
    return {
        'benign': {
            'gt': benign_gt, 'tv': benign_tv, 'hasa': benign_hasa,
            'mask': benign_mask, 'boundary': benign_boundary,
            'tv_stats': benign_tv_stats, 'hasa_stats': benign_hasa_stats
        },
        'malignant': {
            'gt': malignant_gt, 'tv': malignant_tv, 'hasa': malignant_hasa,
            'mask': malignant_mask, 'boundary': malignant_boundary,
            'tv_stats': malignant_tv_stats, 'hasa_stats': malignant_hasa_stats
        }
    }


def main():
    """主函数"""
    results = run_error_analysis(CONFIG)
    return results


if __name__ == '__main__':
    results = main()
