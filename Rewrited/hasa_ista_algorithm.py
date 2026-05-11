"""
HASA-ISTA: 混合自适应稀疏近似 - 迭代软阈值算法
=============================================

本模块实现了 HASA-ISTA (Hybrid Adaptive Sparse Approximation - 
Iterative Soft Thresholding Algorithm) 用于压缩感知超声图像重建。

算法特点：
    1. 自适应权重：基于局部熵和梯度幅度动态调整正则化参数
    2. 双正则化：结合小波稀疏性和 TV 正则化
    3. ISTA 框架：迭代软阈值算法保证收敛

优化问题：
    argmin(x) 0.5*||Φx-y||² + λ₁*||Ψx||₁ + λ₂*TV(x)

其中：
    - Φ: 测量矩阵
    - y: 稀疏测量值
    - Ψ: 小波变换
    - λ₁, λ₂: 自适应正则化参数

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, generic_filter
from scipy.stats import entropy
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pywt
from tqdm import tqdm
import warnings
from typing import Tuple, List, Dict, Optional

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 图像参数
    'target_size': (256, 256),
    
    # 数据集参数
    'n_per_class': 20,                  # 每类图像数量
    'classes': ['benign', 'malignant', 'normal'],
    'dataset_path': 'Dataset_BUSI_with_GT',
    
    # 压缩感知参数
    'sampling_ratio': 0.30,             # 采样率 30%
    'random_seed': 42,
    
    # HASA-ISTA 参数
    'max_iter': 50,                     # 最大迭代次数
    'alpha': 0.001,                     # 梯度下降步长
    'lambda1_base': 0.01,               # 小波正则化基础参数
    'lambda2_base': 0.005,              # TV 正则化基础参数
    
    # 小波参数
    'wavelet': 'db4',
    'wavelet_level': 3,
    
    # 局部统计参数
    'entropy_window': 7,                # 局部熵窗口大小
    'weight_update_interval': 5,        # 权重更新间隔
    
    # 输出参数
    'save_results': True,
    'output_dir': '.',
    'verbose': True,
}


# ============================================================================
# 图像加载和预处理
# ============================================================================

def load_dataset(dataset_path: str, classes: List[str], 
                 n_per_class: int) -> List[Tuple[str, str]]:
    """
    加载数据集图像路径
    
    参数:
        dataset_path: 数据集路径
        classes: 类别列表
        n_per_class: 每类图像数量
        
    返回:
        (类别, 路径) 元组列表
    """
    image_paths = []
    
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.exists(cls_path):
            print(f"警告: 类别路径不存在 - {cls_path}")
            continue
        
        # 获取原始图像（排除 mask）
        images = [f for f in os.listdir(cls_path) 
                 if f.endswith('.png') and '_mask' not in f]
        images = sorted(images)[:n_per_class]
        
        for img in images:
            image_paths.append((cls, os.path.join(cls_path, img)))
    
    return image_paths


def preprocess_image(img_path: str, 
                     target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    预处理图像：加载、灰度化、缩放、归一化
    
    参数:
        img_path: 图像路径
        target_size: 目标尺寸
        
    返回:
        归一化的灰度图像 [0, 1]
    """
    img = Image.open(img_path)
    img_gray = img.convert('L')
    img_resized = img_gray.resize(target_size, Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float64) / 255.0
    return img_array


def load_and_preprocess_all(image_paths: List[Tuple[str, str]], 
                            target_size: Tuple[int, int] = (256, 256)
                            ) -> Tuple[np.ndarray, List[str]]:
    """
    加载并预处理所有图像
    
    参数:
        image_paths: 图像路径列表
        target_size: 目标尺寸
        
    返回:
        (图像数组, 标签列表)
    """
    images = []
    labels = []
    
    for cls, img_path in tqdm(image_paths, desc="加载图像"):
        img = preprocess_image(img_path, target_size)
        images.append(img)
        labels.append(cls)
    
    return np.array(images), labels


# ============================================================================
# 压缩感知测量
# ============================================================================

def create_measurement_matrix(n_measurements: int, n_pixels: int,
                              seed: int = 42) -> np.ndarray:
    """
    创建高斯随机测量矩阵
    
    参数:
        n_measurements: 测量数量
        n_pixels: 像素数量
        seed: 随机种子
        
    返回:
        测量矩阵 Φ (n_measurements x n_pixels)
    """
    np.random.seed(seed)
    Phi = np.random.randn(n_measurements, n_pixels) / np.sqrt(n_measurements)
    return Phi


def generate_sparse_measurements(images: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """
    生成稀疏测量值 y = Φx
    
    参数:
        images: 图像数组 (N, H, W)
        Phi: 测量矩阵
        
    返回:
        测量值数组 (N, M)
    """
    measurements = []
    for img in images:
        x_flat = img.flatten()
        y = Phi @ x_flat
        measurements.append(y)
    return np.array(measurements)


# ============================================================================
# 辅助函数
# ============================================================================

def compute_local_entropy(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    计算局部熵图
    
    高熵区域表示更复杂/纹理丰富的区域
    
    参数:
        image: 输入图像
        window_size: 窗口大小
        
    返回:
        局部熵图
    """
    def local_entropy_func(values):
        hist, _ = np.histogram(values, bins=10, range=(0, 1))
        hist = hist + 1e-10  # 避免 log(0)
        return entropy(hist)
    
    entropy_map = generic_filter(image, local_entropy_func, 
                                 size=window_size, mode='reflect')
    return entropy_map


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    计算梯度幅度图
    
    高梯度区域表示边缘/边界
    
    参数:
        image: 输入图像
        
    返回:
        梯度幅度图
    """
    grad_x = sobel(image, axis=1)
    grad_y = sobel(image, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return grad_magnitude


def normalize_map(map_array: np.ndarray, 
                  min_val: float = 0.1, max_val: float = 1.0) -> np.ndarray:
    """
    将图归一化到指定范围
    
    参数:
        map_array: 输入数组
        min_val: 最小值
        max_val: 最大值
        
    返回:
        归一化的数组
    """
    map_min = map_array.min()
    map_max = map_array.max()
    
    if map_max - map_min < 1e-10:
        return np.ones_like(map_array) * min_val
    
    normalized = (map_array - map_min) / (map_max - map_min)
    return normalized * (max_val - min_val) + min_val


# ============================================================================
# 小波变换和近端算子
# ============================================================================

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    软阈值算子（L1 正则化的近端算子）
    
    参数:
        x: 输入数组
        threshold: 阈值（标量）
        
    返回:
        软阈值后的数组
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def apply_wavelet_soft_threshold(image: np.ndarray, threshold: float,
                                 wavelet: str = 'db4', 
                                 level: int = 3) -> np.ndarray:
    """
    在小波域应用软阈值
    
    参数:
        image: 输入图像
        threshold: 阈值
        wavelet: 小波基
        level: 分解层数
        
    返回:
        阈值处理后的图像
    """
    # 小波分解（使用周期化边界）
    coeffs = pywt.wavedec2(image, wavelet, level=level, mode='periodization')
    
    # 对细节系数进行软阈值处理
    coeffs_thresh = [coeffs[0]]  # 保持近似系数
    
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        cH_thresh = soft_threshold(cH, threshold)
        cV_thresh = soft_threshold(cV, threshold)
        cD_thresh = soft_threshold(cD, threshold)
        coeffs_thresh.append((cH_thresh, cV_thresh, cD_thresh))
    
    # 重建
    reconstructed = pywt.waverec2(coeffs_thresh, wavelet, mode='periodization')
    
    # 确保输出尺寸匹配
    if reconstructed.shape != image.shape:
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
    
    return reconstructed


def apply_tv_denoising(image: np.ndarray, weight: float, 
                       n_iter: int = 5) -> np.ndarray:
    """
    应用 TV 去噪
    
    参数:
        image: 输入图像
        weight: 正则化权重
        n_iter: 迭代次数
        
    返回:
        去噪后的图像
    """
    return denoise_tv_chambolle(image, weight=weight, max_num_iter=n_iter)


# ============================================================================
# HASA-ISTA 算法
# ============================================================================

def hasa_ista_reconstruction(y: np.ndarray, Phi: np.ndarray, 
                             image_shape: Tuple[int, int],
                             max_iter: int = 50,
                             alpha: float = 0.001,
                             lambda1_base: float = 0.01,
                             lambda2_base: float = 0.005,
                             wavelet: str = 'db4',
                             wavelet_level: int = 3,
                             entropy_window: int = 7,
                             weight_update_interval: int = 5,
                             verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    HASA-ISTA 重建算法
    
    混合自适应稀疏近似 - 迭代软阈值算法
    
    算法流程：
        1. 梯度下降步骤：z = x - α * Φᵀ(Φx - y)
        2. 自适应权重计算：基于局部熵和梯度
        3. 双近端步骤：小波软阈值 + TV 去噪
    
    参数:
        y: 测量向量
        Phi: 测量矩阵
        image_shape: 图像形状 (H, W)
        max_iter: 最大迭代次数
        alpha: 梯度下降步长
        lambda1_base: 小波正则化基础参数
        lambda2_base: TV 正则化基础参数
        wavelet: 小波基
        wavelet_level: 小波分解层数
        entropy_window: 局部熵窗口大小
        weight_update_interval: 权重更新间隔
        verbose: 是否打印详细信息
        
    返回:
        (重建图像, 历史记录)
    """
    height, width = image_shape
    n_pixels = height * width
    
    # 初始化：反投影
    x = Phi.T @ y
    
    # 预计算
    Phi_T = Phi.T
    
    # 历史记录
    history = {
        'data_fidelity': [],
        'iterations': [],
        'lambda1_history': [],
        'lambda2_history': []
    }
    
    # 初始化自适应权重
    lambda1_adaptive = lambda1_base
    lambda2_adaptive = lambda2_base
    
    if verbose:
        print(f"HASA-ISTA: shape={image_shape}, iter={max_iter}, "
              f"α={alpha}, λ1={lambda1_base}, λ2={lambda2_base}")
    
    # 主迭代循环
    for k in range(max_iter):
        # (a) 梯度下降步骤
        residual = Phi @ x - y
        gradient = Phi_T @ residual
        z = x - alpha * gradient
        
        # 数据保真度
        data_fidelity = 0.5 * np.linalg.norm(residual)**2
        history['data_fidelity'].append(data_fidelity)
        history['iterations'].append(k)
        
        # 重塑为 2D
        x_2d = z.reshape(image_shape)
        x_2d_clipped = np.clip(x_2d, 0, 1)
        
        # (b) 自适应权重计算（周期性更新以节省时间）
        if k % weight_update_interval == 0 or k == 0:
            # 计算局部统计
            entropy_map = compute_local_entropy(x_2d_clipped, 
                                                window_size=entropy_window)
            gradient_map = compute_gradient_magnitude(x_2d_clipped)
            
            # 归一化
            entropy_normalized = normalize_map(entropy_map, 
                                               min_val=0.5, max_val=1.5)
            gradient_normalized = normalize_map(gradient_map, 
                                                min_val=0.5, max_val=1.5)
            
            # 自适应权重
            lambda1_adaptive = entropy_normalized.mean() * lambda1_base
            lambda2_adaptive = gradient_normalized.mean() * lambda2_base
        
        history['lambda1_history'].append(lambda1_adaptive)
        history['lambda2_history'].append(lambda2_adaptive)
        
        # (c) 双近端步骤
        # 小波软阈值（促进小波域稀疏性）
        x_after_wavelet = apply_wavelet_soft_threshold(
            x_2d, lambda1_adaptive,
            wavelet=wavelet, level=wavelet_level
        )
        
        # TV 去噪（促进平滑区域同时保持边缘）
        x_after_tv = apply_tv_denoising(x_after_wavelet, lambda2_adaptive, n_iter=3)
        
        # 更新
        x = x_after_tv.flatten()
        
        if verbose and (k % 10 == 0 or k == max_iter - 1):
            print(f"  Iter {k:2d}: Fidelity={data_fidelity:.2f}, "
                  f"λ1={lambda1_adaptive:.4f}, λ2={lambda2_adaptive:.4f}")
    
    # 最终重建
    x_recon = x.reshape(image_shape)
    x_recon = np.clip(x_recon, 0, 1)
    
    if verbose:
        print(f"  完成: Fidelity={history['data_fidelity'][-1]:.2f}\n")
    
    return x_recon, history


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_reconstruction(img_true: np.ndarray, 
                            img_recon: np.ndarray) -> Dict:
    """
    评估重建质量
    
    参数:
        img_true: 真实图像
        img_recon: 重建图像
        
    返回:
        指标字典
    """
    psnr_val = psnr(img_true, img_recon, data_range=1.0)
    ssim_val = ssim(img_true, img_recon, data_range=1.0)
    mse_val = np.mean((img_true - img_recon)**2)
    
    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'mse': mse_val
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_reconstruction_comparison(img_true: np.ndarray, 
                                   img_recon: np.ndarray,
                                   metrics: Dict,
                                   title: str = '',
                                   output_path: Optional[str] = None):
    """
    绘制重建对比图
    
    参数:
        img_true: 真实图像
        img_recon: 重建图像
        metrics: 评估指标
        title: 标题
        output_path: 输出路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 真实图像
    axes[0].imshow(img_true, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    
    # 重建图像
    axes[1].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'HASA-ISTA Reconstruction\n'
                      f'PSNR: {metrics["psnr"]:.2f} dB, SSIM: {metrics["ssim"]:.4f}',
                      fontsize=14)
    axes[1].axis('off')
    
    # 差异图
    diff = np.abs(img_true - img_recon)
    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title(f'Absolute Difference\nMSE: {metrics["mse"]:.6f}', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_convergence(history: Dict, output_path: Optional[str] = None):
    """
    绘制收敛曲线
    
    参数:
        history: 历史记录
        output_path: 输出路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 数据保真度
    axes[0].plot(history['iterations'], history['data_fidelity'], 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Data Fidelity')
    axes[0].set_title('Convergence')
    axes[0].grid(True, alpha=0.3)
    
    # 自适应权重
    axes[1].plot(history['iterations'], history['lambda1_history'], 
                 'r-', label='λ₁ (wavelet)', linewidth=2)
    axes[1].plot(history['iterations'], history['lambda2_history'], 
                 'g-', label='λ₂ (TV)', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Regularization Weight')
    axes[1].set_title('Adaptive Weights')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# 主函数
# ============================================================================

def run_hasa_ista_experiment(config: dict = None) -> Tuple[np.ndarray, List[Dict]]:
    """
    运行 HASA-ISTA 重建实验
    
    参数:
        config: 配置参数
        
    返回:
        (重建图像数组, 评估结果列表)
    """
    if config is None:
        config = CONFIG
    
    print("=" * 70)
    print("HASA-ISTA: 混合自适应稀疏近似 - 迭代软阈值算法")
    print("=" * 70)
    
    # 检查数据集
    if not os.path.exists(config['dataset_path']):
        print(f"错误: 数据集路径不存在 - {config['dataset_path']}")
        return None, None
    
    # Step 1: 加载数据集
    print("\n[Step 1] 加载数据集...")
    image_paths = load_dataset(
        config['dataset_path'],
        config['classes'],
        config['n_per_class']
    )
    print(f"总图像数: {len(image_paths)}")
    print(f"每类 {config['n_per_class']} 张 × {len(config['classes'])} 类")
    
    # Step 2: 预处理图像
    print("\n[Step 2] 预处理图像...")
    images, labels = load_and_preprocess_all(image_paths, config['target_size'])
    print(f"图像数组形状: {images.shape}")
    print(f"值范围: [{images.min():.4f}, {images.max():.4f}]")
    
    # Step 3: 创建测量矩阵和稀疏测量
    print("\n[Step 3] 创建稀疏测量...")
    height, width = config['target_size']
    n_pixels = height * width
    n_measurements = int(config['sampling_ratio'] * n_pixels)
    
    print(f"图像尺寸: {height} × {width} = {n_pixels} 像素")
    print(f"采样率: {config['sampling_ratio']:.1%}")
    print(f"测量数量: {n_measurements}")
    print(f"压缩比: {n_pixels / n_measurements:.2f}×")
    
    Phi = create_measurement_matrix(n_measurements, n_pixels, 
                                    seed=config['random_seed'])
    print(f"测量矩阵 Φ 形状: {Phi.shape}")
    
    measurements = generate_sparse_measurements(images, Phi)
    print(f"测量值形状: {measurements.shape}")
    
    # Step 4: 运行 HASA-ISTA 重建
    print("\n[Step 4] 运行 HASA-ISTA 重建...")
    print(f"参数: max_iter={config['max_iter']}, α={config['alpha']}, "
          f"λ1={config['lambda1_base']}, λ2={config['lambda2_base']}")
    print("-" * 70)
    
    reconstructed_images = []
    all_histories = []
    all_metrics = []
    
    for idx in tqdm(range(len(images)), desc="重建图像"):
        y = measurements[idx]
        img_true = images[idx]
        
        # 重建
        img_recon, history = hasa_ista_reconstruction(
            y=y,
            Phi=Phi,
            image_shape=config['target_size'],
            max_iter=config['max_iter'],
            alpha=config['alpha'],
            lambda1_base=config['lambda1_base'],
            lambda2_base=config['lambda2_base'],
            wavelet=config['wavelet'],
            wavelet_level=config['wavelet_level'],
            entropy_window=config['entropy_window'],
            weight_update_interval=config['weight_update_interval'],
            verbose=False
        )
        
        reconstructed_images.append(img_recon)
        all_histories.append(history)
        
        # 评估
        metrics = evaluate_reconstruction(img_true, img_recon)
        metrics['label'] = labels[idx]
        metrics['index'] = idx
        all_metrics.append(metrics)
    
    reconstructed_images = np.array(reconstructed_images)
    
    # Step 5: 汇总结果
    print("\n" + "=" * 70)
    print("重建结果汇总")
    print("=" * 70)
    
    psnr_values = [m['psnr'] for m in all_metrics]
    ssim_values = [m['ssim'] for m in all_metrics]
    
    print(f"\n总体指标:")
    print(f"  PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"  SSIM: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    
    # 按类别统计
    print(f"\n按类别统计:")
    for cls in config['classes']:
        cls_metrics = [m for m in all_metrics if m['label'] == cls]
        if len(cls_metrics) > 0:
            cls_psnr = [m['psnr'] for m in cls_metrics]
            cls_ssim = [m['ssim'] for m in cls_metrics]
            print(f"  {cls}:")
            print(f"    PSNR: {np.mean(cls_psnr):.2f} ± {np.std(cls_psnr):.2f} dB")
            print(f"    SSIM: {np.mean(cls_ssim):.4f} ± {np.std(cls_ssim):.4f}")
    
    print("=" * 70)
    
    return reconstructed_images, all_metrics


def main():
    """主函数"""
    np.random.seed(CONFIG['random_seed'])
    
    # 运行实验
    reconstructed_images, all_metrics = run_hasa_ista_experiment(CONFIG)
    
    if reconstructed_images is None:
        print("实验失败")
        return None
    
    # 保存示例可视化
    if CONFIG['save_results'] and len(all_metrics) > 0:
        # 加载原始图像用于对比
        image_paths = load_dataset(
            CONFIG['dataset_path'],
            CONFIG['classes'],
            CONFIG['n_per_class']
        )
        
        # 保存前几张图像的对比
        n_examples = min(3, len(reconstructed_images))
        for i in range(n_examples):
            img_true = preprocess_image(image_paths[i][1], CONFIG['target_size'])
            img_recon = reconstructed_images[i]
            metrics = all_metrics[i]
            
            output_path = os.path.join(
                CONFIG['output_dir'], 
                f'hasa_ista_result_{i}.png'
            )
            plot_reconstruction_comparison(
                img_true, img_recon, metrics,
                title=f"HASA-ISTA Reconstruction - {metrics['label']}",
                output_path=output_path
            )
            print(f"保存对比图: {output_path}")
    
    return reconstructed_images, all_metrics


if __name__ == '__main__':
    results = main()
