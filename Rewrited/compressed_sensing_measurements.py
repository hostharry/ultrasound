"""
压缩感知测量数据准备与 Blended-HASA 重建
==========================================

本模块实现了压缩感知测量数据的生成和 Blended-HASA 重建方法。

Blended-HASA 方法：
    λ_final = λ_static + β * λ_adaptive

其中 β 控制静态和自适应正则化的混合比例。

分析计划：
    1. 加载 30 张图像研究集，生成 15% 采样率的测量数据
    2. 实现 Blended-HASA 方法
    3. 对 β=0.25 和 β=0.5 进行重建
    4. 与 Static-HASA 和 Adaptive-HASA 基准进行比较
    5. 评估是否满足两项指标均在 5% 以内的标准

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import generic_filter
import time
import warnings
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 图像参数
    'target_size': (256, 256),
    
    # 压缩感知参数
    'sampling_rate': 0.15,              # 采样率 15%
    'snr_db': 25,                       # 信噪比 25 dB
    'random_seed': 42,                  # 随机种子
    
    # Blended-HASA 参数
    'beta_values': [0.25, 0.5],         # β 混合系数
    'lambda_static': 0.01,              # 静态正则化参数
    'lambda_adaptive_base': 0.01,       # 自适应正则化基础参数
    
    # 重建参数
    'max_iter': 50,                     # 最大迭代次数
    'tol': 1e-4,                        # 收敛容差
    
    # 小波参数
    'wavelet': 'db4',
    'wavelet_level': 3,
    
    # 数据集路径
    'dataset_path': 'Dataset_BUSI_with_GT',
    
    # 基准结果文件
    'benchmark_csv': 'reconstruction_results_15percent_all_methods.csv',
    
    # 输出参数
    'save_results': True,
    'output_csv': 'blended_hasa_results.csv',
    'verbose': True,
}

# ============================================================================
# 预定义的图像列表
# ============================================================================

# 30 张图像研究集（15 良性 + 15 恶性）
BENIGN_IMAGES = [
    'benign (1).png', 'benign (10).png', 'benign (100).png', 'benign (101).png',
    'benign (102).png', 'benign (103).png', 'benign (104).png', 'benign (105).png',
    'benign (106).png', 'benign (107).png', 'benign (108).png', 'benign (109).png',
    'benign (11).png', 'benign (110).png', 'benign (111).png'
]

MALIGNANT_IMAGES = [
    'malignant (1).png', 'malignant (10).png', 'malignant (100).png',
    'malignant (101).png', 'malignant (102).png', 'malignant (103).png',
    'malignant (104).png', 'malignant (105).png', 'malignant (106).png',
    'malignant (107).png', 'malignant (108).png', 'malignant (109).png',
    'malignant (11).png', 'malignant (110).png', 'malignant (111).png'
]


# ============================================================================
# 图像加载函数
# ============================================================================

def load_and_preprocess_image(image_path: str, 
                              target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    加载图像，转换为灰度图，缩放，归一化到 [0, 1]
    
    参数:
        image_path: 图像路径
        target_size: 目标尺寸 (height, width)
        
    返回:
        归一化的灰度图像数组
    """
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float64) / 255.0
    return img_array


def load_mask(mask_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    加载并缩放 mask
    
    参数:
        mask_path: mask 路径
        target_size: 目标尺寸 (height, width)
        
    返回:
        布尔型 mask 数组
    """
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(target_size, Image.NEAREST)
    mask_array = np.array(mask, dtype=bool)
    return mask_array


def build_image_list(dataset_path: str) -> List[Dict]:
    """
    构建图像列表
    
    参数:
        dataset_path: 数据集路径
        
    返回:
        包含图像信息的字典列表
    """
    dataset_path = Path(dataset_path)
    image_list = []
    
    # 添加良性图像
    for img_name in BENIGN_IMAGES:
        image_list.append({
            'class': 'benign',
            'image_path': dataset_path / 'benign' / img_name,
            'mask_path': dataset_path / 'benign' / img_name.replace('.png', '_mask.png')
        })
    
    # 添加恶性图像
    for img_name in MALIGNANT_IMAGES:
        image_list.append({
            'class': 'malignant',
            'image_path': dataset_path / 'malignant' / img_name,
            'mask_path': dataset_path / 'malignant' / img_name.replace('.png', '_mask.png')
        })
    
    return image_list


def load_all_data(image_list: List[Dict], 
                  target_size: Tuple[int, int] = (256, 256)) -> List[Dict]:
    """
    加载所有图像和 mask
    
    参数:
        image_list: 图像列表
        target_size: 目标尺寸
        
    返回:
        包含加载数据的记录列表
    """
    data_records = []
    
    for idx, item in enumerate(tqdm(image_list, desc="加载图像")):
        # 检查文件是否存在
        if not item['image_path'].exists():
            print(f"警告: 图像不存在 - {item['image_path']}")
            continue
            
        img = load_and_preprocess_image(str(item['image_path']), target_size)
        
        # 尝试加载 mask
        mask = None
        if item['mask_path'].exists():
            mask = load_mask(str(item['mask_path']), target_size)
        else:
            # 尝试替代命名
            alt_mask_path = str(item['mask_path']).replace('_mask.png', '_mask_1.png')
            if Path(alt_mask_path).exists():
                mask = load_mask(alt_mask_path, target_size)
        
        data_records.append({
            'index': idx,
            'class': item['class'],
            'image_name': item['image_path'].name,
            'original_image': img,
            'mask': mask
        })
    
    return data_records


# ============================================================================
# 压缩感知测量函数
# ============================================================================

def create_measurement_matrix(n_measurements: int, n_pixels: int, 
                              seed: int = 42) -> np.ndarray:
    """
    创建随机高斯测量矩阵
    
    参数:
        n_measurements: 测量数量
        n_pixels: 像素数量
        seed: 随机种子
        
    返回:
        测量矩阵 (n_measurements x n_pixels)
    """
    np.random.seed(seed)
    A = np.random.randn(n_measurements, n_pixels) / np.sqrt(n_measurements)
    return A


def generate_measurements(image: np.ndarray, A: np.ndarray, 
                          snr_db: float = 25, seed: int = 43) -> np.ndarray:
    """
    生成压缩感知测量值
    
    参数:
        image: 原始图像
        A: 测量矩阵
        snr_db: 目标信噪比 (dB)
        seed: 噪声随机种子
        
    返回:
        带噪声的测量值
    """
    x = image.flatten()
    y_clean = A @ x
    
    # 添加噪声
    np.random.seed(seed)
    signal_power = np.mean(y_clean ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(y_clean))
    
    return y_clean + noise


# ============================================================================
# 自适应权重计算
# ============================================================================

def compute_local_entropy(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    计算局部熵
    
    参数:
        image: 输入图像
        window_size: 窗口大小
        
    返回:
        局部熵图
    """
    def entropy_func(window):
        hist, _ = np.histogram(window, bins=10, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        return -np.sum(hist * np.log(hist + 1e-10))
    
    return generic_filter(image, entropy_func, size=window_size)


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    计算梯度幅度
    
    参数:
        image: 输入图像
        
    返回:
        梯度幅度图
    """
    grad_x = np.gradient(image, axis=1)
    grad_y = np.gradient(image, axis=0)
    return np.sqrt(grad_x**2 + grad_y**2)


def compute_adaptive_weights(image: np.ndarray, 
                             lambda_base: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算自适应权重图
    
    参数:
        image: 输入图像（通常是反投影结果）
        lambda_base: 基础正则化参数
        
    返回:
        (TV权重图, 小波权重图)
    """
    # 计算梯度和熵
    gradient_map = compute_gradient_magnitude(image)
    entropy_map = compute_local_entropy(image, window_size=5)
    
    # 归一化
    grad_norm = (gradient_map - gradient_map.min()) / (gradient_map.max() - gradient_map.min() + 1e-10)
    entropy_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-10)
    
    # 计算权重图
    # TV 权重：梯度大的区域需要更强的保护（较小的正则化）
    lambda_tv_map = lambda_base * (1 - 0.5 * grad_norm)
    
    # 小波权重：熵高的区域（纹理丰富）需要更强的保护
    lambda_wav_map = lambda_base * (1 - 0.5 * entropy_norm)
    
    return lambda_tv_map, lambda_wav_map


# ============================================================================
# Blended-HASA 重建
# ============================================================================

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """软阈值算子"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def blended_hasa_reconstruction(y: np.ndarray, A: np.ndarray, 
                                shape: Tuple[int, int],
                                lambda_static: float = 0.01,
                                lambda_adaptive_base: float = 0.01,
                                beta: float = 0.5,
                                max_iter: int = 50, tol: float = 1e-4,
                                wavelet: str = 'db4', 
                                wavelet_level: int = 3) -> np.ndarray:
    """
    Blended-HASA 重建算法
    
    λ_final = λ_static + β * λ_adaptive
    
    参数:
        y: 测量向量
        A: 测量矩阵
        shape: 图像形状 (h, w)
        lambda_static: 静态正则化参数
        lambda_adaptive_base: 自适应正则化基础参数
        beta: 混合系数
        max_iter: 最大迭代次数
        tol: 收敛容差
        wavelet: 小波基
        wavelet_level: 小波分解层数
        
    返回:
        重建图像 (h, w)
    """
    n_pixels = shape[0] * shape[1]
    
    # 初始化：反投影
    x = A.T @ y
    x_img = x.reshape(shape)
    
    # 计算自适应权重
    lambda_tv_adaptive, lambda_wav_adaptive = compute_adaptive_weights(
        x_img, lambda_adaptive_base
    )
    
    # 混合权重
    lambda_tv = lambda_static + beta * lambda_tv_adaptive.mean()
    lambda_wav = lambda_static + beta * lambda_wav_adaptive.mean()
    
    # ISTA 迭代
    # 估计 Lipschitz 常数
    L = np.linalg.norm(A.T @ A, ord=2)
    step_size = 1.0 / L
    
    AtA = A.T @ A
    Aty = A.T @ y
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # 梯度下降步骤
        gradient = AtA @ x - Aty
        x = x - step_size * gradient
        
        # TV 去噪（近端算子近似）
        x_img = x.reshape(shape)
        x_img = denoise_tv_chambolle(x_img, weight=lambda_tv * step_size, max_num_iter=5)
        
        # 小波软阈值
        coeffs = pywt.wavedec2(x_img, wavelet, level=wavelet_level)
        coeffs_thresh = [coeffs[0]]  # 保持近似系数
        for detail in coeffs[1:]:
            coeffs_thresh.append(tuple([
                soft_threshold(c, lambda_wav * step_size) for c in detail
            ]))
        x_img = pywt.waverec2(coeffs_thresh, wavelet)
        x_img = x_img[:shape[0], :shape[1]]  # 裁剪到正确尺寸
        
        x = x_img.flatten()
        
        # 检查收敛
        rel_change = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-10)
        if rel_change < tol:
            break
    
    return x.reshape(shape)


# ============================================================================
# 评估函数
# ============================================================================

def calculate_metrics(img_true: np.ndarray, img_recon: np.ndarray, 
                      mask: Optional[np.ndarray] = None) -> Dict:
    """
    计算重建质量指标
    
    参数:
        img_true: 真实图像
        img_recon: 重建图像
        mask: 肿瘤 mask（可选）
        
    返回:
        指标字典
    """
    # 全图指标
    psnr_val = psnr(img_true, img_recon, data_range=1.0)
    ssim_val = ssim(img_true, img_recon, data_range=1.0)
    
    metrics = {
        'psnr': psnr_val,
        'ssim': ssim_val,
    }
    
    # ROI 指标（如果有 mask）
    if mask is not None and np.sum(mask) > 0:
        # 找到边界框
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            roi_true = img_true[rmin:rmax+1, cmin:cmax+1]
            roi_recon = img_recon[rmin:rmax+1, cmin:cmax+1]
            
            if roi_true.shape[0] >= 7 and roi_true.shape[1] >= 7:
                metrics['ssim_roi'] = ssim(roi_true, roi_recon, data_range=1.0)
            else:
                metrics['ssim_roi'] = np.nan
        else:
            metrics['ssim_roi'] = np.nan
    else:
        metrics['ssim_roi'] = np.nan
    
    return metrics


# ============================================================================
# 主处理函数
# ============================================================================

def run_blended_hasa_experiment(config: dict = None) -> pd.DataFrame:
    """
    运行 Blended-HASA 实验
    
    参数:
        config: 配置参数
        
    返回:
        结果 DataFrame
    """
    if config is None:
        config = CONFIG
    
    print("=" * 70)
    print("Blended-HASA 压缩感知重建实验")
    print("=" * 70)
    
    # 检查数据集
    if not Path(config['dataset_path']).exists():
        print(f"错误: 数据集路径不存在 - {config['dataset_path']}")
        return None
    
    # 构建图像列表
    print("\n[Step 1] 构建图像列表...")
    image_list = build_image_list(config['dataset_path'])
    print(f"总图像数: {len(image_list)}")
    print(f"良性: {len(BENIGN_IMAGES)}, 恶性: {len(MALIGNANT_IMAGES)}")
    
    # 加载数据
    print("\n[Step 2] 加载图像和 mask...")
    data_records = load_all_data(image_list, config['target_size'])
    print(f"成功加载 {len(data_records)} 张图像")
    
    if len(data_records) == 0:
        print("错误: 没有成功加载任何图像")
        return None
    
    print(f"图像尺寸: {data_records[0]['original_image'].shape}")
    
    # 创建测量矩阵
    print("\n[Step 3] 创建测量矩阵...")
    n_pixels = config['target_size'][0] * config['target_size'][1]
    n_measurements = int(config['sampling_rate'] * n_pixels)
    A = create_measurement_matrix(n_measurements, n_pixels, seed=config['random_seed'])
    print(f"测量矩阵形状: {A.shape}")
    print(f"采样率: {config['sampling_rate']*100}%")
    
    # 对每个 β 值进行实验
    print("\n[Step 4] 运行 Blended-HASA 重建...")
    all_results = []
    
    for beta in config['beta_values']:
        print(f"\n--- β = {beta} ---")
        
        for record in tqdm(data_records, desc=f"β={beta}"):
            # 生成测量值
            y = generate_measurements(
                record['original_image'], A, 
                snr_db=config['snr_db'], 
                seed=config['random_seed'] + 1
            )
            
            # 重建
            start_time = time.time()
            img_recon = blended_hasa_reconstruction(
                y, A, config['target_size'],
                lambda_static=config['lambda_static'],
                lambda_adaptive_base=config['lambda_adaptive_base'],
                beta=beta,
                max_iter=config['max_iter'],
                tol=config['tol'],
                wavelet=config['wavelet'],
                wavelet_level=config['wavelet_level']
            )
            recon_time = time.time() - start_time
            
            # 计算指标
            metrics = calculate_metrics(
                record['original_image'], 
                img_recon, 
                record['mask']
            )
            
            # 保存结果
            all_results.append({
                'image_name': record['image_name'],
                'class': record['class'],
                'beta': beta,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'ssim_roi': metrics['ssim_roi'],
                'time': recon_time
            })
    
    # 转换为 DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df


def compare_with_baselines(results_df: pd.DataFrame, 
                           benchmark_path: str) -> Optional[pd.DataFrame]:
    """
    与基准方法进行比较
    
    参数:
        results_df: Blended-HASA 结果
        benchmark_path: 基准结果 CSV 路径
        
    返回:
        比较结果 DataFrame
    """
    if not Path(benchmark_path).exists():
        print(f"警告: 基准文件不存在 - {benchmark_path}")
        return None
    
    benchmark_df = pd.read_csv(benchmark_path)
    
    print("\n" + "=" * 70)
    print("与基准方法比较")
    print("=" * 70)
    
    # 提取基准方法的平均指标
    if 'Method' in benchmark_df.columns:
        for method in ['Static-HASA', 'Adaptive-HASA']:
            method_data = benchmark_df[benchmark_df['Method'] == method]
            if len(method_data) > 0:
                print(f"\n{method}:")
                print(f"  PSNR: {method_data['PSNR'].mean():.2f} dB")
                print(f"  SSIM: {method_data['SSIM'].mean():.4f}")
    
    # Blended-HASA 结果
    print("\nBlended-HASA 结果:")
    for beta in results_df['beta'].unique():
        beta_data = results_df[results_df['beta'] == beta]
        print(f"\n  β = {beta}:")
        print(f"    PSNR: {beta_data['psnr'].mean():.2f} dB (±{beta_data['psnr'].std():.2f})")
        print(f"    SSIM: {beta_data['ssim'].mean():.4f} (±{beta_data['ssim'].std():.4f})")
        if not beta_data['ssim_roi'].isna().all():
            print(f"    SSIM_ROI: {beta_data['ssim_roi'].mean():.4f}")
    
    return benchmark_df


def print_summary(results_df: pd.DataFrame):
    """
    打印结果汇总
    
    参数:
        results_df: 结果 DataFrame
    """
    print("\n" + "=" * 70)
    print("Blended-HASA 结果汇总")
    print("=" * 70)
    
    for beta in results_df['beta'].unique():
        beta_data = results_df[results_df['beta'] == beta]
        print(f"\n[β = {beta}]")
        print(f"  样本数: {len(beta_data)}")
        print(f"  PSNR: {beta_data['psnr'].mean():.2f} ± {beta_data['psnr'].std():.2f} dB")
        print(f"  SSIM: {beta_data['ssim'].mean():.4f} ± {beta_data['ssim'].std():.4f}")
        
        if not beta_data['ssim_roi'].isna().all():
            valid_roi = beta_data['ssim_roi'].dropna()
            print(f"  SSIM_ROI: {valid_roi.mean():.4f} ± {valid_roi.std():.4f} (n={len(valid_roi)})")
        
        print(f"  平均重建时间: {beta_data['time'].mean():.2f} 秒")
    
    print("=" * 70)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 运行实验
    results_df = run_blended_hasa_experiment(CONFIG)
    
    if results_df is None or len(results_df) == 0:
        print("实验失败或没有有效结果")
        return None
    
    # 打印汇总
    print_summary(results_df)
    
    # 与基准比较
    compare_with_baselines(results_df, CONFIG['benchmark_csv'])
    
    # 保存结果
    if CONFIG['save_results']:
        output_path = CONFIG['output_csv']
        results_df.to_csv(output_path, index=False)
        print(f"\n结果已保存至: {output_path}")
    
    return results_df


if __name__ == '__main__':
    results = main()
