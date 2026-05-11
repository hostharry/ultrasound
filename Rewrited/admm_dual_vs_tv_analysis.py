"""
ADMM 双正则化 vs TV-only 性能对比分析
=====================================

本模块对比分析 ADMM 双正则化（TV + Wavelet）和仅 TV 正则化两种方法
在压缩感知超声图像重建中的性能表现。

主要功能：
    1. 在 30 张 BUSI 数据集图像上进行重建
    2. 分别使用双正则化 ADMM 和 TV-only ADMM 进行重建
    3. 计算肿瘤 ROI 区域的 SSIM 指标
    4. 统计分析两种方法的性能差异

数据集：
    - BUSI (Breast Ultrasound Images) 数据集
    - 15 张良性图像 + 15 张恶性图像

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.sparse import linalg as splinalg
from skimage import io, transform
from skimage.metrics import structural_similarity as ssim
import pywt
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
    
    # 压缩感知参数
    'sampling_rate': 0.15,              # 采样率 15%
    'snr_db': 25,                       # 信噪比 25 dB
    'random_seed_matrix': 42,           # 测量矩阵随机种子
    'random_seed_noise': 43,            # 噪声随机种子
    
    # ADMM 参数 - 双正则化
    'lambda_tv_dual': 0.01,             # TV 正则化参数
    'lambda_wavelet_dual': 0.005,       # 小波正则化参数
    
    # ADMM 参数 - TV-only
    'lambda_tv_only': 0.01,             # TV 正则化参数
    
    # ADMM 通用参数
    'rho': 1.0,                         # ADMM 惩罚参数
    'max_iter': 50,                     # 最大迭代次数
    'tol': 1e-3,                        # 收敛容差
    'cg_rtol': 1e-2,                    # CG 相对容差
    'cg_max_iter': 10,                  # CG 最大迭代次数
    
    # 小波参数
    'wavelet': 'db4',
    'wavelet_level': 3,
    
    # 数据集参数
    'n_benign': 15,                     # 良性图像数量
    'n_malignant': 15,                  # 恶性图像数量
    'dataset_path': 'Dataset_BUSI_with_GT',
    
    # 输出参数
    'save_results': True,
    'output_csv': 'admm_dual_vs_tv_results.csv',
    'verbose': True,
}


# ============================================================================
# 图像预处理函数
# ============================================================================

def preprocess_image(img_path: str, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    加载、缩放、灰度化并归一化图像到 [0, 1]
    
    参数:
        img_path: 图像路径
        target_size: 目标尺寸 (height, width)
        
    返回:
        归一化的灰度图像数组
    """
    img = io.imread(img_path)
    
    # 转换为灰度图
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    
    # 缩放
    img_resized = transform.resize(img, target_size, anti_aliasing=True, preserve_range=True)
    
    # 归一化到 [0, 1]
    img_normalized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-10)
    
    return img_normalized


def preprocess_mask(mask_path: str, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    加载并缩放 mask
    
    参数:
        mask_path: mask 路径
        target_size: 目标尺寸 (height, width)
        
    返回:
        二值化的 mask 数组
    """
    mask = io.imread(mask_path)
    
    # 转换为灰度图
    if len(mask.shape) == 3:
        mask = np.mean(mask, axis=2)
    
    # 转换为浮点数
    mask = mask.astype(float)
    
    # 使用最近邻插值缩放（保持二值性）
    mask_resized = transform.resize(mask, target_size, order=0, anti_aliasing=False, preserve_range=True)
    
    # 二值化
    mask_binary = (mask_resized > 0.5 * mask_resized.max()).astype(float)
    
    return mask_binary


# ============================================================================
# 压缩感知相关函数
# ============================================================================

def create_measurement_matrix(n_measurements: int, n_pixels: int, seed: int = 42) -> np.ndarray:
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


def add_noise_snr(signal: np.ndarray, target_snr_db: float = 25, seed: int = 43) -> np.ndarray:
    """
    添加高斯噪声以达到目标 SNR
    
    参数:
        signal: 输入信号
        target_snr_db: 目标信噪比 (dB)
        seed: 随机种子
        
    返回:
        带噪声的信号
    """
    np.random.seed(seed)
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)
    return signal + noise


# ============================================================================
# ADMM 算子
# ============================================================================

def gradient_op(x: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    计算图像梯度（前向差分）
    
    参数:
        x: 展平的图像向量
        shape: 图像形状 (h, w)
        
    返回:
        梯度数组 (h, w, 2)，包含水平和垂直方向梯度
    """
    x_2d = x.reshape(shape)
    grad_x = np.zeros((shape[0], shape[1], 2))
    
    # 水平梯度
    grad_x[:-1, :, 0] = x_2d[1:, :] - x_2d[:-1, :]
    # 垂直梯度
    grad_x[:, :-1, 1] = x_2d[:, 1:] - x_2d[:, :-1]
    
    return grad_x


def divergence_op(grad: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    计算散度（梯度的负伴随算子）
    
    参数:
        grad: 梯度数组 (h, w, 2)
        shape: 图像形状 (h, w)
        
    返回:
        散度向量
    """
    div = np.zeros(shape)
    
    # 水平分量（负伴随）
    div[:-1, :] -= grad[:-1, :, 0]
    div[1:, :] += grad[:-1, :, 0]
    
    # 垂直分量（负伴随）
    div[:, :-1] -= grad[:, :-1, 1]
    div[:, 1:] += grad[:, :-1, 1]
    
    return div.flatten()


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    软阈值算子
    
    参数:
        x: 输入数组
        threshold: 阈值
        
    返回:
        软阈值处理后的数组
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def wavelet_soft_threshold(x: np.ndarray, threshold: float, shape: Tuple[int, int],
                           wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """
    在小波域应用软阈值
    
    参数:
        x: 展平的图像向量
        threshold: 阈值
        shape: 图像形状 (h, w)
        wavelet: 小波基名称
        level: 分解层数
        
    返回:
        小波软阈值处理后的图像向量
    """
    x_2d = x.reshape(shape)
    coeffs = pywt.wavedec2(x_2d, wavelet, level=level)
    
    # 对除近似系数外的所有系数进行阈值处理
    coeffs_thresh = [coeffs[0]]
    for detail_level in coeffs[1:]:
        coeffs_thresh.append(tuple([soft_threshold(c, threshold) for c in detail_level]))
    
    x_reconstructed = pywt.waverec2(coeffs_thresh, wavelet)
    
    # 处理小波分解导致的尺寸不匹配
    x_reconstructed = x_reconstructed[:shape[0], :shape[1]]
    
    return x_reconstructed.flatten()


# ============================================================================
# ADMM 求解器
# ============================================================================

def admm_cs_reconstruction(y: np.ndarray, A: np.ndarray, shape: Tuple[int, int],
                           lambda_tv: float = 0.01, lambda_wavelet: float = 0.005,
                           rho: float = 0.5, max_iter: int = 20, tol: float = 1e-3,
                           cg_rtol: float = 1e-2, cg_max_iter: int = 10,
                           wavelet: str = 'db4', wavelet_level: int = 3,
                           verbose: bool = False) -> np.ndarray:
    """
    ADMM 压缩感知重建求解器（TV + 小波正则化）
    
    优化问题:
        argmin(x) 0.5*||Ax-y||^2 + lambda_tv*||grad(x)||_1 + lambda_wavelet*||Psi(x)||_1
    
    参数:
        y: 测量向量
        A: 测量矩阵
        shape: 图像形状 (h, w)
        lambda_tv: TV 正则化参数
        lambda_wavelet: 小波正则化参数（0 表示仅使用 TV）
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛容差
        cg_rtol: CG 相对容差
        cg_max_iter: CG 最大迭代次数
        wavelet: 小波基名称
        wavelet_level: 小波分解层数
        verbose: 是否打印详细信息
        
    返回:
        重建的图像 (h, w)
    """
    n_pixels = shape[0] * shape[1]
    
    # 初始化变量
    x = A.T @ y  # 反投影初始化
    z_tv = np.zeros((shape[0], shape[1], 2))
    u_tv = np.zeros((shape[0], shape[1], 2))
    z_wavelet = x.copy()
    u_wavelet = np.zeros(n_pixels)
    
    ATy = A.T @ y
    ATA = A.T @ A
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # x 更新：求解线性系统
        div_term = divergence_op(z_tv - u_tv, shape)
        b = ATy + rho * (div_term + z_wavelet - u_wavelet)
        
        def matvec(v):
            grad_v = gradient_op(v, shape)
            div_grad_v = divergence_op(grad_v, shape)
            return ATA @ v + 2 * rho * v - rho * div_grad_v
        
        linear_op = splinalg.LinearOperator((n_pixels, n_pixels), matvec=matvec)
        x, info = splinalg.cg(linear_op, b, x0=x, rtol=cg_rtol, atol=1e-8, maxiter=cg_max_iter)
        
        if verbose and info != 0:
            print(f"CG 在迭代 {iteration} 未收敛 (info={info})")
        
        # z_tv 更新：各向异性 TV 软阈值
        grad_x = gradient_op(x, shape)
        v_tv = grad_x + u_tv
        v_tv_norm = np.sqrt(v_tv[:, :, 0]**2 + v_tv[:, :, 1]**2)
        shrinkage = np.maximum(1 - lambda_tv / (rho * v_tv_norm + 1e-10), 0)
        z_tv = v_tv * shrinkage[:, :, np.newaxis]
        
        # z_wavelet 更新：仅当 lambda_wavelet > 0 时
        if lambda_wavelet > 0:
            v_wavelet = x + u_wavelet
            z_wavelet = wavelet_soft_threshold(v_wavelet, lambda_wavelet / rho, shape,
                                               wavelet=wavelet, level=wavelet_level)
        else:
            z_wavelet = x + u_wavelet
        
        # u 更新
        u_tv = u_tv + grad_x - z_tv
        u_wavelet = u_wavelet + x - z_wavelet
        
        # 检查收敛
        residual = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-10)
        if residual < tol:
            if verbose:
                print(f"在迭代 {iteration} 收敛")
            break
    
    return x.reshape(shape)


# ============================================================================
# 评估函数
# ============================================================================

def calculate_roi_ssim(img_recon: np.ndarray, img_true: np.ndarray, 
                       mask: np.ndarray, use_bbox: bool = True) -> float:
    """
    计算肿瘤 ROI 区域的 SSIM（使用边界框方法）
    
    参数:
        img_recon: 重建图像
        img_true: 真实图像
        mask: 二值 mask
        use_bbox: 是否使用边界框
        
    返回:
        ROI 区域的 SSIM 值
    """
    if np.sum(mask) == 0:
        return np.nan
    
    if use_bbox:
        # 找到肿瘤区域的边界框
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 提取边界框区域
        roi_recon = img_recon[rmin:rmax+1, cmin:cmax+1]
        roi_true = img_true[rmin:rmax+1, cmin:cmax+1]
        
        # 检查 ROI 是否足够大以计算 SSIM
        if roi_recon.shape[0] < 7 or roi_recon.shape[1] < 7:
            return np.nan
        
        # 计算边界框上的 SSIM
        return ssim(roi_true, roi_recon, data_range=1.0)
    else:
        # 遮罩 SSIM（对小 ROI 可靠性较低）
        roi_recon = img_recon[mask > 0.5]
        roi_true = img_true[mask > 0.5]
        
        if len(roi_recon) < 49:  # 小于 7x7
            return np.nan
        
        return ssim(roi_true.reshape(-1, 1), roi_recon.reshape(-1, 1), data_range=1.0)


def calculate_psnr(img_true: np.ndarray, img_recon: np.ndarray) -> float:
    """计算 PSNR"""
    mse = np.mean((img_true - img_recon) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_full_ssim(img_true: np.ndarray, img_recon: np.ndarray) -> float:
    """计算全图 SSIM"""
    return ssim(img_true, img_recon, data_range=1.0)


# ============================================================================
# 数据集加载
# ============================================================================

def load_dataset(dataset_path: str, n_benign: int = 15, n_malignant: int = 15,
                 seed: int = 42) -> List[Path]:
    """
    加载数据集并选择指定数量的图像
    
    参数:
        dataset_path: 数据集路径
        n_benign: 良性图像数量
        n_malignant: 恶性图像数量
        seed: 随机种子
        
    返回:
        选中的图像路径列表
    """
    dataset_path = Path(dataset_path)
    
    # 获取所有良性和恶性图像
    benign_images = sorted([f for f in (dataset_path / 'benign').glob('*.png') 
                           if '_mask' not in f.name])
    malignant_images = sorted([f for f in (dataset_path / 'malignant').glob('*.png') 
                              if '_mask' not in f.name])
    
    print(f"总良性图像数: {len(benign_images)}")
    print(f"总恶性图像数: {len(malignant_images)}")
    
    # 使用固定随机种子选择图像
    np.random.seed(seed)
    selected_benign_idx = np.random.choice(len(benign_images), n_benign, replace=False)
    selected_malignant_idx = np.random.choice(len(malignant_images), n_malignant, replace=False)
    
    selected_benign = [benign_images[i] for i in selected_benign_idx]
    selected_malignant = [malignant_images[i] for i in selected_malignant_idx]
    
    selected_images = selected_benign + selected_malignant
    
    print(f"\n选中图像总数: {len(selected_images)}")
    print(f"前5张选中图像: {[img.name for img in selected_images[:5]]}")
    
    return selected_images


def find_mask_path(img_path: Path) -> Optional[Path]:
    """
    查找图像对应的 mask 路径
    
    参数:
        img_path: 图像路径
        
    返回:
        mask 路径（如果存在）
    """
    # 尝试标准命名
    mask_path = img_path.parent / img_path.name.replace('.png', '_mask.png')
    if mask_path.exists():
        return mask_path
    
    # 尝试替代命名
    mask_path = img_path.parent / img_path.name.replace('.png', '_mask_1.png')
    if mask_path.exists():
        return mask_path
    
    return None


# ============================================================================
# 主处理函数
# ============================================================================

def process_single_image(img_path: Path, A: np.ndarray, config: dict,
                         verbose: bool = False) -> Optional[Dict]:
    """
    处理单张图像
    
    参数:
        img_path: 图像路径
        A: 测量矩阵
        config: 配置参数
        verbose: 是否打印详细信息
        
    返回:
        结果字典（如果成功）
    """
    # 查找 mask
    mask_path = find_mask_path(img_path)
    if mask_path is None:
        if verbose:
            print(f"  警告: 未找到 {img_path.name} 的 mask，跳过...")
        return None
    
    # 预处理图像和 mask
    img_true = preprocess_image(str(img_path), config['target_size'])
    mask = preprocess_mask(str(mask_path), config['target_size'])
    
    # 生成测量值
    y = A @ img_true.flatten()
    y_noisy = add_noise_snr(y, target_snr_db=config['snr_db'], seed=config['random_seed_noise'])
    
    # 双正则化 ADMM 重建
    if verbose:
        print(f"  运行双正则化 ADMM...")
    img_recon_dual = admm_cs_reconstruction(
        y_noisy, A, config['target_size'],
        lambda_tv=config['lambda_tv_dual'],
        lambda_wavelet=config['lambda_wavelet_dual'],
        rho=config['rho'],
        max_iter=config['max_iter'],
        tol=config['tol'],
        cg_rtol=config['cg_rtol'],
        cg_max_iter=config['cg_max_iter'],
        wavelet=config['wavelet'],
        wavelet_level=config['wavelet_level'],
        verbose=False
    )
    
    # TV-only ADMM 重建
    if verbose:
        print(f"  运行 TV-only ADMM...")
    img_recon_tv_only = admm_cs_reconstruction(
        y_noisy, A, config['target_size'],
        lambda_tv=config['lambda_tv_only'],
        lambda_wavelet=0.0,  # 无小波正则化
        rho=config['rho'],
        max_iter=config['max_iter'],
        tol=config['tol'],
        cg_rtol=config['cg_rtol'],
        cg_max_iter=config['cg_max_iter'],
        verbose=False
    )
    
    # 计算 ROI SSIM
    ssim_dual = calculate_roi_ssim(img_recon_dual, img_true, mask, use_bbox=True)
    ssim_tv_only = calculate_roi_ssim(img_recon_tv_only, img_true, mask, use_bbox=True)
    ssim_gain = ssim_dual - ssim_tv_only
    
    # 计算全图指标
    psnr_dual = calculate_psnr(img_true, img_recon_dual)
    psnr_tv_only = calculate_psnr(img_true, img_recon_tv_only)
    full_ssim_dual = calculate_full_ssim(img_true, img_recon_dual)
    full_ssim_tv_only = calculate_full_ssim(img_true, img_recon_tv_only)
    
    # 计算原始 mask 的肿瘤面积
    mask_original = io.imread(str(mask_path))
    if len(mask_original.shape) == 3:
        mask_original = np.mean(mask_original, axis=2)
    tumor_area = np.sum(mask_original > 0.5 * mask_original.max())
    
    return {
        'image_name': img_path.name,
        'class': img_path.parent.name,
        'tumor_area': tumor_area,
        'ssim_roi_dual': ssim_dual,
        'ssim_roi_tv_only': ssim_tv_only,
        'ssim_roi_gain': ssim_gain,
        'psnr_dual': psnr_dual,
        'psnr_tv_only': psnr_tv_only,
        'ssim_full_dual': full_ssim_dual,
        'ssim_full_tv_only': full_ssim_tv_only,
    }


def run_analysis(config: dict = None) -> pd.DataFrame:
    """
    运行完整的对比分析
    
    参数:
        config: 配置参数（如果为 None，使用默认配置）
        
    返回:
        结果 DataFrame
    """
    if config is None:
        config = CONFIG
    
    print("=" * 70)
    print("ADMM 双正则化 vs TV-only 性能对比分析")
    print("=" * 70)
    
    # 检查数据集路径
    if not Path(config['dataset_path']).exists():
        print(f"错误: 数据集路径不存在 - {config['dataset_path']}")
        return None
    
    # 加载数据集
    print("\n[Step 1] 加载数据集...")
    selected_images = load_dataset(
        config['dataset_path'],
        n_benign=config['n_benign'],
        n_malignant=config['n_malignant'],
        seed=config['random_seed_matrix']
    )
    
    # 创建测量矩阵
    print("\n[Step 2] 创建测量矩阵...")
    n_pixels = config['target_size'][0] * config['target_size'][1]
    n_measurements = int(config['sampling_rate'] * n_pixels)
    A = create_measurement_matrix(n_measurements, n_pixels, seed=config['random_seed_matrix'])
    print(f"  测量矩阵形状: {A.shape}")
    print(f"  采样率: {config['sampling_rate']*100}%")
    
    # 打印参数
    print("\n[Step 3] 参数设置:")
    print(f"  SNR: {config['snr_db']} dB")
    print(f"  双正则化 ADMM: λ_TV={config['lambda_tv_dual']}, λ_wavelet={config['lambda_wavelet_dual']}")
    print(f"  TV-only ADMM: λ_TV={config['lambda_tv_only']}, λ_wavelet=0")
    print(f"  ADMM 迭代次数: {config['max_iter']}")
    
    # 处理所有图像
    print(f"\n[Step 4] 处理 {len(selected_images)} 张图像...")
    print("-" * 70)
    
    results = []
    for idx, img_path in enumerate(tqdm(selected_images, desc="处理进度")):
        if config['verbose']:
            print(f"\n处理图像 {idx+1}/{len(selected_images)}: {img_path.name}")
        
        result = process_single_image(img_path, A, config, verbose=config['verbose'])
        
        if result is not None:
            results.append(result)
            if config['verbose']:
                print(f"  肿瘤面积: {result['tumor_area']:.0f} 像素")
                print(f"  ROI SSIM (双正则): {result['ssim_roi_dual']:.4f}, "
                      f"ROI SSIM (TV-only): {result['ssim_roi_tv_only']:.4f}, "
                      f"增益: {result['ssim_roi_gain']:.4f}")
    
    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "-" * 70)
    print(f"处理完成! 共处理 {len(results)} 张图像")
    
    return results_df


def print_summary(results_df: pd.DataFrame):
    """
    打印结果汇总
    
    参数:
        results_df: 结果 DataFrame
    """
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    # 基本统计
    print("\n[ROI SSIM 统计]")
    print(f"  双正则化 - 均值: {results_df['ssim_roi_dual'].mean():.4f}, "
          f"标准差: {results_df['ssim_roi_dual'].std():.4f}")
    print(f"  TV-only  - 均值: {results_df['ssim_roi_tv_only'].mean():.4f}, "
          f"标准差: {results_df['ssim_roi_tv_only'].std():.4f}")
    print(f"  SSIM 增益 - 均值: {results_df['ssim_roi_gain'].mean():.4f}, "
          f"标准差: {results_df['ssim_roi_gain'].std():.4f}")
    
    print("\n[全图 PSNR 统计]")
    print(f"  双正则化 - 均值: {results_df['psnr_dual'].mean():.2f} dB, "
          f"标准差: {results_df['psnr_dual'].std():.2f} dB")
    print(f"  TV-only  - 均值: {results_df['psnr_tv_only'].mean():.2f} dB, "
          f"标准差: {results_df['psnr_tv_only'].std():.2f} dB")
    
    print("\n[全图 SSIM 统计]")
    print(f"  双正则化 - 均值: {results_df['ssim_full_dual'].mean():.4f}, "
          f"标准差: {results_df['ssim_full_dual'].std():.4f}")
    print(f"  TV-only  - 均值: {results_df['ssim_full_tv_only'].mean():.4f}, "
          f"标准差: {results_df['ssim_full_tv_only'].std():.4f}")
    
    # 按类别统计
    print("\n[按类别分组统计 - ROI SSIM 增益]")
    for cls in ['benign', 'malignant']:
        cls_data = results_df[results_df['class'] == cls]
        if len(cls_data) > 0:
            print(f"  {cls}: 均值={cls_data['ssim_roi_gain'].mean():.4f}, "
                  f"标准差={cls_data['ssim_roi_gain'].std():.4f}, "
                  f"数量={len(cls_data)}")
    
    # 双正则化优于 TV-only 的比例
    n_dual_better = (results_df['ssim_roi_gain'] > 0).sum()
    n_total = len(results_df)
    print(f"\n[双正则化优于 TV-only 的比例]")
    print(f"  {n_dual_better}/{n_total} ({n_dual_better/n_total*100:.1f}%)")
    
    print("=" * 70)


def plot_results(results_df: pd.DataFrame, output_dir: str = '.'):
    """
    绘制结果可视化
    
    参数:
        results_df: 结果 DataFrame
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROI SSIM 对比箱线图
    ax1 = axes[0, 0]
    data_to_plot = [results_df['ssim_roi_dual'].dropna(), 
                    results_df['ssim_roi_tv_only'].dropna()]
    bp = ax1.boxplot(data_to_plot, labels=['Dual (TV+Wavelet)', 'TV-only'])
    ax1.set_ylabel('ROI SSIM')
    ax1.set_title('ROI SSIM Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 2. SSIM 增益分布直方图
    ax2 = axes[0, 1]
    gains = results_df['ssim_roi_gain'].dropna()
    ax2.hist(gains, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero gain')
    ax2.axvline(x=gains.mean(), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {gains.mean():.4f}')
    ax2.set_xlabel('SSIM Gain (Dual - TV-only)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of ROI SSIM Gain')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. PSNR 对比散点图
    ax3 = axes[1, 0]
    ax3.scatter(results_df['psnr_tv_only'], results_df['psnr_dual'], 
                c=results_df['class'].map({'benign': 'blue', 'malignant': 'red'}),
                alpha=0.7, s=50)
    min_val = min(results_df['psnr_tv_only'].min(), results_df['psnr_dual'].min()) - 1
    max_val = max(results_df['psnr_tv_only'].max(), results_df['psnr_dual'].max()) + 1
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    ax3.set_xlabel('PSNR TV-only (dB)')
    ax3.set_ylabel('PSNR Dual (dB)')
    ax3.set_title('PSNR Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 肿瘤面积 vs SSIM 增益
    ax4 = axes[1, 1]
    ax4.scatter(results_df['tumor_area'], results_df['ssim_roi_gain'],
                c=results_df['class'].map({'benign': 'blue', 'malignant': 'red'}),
                alpha=0.7, s=50)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Tumor Area (pixels)')
    ax4.set_ylabel('ROI SSIM Gain')
    ax4.set_title('Tumor Area vs SSIM Gain')
    ax4.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Benign'),
                       Patch(facecolor='red', label='Malignant')]
    ax3.legend(handles=legend_elements, loc='lower right')
    ax4.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'admm_dual_vs_tv_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n结果图表已保存至: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 运行分析
    results_df = run_analysis(CONFIG)
    
    if results_df is None or len(results_df) == 0:
        print("分析失败或没有有效结果")
        return None
    
    # 打印汇总
    print_summary(results_df)
    
    # 保存结果
    if CONFIG['save_results']:
        output_path = CONFIG['output_csv']
        results_df.to_csv(output_path, index=False)
        print(f"\n结果已保存至: {output_path}")
        
        # 绘制可视化
        plot_results(results_df, '.')
    
    return results_df


if __name__ == '__main__':
    results = main()
