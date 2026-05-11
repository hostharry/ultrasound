"""
Scalar-Averaged Adaptive ADMM 压缩感知重建
==========================================

本模块实现了标量平均自适应 ADMM (Alternating Direction Method of 
Multipliers) 求解器，用于超声图像的压缩感知重建。

核心创新：
    在每次迭代中，从当前估计计算空间权重图，然后将其**平均**为标量值，
    再用于近端算子。这种方法结合了空间自适应性和算法稳定性。

ADMM 问题：
    argmin_x  0.5*||Ax-y||_2^2 + λ_TV*||∇x||_1 + λ_wav*||Wx||_1
    
其中：
    - A: 测量矩阵
    - λ_TV, λ_wav: 通过平均空间权重图得到的标量正则化参数

实验设置：
    - 30 张图像（15 良性 + 15 恶性）
    - 采样率：15%
    - SNR：25 dB
    - 图像尺寸：128×128

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
from skimage.restoration import denoise_tv_chambolle
from scipy.sparse.linalg import LinearOperator, cg
from scipy.ndimage import uniform_filter
import pywt
from tqdm import tqdm
import time
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 数据集参数
    'data_dir': 'Dataset_BUSI_with_GT',
    'baseline_csv': 'reconstruction_results_15percent_all_methods.csv',
    'target_size': (128, 128),
    
    # 压缩感知参数
    'sampling_rate': 0.15,              # 采样率 15%
    'snr_db': 25,                       # 信噪比 25 dB
    'meas_seed': 42,                    # 测量矩阵种子
    'noise_seed': 43,                   # 噪声种子
    
    # ADMM 参数
    'rho': 1.0,                         # ADMM 惩罚参数
    'max_iter': 100,                    # 最大迭代次数
    'tol': 1e-4,                        # 收敛容差
    'cg_maxiter': 50,                   # CG 最大迭代次数
    
    # 正则化参数
    'base_lambda_tv': 0.01,             # TV 基础参数
    'base_lambda_wav': 0.01,            # 小波基础参数
    
    # 小波参数
    'wavelet': 'db4',
    'wavelet_level': 3,
    
    # 输出参数
    'verbose': True,
    'save_results': True,
}


# ============================================================================
# 图像加载函数
# ============================================================================

def load_and_preprocess_image(img_path: str, 
                               target_size: Tuple[int, int] = (128, 128)
                               ) -> np.ndarray:
    """
    加载图像，缩放，转换为灰度，归一化到 [0, 1]
    
    参数:
        img_path: 图像路径
        target_size: 目标尺寸
        
    返回:
        归一化的灰度图像
    """
    img = Image.open(img_path).convert('L')
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float64) / 255.0
    return img_array


def load_mask(mask_path: str, 
              target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    加载并预处理掩模
    
    参数:
        mask_path: 掩模路径
        target_size: 目标尺寸
        
    返回:
        二值掩模
    """
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(target_size, Image.NEAREST)
    mask_array = np.array(mask, dtype=np.float64) / 255.0
    mask_array = (mask_array > 0.5).astype(np.float64)
    return mask_array


# ============================================================================
# 压缩感知测量
# ============================================================================

def create_measurement_matrix(n: int, m: int, seed: int = 42) -> np.ndarray:
    """
    创建高斯随机测量矩阵
    
    参数:
        n: 图像像素数
        m: 测量数量
        seed: 随机种子
        
    返回:
        测量矩阵 (m × n)
    """
    rng = np.random.RandomState(seed)
    A = rng.randn(m, n) / np.sqrt(m)
    return A


def add_noise_to_measurements(y: np.ndarray, target_snr_db: float = 25, 
                               seed: int = 43) -> np.ndarray:
    """
    向测量添加高斯噪声以达到目标 SNR
    
    参数:
        y: 原始测量
        target_snr_db: 目标 SNR (dB)
        seed: 随机种子
        
    返回:
        带噪声的测量
    """
    rng = np.random.RandomState(seed)
    signal_power = np.mean(y ** 2)
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.randn(len(y)) * np.sqrt(noise_power)
    return y + noise


def compute_lipschitz_constant(A: np.ndarray, max_iter: int = 100) -> float:
    """
    使用幂迭代估计 Lipschitz 常数（A^T A 的最大特征值）
    
    参数:
        A: 测量矩阵
        max_iter: 迭代次数
        
    返回:
        Lipschitz 常数
    """
    n = A.shape[1]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        x = A.T @ (A @ x)
        x = x / np.linalg.norm(x)
    
    L = np.linalg.norm(A @ x) ** 2
    return L


# ============================================================================
# 自适应权重计算
# ============================================================================

def compute_local_gradient_strength(img: np.ndarray, 
                                     window_size: int = 3) -> np.ndarray:
    """
    计算局部梯度幅度（用于自适应 TV 权重）
    
    参数:
        img: 输入图像
        window_size: 窗口大小
        
    返回:
        局部梯度图
    """
    grad_y, grad_x = np.gradient(img)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    local_grad = uniform_filter(grad_mag, size=window_size)
    return local_grad


def compute_local_entropy(img: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    计算局部熵（用于自适应小波权重）- 优化版本
    
    参数:
        img: 输入图像
        window_size: 窗口大小
        
    返回:
        局部熵图
    """
    # 转换为 uint8
    img_uint8 = (img * 255).astype(np.uint8)
    
    # 使用 skimage 优化的 rank entropy
    entropy_map = rank_entropy(img_uint8, disk(window_size // 2))
    
    # 归一化到 [0, 1]
    entropy_map = entropy_map.astype(np.float64) / 255.0
    
    return entropy_map


def compute_adaptive_weights(img: np.ndarray, 
                              base_lambda_tv: float = 0.01,
                              base_lambda_wav: float = 0.01
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算空间自适应权重图
    
    参数:
        img: 当前图像估计
        base_lambda_tv: TV 基础参数
        base_lambda_wav: 小波基础参数
        
    返回:
        (TV 权重图, 小波权重图)
    """
    # 计算局部特征
    grad_strength = compute_local_gradient_strength(img)
    local_ent = compute_local_entropy(img)
    
    # 归一化到 [0, 1]
    grad_norm = (grad_strength - grad_strength.min()) / (grad_strength.max() - grad_strength.min() + 1e-10)
    ent_norm = (local_ent - local_ent.min()) / (local_ent.max() - local_ent.min() + 1e-10)
    
    # 创建权重图：在边缘（高梯度）和高熵区域增加正则化
    tv_weight_map = base_lambda_tv * (1.0 + grad_norm)
    wav_weight_map = base_lambda_wav * (1.0 + ent_norm)
    
    return tv_weight_map, wav_weight_map


# ============================================================================
# 近端算子
# ============================================================================

def tv_prox_operator(v: np.ndarray, lambda_tv: float, 
                     max_iter: int = 100) -> np.ndarray:
    """
    TV 近端算子（使用 Chambolle 算法）
    
    参数:
        v: 输入图像
        lambda_tv: TV 参数
        max_iter: 最大迭代次数
        
    返回:
        TV 去噪后的图像
    """
    weight = 1.0 / lambda_tv if lambda_tv > 0 else 1e10
    return denoise_tv_chambolle(v, weight=weight, max_num_iter=max_iter)


def wavelet_soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """
    对小波系数应用软阈值
    
    参数:
        coeffs: 小波系数
        threshold: 阈值
        
    返回:
        阈值后的系数
    """
    return pywt.threshold(coeffs, threshold, mode='soft')


def wavelet_prox_operator(v: np.ndarray, lambda_wav: float, 
                          wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """
    小波近端算子（软阈值）
    
    参数:
        v: 输入（图像或向量）
        lambda_wav: 小波参数
        wavelet: 小波基
        level: 分解层数
        
    返回:
        小波去噪后的图像
    """
    # 如果是向量，reshape 为 2D
    shape = v.shape
    if len(shape) == 1:
        n = int(np.sqrt(len(v)))
        v_2d = v.reshape(n, n)
    else:
        v_2d = v
    
    # 小波分解
    coeffs = pywt.wavedec2(v_2d, wavelet=wavelet, level=level, mode='periodization')
    
    # 对细节系数应用软阈值（保持近似系数不变）
    coeffs_thresh = [coeffs[0]]
    for detail_level in coeffs[1:]:
        coeffs_thresh.append(tuple(wavelet_soft_threshold(c, lambda_wav) for c in detail_level))
    
    # 重建
    v_denoised = pywt.waverec2(coeffs_thresh, wavelet=wavelet, mode='periodization')
    
    # 处理尺寸不匹配
    if v_denoised.shape != v_2d.shape:
        v_denoised = v_denoised[:v_2d.shape[0], :v_2d.shape[1]]
    
    # 返回原始形状
    if len(shape) == 1:
        return v_denoised.flatten()
    else:
        return v_denoised


# ============================================================================
# Scalar-Averaged Adaptive ADMM 求解器
# ============================================================================

def admm_scalar_averaged_adaptive(A: np.ndarray, y: np.ndarray, 
                                   img_shape: Tuple[int, int],
                                   rho: float = 1.0, max_iter: int = 100,
                                   base_lambda_tv: float = 0.01,
                                   base_lambda_wav: float = 0.01,
                                   tol: float = 1e-4,
                                   cg_maxiter: int = 100
                                   ) -> Tuple[np.ndarray, Dict]:
    """
    Scalar-Averaged Adaptive ADMM 求解器
    
    核心创新：在每次迭代中，计算空间权重图后将其**平均为标量**，
    再用于近端算子。
    
    参数:
        A: 测量矩阵 (m × n)
        y: 带噪声测量 (m,)
        img_shape: 图像形状 (H, W)
        rho: ADMM 惩罚参数
        max_iter: ADMM 最大迭代次数
        base_lambda_tv: TV 基础参数
        base_lambda_wav: 小波基础参数
        tol: 收敛容差
        cg_maxiter: CG 最大迭代次数
        
    返回:
        (重建图像向量, 收敛信息字典)
    """
    m, n = A.shape
    
    # 初始化变量
    x = A.T @ y  # 反投影初始化
    z1 = x.copy()  # TV 辅助变量
    z2 = x.copy()  # 小波辅助变量
    u1 = np.zeros(n)  # TV 对偶变量
    u2 = np.zeros(n)  # 小波对偶变量
    
    # 收敛跟踪
    convergence_info = {
        'iterations': 0,
        'cg_failures': 0,
        'residuals': [],
        'lambda_tv_history': [],
        'lambda_wav_history': []
    }
    
    # 预计算线性算子
    def AtA_plus_2rho(v):
        return A.T @ (A @ v) + 2 * rho * v
    
    AtA_op = LinearOperator((n, n), matvec=AtA_plus_2rho)
    
    # ADMM 迭代
    for k in range(max_iter):
        x_old = x.copy()
        
        # Step 1: x-update（使用共轭梯度）
        # 求解: (A^T A + 2*rho*I) x = A^T y + rho(z1 + z2 - u1 - u2)
        b = A.T @ y + rho * (z1 + z2 - u1 - u2)
        x, cg_info = cg(AtA_op, b, x0=x, maxiter=cg_maxiter, atol=1e-6)
        
        if cg_info > 0:
            convergence_info['cg_failures'] += 1
        
        # Step 2: 计算自适应权重
        x_2d = x.reshape(img_shape)
        tv_weight_map, wav_weight_map = compute_adaptive_weights(
            x_2d, base_lambda_tv=base_lambda_tv, base_lambda_wav=base_lambda_wav
        )
        
        # **关键步骤**: 将空间权重图平均为标量
        lambda_tv_scalar = np.mean(tv_weight_map)
        lambda_wav_scalar = np.mean(wav_weight_map)
        
        convergence_info['lambda_tv_history'].append(lambda_tv_scalar)
        convergence_info['lambda_wav_history'].append(lambda_wav_scalar)
        
        # Step 3: z1-update（TV 近端算子，使用标量权重）
        v1 = x + u1
        v1_2d = v1.reshape(img_shape)
        z1_2d = tv_prox_operator(v1_2d, rho * lambda_tv_scalar)
        z1 = z1_2d.flatten()
        
        # Step 4: z2-update（小波近端算子，使用标量权重）
        v2 = x + u2
        v2_2d = v2.reshape(img_shape)
        z2_2d = wavelet_prox_operator(v2_2d, rho * lambda_wav_scalar)
        z2 = z2_2d.flatten()
        
        # Step 5: 对偶变量更新
        u1 = u1 + (x - z1)
        u2 = u2 + (x - z2)
        
        # 检查收敛
        primal_residual = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-10)
        convergence_info['residuals'].append(primal_residual)
        
        if primal_residual < tol and k > 10:
            convergence_info['iterations'] = k + 1
            break
    else:
        convergence_info['iterations'] = max_iter
    
    return x, convergence_info


# ============================================================================
# 评估函数
# ============================================================================

def compute_tumor_roi_metrics(recon_img: np.ndarray, ground_truth: np.ndarray,
                               mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    计算肿瘤 ROI 的 PSNR 和 SSIM（使用边界框方法）
    
    参数:
        recon_img: 重建图像
        ground_truth: 真实图像
        mask: 肿瘤掩模
        
    返回:
        (PSNR, SSIM) 或 (None, None)
    """
    if mask.sum() == 0:
        return None, None
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None, None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    roi_recon = recon_img[rmin:rmax+1, cmin:cmax+1]
    roi_gt = ground_truth[rmin:rmax+1, cmin:cmax+1]
    
    if roi_recon.size > 0 and roi_gt.size > 0:
        roi_psnr = psnr(roi_gt, roi_recon, data_range=1.0)
        roi_ssim = ssim(roi_gt, roi_recon, data_range=1.0)
        return roi_psnr, roi_ssim
    else:
        return None, None


def compute_full_image_metrics(recon_img: np.ndarray, 
                                ground_truth: np.ndarray) -> Tuple[float, float]:
    """
    计算全图的 PSNR 和 SSIM
    
    参数:
        recon_img: 重建图像
        ground_truth: 真实图像
        
    返回:
        (PSNR, SSIM)
    """
    img_psnr = psnr(ground_truth, recon_img, data_range=1.0)
    img_ssim = ssim(ground_truth, recon_img, data_range=1.0)
    return img_psnr, img_ssim


# ============================================================================
# 主函数
# ============================================================================

def run_scalar_averaged_admm_experiment(config: dict = None) -> pd.DataFrame:
    """
    运行 Scalar-Averaged Adaptive ADMM 实验
    
    参数:
        config: 配置参数
        
    返回:
        结果 DataFrame
    """
    if config is None:
        config = CONFIG
    
    print("=" * 70)
    print("Scalar-Averaged Adaptive ADMM 压缩感知重建")
    print("=" * 70)
    
    data_dir = Path(config['data_dir'])
    
    # Step 1: 加载基准结果获取图像列表
    print("\n[Step 1] 加载基准结果...")
    try:
        baseline_df = pd.read_csv(config['baseline_csv'])
        fista_hasa_df = baseline_df[baseline_df['Method'] == 'Adaptive-HASA'].copy()
        test_images = sorted(fista_hasa_df['Image'].unique())
        print(f"找到 {len(test_images)} 张测试图像")
    except FileNotFoundError:
        print(f"警告: 基准文件不存在，使用默认图像列表")
        # 默认图像列表
        test_images = [f'benign_{i}' for i in range(1, 16)] + [f'malignant_{i}' for i in range(1, 16)]
    
    # Step 2: 构建图像路径
    print("\n[Step 2] 构建图像路径...")
    image_paths = {}
    for img_name in test_images:
        parts = img_name.split('_')
        img_type = parts[0]
        img_num = parts[1]
        
        img_file = data_dir / img_type / f"{img_type} ({img_num}).png"
        mask_file = data_dir / img_type / f"{img_type} ({img_num})_mask.png"
        
        if img_file.exists() and mask_file.exists():
            image_paths[img_name] = {
                'image': img_file,
                'mask': mask_file,
                'type': img_type
            }
    
    print(f"有效图像: {len(image_paths)}/{len(test_images)}")
    
    if len(image_paths) == 0:
        print("错误: 没有找到有效图像")
        return None
    
    # Step 3: 创建测量矩阵
    print("\n[Step 3] 创建测量矩阵...")
    target_size = config['target_size']
    img_size = target_size[0] * target_size[1]
    num_measurements = int(img_size * config['sampling_rate'])
    
    print(f"  图像大小: {img_size} 像素")
    print(f"  采样率: {config['sampling_rate']*100}%")
    print(f"  测量数量: {num_measurements}")
    
    A = create_measurement_matrix(img_size, num_measurements, seed=config['meas_seed'])
    print(f"  测量矩阵: {A.shape}")
    
    # Step 4: 运行重建
    print("\n[Step 4] 运行 Scalar-Averaged Adaptive ADMM 重建...")
    print(f"  ADMM 参数: rho={config['rho']}, max_iter={config['max_iter']}")
    print(f"  基础正则化: λ_TV={config['base_lambda_tv']}, λ_wav={config['base_lambda_wav']}")
    
    results = []
    total_cg_failures = 0
    
    for img_name in tqdm(list(image_paths.keys()), desc="重建进度"):
        try:
            # 加载图像和掩模
            img_path = image_paths[img_name]['image']
            mask_path = image_paths[img_name]['mask']
            
            ground_truth = load_and_preprocess_image(str(img_path), target_size)
            mask = load_mask(str(mask_path), target_size)
            
            # 生成压缩感知测量
            x_true = ground_truth.flatten()
            y_clean = A @ x_true
            y_noisy = add_noise_to_measurements(y_clean, 
                                                 target_snr_db=config['snr_db'],
                                                 seed=config['noise_seed'])
            
            # ADMM 重建
            x_recon, conv_info = admm_scalar_averaged_adaptive(
                A, y_noisy, target_size,
                rho=config['rho'],
                max_iter=config['max_iter'],
                base_lambda_tv=config['base_lambda_tv'],
                base_lambda_wav=config['base_lambda_wav'],
                tol=config['tol'],
                cg_maxiter=config['cg_maxiter']
            )
            
            recon_img = x_recon.reshape(target_size)
            
            # 计算指标
            full_psnr, full_ssim = compute_full_image_metrics(recon_img, ground_truth)
            tumor_psnr, tumor_ssim = compute_tumor_roi_metrics(recon_img, ground_truth, mask)
            
            total_cg_failures += conv_info['cg_failures']
            
            results.append({
                'Image': img_name,
                'Type': image_paths[img_name]['type'],
                'PSNR_Full': full_psnr,
                'SSIM_Full': full_ssim,
                'PSNR_Tumor': tumor_psnr,
                'SSIM_Tumor': tumor_ssim,
                'ADMM_Iterations': conv_info['iterations'],
                'CG_Failures': conv_info['cg_failures'],
                'Final_Lambda_TV': conv_info['lambda_tv_history'][-1] if conv_info['lambda_tv_history'] else None,
                'Final_Lambda_Wav': conv_info['lambda_wav_history'][-1] if conv_info['lambda_wav_history'] else None
            })
            
        except Exception as e:
            print(f"\n处理 {img_name} 出错: {str(e)}")
            results.append({
                'Image': img_name,
                'Type': image_paths[img_name]['type'],
                'Error': str(e)
            })
    
    # Step 5: 创建结果 DataFrame
    results_df = pd.DataFrame(results)
    
    # Step 6: 打印汇总
    print("\n" + "=" * 70)
    print("重建完成")
    print("=" * 70)
    print(f"处理图像: {len(results_df)}")
    print(f"CG 失败总数: {total_cg_failures}")
    
    # 过滤成功的结果
    success_df = results_df[~results_df['PSNR_Full'].isna()]
    
    if len(success_df) > 0:
        print(f"\n成功重建: {len(success_df)}/{len(results_df)}")
        print(f"\n全图指标:")
        print(f"  PSNR: {success_df['PSNR_Full'].mean():.2f} ± {success_df['PSNR_Full'].std():.2f} dB")
        print(f"  SSIM: {success_df['SSIM_Full'].mean():.4f} ± {success_df['SSIM_Full'].std():.4f}")
        
        tumor_df = success_df[~success_df['SSIM_Tumor'].isna()]
        if len(tumor_df) > 0:
            print(f"\n肿瘤 ROI 指标 ({len(tumor_df)} 张图像):")
            print(f"  PSNR: {tumor_df['PSNR_Tumor'].mean():.2f} ± {tumor_df['PSNR_Tumor'].std():.2f} dB")
            print(f"  SSIM: {tumor_df['SSIM_Tumor'].mean():.4f} ± {tumor_df['SSIM_Tumor'].std():.4f}")
        
        # 按类型统计
        print(f"\n按类型统计:")
        for img_type in ['benign', 'malignant']:
            type_df = success_df[success_df['Type'] == img_type]
            if len(type_df) > 0:
                print(f"  {img_type}: PSNR={type_df['PSNR_Full'].mean():.2f} dB, "
                      f"SSIM={type_df['SSIM_Full'].mean():.4f}")
    
    # 保存结果
    if config['save_results']:
        output_file = 'scalar_averaged_admm_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")
    
    return results_df


def main():
    """主函数"""
    results = run_scalar_averaged_admm_experiment(CONFIG)
    return results


if __name__ == '__main__':
    results = main()
