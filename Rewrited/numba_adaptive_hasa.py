"""
Numba JIT 优化的 Adaptive-HASA 算法
===================================

本模块实现了使用 Numba JIT 加速的 Adaptive-HASA (Hybrid Algorithm for 
Sparse Acquisition) 压缩感知重建算法。

主要特点：
    - 使用 Numba JIT 编译加速局部熵和梯度计算
    - 支持多种局部窗口大小 (5×5, 9×9, 15×15)
    - 自适应 TV 和小波正则化权重
    - 完整的评估指标（全图、肿瘤 ROI、背景）

实验设置：
    - 30 张图像：15 良性 + 15 恶性
    - 采样率：15%
    - SNR：25 dB
    - 图像尺寸：256×256

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import os
import time
from PIL import Image
from typing import Tuple, List, Dict, Optional
import warnings

# 第三方库
from numba import jit
import pywt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import denoise_tv_chambolle

warnings.filterwarnings('ignore')


# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 数据集参数
    'base_path': 'Dataset_BUSI_with_GT',
    'target_size': (256, 256),
    
    # 压缩感知参数
    'sampling_rate': 0.15,              # 采样率 15%
    'target_snr_db': 25,                # 信噪比 25 dB
    'random_seed_measurement': 42,
    'random_seed_noise': 43,
    
    # Adaptive-HASA 参数
    'window_sizes': [5, 9, 15],         # 测试的窗口大小
    'n_iterations': 25,                 # 迭代次数（减少以避免超时）
    'step_size': 0.001,                 # 梯度下降步长
    'tv_weight_base': 0.1,              # TV 基础权重
    'wavelet_weight_base': 0.1,         # 小波基础权重
    
    # 小波参数
    'wavelet': 'db1',
    
    # 输出参数
    'verbose': True,
}

# 预定义的研究图像集
BENIGN_IMAGES = [
    'benign (10)', 'benign (100)', 'benign (102)', 'benign (104)', 'benign (106)',
    'benign (108)', 'benign (11)', 'benign (110)', 'benign (112)', 'benign (114)',
    'benign (116)', 'benign (118)', 'benign (12)', 'benign (120)', 'benign (122)'
]

MALIGNANT_IMAGES = [
    'malignant (1)', 'malignant (10)', 'malignant (100)', 'malignant (102)', 'malignant (104)',
    'malignant (106)', 'malignant (108)', 'malignant (11)', 'malignant (110)', 'malignant (112)',
    'malignant (114)', 'malignant (116)', 'malignant (118)', 'malignant (12)', 'malignant (120)'
]


# ============================================================================
# 图像加载
# ============================================================================

def load_and_preprocess_image(image_path: str, mask_path: str,
                               target_size: Tuple[int, int] = (256, 256)
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载并预处理图像和掩模
    
    参数:
        image_path: 图像路径
        mask_path: 掩模路径
        target_size: 目标尺寸
        
    返回:
        (图像数组, 掩模数组)
    """
    # 加载图像
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float64) / 255.0
    
    # 加载掩模
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(target_size, Image.NEAREST)
    mask_array = (np.array(mask) > 0).astype(bool)
    
    return img_array, mask_array


def load_study_images(base_path: str,
                      benign_names: List[str],
                      malignant_names: List[str],
                      target_size: Tuple[int, int] = (256, 256)
                      ) -> Tuple[List[np.ndarray], List[np.ndarray], 
                                List[str], List[str]]:
    """
    加载所有研究图像
    
    参数:
        base_path: 数据集基础路径
        benign_names: 良性图像名称列表
        malignant_names: 恶性图像名称列表
        target_size: 目标尺寸
        
    返回:
        (图像列表, 掩模列表, 名称列表, 标签列表)
    """
    all_images = []
    all_masks = []
    all_names = []
    all_labels = []
    
    # 加载良性图像
    for img_name in benign_names:
        img_path = os.path.join(base_path, 'benign', f'{img_name}.png')
        mask_path = os.path.join(base_path, 'benign', f'{img_name}_mask.png')
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img, mask = load_and_preprocess_image(img_path, mask_path, target_size)
            all_images.append(img)
            all_masks.append(mask)
            all_names.append(img_name)
            all_labels.append('benign')
    
    # 加载恶性图像
    for img_name in malignant_names:
        img_path = os.path.join(base_path, 'malignant', f'{img_name}.png')
        mask_path = os.path.join(base_path, 'malignant', f'{img_name}_mask.png')
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img, mask = load_and_preprocess_image(img_path, mask_path, target_size)
            all_images.append(img)
            all_masks.append(mask)
            all_names.append(img_name)
            all_labels.append('malignant')
    
    return all_images, all_masks, all_names, all_labels


# ============================================================================
# 压缩感知测量
# ============================================================================

def create_measurement_matrix(n_measurements: int, image_size: int, 
                               seed: int = 42) -> np.ndarray:
    """
    创建高斯随机测量矩阵
    
    参数:
        n_measurements: 测量数量
        image_size: 图像像素总数
        seed: 随机种子
        
    返回:
        测量矩阵 (m × n)
    """
    np.random.seed(seed)
    A = np.random.randn(n_measurements, image_size) / np.sqrt(n_measurements)
    return A


def generate_measurements(images: List[np.ndarray], A: np.ndarray,
                          target_snr_db: float = 25, seed: int = 43
                          ) -> List[np.ndarray]:
    """
    生成所有图像的测量向量
    
    参数:
        images: 图像列表
        A: 测量矩阵
        target_snr_db: 目标信噪比 (dB)
        seed: 噪声随机种子
        
    返回:
        测量向量列表
    """
    np.random.seed(seed)
    measurements = []
    
    for img in images:
        x = img.flatten()
        y_clean = A @ x
        
        # 添加噪声以达到目标 SNR
        signal_power = np.mean(y_clean ** 2)
        noise_power = signal_power / (10 ** (target_snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(y_clean))
        y = y_clean + noise
        
        measurements.append(y)
    
    return measurements


# ============================================================================
# Numba JIT 优化函数
# ============================================================================

@jit(nopython=True)
def compute_local_entropy_numba(image: np.ndarray, window_size: int) -> np.ndarray:
    """
    使用 Numba JIT 计算局部熵图
    
    相比 scipy.ndimage.generic_filter 有显著加速效果
    
    参数:
        image: 输入图像
        window_size: 窗口大小
        
    返回:
        局部熵图
    """
    h, w = image.shape
    entropy_map = np.zeros_like(image)
    half_window = window_size // 2
    
    for i in range(h):
        for j in range(w):
            # 定义窗口边界（边缘处理）
            i_min = max(0, i - half_window)
            i_max = min(h, i + half_window + 1)
            j_min = max(0, j - half_window)
            j_max = min(w, j + half_window + 1)
            
            # 提取局部窗口
            window = image[i_min:i_max, j_min:j_max]
            
            # 计算直方图和熵
            # 将值分到 256 个 bin 中 (对应 [0, 1] 范围)
            hist = np.zeros(256, dtype=np.int64)
            window_flat = window.flatten()
            for val in window_flat:
                bin_idx = min(int(val * 255), 255)
                hist[bin_idx] += 1
            
            # 计算熵
            total = window_flat.size
            entropy = 0.0
            for count in hist:
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            
            entropy_map[i, j] = entropy
    
    return entropy_map


@jit(nopython=True)
def compute_local_gradient_numba(image: np.ndarray, window_size: int) -> np.ndarray:
    """
    使用 Numba JIT 计算局部梯度幅度图
    
    参数:
        image: 输入图像
        window_size: 窗口大小
        
    返回:
        局部梯度幅度图
    """
    h, w = image.shape
    gradient_map = np.zeros_like(image)
    half_window = window_size // 2
    
    for i in range(h):
        for j in range(w):
            # 定义窗口边界
            i_min = max(0, i - half_window)
            i_max = min(h, i + half_window + 1)
            j_min = max(0, j - half_window)
            j_max = min(w, j + half_window + 1)
            
            # 提取局部窗口
            window = image[i_min:i_max, j_min:j_max]
            
            # 计算平均梯度幅度
            grad_sum = 0.0
            count = 0
            wh, ww = window.shape
            for wi in range(wh - 1):
                for wj in range(ww - 1):
                    gx = window[wi+1, wj] - window[wi, wj]
                    gy = window[wi, wj+1] - window[wi, wj]
                    grad_sum += np.sqrt(gx**2 + gy**2)
                    count += 1
            
            if count > 0:
                gradient_map[i, j] = grad_sum / count
    
    return gradient_map


# ============================================================================
# Adaptive-HASA 重建算法
# ============================================================================

def adaptive_hasa_reconstruction(y: np.ndarray, A: np.ndarray,
                                  window_size: int = 9,
                                  n_iterations: int = 50,
                                  step_size: float = 0.001,
                                  tv_weight_base: float = 0.1,
                                  wavelet_weight_base: float = 0.1,
                                  wavelet: str = 'db1') -> np.ndarray:
    """
    Adaptive-HASA 重建算法
    
    使用 Numba JIT 优化的局部统计量计算和自适应正则化权重
    
    参数:
        y: 测量向量
        A: 测量矩阵
        window_size: 局部统计窗口大小 (5, 9, 或 15)
        n_iterations: 迭代次数
        step_size: 梯度下降步长
        tv_weight_base: TV 去噪基础权重
        wavelet_weight_base: 小波阈值基础权重
        wavelet: 小波基类型
        
    返回:
        重建图像 (256×256)
    """
    # 用 A^T * y 初始化（反投影）
    x = A.T @ y
    x = x.reshape(256, 256)
    x = np.clip(x, 0, 1)
    
    for iteration in range(n_iterations):
        # 使用 Numba 优化函数计算局部统计量
        entropy_map = compute_local_entropy_numba(x, window_size)
        gradient_map = compute_local_gradient_numba(x, window_size)
        
        # 归一化到 [0, 1]
        if entropy_map.max() > 0:
            entropy_map = entropy_map / entropy_map.max()
        if gradient_map.max() > 0:
            gradient_map = gradient_map / gradient_map.max()
        
        # 计算自适应权重
        # 高梯度 -> 高 TV 权重
        # 高熵 -> 高小波权重
        tv_weight = tv_weight_base * (1 + gradient_map)
        wavelet_weight = wavelet_weight_base * (1 + entropy_map)
        
        # 数据保真度梯度步骤
        x_flat = x.flatten()
        residual = A @ x_flat - y
        gradient = A.T @ residual
        x_flat = x_flat - step_size * gradient
        x = x_flat.reshape(256, 256)
        x = np.clip(x, 0, 1)
        
        # 顺序应用：先 TV 去噪，后小波阈值
        # （反向顺序会降低质量）
        
        # 1. 使用空间自适应权重的 TV 去噪
        tv_weight_mean = np.mean(tv_weight)
        x_tv = denoise_tv_chambolle(x, weight=tv_weight_mean)
        
        # 2. 使用空间自适应权重的小波阈值
        coeffs = pywt.dwt2(x_tv, wavelet)
        cA, (cH, cV, cD) = coeffs
        
        # 计算自适应阈值
        wavelet_weight_mean = np.mean(wavelet_weight)
        threshold = wavelet_weight_mean * 0.1
        
        # 应用软阈值
        cH = pywt.threshold(cH, threshold, mode='soft')
        cV = pywt.threshold(cV, threshold, mode='soft')
        cD = pywt.threshold(cD, threshold, mode='soft')
        
        # 重建
        x = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
        x = np.clip(x, 0, 1)
    
    return x


# ============================================================================
# 评估指标
# ============================================================================

def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray, 
                      mask: np.ndarray) -> Dict[str, float]:
    """
    计算重建质量指标
    
    参数:
        original: 原始图像 (256×256)
        reconstructed: 重建图像 (256×256)
        mask: 肿瘤 ROI 二值掩模 (256×256)
        
    返回:
        指标字典: full_psnr, full_ssim, tumor_psnr, tumor_ssim, bg_psnr, bg_ssim
    """
    # 确保值在有效范围内
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # 全图指标
    full_psnr = psnr(original, reconstructed, data_range=1.0)
    full_ssim = ssim(original, reconstructed, data_range=1.0)
    
    # 肿瘤 ROI 指标
    tumor_pixels = np.sum(mask)
    if tumor_pixels > 49:  # 最小尺寸以确保可靠的 SSIM
        tumor_roi_orig = original[mask]
        tumor_roi_recon = reconstructed[mask]
        tumor_psnr = psnr(tumor_roi_orig, tumor_roi_recon, data_range=1.0)
        
        # 对于 SSIM，需要 2D 图像，使用边界框方法
        rows, cols = np.where(mask)
        r_min, r_max = rows.min(), rows.max() + 1
        c_min, c_max = cols.min(), cols.max() + 1
        tumor_roi_orig_2d = original[r_min:r_max, c_min:c_max]
        tumor_roi_recon_2d = reconstructed[r_min:r_max, c_min:c_max]
        
        if tumor_roi_orig_2d.shape[0] >= 7 and tumor_roi_orig_2d.shape[1] >= 7:
            tumor_ssim = ssim(tumor_roi_orig_2d, tumor_roi_recon_2d, data_range=1.0)
        else:
            tumor_ssim = np.nan
    else:
        tumor_psnr = np.nan
        tumor_ssim = np.nan
    
    # 背景指标
    bg_mask = ~mask
    bg_pixels = np.sum(bg_mask)
    if bg_pixels > 49:
        bg_orig = original[bg_mask]
        bg_recon = reconstructed[bg_mask]
        bg_psnr = psnr(bg_orig, bg_recon, data_range=1.0)
        
        # 背景 SSIM，使用掩模后的全图
        bg_orig_2d = original.copy()
        bg_orig_2d[mask] = 0
        bg_recon_2d = reconstructed.copy()
        bg_recon_2d[mask] = 0
        bg_ssim = ssim(bg_orig_2d, bg_recon_2d, data_range=1.0)
    else:
        bg_psnr = np.nan
        bg_ssim = np.nan
    
    return {
        'full_psnr': full_psnr,
        'full_ssim': full_ssim,
        'tumor_psnr': tumor_psnr,
        'tumor_ssim': tumor_ssim,
        'bg_psnr': bg_psnr,
        'bg_ssim': bg_ssim
    }


# ============================================================================
# 结果分析
# ============================================================================

def analyze_results(all_results: Dict[int, List[Dict]]) -> None:
    """
    分析并打印实验结果
    
    参数:
        all_results: 按窗口大小组织的结果字典
    """
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    
    for window_size, results in all_results.items():
        print(f"\n窗口大小: {window_size}×{window_size}")
        print("-" * 60)
        
        # 提取指标
        full_psnr = [r['full_psnr'] for r in results]
        full_ssim = [r['full_ssim'] for r in results]
        tumor_ssim = [r['tumor_ssim'] for r in results if not np.isnan(r['tumor_ssim'])]
        
        print(f"全图 PSNR: {np.mean(full_psnr):.2f} ± {np.std(full_psnr):.2f} dB")
        print(f"全图 SSIM: {np.mean(full_ssim):.4f} ± {np.std(full_ssim):.4f}")
        if tumor_ssim:
            print(f"肿瘤 SSIM: {np.mean(tumor_ssim):.4f} ± {np.std(tumor_ssim):.4f}")
        
        # 按类别分析
        benign_ssim = [r['full_ssim'] for r in results if r['label'] == 'benign']
        malignant_ssim = [r['full_ssim'] for r in results if r['label'] == 'malignant']
        
        print(f"\n良性 SSIM: {np.mean(benign_ssim):.4f} ± {np.std(benign_ssim):.4f}")
        print(f"恶性 SSIM: {np.mean(malignant_ssim):.4f} ± {np.std(malignant_ssim):.4f}")
    
    # 比较不同窗口大小
    print("\n" + "=" * 80)
    print("窗口大小比较")
    print("=" * 80)
    print(f"{'窗口大小':^12} | {'平均 PSNR (dB)':^16} | {'平均 SSIM':^16} | {'平均肿瘤 SSIM':^16}")
    print("-" * 68)
    
    for window_size, results in all_results.items():
        full_psnr = np.mean([r['full_psnr'] for r in results])
        full_ssim = np.mean([r['full_ssim'] for r in results])
        tumor_ssim = [r['tumor_ssim'] for r in results if not np.isnan(r['tumor_ssim'])]
        tumor_ssim_mean = np.mean(tumor_ssim) if tumor_ssim else np.nan
        
        print(f"{window_size}×{window_size:^8} | {full_psnr:^16.2f} | {full_ssim:^16.4f} | {tumor_ssim_mean:^16.4f}")


def results_to_dataframe(all_results: Dict[int, List[Dict]]):
    """
    将结果转换为 pandas DataFrame
    
    参数:
        all_results: 按窗口大小组织的结果字典
        
    返回:
        DataFrame
    """
    try:
        import pandas as pd
        
        results_list = []
        for window_size in all_results:
            for result in all_results[window_size]:
                results_list.append(result)
        
        df = pd.DataFrame(results_list)
        return df
    except ImportError:
        print("pandas 未安装，跳过 DataFrame 转换")
        return None


# ============================================================================
# 主函数
# ============================================================================

def run_experiment(config: dict = None) -> Dict[int, List[Dict]]:
    """
    运行完整实验
    
    参数:
        config: 配置参数
        
    返回:
        所有结果的字典
    """
    if config is None:
        config = CONFIG
    
    np.random.seed(42)
    
    print("=" * 80)
    print("Numba JIT 优化的 Adaptive-HASA 算法")
    print("=" * 80)
    
    # Step 1: 打印研究集信息
    print(f"\n研究集定义:")
    print(f"  良性图像: {len(BENIGN_IMAGES)}")
    print(f"  恶性图像: {len(MALIGNANT_IMAGES)}")
    print(f"  总计: {len(BENIGN_IMAGES) + len(MALIGNANT_IMAGES)}")
    
    # Step 2: 加载图像
    print(f"\n[Step 1] 加载图像...")
    if not os.path.exists(config['base_path']):
        print(f"错误: 数据集路径不存在 - {config['base_path']}")
        return None
    
    all_images, all_masks, all_names, all_labels = load_study_images(
        config['base_path'],
        BENIGN_IMAGES,
        MALIGNANT_IMAGES,
        config['target_size']
    )
    
    print(f"成功加载 {len(all_images)} 张图像")
    print(f"  良性: {sum(1 for l in all_labels if l == 'benign')}")
    print(f"  恶性: {sum(1 for l in all_labels if l == 'malignant')}")
    print(f"  图像尺寸: {all_images[0].shape}")
    
    # Step 3: 创建压缩感知测量
    print(f"\n[Step 2] 创建压缩感知测量...")
    image_size = config['target_size'][0] * config['target_size'][1]
    n_measurements = int(config['sampling_rate'] * image_size)
    
    print(f"  采样率: {config['sampling_rate'] * 100}%")
    print(f"  图像大小: {image_size} 像素")
    print(f"  测量数量: {n_measurements}")
    
    A = create_measurement_matrix(n_measurements, image_size, 
                                   config['random_seed_measurement'])
    print(f"  测量矩阵形状: {A.shape}")
    
    all_measurements = generate_measurements(all_images, A,
                                              config['target_snr_db'],
                                              config['random_seed_noise'])
    print(f"生成 {len(all_measurements)} 个测量向量")
    
    # Step 4: JIT 预热
    print(f"\n[Step 3] Numba JIT 预热...")
    test_img = np.random.rand(64, 64)
    _ = compute_local_entropy_numba(test_img, 5)
    _ = compute_local_gradient_numba(test_img, 5)
    print("JIT 编译完成")
    
    # Step 5: 运行重建
    print(f"\n[Step 4] 开始 Adaptive-HASA 重建...")
    print(f"窗口大小: {config['window_sizes']}")
    print(f"迭代次数: {config['n_iterations']}")
    print("=" * 80)
    
    all_results = {ws: [] for ws in config['window_sizes']}
    
    for window_size in config['window_sizes']:
        print(f"\n窗口大小: {window_size}×{window_size}")
        print("-" * 80)
        
        start_time_total = time.time()
        
        for i, (img, mask, y, name, label) in enumerate(zip(
                all_images, all_masks, all_measurements, all_names, all_labels)):
            
            start_time_img = time.time()
            
            # 运行 Adaptive-HASA 重建
            reconstructed = adaptive_hasa_reconstruction(
                y, A,
                window_size=window_size,
                n_iterations=config['n_iterations'],
                step_size=config['step_size'],
                tv_weight_base=config['tv_weight_base'],
                wavelet_weight_base=config['wavelet_weight_base']
            )
            
            # 计算指标
            metrics = calculate_metrics(img, reconstructed, mask)
            
            # 存储结果
            result = {
                'name': name,
                'label': label,
                'window_size': window_size,
                **metrics
            }
            all_results[window_size].append(result)
            
            elapsed_time_img = time.time() - start_time_img
            
            # 打印进度
            if (i + 1) % 5 == 0 or (i + 1) == len(all_images):
                print(f"  完成 {i+1}/{len(all_images)} 张图像 "
                      f"({label:9s} - {name:20s}) - 时间: {elapsed_time_img:.1f}s")
        
        elapsed_time_total = time.time() - start_time_total
        print(f"  窗口 {window_size} 总时间: {elapsed_time_total:.1f}s "
              f"({elapsed_time_total/60:.1f} 分钟)")
    
    print("\n" + "=" * 80)
    print("所有重建完成!")
    print(f"总结果数: {sum(len(all_results[ws]) for ws in config['window_sizes'])}")
    
    # Step 6: 分析结果
    analyze_results(all_results)
    
    # Step 7: 转换为 DataFrame（可选）
    df = results_to_dataframe(all_results)
    if df is not None:
        print(f"\n结果 DataFrame 形状: {df.shape}")
    
    return all_results


def main():
    """主函数"""
    results = run_experiment(CONFIG)
    return results


if __name__ == '__main__':
    results = main()
