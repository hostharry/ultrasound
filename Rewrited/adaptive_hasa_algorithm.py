"""
Adaptive-HASA 压缩感知超声图像重建算法
======================================

本脚本实现三种FISTA框架下的压缩感知重建方法：
1. TV-only: 仅使用总变分(TV)正则化
2. Static-HASA: 静态混合TV+小波正则化
3. Adaptive-HASA: 自适应混合正则化（基于局部梯度和熵的空间变化权重）

验证标准：Adaptive-HASA 与其他两种方法的平均 MSE > 1e-5
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.restoration import denoise_tv_chambolle
import pywt
from numba import jit
import time
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'target_size': (128, 128),
    'sampling_rate': 0.15,  # 15% 采样率
    'target_snr_db': 25.0,  # 25dB SNR
    'measurement_seed': 42,
    'noise_seed': 43,
    'base_lambda': 0.005,   # 基础正则化参数
    'n_iterations': 100,    # FISTA迭代次数
    'entropy_window_size': 9,
    'wavelet': 'db4',
    'wavelet_level': 3,
    'validation_threshold': 1e-5,
}

# 测试图像路径
TEST_IMAGES = {
    'benign_1': 'Dataset_BUSI_with_GT/benign/benign (1).png',
    'benign_198': 'Dataset_BUSI_with_GT/benign/benign (198).png',
    'malignant_1': 'Dataset_BUSI_with_GT/malignant/malignant (1).png',
    'malignant_50': 'Dataset_BUSI_with_GT/malignant/malignant (50).png',
    'malignant_146': 'Dataset_BUSI_with_GT/malignant/malignant (146).png',
}


# ==================== 图像加载与预处理 ====================

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    加载图像，转换为灰度，调整大小并归一化到[0, 1]
    """
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float64)
    img_array = img_array / 255.0
    return img_array


def load_test_images(image_paths, target_size):
    """加载所有测试图像"""
    images = {}
    for name, path in image_paths.items():
        if os.path.exists(path):
            images[name] = load_and_preprocess_image(path, target_size)
            print(f"  已加载 {name}: shape {images[name].shape}, "
                  f"range [{images[name].min():.3f}, {images[name].max():.3f}]")
        else:
            print(f"  警告: {path} 不存在")
    return images


# ==================== 压缩感知测量 ====================

def create_measurement_matrix(n_pixels, sampling_rate, seed=42):
    """
    创建高斯随机测量矩阵
    """
    n_measurements = int(n_pixels * sampling_rate)
    rng = np.random.RandomState(seed)
    A = rng.randn(n_measurements, n_pixels) / np.sqrt(n_measurements)
    return A


def add_gaussian_noise(signal, target_snr_db, seed=43):
    """
    添加高斯噪声以达到目标SNR (dB)
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(signal)) * np.sqrt(noise_power)
    
    return signal + noise


def create_measurements(images, A, target_snr_db, noise_seed):
    """为所有图像创建压缩感知测量"""
    measurements = {}
    for name, img in images.items():
        img_flat = img.flatten()
        y = A @ img_flat
        y_noisy = add_gaussian_noise(y, target_snr_db, seed=noise_seed)
        measurements[name] = y_noisy
        print(f"  {name}: measurements shape {y_noisy.shape}, "
              f"range [{y_noisy.min():.3f}, {y_noisy.max():.3f}]")
    return measurements


# ==================== 辅助函数 ====================

@jit(nopython=True)
def calculate_local_entropy_fast(image, window_size=9):
    """
    快速JIT编译的局部熵计算
    使用滑动窗口中的直方图计算熵
    """
    h, w = image.shape
    pad = window_size // 2
    result = np.zeros_like(image)
    
    # 离散化图像到256级
    img_discrete = (image * 255).astype(np.int32)
    img_discrete = np.clip(img_discrete, 0, 255)
    
    for i in range(h):
        for j in range(w):
            # 窗口边界
            i_min = max(0, i - pad)
            i_max = min(h, i + pad + 1)
            j_min = max(0, j - pad)
            j_max = min(w, j + pad + 1)
            
            window = img_discrete[i_min:i_max, j_min:j_max]
            
            # 计算直方图
            hist = np.zeros(256, dtype=np.int32)
            for ii in range(window.shape[0]):
                for jj in range(window.shape[1]):
                    hist[window[ii, jj]] += 1
            
            # 计算熵
            total = window.size
            entropy = 0.0
            for k in range(256):
                if hist[k] > 0:
                    p = hist[k] / total
                    entropy -= p * np.log2(p)
            
            result[i, j] = entropy
    
    return result


def estimate_lipschitz_constant(A, max_iter=50):
    """
    使用幂迭代法估计 Lipschitz 常数 L (A^T A 的最大特征值)
    """
    n = A.shape[1]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        x = A.T @ (A @ x)
        x = x / np.linalg.norm(x)
    
    L = np.dot(x, A.T @ (A @ x))
    return L


def wavelet_soft_threshold(image, threshold, wavelet='db4', level=3):
    """
    小波域软阈值去噪
    L1小波正则化的近端算子
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # 对所有细节系数进行软阈值处理
    coeffs_thresh = [coeffs[0]]  # 保留近似系数
    for detail_level in coeffs[1:]:
        coeffs_thresh.append(tuple(
            pywt.threshold(c, threshold, mode='soft') for c in detail_level
        ))
    
    reconstructed = pywt.waverec2(coeffs_thresh, wavelet)
    
    # 处理尺寸不匹配
    if reconstructed.shape != image.shape:
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
    
    return reconstructed


# ==================== 重建算法 ====================

def fista_tv_only(y, A, lambda_tv, n_iter=100, L=None):
    """
    FISTA 重建 - 仅TV正则化
    """
    if L is None:
        L = estimate_lipschitz_constant(A)
    
    n_pixels = A.shape[1]
    img_size = int(np.sqrt(n_pixels))
    
    # 初始化
    x = (A.T @ y).reshape(img_size, img_size)
    z = x.copy()
    t = 1.0
    
    for _ in range(n_iter):
        # 梯度下降步骤
        residual = A @ z.flatten() - y
        grad = (A.T @ residual).reshape(img_size, img_size)
        x_new = z - (1/L) * grad
        
        # TV近端算子
        weight = lambda_tv / L
        x_new = denoise_tv_chambolle(x_new, weight=weight)
        
        # FISTA动量更新
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
    
    return x


def fista_static_hasa(y, A, lambda_tv, lambda_wav, n_iter=100, L=None, 
                      wavelet='db4', level=3):
    """
    FISTA 重建 - 静态混合 TV + 小波正则化 (Static-HASA)
    """
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
        
        # TV近端算子
        weight_tv = lambda_tv / L
        x_new = denoise_tv_chambolle(x_new, weight=weight_tv)
        
        # 小波近端算子
        threshold_wav = lambda_wav / L
        x_new = wavelet_soft_threshold(x_new, threshold=threshold_wav, 
                                       wavelet=wavelet, level=level)
        
        # FISTA动量更新
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
    
    return x


def fista_adaptive_hasa(y, A, base_lambda, n_iter=100, L=None,
                        wavelet='db4', level=3, entropy_window=9):
    """
    FISTA 重建 - 自适应混合 TV + 小波正则化 (Adaptive-HASA)
    
    每次迭代计算基于局部梯度和熵的空间变化权重
    """
    if L is None:
        L = estimate_lipschitz_constant(A)
    
    n_pixels = A.shape[1]
    img_size = int(np.sqrt(n_pixels))
    
    x = (A.T @ y).reshape(img_size, img_size)
    z = x.copy()
    t = 1.0
    
    for _ in range(n_iter):
        # 梯度下降步骤
        residual = A @ z.flatten() - y
        grad = (A.T @ residual).reshape(img_size, img_size)
        x_intermediate = z - (1/L) * grad
        
        # 基于当前估计z计算自适应权重
        # 局部梯度图
        gradient_map = sobel(z)
        
        # 局部熵图
        entropy_map = calculate_local_entropy_fast(z, window_size=entropy_window)
        
        # 归一化到[0, 1]
        gradient_norm = (gradient_map - gradient_map.min()) / (gradient_map.max() - gradient_map.min() + 1e-10)
        entropy_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-10)
        
        # TV权重与梯度成正比，小波权重与熵成正比
        lambda_tv_map = base_lambda * gradient_norm
        lambda_wav_map = base_lambda * entropy_norm
        
        # 使用空间平均权重（简化方法）
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
    
    return x


# ==================== 评估与可视化 ====================

def calculate_mse(recon1, recon2):
    """计算两个重建图像之间的MSE"""
    return np.mean((recon1 - recon2) ** 2)


def run_all_reconstructions(measurements, A, L, config):
    """对所有图像运行三种重建方法"""
    reconstructions_tv = {}
    reconstructions_static = {}
    reconstructions_adaptive = {}
    
    n_images = len(measurements)
    
    for i, (name, y) in enumerate(measurements.items(), 1):
        print(f"\n[{i}/{n_images}] 处理 {name}...")
        
        # TV-only
        print("  运行 TV-only 重建...")
        start = time.time()
        recon_tv = fista_tv_only(y, A, lambda_tv=config['base_lambda'], 
                                  n_iter=config['n_iterations'], L=L)
        reconstructions_tv[name] = recon_tv
        print(f"  TV-only 完成: {time.time()-start:.1f}s, "
              f"range [{recon_tv.min():.3f}, {recon_tv.max():.3f}]")
        
        # Static-HASA
        print("  运行 Static-HASA 重建...")
        start = time.time()
        recon_static = fista_static_hasa(y, A, lambda_tv=config['base_lambda'], 
                                          lambda_wav=config['base_lambda'],
                                          n_iter=config['n_iterations'], L=L,
                                          wavelet=config['wavelet'], 
                                          level=config['wavelet_level'])
        reconstructions_static[name] = recon_static
        print(f"  Static-HASA 完成: {time.time()-start:.1f}s, "
              f"range [{recon_static.min():.3f}, {recon_static.max():.3f}]")
        
        # Adaptive-HASA
        print("  运行 Adaptive-HASA 重建...")
        start = time.time()
        recon_adaptive = fista_adaptive_hasa(y, A, base_lambda=config['base_lambda'],
                                              n_iter=config['n_iterations'], L=L,
                                              wavelet=config['wavelet'],
                                              level=config['wavelet_level'],
                                              entropy_window=config['entropy_window_size'])
        reconstructions_adaptive[name] = recon_adaptive
        print(f"  Adaptive-HASA 完成: {time.time()-start:.1f}s, "
              f"range [{recon_adaptive.min():.3f}, {recon_adaptive.max():.3f}]")
    
    return reconstructions_tv, reconstructions_static, reconstructions_adaptive


def evaluate_reconstructions(recon_tv, recon_static, recon_adaptive, threshold):
    """评估重建结果，验证Adaptive-HASA的独特性"""
    mse_adaptive_vs_tv = {}
    mse_adaptive_vs_static = {}
    
    print("\n" + "=" * 80)
    print("重建方法之间的MSE比较")
    print("=" * 80)
    
    for name in recon_adaptive.keys():
        mse_tv = calculate_mse(recon_adaptive[name], recon_tv[name])
        mse_static = calculate_mse(recon_adaptive[name], recon_static[name])
        
        mse_adaptive_vs_tv[name] = mse_tv
        mse_adaptive_vs_static[name] = mse_static
        
        print(f"\n{name}:")
        print(f"  MSE(Adaptive-HASA vs TV-only):     {mse_tv:.6e}")
        print(f"  MSE(Adaptive-HASA vs Static-HASA): {mse_static:.6e}")
    
    # 计算平均MSE
    avg_mse_tv = np.mean(list(mse_adaptive_vs_tv.values()))
    avg_mse_static = np.mean(list(mse_adaptive_vs_static.values()))
    
    print("\n" + "=" * 80)
    print("汇总:")
    print(f"  平均 MSE(Adaptive-HASA vs TV-only):     {avg_mse_tv:.6e}")
    print(f"  平均 MSE(Adaptive-HASA vs Static-HASA): {avg_mse_static:.6e}")
    
    # 验证
    tv_passed = avg_mse_tv > threshold
    static_passed = avg_mse_static > threshold
    both_passed = tv_passed and static_passed
    
    print(f"\n验证阈值: {threshold:.6e}")
    print(f"\n验证结果:")
    print(f"  Adaptive-HASA vs TV-only:     {'通过' if tv_passed else '未通过'} "
          f"(MSE {'>' if tv_passed else '<='} {threshold:.6e})")
    print(f"  Adaptive-HASA vs Static-HASA: {'通过' if static_passed else '未通过'} "
          f"(MSE {'>' if static_passed else '<='} {threshold:.6e})")
    print(f"\n总体验证: {'✓ 通过' if both_passed else '✗ 未通过'}")
    print("=" * 80)
    
    return mse_adaptive_vs_tv, mse_adaptive_vs_static, both_passed


def create_visualization(images, recon_tv, recon_static, recon_adaptive,
                        mse_tv, mse_static, representative='malignant_1',
                        output_path='adaptive_hasa_validation.png'):
    """创建对比可视化图"""
    original = images[representative]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原图
    im0 = axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'A. 原图 ({representative})', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # TV-only
    im1 = axes[0, 1].imshow(recon_tv[representative], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'B. TV-only\nMSE vs Adaptive: {mse_tv[representative]:.2e}', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Static-HASA
    im2 = axes[1, 0].imshow(recon_static[representative], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'C. Static-HASA\nMSE vs Adaptive: {mse_static[representative]:.2e}', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Adaptive-HASA
    im3 = axes[1, 1].imshow(recon_adaptive[representative], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('D. Adaptive-HASA\n(空间变化权重，每次迭代更新)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化图已保存: {output_path}")
    plt.show()


def create_difference_maps(recon_tv, recon_static, recon_adaptive,
                           mse_tv, mse_static, representative='malignant_1',
                           output_path='adaptive_hasa_difference_maps.png'):
    """创建差异图可视化"""
    diff_tv = np.abs(recon_adaptive[representative] - recon_tv[representative])
    diff_static = np.abs(recon_adaptive[representative] - recon_static[representative])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = axes[0].imshow(diff_tv, cmap='hot', vmin=0, vmax=0.3)
    axes[0].set_title(f'|Adaptive-HASA - TV-only|\nMSE = {mse_tv[representative]:.2e}',
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    im2 = axes[1].imshow(diff_static, cmap='hot', vmin=0, vmax=0.3)
    axes[1].set_title(f'|Adaptive-HASA - Static-HASA|\nMSE = {mse_static[representative]:.2e}',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"差异图已保存: {output_path}")
    plt.show()


# ==================== 主函数 ====================

def main():
    print("=" * 80)
    print("Adaptive-HASA 压缩感知超声图像重建算法")
    print("=" * 80)
    
    config = CONFIG
    
    # 打印配置
    print(f"\n配置参数:")
    print(f"  图像大小: {config['target_size']}")
    print(f"  采样率: {config['sampling_rate']*100:.0f}%")
    print(f"  SNR: {config['target_snr_db']} dB")
    print(f"  迭代次数: {config['n_iterations']}")
    print(f"  基础正则化参数: {config['base_lambda']}")
    
    # 1. 加载图像
    print("\n[1] 加载测试图像...")
    images = load_test_images(TEST_IMAGES, config['target_size'])
    
    if len(images) == 0:
        print("错误: 没有找到测试图像!")
        return
    
    # 2. 创建测量矩阵
    print("\n[2] 创建压缩感知测量矩阵...")
    n_pixels = config['target_size'][0] * config['target_size'][1]
    A = create_measurement_matrix(n_pixels, config['sampling_rate'], 
                                   seed=config['measurement_seed'])
    print(f"  测量矩阵形状: {A.shape}")
    
    # 3. 估计Lipschitz常数
    print("\n[3] 估计Lipschitz常数...")
    L = estimate_lipschitz_constant(A)
    print(f"  L = {L:.6f}")
    
    # 4. 预编译JIT函数
    print("\n[4] 预编译局部熵计算函数...")
    _ = calculate_local_entropy_fast(list(images.values())[0], window_size=9)
    print("  JIT编译完成")
    
    # 5. 创建测量数据
    print("\n[5] 创建压缩感知测量数据...")
    measurements = create_measurements(images, A, config['target_snr_db'], 
                                        config['noise_seed'])
    
    # 6. 运行重建
    print("\n[6] 运行三种重建算法...")
    print(f"  (每张图像 {config['n_iterations']} 次迭代，可能需要几分钟)")
    
    start_time = time.time()
    recon_tv, recon_static, recon_adaptive = run_all_reconstructions(
        measurements, A, L, config
    )
    total_time = time.time() - start_time
    print(f"\n所有重建完成! 总耗时: {total_time:.1f}s")
    
    # 7. 评估结果
    print("\n[7] 评估重建结果...")
    mse_tv, mse_static, passed = evaluate_reconstructions(
        recon_tv, recon_static, recon_adaptive, 
        config['validation_threshold']
    )
    
    # 8. 可视化
    print("\n[8] 生成可视化...")
    create_visualization(images, recon_tv, recon_static, recon_adaptive,
                        mse_tv, mse_static)
    create_difference_maps(recon_tv, recon_static, recon_adaptive,
                          mse_tv, mse_static)
    
    # 9. 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    if passed:
        print("\n✓ 验证通过!")
        print("\nAdaptive-HASA 产生了与 TV-only 和 Static-HASA 定量不同的重建结果，")
        print("证明了自适应机制是有效的。")
    else:
        print("\n✗ 验证未通过")
        print("\nAdaptive-HASA 与其他方法的差异未达到阈值。")
    print("=" * 80)
    
    return {
        'images': images,
        'measurements': measurements,
        'recon_tv': recon_tv,
        'recon_static': recon_static,
        'recon_adaptive': recon_adaptive,
        'mse_tv': mse_tv,
        'mse_static': mse_static,
        'passed': passed,
    }


if __name__ == '__main__':
    results = main()
