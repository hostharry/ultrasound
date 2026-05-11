"""
双正则化 ADMM 求解器：用于压缩感知超声图像重建
===============================================

本模块实现了基于 ADMM（交替方向乘子法）框架的双正则化压缩感知重建算法，
同时整合了 TV（全变分）和小波正则化，并使用空间变化的自适应权重。

优化问题：
    argmin(x) 0.5*||Ax-y||^2 + ||W_tv*grad(x)||_1 + ||W_wav*Psi(x)||_1

其中：
    - A: 测量矩阵
    - y: 测量值
    - W_tv: TV 空间权重图（基于梯度）
    - W_wav: 小波空间权重图（基于局部熵）
    - Psi: 小波变换

主要特性：
    1. 空间变化的 TV 权重（边缘区域更强的正则化）
    2. 空间变化的小波权重（纹理区域更强的正则化）
    3. 共轭梯度法求解 x 更新子问题
    4. 加权软阈值处理 TV 和小波系数

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import cg, LinearOperator
from scipy.ndimage import generic_filter
from scipy.stats import entropy as shannon_entropy
from skimage.metrics import structural_similarity as ssim
import pywt
import os
from typing import Tuple, List, Optional

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 图像参数
    'image_size': (128, 128),
    
    # 压缩感知参数
    'sampling_rate': 0.15,          # 采样率 15%
    'target_snr_db': 25,            # 目标信噪比 25 dB
    'random_seed_measurement': 42,  # 测量矩阵随机种子
    'random_seed_noise': 43,        # 噪声随机种子
    
    # 小波参数
    'wavelet_name': 'db4',          # 小波基
    'wavelet_level': 3,             # 分解层数
    
    # ADMM 参数
    'lambda_tv': 0.005,             # TV 正则化强度
    'lambda_wav': 0.005,            # 小波正则化强度
    'rho1': 0.1,                    # TV 惩罚参数
    'rho2': 0.1,                    # 小波惩罚参数
    'max_iter': 100,                # 最大迭代次数
    'tol': 1e-4,                    # 收敛容差
    'cg_max_iter': 50,              # CG 最大迭代次数
    
    # 权重图参数
    'entropy_window_size': 5,       # 局部熵窗口大小
    'weight_baseline': 0.1,         # 权重基线值
    
    # 评估参数
    'psnr_threshold': 18.0,         # PSNR 阈值 (dB)
    
    # 输出参数
    'save_figures': True,
    'output_dir': '.',
}


# ============================================================================
# 辅助函数
# ============================================================================

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """
    加载并预处理图像
    
    参数:
        image_path: 图像路径
        target_size: 目标尺寸 (height, width)
        
    返回:
        归一化的灰度图像数组 [0, 1]
    """
    img = Image.open(image_path)
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img = img.convert('L')
    img_array = np.array(img, dtype=np.float64) / 255.0
    return img_array


def create_measurement_matrix(M: int, N: int, seed: int = 42) -> np.ndarray:
    """
    创建随机高斯测量矩阵
    
    参数:
        M: 测量数量
        N: 信号维度
        seed: 随机种子
        
    返回:
        测量矩阵 (M x N)
    """
    np.random.seed(seed)
    A = np.random.randn(M, N) / np.sqrt(M)
    return A


def add_noise_to_measurements(y_clean: np.ndarray, target_snr_db: float, 
                              seed: int = 43) -> Tuple[np.ndarray, float]:
    """
    向测量值添加高斯噪声以达到目标 SNR
    
    参数:
        y_clean: 干净的测量值
        target_snr_db: 目标信噪比 (dB)
        seed: 随机种子
        
    返回:
        (带噪声的测量值, 实际SNR)
    """
    np.random.seed(seed)
    signal_power = np.mean(y_clean ** 2)
    target_snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / target_snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(y_clean))
    y = y_clean + noise
    actual_snr = 10 * np.log10(signal_power / np.mean(noise ** 2))
    return y, actual_snr


def local_entropy(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    计算局部熵
    
    参数:
        img: 输入图像
        window_size: 滑动窗口大小
        
    返回:
        局部熵图
    """
    def window_entropy(window):
        hist, _ = np.histogram(window, bins=10, range=(0, 1))
        hist = hist / hist.sum()
        return shannon_entropy(hist + 1e-10)
    
    return generic_filter(img, window_entropy, size=window_size)


def create_tv_weight_map(img: np.ndarray, baseline: float = 0.1) -> np.ndarray:
    """
    基于梯度幅度创建 TV 权重图
    
    参数:
        img: 输入图像
        baseline: 基线权重值
        
    返回:
        TV 权重图（均值归一化为1）
    """
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    W_tv = grad_mag / (grad_mag.max() + 1e-8)
    W_tv = W_tv + baseline
    W_tv = W_tv / W_tv.mean()
    return W_tv


def create_wavelet_weight_map(img: np.ndarray, window_size: int = 5, 
                               baseline: float = 0.1) -> np.ndarray:
    """
    基于局部熵创建小波权重图
    
    参数:
        img: 输入图像
        window_size: 局部熵窗口大小
        baseline: 基线权重值
        
    返回:
        小波权重图（均值归一化为1）
    """
    W_wav = local_entropy(img, window_size=window_size)
    W_wav = W_wav / (W_wav.max() + 1e-8)
    W_wav = W_wav + baseline
    W_wav = W_wav / W_wav.mean()
    return W_wav


# ============================================================================
# 梯度和散度算子
# ============================================================================

def gradient_op(x: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    计算图像梯度
    
    参数:
        x: 展平的图像向量
        shape: 图像形状 (h, w)
        
    返回:
        梯度向量 [gx; gy]
    """
    x_img = x.reshape(shape)
    # 前向差分，周期边界条件
    gx = np.roll(x_img, -1, axis=1) - x_img
    gy = np.roll(x_img, -1, axis=0) - x_img
    return np.concatenate([gx.flatten(), gy.flatten()])


def divergence_op(g: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    计算散度（梯度的负伴随算子）
    
    参数:
        g: 梯度向量 [gx; gy]
        shape: 图像形状 (h, w)
        
    返回:
        散度向量
    """
    n_pixels = shape[0] * shape[1]
    gx = g[:n_pixels].reshape(shape)
    gy = g[n_pixels:].reshape(shape)
    
    # 后向差分（前向差分的伴随）
    div_x = gx - np.roll(gx, 1, axis=1)
    div_y = gy - np.roll(gy, 1, axis=0)
    
    return -(div_x + div_y).flatten()


# ============================================================================
# 小波变换算子
# ============================================================================

def wavelet_forward(x: np.ndarray, shape: Tuple[int, int], 
                    wavelet: str = 'db4', level: int = 3) -> Tuple[np.ndarray, list, Tuple]:
    """
    应用 2D 小波变换
    
    参数:
        x: 展平的图像向量
        shape: 图像形状 (h, w)
        wavelet: 小波基名称
        level: 分解层数
        
    返回:
        (展平的系数向量, 系数切片, 系数数组形状)
    """
    x_img = x.reshape(shape)
    coeffs = pywt.wavedec2(x_img, wavelet=wavelet, level=level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return coeff_arr.flatten(), coeff_slices, coeff_arr.shape


def wavelet_inverse(c: np.ndarray, coeff_shape: Tuple, coeff_slices: list,
                    wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """
    应用逆 2D 小波变换
    
    参数:
        c: 展平的系数向量
        coeff_shape: 系数数组形状
        coeff_slices: 系数切片
        wavelet: 小波基名称
        level: 分解层数
        
    返回:
        重建的图像向量
    """
    coeff_arr = c.reshape(coeff_shape)
    coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
    x_img = pywt.waverec2(coeffs, wavelet=wavelet)
    return x_img.flatten()


# ============================================================================
# 软阈值算子
# ============================================================================

def soft_threshold(x: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """
    软阈值算子（用于 L1 正则化）
    
    参数:
        x: 输入向量
        threshold: 阈值（可以是标量或与 x 同形状的数组）
        
    返回:
        软阈值后的向量
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


# ============================================================================
# ADMM 求解器
# ============================================================================

def dual_admm_solver(A: np.ndarray, y: np.ndarray, img_shape: Tuple[int, int],
                     W_tv: np.ndarray, W_wav: np.ndarray,
                     lambda_tv: float = 0.01, lambda_wav: float = 0.01,
                     rho1: float = 0.1, rho2: float = 0.1,
                     max_iter: int = 100, tol: float = 1e-4,
                     cg_max_iter: int = 50,
                     wavelet_name: str = 'db4', wavelet_level: int = 3,
                     verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
    """
    双正则化 ADMM 求解器
    
    优化问题: argmin(x) 0.5*||Ax-y||^2 + ||W_tv*grad(x)||_1 + ||W_wav*Psi(x)||_1
    
    参数:
        A: 测量矩阵 (M x N)
        y: 测量向量 (M,)
        img_shape: 图像形状 (h, w)
        W_tv: TV 权重图 (h, w)
        W_wav: 小波权重图 (h, w)
        lambda_tv: TV 正则化参数
        lambda_wav: 小波正则化参数
        rho1: TV 的 ADMM 惩罚参数
        rho2: 小波的 ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛容差
        cg_max_iter: CG 最大迭代次数
        wavelet_name: 小波基名称
        wavelet_level: 小波分解层数
        verbose: 是否打印进度
        
    返回:
        (重建图像向量, 目标函数历史)
    """
    N = img_shape[0] * img_shape[1]
    
    # 初始化变量
    x = np.zeros(N)
    
    # 获取初始小波系数结构
    _, coeff_slices, coeff_shape = wavelet_forward(x, img_shape, 
                                                    wavelet=wavelet_name, 
                                                    level=wavelet_level)
    n_coeffs = coeff_shape[0] * coeff_shape[1]
    
    # 辅助变量
    z1 = np.zeros(2 * N)      # 梯度 [gx; gy]
    z2 = np.zeros(n_coeffs)   # 小波系数
    
    # 对偶变量（拉格朗日乘子）
    u1 = np.zeros(2 * N)
    u2 = np.zeros(n_coeffs)
    
    # 预计算 A^T @ A 和 A^T @ y
    AtA = A.T @ A
    Aty = A.T @ y
    
    # 目标函数历史
    obj_history = []
    
    # 预计算小波域权重图
    W_wav_resized = np.array(Image.fromarray(W_wav).resize(coeff_shape[::-1], Image.BILINEAR))
    W_wav_flat = W_wav_resized.flatten()
    
    # ADMM 迭代
    for iter_num in range(max_iter):
        # --- x 更新：求解线性系统 ---
        # (A^T*A + rho1*divgrad + rho2*Psi^T*Psi)*x = A^T*y + rho1*div(z1-u1) + rho2*Psi^T(z2-u2)
        
        # 右侧
        rhs = Aty.copy()
        rhs += rho1 * divergence_op(z1 - u1, img_shape)
        
        # 小波项: Psi^T(z2-u2) = 逆小波变换
        wav_term = wavelet_inverse(z2 - u2, coeff_shape, coeff_slices, 
                                   wavelet=wavelet_name, level=wavelet_level)
        rhs += rho2 * wav_term[:N]
        
        # 定义 CG 的线性算子
        def matvec(v):
            result = AtA @ v
            result += rho1 * divergence_op(gradient_op(v, img_shape), img_shape)
            
            # Psi^T*Psi: 前向后逆小波变换
            wav_v, _, _ = wavelet_forward(v, img_shape, 
                                          wavelet=wavelet_name, level=wavelet_level)
            wav_back = wavelet_inverse(wav_v, coeff_shape, coeff_slices, 
                                       wavelet=wavelet_name, level=wavelet_level)
            result += rho2 * wav_back[:N]
            
            return result
        
        A_op = LinearOperator((N, N), matvec=matvec)
        x, info = cg(A_op, rhs, x0=x, maxiter=cg_max_iter)
        
        if info != 0 and verbose:
            print(f"警告: CG 在迭代 {iter_num} 未收敛, info={info}")
        
        # --- z1 更新 (TV): 对梯度进行加权软阈值 ---
        grad_x = gradient_op(x, img_shape)
        v1 = grad_x + u1
        
        # 分离 gx 和 gy 分量
        gx_part = v1[:N].reshape(img_shape)
        gy_part = v1[N:].reshape(img_shape)
        
        # 加权软阈值
        threshold_tv = (lambda_tv * W_tv) / rho1
        gx_thresh = soft_threshold(gx_part, threshold_tv)
        gy_thresh = soft_threshold(gy_part, threshold_tv)
        
        z1 = np.concatenate([gx_thresh.flatten(), gy_thresh.flatten()])
        
        # --- z2 更新 (小波): 对小波系数进行加权软阈值 ---
        psi_x, _, _ = wavelet_forward(x, img_shape, 
                                      wavelet=wavelet_name, level=wavelet_level)
        v2 = psi_x + u2
        
        threshold_wav = (lambda_wav * W_wav_flat) / rho2
        z2 = soft_threshold(v2, threshold_wav)
        
        # --- u1 更新 (TV 的对偶变量) ---
        u1 = u1 + grad_x - z1
        
        # --- u2 更新 (小波的对偶变量) ---
        u2 = u2 + psi_x - z2
        
        # --- 计算目标函数 ---
        residual = A @ x - y
        data_fidelity = 0.5 * np.sum(residual ** 2)
        
        # TV 项
        grad_x_img = grad_x[:N].reshape(img_shape)
        grad_y_img = grad_x[N:].reshape(img_shape)
        grad_mag = np.sqrt(grad_x_img**2 + grad_y_img**2)
        tv_term = lambda_tv * np.sum(W_tv * grad_mag)
        
        # 小波项
        wav_term_val = lambda_wav * np.sum(W_wav_flat * np.abs(psi_x))
        
        objective = data_fidelity + tv_term + wav_term_val
        obj_history.append(objective)
        
        if verbose and iter_num % 10 == 0:
            print(f"Iter {iter_num:3d}: Obj = {objective:.6f}, "
                  f"Data = {data_fidelity:.6f}, TV = {tv_term:.6f}, Wav = {wav_term_val:.6f}")
        
        # 检查收敛
        if iter_num > 0 and abs(obj_history[-1] - obj_history[-2]) / obj_history[-2] < tol:
            if verbose:
                print(f"在迭代 {iter_num} 收敛")
            break
    
    return x, obj_history


# ============================================================================
# 评估函数
# ============================================================================

def calculate_psnr(img_true: np.ndarray, img_recon: np.ndarray) -> float:
    """计算峰值信噪比 (PSNR)"""
    mse = np.mean((img_true - img_recon) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img_true: np.ndarray, img_recon: np.ndarray) -> float:
    """计算结构相似性指数 (SSIM)"""
    return ssim(img_true, img_recon, data_range=1.0)


# ============================================================================
# 可视化函数
# ============================================================================

def plot_convergence(obj_history: List[float], output_path: str):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(obj_history, 'b-', linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('目标函数', fontsize=12)
    plt.title('ADMM 收敛曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(obj_history, 'b-', linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('目标函数 (对数尺度)', fontsize=12)
    plt.title('ADMM 收敛曲线 (对数尺度)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"收敛曲线已保存至: {output_path}")


def plot_reconstruction_comparison(img_true: np.ndarray, img_backproj: np.ndarray,
                                   img_recon: np.ndarray, psnr_bp: float,
                                   psnr_recon: float, ssim_recon: float,
                                   output_path: str):
    """绘制重建结果对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_true, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_backproj, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Back-projection\nPSNR: {psnr_bp:.2f} dB', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Dual-Regularizer ADMM\nPSNR: {psnr_recon:.2f} dB, SSIM: {ssim_recon:.3f}',
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"重建对比图已保存至: {output_path}")


def plot_comprehensive_results(img_true: np.ndarray, img_recon: np.ndarray,
                               obj_history: List[float], psnr_val: float,
                               ssim_val: float, sampling_rate: float,
                               snr: float, output_path: str):
    """绘制综合结果图"""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 收敛曲线
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(obj_history, 'b-', linewidth=2.5, marker='o', markersize=4, markevery=5)
    ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Objective Function', fontsize=14, fontweight='bold')
    ax1.set_title('A. ADMM Convergence: Dual-Regularizer (TV + Wavelet)',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    reduction = (obj_history[0] - obj_history[-1]) / obj_history[0] * 100
    ax1.text(0.98, 0.95, f'Initial: {obj_history[0]:.2f}\nFinal: {obj_history[-1]:.2f}\n'
             f'Reduction: {reduction:.1f}%',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Ground Truth
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(img_true, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('B. Ground Truth', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # 重建结果
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(img_recon, cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f'C. ADMM Reconstruction\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.3f}',
                  fontsize=14, fontweight='bold', pad=10)
    ax3.axis('off')
    
    fig.suptitle(f'Dual-Regularizer ADMM Solver for Compressed Sensing\n'
                 f'({sampling_rate*100:.0f}% Sampling, {snr:.0f} dB SNR)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"综合结果图已保存至: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main(image_path: Optional[str] = None):
    """
    主函数：执行双正则化 ADMM 压缩感知重建
    
    参数:
        image_path: 输入图像路径（如果为 None，则使用默认路径）
    """
    print("=" * 70)
    print("双正则化 ADMM 求解器 - 压缩感知超声图像重建")
    print("=" * 70)
    
    # 设置默认图像路径
    if image_path is None:
        image_path = 'Dataset_BUSI_with_GT/benign/benign (1).png'
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像不存在 - {image_path}")
        print("请提供有效的图像路径")
        return None
    
    # ========================================================================
    # Step 1: 加载和预处理图像
    # ========================================================================
    print("\n[Step 1] 加载和预处理图像...")
    img_array = load_and_preprocess_image(image_path, CONFIG['image_size'])
    print(f"  图像形状: {img_array.shape}")
    print(f"  值范围: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    x_true = img_array.flatten()
    N = len(x_true)
    img_shape = img_array.shape
    print(f"  展平后大小: {N}")
    
    # ========================================================================
    # Step 2: 模拟压缩感知采集
    # ========================================================================
    print("\n[Step 2] 模拟压缩感知采集...")
    M = int(CONFIG['sampling_rate'] * N)
    print(f"  测量数量: {M} ({CONFIG['sampling_rate']*100}% of {N})")
    
    A = create_measurement_matrix(M, N, seed=CONFIG['random_seed_measurement'])
    y_clean = A @ x_true
    y, actual_snr = add_noise_to_measurements(y_clean, CONFIG['target_snr_db'],
                                               seed=CONFIG['random_seed_noise'])
    print(f"  目标 SNR: {CONFIG['target_snr_db']} dB, 实际 SNR: {actual_snr:.2f} dB")
    
    # ========================================================================
    # Step 3: 创建空间变化的权重图
    # ========================================================================
    print("\n[Step 3] 创建空间变化的权重图...")
    W_tv = create_tv_weight_map(img_array, baseline=CONFIG['weight_baseline'])
    print(f"  TV 权重范围: [{W_tv.min():.3f}, {W_tv.max():.3f}], 均值: {W_tv.mean():.3f}")
    
    W_wav = create_wavelet_weight_map(img_array, 
                                       window_size=CONFIG['entropy_window_size'],
                                       baseline=CONFIG['weight_baseline'])
    print(f"  小波权重范围: [{W_wav.min():.3f}, {W_wav.max():.3f}], 均值: {W_wav.mean():.3f}")
    
    # ========================================================================
    # Step 4: 运行 ADMM 重建
    # ========================================================================
    print("\n[Step 4] 运行双正则化 ADMM 求解器...")
    print(f"  参数: lambda_tv={CONFIG['lambda_tv']}, lambda_wav={CONFIG['lambda_wav']}, "
          f"rho1={CONFIG['rho1']}, rho2={CONFIG['rho2']}")
    print("-" * 70)
    
    x_recon, obj_history = dual_admm_solver(
        A, y, img_shape, W_tv, W_wav,
        lambda_tv=CONFIG['lambda_tv'],
        lambda_wav=CONFIG['lambda_wav'],
        rho1=CONFIG['rho1'],
        rho2=CONFIG['rho2'],
        max_iter=CONFIG['max_iter'],
        tol=CONFIG['tol'],
        cg_max_iter=CONFIG['cg_max_iter'],
        wavelet_name=CONFIG['wavelet_name'],
        wavelet_level=CONFIG['wavelet_level'],
        verbose=True
    )
    
    print("-" * 70)
    print(f"重建完成. 总迭代次数: {len(obj_history)}")
    
    # ========================================================================
    # Step 5: 评估重建质量
    # ========================================================================
    print("\n[Step 5] 评估重建质量...")
    img_recon = x_recon.reshape(img_shape)
    
    psnr_val = calculate_psnr(img_array, img_recon)
    ssim_val = calculate_ssim(img_array, img_recon)
    
    # 计算反投影结果用于比较
    x_backproj = A.T @ y
    img_backproj = x_backproj.reshape(img_shape)
    psnr_bp = calculate_psnr(img_array, img_backproj)
    
    print("\n" + "=" * 70)
    print("重建质量指标:")
    print("=" * 70)
    print(f"  反投影 PSNR: {psnr_bp:.4f} dB")
    print(f"  ADMM PSNR: {psnr_val:.4f} dB")
    print(f"  ADMM SSIM: {ssim_val:.4f}")
    print(f"  相比反投影的提升: {psnr_val - psnr_bp:.4f} dB")
    print("=" * 70)
    
    # 验证假设
    if psnr_val > CONFIG['psnr_threshold']:
        print(f"\n✓ 成功: PSNR ({psnr_val:.4f} dB) 超过阈值 ({CONFIG['psnr_threshold']} dB)")
        print("  假设支持: 双正则化 ADMM 求解器成功收敛")
    else:
        print(f"\n✗ 失败: PSNR ({psnr_val:.4f} dB) 未超过阈值 ({CONFIG['psnr_threshold']} dB)")
        print("  假设不支持")
    
    # ========================================================================
    # Step 6: 保存可视化结果
    # ========================================================================
    if CONFIG['save_figures']:
        print("\n[Step 6] 保存可视化结果...")
        output_dir = CONFIG['output_dir']
        
        plot_convergence(obj_history, os.path.join(output_dir, 'convergence_plot.png'))
        
        plot_reconstruction_comparison(
            img_array, img_backproj, img_recon,
            psnr_bp, psnr_val, ssim_val,
            os.path.join(output_dir, 'reconstruction_comparison.png')
        )
        
        plot_comprehensive_results(
            img_array, img_recon, obj_history,
            psnr_val, ssim_val, CONFIG['sampling_rate'], actual_snr,
            os.path.join(output_dir, 'final_admm_dual_regularizer_results.png')
        )
    
    # ========================================================================
    # 返回结果汇总
    # ========================================================================
    results = {
        'x_recon': x_recon,
        'img_recon': img_recon,
        'obj_history': obj_history,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'psnr_backproj': psnr_bp,
        'actual_snr': actual_snr,
        'iterations': len(obj_history),
    }
    
    print("\n" + "=" * 70)
    print("结果汇总:")
    print("=" * 70)
    print(f"  采样率: {CONFIG['sampling_rate']*100}%")
    print(f"  SNR: {actual_snr:.2f} dB")
    print(f"  ADMM 迭代次数: {len(obj_history)}")
    print(f"  目标函数下降: {(obj_history[0] - obj_history[-1]) / obj_history[0] * 100:.2f}%")
    print(f"  最终 PSNR: {psnr_val:.4f} dB")
    print(f"  最终 SSIM: {ssim_val:.4f}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    import sys
    
    # 如果提供了命令行参数，使用该路径作为图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = None
    
    results = main(image_path)
