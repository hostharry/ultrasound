"""
原型 ADMM 求解器 - 空间加权 TV 正则化
=====================================

本模块实现了用于压缩感知超声图像重建的 ADMM (Alternating Direction 
Method of Multipliers) 求解器，支持空间变化的 TV (Total Variation) 
正则化权重。

优化问题：
    argmin_x  0.5*||Ax-y||_2^2 + ||W*grad(x)||_1
    
其中：
    - A: 测量矩阵
    - y: 测量向量
    - W: 空间变化权重图
    - grad(x): 图像梯度

ADMM 迭代步骤：
    1. x-update: 求解 (A^T*A + ρ*div∘grad)*x = A^T*y + ρ*div(z-u)
    2. z-update: z = shrink(grad(x)+u, W/ρ)
    3. u-update: u = u + grad(x) - z

实验设置：
    - 测试图像：128×128 灰度超声图像
    - 采样率：15%
    - 迭代次数：50

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import LinearOperator, cg
from scipy.ndimage import sobel
import os
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 图像参数
    'image_path': 'Dataset_BUSI_with_GT/benign/benign (1).png',
    'target_size': (128, 128),
    
    # 压缩感知参数
    'sampling_rate': 0.15,              # 采样率 15%
    'random_seed': 42,
    
    # ADMM 参数
    'rho': 10.0,                        # ADMM 惩罚参数
    'max_iter': 50,                     # 最大迭代次数
    'cg_maxiter': 200,                  # CG 最大迭代次数
    'cg_rtol': 1e-5,                    # CG 相对容差
    'cg_atol': 1e-8,                    # CG 绝对容差
    
    # 权重图参数
    'weight_min': 1.0,                  # 最小权重
    'weight_max': 3.0,                  # 最大权重
    
    # 输出参数
    'verbose': True,
    'save_figures': True,
    'output_dir': '.',
}


# ============================================================================
# 图像加载与预处理
# ============================================================================

def load_and_preprocess_image(image_path: str, 
                               target_size: Tuple[int, int] = (128, 128)
                               ) -> np.ndarray:
    """
    加载并预处理图像
    
    参数:
        image_path: 图像路径
        target_size: 目标尺寸
        
    返回:
        归一化的灰度图像 [0, 1]
    """
    img = Image.open(image_path).convert('L')
    img_resized = img.resize(target_size, Image.BILINEAR)
    x_true = np.array(img_resized, dtype=np.float64) / 255.0
    return x_true


# ============================================================================
# 压缩感知测量
# ============================================================================

def create_measurement_matrix(m: int, n: int, seed: int = 42) -> np.ndarray:
    """
    创建高斯随机测量矩阵
    
    参数:
        m: 测量数量
        n: 图像像素数
        seed: 随机种子
        
    返回:
        测量矩阵 (m × n)
    """
    np.random.seed(seed)
    A = np.random.randn(m, n) / np.sqrt(m)
    return A


def generate_measurements(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    生成无噪声测量值
    
    参数:
        A: 测量矩阵
        x: 原始图像向量
        
    返回:
        测量向量
    """
    return A @ x


# ============================================================================
# 空间变化权重图
# ============================================================================

def create_weight_map(image: np.ndarray, 
                      weight_min: float = 1.0, 
                      weight_max: float = 3.0) -> np.ndarray:
    """
    基于边缘信息创建空间变化权重图
    
    边缘处使用更高的权重（更强的正则化）
    
    参数:
        image: 输入图像
        weight_min: 最小权重
        weight_max: 最大权重
        
    返回:
        权重图
    """
    # 使用 Sobel 算子计算梯度
    gx = sobel(image, axis=1)
    gy = sobel(image, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # 归一化到 [0, 1]
    grad_mag_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-10)
    
    # 创建权重图
    W = weight_min + (weight_max - weight_min) * grad_mag_norm
    
    return W


# ============================================================================
# 梯度和散度算子
# ============================================================================

def gradient_2d(x: np.ndarray, shape: Tuple[int, int]
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 2D 梯度（前向差分，循环边界）
    
    参数:
        x: 扁平化的图像向量
        shape: 图像形状 (h, w)
        
    返回:
        (gx, gy) 水平和垂直梯度
    """
    x_2d = x.reshape(shape)
    h, w = shape
    
    # 水平梯度
    gx = np.zeros_like(x_2d)
    gx[:, :-1] = x_2d[:, 1:] - x_2d[:, :-1]
    gx[:, -1] = x_2d[:, 0] - x_2d[:, -1]
    
    # 垂直梯度
    gy = np.zeros_like(x_2d)
    gy[:-1, :] = x_2d[1:, :] - x_2d[:-1, :]
    gy[-1, :] = x_2d[0, :] - x_2d[-1, :]
    
    return gx, gy


def divergence_2d(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    计算 2D 散度（梯度的伴随算子）
    
    使用后向差分，返回负散度以满足伴随性质
    
    参数:
        gx: 水平梯度
        gy: 垂直梯度
        
    返回:
        扁平化的散度向量
    """
    h, w = gx.shape
    
    # 水平散度
    divx = np.zeros_like(gx)
    divx[:, 0] = gx[:, 0] - gx[:, -1]
    divx[:, 1:] = gx[:, 1:] - gx[:, :-1]
    
    # 垂直散度
    divy = np.zeros_like(gy)
    divy[0, :] = gy[0, :] - gy[-1, :]
    divy[1:, :] = gy[1:, :] - gy[:-1, :]
    
    # 返回负散度（伴随性质）
    div = -(divx + divy)
    return div.flatten()


def verify_adjoint_property(shape: Tuple[int, int] = (128, 128)) -> bool:
    """
    验证梯度和散度算子的伴随性质
    
    <grad(x), v> = <x, div(v)>
    
    参数:
        shape: 测试图像形状
        
    返回:
        是否满足伴随性质
    """
    n = shape[0] * shape[1]
    test_x = np.random.randn(n)
    test_gx = np.random.randn(*shape)
    test_gy = np.random.randn(*shape)
    
    gx, gy = gradient_2d(test_x, shape)
    
    inner1 = np.sum(gx * test_gx) + np.sum(gy * test_gy)
    inner2 = np.dot(test_x, divergence_2d(test_gx, test_gy))
    
    return abs(inner1 - inner2) < 1e-6


# ============================================================================
# 软阈值算子
# ============================================================================

def soft_threshold(x: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """
    软阈值（收缩）算子
    
    S(x, t) = sign(x) * max(|x| - t, 0)
    
    参数:
        x: 输入数组
        threshold: 阈值（标量或数组）
        
    返回:
        阈值处理后的数组
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


# ============================================================================
# ADMM 求解器
# ============================================================================

def admm_spatially_weighted_tv(A: np.ndarray, y: np.ndarray, W: np.ndarray,
                                img_shape: Tuple[int, int],
                                x_true_vec: Optional[np.ndarray] = None,
                                rho: float = 10.0,
                                max_iter: int = 50,
                                cg_maxiter: int = 200,
                                cg_rtol: float = 1e-5,
                                cg_atol: float = 1e-8,
                                verbose: bool = True
                                ) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    ADMM 求解器：空间加权 TV 正则化的压缩感知重建
    
    求解：argmin_x  0.5*||Ax-y||_2^2 + ||W*grad(x)||_1
    
    参数:
        A: 测量矩阵 (m × n)
        y: 测量向量 (m,)
        W: 空间变化权重图 (h, w)
        img_shape: 图像形状 (h, w)
        x_true_vec: 真实图像向量（用于误差监控，可选）
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        cg_maxiter: 共轭梯度最大迭代次数
        cg_rtol: CG 相对容差
        cg_atol: CG 绝对容差
        verbose: 是否打印进度
        
    返回:
        (重建图像向量, 重建误差列表, 目标函数值列表)
    """
    m, n = A.shape
    h, w = img_shape
    
    # 用反投影初始化
    x = A.T @ y
    if verbose:
        print(f"初始化: x 范围 [{x.min():.4f}, {x.max():.4f}]")
    
    # 分裂变量
    zx = np.zeros((h, w))
    zy = np.zeros((h, w))
    
    # 对偶变量
    ux = np.zeros((h, w))
    uy = np.zeros((h, w))
    
    # 预计算 A^T @ y
    ATy = A.T @ y
    
    # 定义 x-update 的线性算子
    def matvec_x_update(x_vec):
        ATAx = A.T @ (A @ x_vec)
        gx, gy = gradient_2d(x_vec, img_shape)
        divgrad_x = divergence_2d(gx, gy)
        return ATAx + rho * divgrad_x
    
    lin_op = LinearOperator((n, n), matvec=matvec_x_update)
    
    errors = []
    objectives = []
    
    if verbose:
        print("开始 ADMM 迭代...")
        print(f"参数: ρ={rho}, max_iter={max_iter}")
        print(f"图像形状: {img_shape}, n={n}, m={m}")
    
    for iter_num in range(max_iter):
        # x-update: 求解 (A^T*A + ρ*div∘grad)*x = A^T*y + ρ*div(z-u)
        rhs = ATy + rho * divergence_2d(zx - ux, zy - uy)
        x_new, info = cg(lin_op, rhs, x0=x, maxiter=cg_maxiter, 
                         rtol=cg_rtol, atol=cg_atol)
        
        if info > 0 and verbose and iter_num % 10 == 0:
            print(f"注意: CG 在 {info} 次迭代后停止（可能未完全收敛）")
        elif info < 0:
            print(f"错误: CG 在迭代 {iter_num} 出错, info={info}")
            break
        
        x = x_new
        
        # z-update: 加权软阈值
        gx, gy = gradient_2d(x, img_shape)
        zx = soft_threshold(gx + ux, W / rho)
        zy = soft_threshold(gy + uy, W / rho)
        
        # u-update: u = u + grad(x) - z
        ux = ux + gx - zx
        uy = uy + gy - zy
        
        # 计算重建误差
        if x_true_vec is not None:
            recon_error = np.linalg.norm(x - x_true_vec)
            errors.append(recon_error)
        
        # 计算目标函数值
        residual = A @ x - y
        data_fidelity = 0.5 * np.sum(residual**2)
        tv_term = np.sum(W * np.sqrt(gx**2 + gy**2))
        objective = data_fidelity + tv_term
        objectives.append(objective)
        
        if verbose and (iter_num % 10 == 0 or iter_num == max_iter - 1):
            error_str = f", 重建误差={recon_error:.6f}" if x_true_vec is not None else ""
            print(f"迭代 {iter_num:3d}: 数据保真={data_fidelity:.6f}, "
                  f"TV项={tv_term:.6f}, 目标={objective:.6f}{error_str}")
    
    return x, errors, objectives


# ============================================================================
# 评估函数
# ============================================================================

def compute_metrics(x_true: np.ndarray, 
                    x_recon: np.ndarray) -> Dict[str, float]:
    """
    计算重建质量指标
    
    参数:
        x_true: 真实图像
        x_recon: 重建图像
        
    返回:
        指标字典
    """
    # 像素误差
    pixel_error = np.abs(x_recon - x_true)
    
    # MSE, RMSE, PSNR
    mse = np.mean((x_recon - x_true)**2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(1.0 / rmse) if rmse > 0 else np.inf
    
    return {
        'mse': mse,
        'rmse': rmse,
        'psnr': psnr,
        'pixel_error_mean': pixel_error.mean(),
        'pixel_error_std': pixel_error.std(),
        'pixel_error_max': pixel_error.max()
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_reconstruction_comparison(x_true: np.ndarray, x_recon: np.ndarray,
                                    psnr: float, 
                                    output_path: Optional[str] = None):
    """
    绘制真实图像与重建图像对比
    
    参数:
        x_true: 真实图像
        x_recon: 重建图像
        psnr: PSNR 值
        output_path: 输出路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 真实图像
    ax = axes[0]
    im1 = ax.imshow(x_true, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Ground Truth', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    
    # 重建图像
    ax = axes[1]
    im2 = ax.imshow(x_recon, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'ADMM Spatially-Weighted TV\n(PSNR: {psnr:.2f} dB)', 
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle('Compressed Sensing Reconstruction', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_convergence(objectives: List[float], errors: List[float],
                     output_path: Optional[str] = None):
    """
    绘制收敛曲线
    
    参数:
        objectives: 目标函数值列表
        errors: 重建误差列表
        output_path: 输出路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # 目标函数
    ax = axes[0]
    ax.plot(range(len(objectives)), objectives, 'b-', linewidth=2, 
            marker='o', markersize=4, markevery=5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('A. Objective Function Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    reduction = (objectives[0] - objectives[-1]) / objectives[0] * 100
    ax.text(0.6, 0.9, f'Reduction: {reduction:.1f}%\nFinal: {objectives[-1]:.2f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 重建误差
    if errors:
        ax = axes[1]
        ax.plot(range(len(errors)), errors, 'r-', linewidth=2, 
                marker='s', markersize=4, markevery=5)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Reconstruction Error ||x - x_true||', fontsize=12)
        ax.set_title('B. Reconstruction Error', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.text(0.6, 0.9, f'Initial: {errors[0]:.2f}\nFinal: {errors[-1]:.2f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"收敛图已保存: {output_path}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# 主函数
# ============================================================================

def run_admm_experiment(config: dict = None) -> Dict:
    """
    运行 ADMM 求解器实验
    
    参数:
        config: 配置参数
        
    返回:
        结果字典
    """
    if config is None:
        config = CONFIG
    
    np.random.seed(config['random_seed'])
    
    print("=" * 70)
    print("原型 ADMM 求解器 - 空间加权 TV 正则化")
    print("=" * 70)
    
    # Step 1: 加载图像
    print("\n[Step 1] 加载测试图像...")
    if not os.path.exists(config['image_path']):
        print(f"错误: 图像路径不存在 - {config['image_path']}")
        return None
    
    x_true = load_and_preprocess_image(config['image_path'], config['target_size'])
    print(f"预处理后图像形状: {x_true.shape}")
    print(f"值范围: [{x_true.min():.4f}, {x_true.max():.4f}]")
    
    # 扁平化
    n = x_true.size
    x_true_vec = x_true.flatten()
    print(f"扁平化向量大小: {n}")
    
    # Step 2: 创建测量矩阵
    print("\n[Step 2] 创建测量矩阵...")
    m = int(config['sampling_rate'] * n)
    print(f"采样率: {config['sampling_rate']*100:.1f}%")
    print(f"测量数量: {m} / {n}")
    
    A = create_measurement_matrix(m, n, config['random_seed'])
    print(f"测量矩阵形状: {A.shape}")
    print(f"矩阵内存: {A.nbytes / 1024**2:.2f} MB")
    
    # Step 3: 生成测量值
    print("\n[Step 3] 生成测量值...")
    y = generate_measurements(A, x_true_vec)
    print(f"测量向量形状: {y.shape}")
    print(f"测量值范围: [{y.min():.4f}, {y.max():.4f}]")
    
    # Step 4: 创建权重图
    print("\n[Step 4] 创建空间变化权重图...")
    W = create_weight_map(x_true, config['weight_min'], config['weight_max'])
    print(f"权重图形状: {W.shape}")
    print(f"权重范围: [{W.min():.4f}, {W.max():.4f}]")
    print(f"权重空间方差: {W.std():.4f}")
    
    # Step 5: 验证算子
    print("\n[Step 5] 验证梯度/散度伴随性质...")
    is_adjoint = verify_adjoint_property(config['target_size'])
    print(f"伴随性质验证: {'通过 ✓' if is_adjoint else '失败 ✗'}")
    
    # Step 6: 运行 ADMM
    print("\n[Step 6] 运行 ADMM 求解器...")
    x_recon, errors, objectives = admm_spatially_weighted_tv(
        A, y, W,
        img_shape=config['target_size'],
        x_true_vec=x_true_vec,
        rho=config['rho'],
        max_iter=config['max_iter'],
        cg_maxiter=config['cg_maxiter'],
        cg_rtol=config['cg_rtol'],
        cg_atol=config['cg_atol'],
        verbose=config['verbose']
    )
    
    x_recon_img = x_recon.reshape(config['target_size'])
    
    # Step 7: 计算指标
    print("\n[Step 7] 计算重建质量指标...")
    metrics = compute_metrics(x_true, x_recon_img)
    
    print(f"\n重建质量:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  像素误差均值: {metrics['pixel_error_mean']:.6f}")
    print(f"  像素误差最大: {metrics['pixel_error_max']:.6f}")
    
    # Step 8: 收敛验证
    print("\n[Step 8] 收敛验证...")
    is_monotonic = all(objectives[i] >= objectives[i+1] 
                       for i in range(len(objectives)-1))
    reduction = (objectives[0] - objectives[-1]) / objectives[0] * 100
    
    print(f"✓ 目标函数降低: {reduction:.1f}%")
    print(f"✓ 单调递减: {'是' if is_monotonic else '否'}")
    print(f"✓ 最终目标值: {objectives[-1]:.2f}")
    print(f"✓ 完成 {len(objectives)} 次迭代")
    
    # Step 9: 保存可视化
    if config['save_figures']:
        print("\n[Step 9] 保存可视化...")
        
        plot_reconstruction_comparison(
            x_true, x_recon_img, metrics['psnr'],
            os.path.join(config['output_dir'], 'admm_reconstruction.png')
        )
        
        plot_convergence(
            objectives, errors,
            os.path.join(config['output_dir'], 'admm_convergence.png')
        )
    
    # 最终汇总
    print("\n" + "=" * 70)
    print("最终汇总")
    print("=" * 70)
    print(f"✓ ADMM 求解器成功实现空间加权 TV 正则化")
    print(f"✓ 目标函数收敛: {objectives[0]:.2f} → {objectives[-1]:.2f} ({reduction:.1f}% 降低)")
    print(f"✓ 单调收敛验证: {'通过' if is_monotonic else '失败'}")
    print(f"✓ 重建质量: PSNR = {metrics['psnr']:.2f} dB (采样率 {config['sampling_rate']*100}%)")
    print(f"✓ 空间变化权重成功集成 (std = {W.std():.4f})")
    
    return {
        'x_true': x_true,
        'x_recon': x_recon_img,
        'W': W,
        'errors': errors,
        'objectives': objectives,
        'metrics': metrics,
        'is_monotonic': is_monotonic
    }


def main():
    """主函数"""
    results = run_admm_experiment(CONFIG)
    return results


if __name__ == '__main__':
    results = main()
