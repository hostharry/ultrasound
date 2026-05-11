"""
域特定数据增强改善 Fast-Adaptive-HASA 重建
==========================================

本模块实现了基于数据增强的 Fast-Adaptive-HASA 方法，
使用轻量级 U-Net 网络预测自适应权重图。

主要内容：
    1. BUSI 数据集加载和分割
    2. 压缩感知采集模拟
    3. 域特定数据增强（旋转、平移、伽马校正）
    4. Slow Adaptive-HASA 基准算法
    5. 轻量级 U-Net 权重预测网络
    6. Fast-Adaptive-HASA 实现

核心思想：
    - Slow Adaptive-HASA：每次迭代计算局部梯度和熵，计算密集
    - Fast-Adaptive-HASA：使用 CNN 直接从反投影预测权重图，大幅加速
    - 数据增强：通过旋转、平移、伽马校正扩展训练数据

作者: Auto-generated from Jupyter notebook
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter, rotate, shift, sobel, uniform_filter
from scipy.stats import entropy
from skimage.restoration import denoise_tv_chambolle
from skimage.exposure import adjust_gamma
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters.rank import entropy as skimage_entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import time

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
    'measurement_seed': 42,             # 测量矩阵随机种子
    'noise_seed': 43,                   # 噪声随机种子
    
    # 数据增强参数
    'rotation_range': 10,               # 旋转角度范围 ±10°
    'translation_range': 5,             # 平移范围 ±5 像素
    'gamma_range': (0.8, 1.2),          # 伽马校正范围
    'n_augmentations': 10,              # 每张图像的增强数量
    
    # 训练/测试分割
    'n_train_benign': 14,
    'n_test_benign': 3,
    'n_train_malignant': 9,
    'n_test_malignant': 1,
    'n_train_normal': 2,
    'n_test_normal': 1,
    
    # Adaptive-HASA 参数
    'n_iterations': 50,                 # 迭代次数
    'step_size': 0.001,                 # 梯度下降步长
    'tv_weight_base': 0.1,              # TV 权重基础值
    'wavelet_threshold_base': 0.05,     # 小波阈值基础值
    'window_size': 9,                   # 局部统计窗口大小
    'update_weights_every': 5,          # 权重更新频率
    
    # 小波参数
    'wavelet': 'db4',
    'wavelet_level': 3,
    
    # U-Net 参数
    'unet_base_channels': 16,
    
    # 训练参数
    'batch_size': 8,
    'learning_rate': 1e-3,
    'n_epochs': 50,
    
    # 数据集路径
    'dataset_path': 'Dataset_BUSI_with_GT',
    
    # 输出参数
    'save_model': True,
    'model_path': 'fast_adaptive_hasa_unet.pth',
}


# ============================================================================
# 数据集加载函数
# ============================================================================

def load_busi_images(base_path: str = 'Dataset_BUSI_with_GT') -> List[Tuple]:
    """
    加载 BUSI 数据集所有图像
    
    参数:
        base_path: 数据集路径
        
    返回:
        图像列表，每个元素为 (图像数组, 类别, 文件名)
    """
    images = []
    
    for category in ['benign', 'malignant', 'normal']:
        cat_path = os.path.join(base_path, category)
        if not os.path.exists(cat_path):
            continue
        
        files = sorted([f for f in os.listdir(cat_path) 
                       if f.endswith('.png') and '_mask' not in f])
        
        for fname in files:
            img_path = os.path.join(cat_path, fname)
            img = Image.open(img_path).convert('L')  # 转换为灰度图
            images.append((np.array(img), category, fname))
    
    return images


def create_train_test_split(all_images: List[Tuple], 
                            config: dict) -> Tuple[List, List]:
    """
    创建分层训练/测试集分割
    
    参数:
        all_images: 所有图像列表
        config: 配置参数
        
    返回:
        (训练集, 测试集)
    """
    np.random.seed(42)  # 固定随机种子保证可重复性
    
    # 按类别分离
    benign_imgs = [(img, cat, fname) for img, cat, fname in all_images if cat == 'benign']
    malignant_imgs = [(img, cat, fname) for img, cat, fname in all_images if cat == 'malignant']
    normal_imgs = [(img, cat, fname) for img, cat, fname in all_images if cat == 'normal']
    
    # 打乱每个类别
    np.random.shuffle(benign_imgs)
    np.random.shuffle(malignant_imgs)
    np.random.shuffle(normal_imgs)
    
    # 分层分割
    train_images = []
    test_images = []
    
    train_images.extend(benign_imgs[:config['n_train_benign']])
    test_images.extend(benign_imgs[config['n_train_benign']:
                                   config['n_train_benign'] + config['n_test_benign']])
    
    train_images.extend(malignant_imgs[:config['n_train_malignant']])
    test_images.extend(malignant_imgs[config['n_train_malignant']:
                                      config['n_train_malignant'] + config['n_test_malignant']])
    
    train_images.extend(normal_imgs[:config['n_train_normal']])
    test_images.extend(normal_imgs[config['n_train_normal']:
                                   config['n_train_normal'] + config['n_test_normal']])
    
    return train_images, test_images


# ============================================================================
# 图像预处理函数
# ============================================================================

def preprocess_image(img: np.ndarray, 
                     target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    缩放并归一化图像到 [0, 1]
    
    参数:
        img: 输入图像数组
        target_size: 目标尺寸
        
    返回:
        归一化的图像数组
    """
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize(target_size, Image.BILINEAR)
    img_normalized = np.array(img_resized, dtype=np.float64) / 255.0
    return img_normalized


# ============================================================================
# 压缩感知模拟函数
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
        测量矩阵 (n_measurements x n_pixels)
    """
    np.random.seed(seed)
    A = np.random.randn(n_measurements, n_pixels) / np.sqrt(n_measurements)
    return A


def add_gaussian_noise(y: np.ndarray, target_snr_db: float = 25, 
                       seed: int = 43) -> np.ndarray:
    """
    添加高斯噪声以达到目标 SNR
    
    参数:
        y: 输入信号
        target_snr_db: 目标信噪比 (dB)
        seed: 随机种子
        
    返回:
        带噪声的信号
    """
    np.random.seed(seed)
    signal_power = np.mean(y ** 2)
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
    return y + noise


def simulate_cs_acquisition(img: np.ndarray, sampling_rate: float = 0.15,
                            target_snr_db: float = 25,
                            measurement_seed: int = 42,
                            noise_seed: int = 43) -> Tuple[np.ndarray, np.ndarray]:
    """
    完整的压缩感知采集模拟
    
    参数:
        img: 输入图像
        sampling_rate: 采样率
        target_snr_db: 目标 SNR
        measurement_seed: 测量矩阵随机种子
        noise_seed: 噪声随机种子
        
    返回:
        (测量矩阵, 带噪声测量值)
    """
    img_flat = img.flatten()
    n_pixels = len(img_flat)
    n_measurements = int(sampling_rate * n_pixels)
    
    A = create_measurement_matrix(n_measurements, n_pixels, seed=measurement_seed)
    y = A @ img_flat
    y_noisy = add_gaussian_noise(y, target_snr_db, seed=noise_seed)
    
    return A, y_noisy


# ============================================================================
# 数据增强函数
# ============================================================================

def augment_image(img: np.ndarray, aug_seed: int,
                  rotation_range: float = 10,
                  translation_range: float = 5,
                  gamma_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    对图像应用随机增强
    
    增强包括：
        - 随机旋转：±rotation_range 度
        - 随机平移：±translation_range 像素
        - 随机伽马校正：gamma_range[0] 到 gamma_range[1]
    
    参数:
        img: 输入图像
        aug_seed: 随机种子
        rotation_range: 旋转范围（度）
        translation_range: 平移范围（像素）
        gamma_range: 伽马校正范围
        
    返回:
        增强后的图像
    """
    np.random.seed(aug_seed)
    
    # 随机旋转
    angle = np.random.uniform(-rotation_range, rotation_range)
    img_aug = rotate(img, angle, reshape=False, mode='nearest')
    
    # 随机平移
    shift_x = np.random.uniform(-translation_range, translation_range)
    shift_y = np.random.uniform(-translation_range, translation_range)
    img_aug = shift(img_aug, shift=[shift_y, shift_x], mode='nearest')
    
    # 随机伽马校正
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    img_aug = adjust_gamma(img_aug, gamma=gamma)
    
    # 裁剪到有效范围 [0, 1]
    img_aug = np.clip(img_aug, 0, 1)
    
    return img_aug


# ============================================================================
# 重建辅助函数
# ============================================================================

def initial_backprojection(A: np.ndarray, y: np.ndarray, 
                           img_shape: Tuple[int, int]) -> np.ndarray:
    """
    使用反投影进行初始重建 (A^T @ y)
    
    参数:
        A: 测量矩阵
        y: 测量值
        img_shape: 图像形状
        
    返回:
        初始重建图像
    """
    x_init = A.T @ y
    return x_init.reshape(img_shape)


def wavelet_soft_threshold(img: np.ndarray, threshold: float,
                           wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """
    应用小波软阈值
    
    参数:
        img: 输入图像
        threshold: 阈值
        wavelet: 小波基
        level: 分解层数
        
    返回:
        阈值处理后的图像
    """
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    
    # 对细节系数进行阈值处理
    coeffs_thresh = [coeffs[0]]  # 保持近似系数
    for detail_level in coeffs[1:]:
        coeffs_thresh.append(tuple([
            pywt.threshold(c, threshold, mode='soft') for c in detail_level
        ]))
    
    result = pywt.waverec2(coeffs_thresh, wavelet=wavelet)
    return result[:img.shape[0], :img.shape[1]]


def compute_local_entropy(img: np.ndarray, window_size: int = 9) -> np.ndarray:
    """
    计算局部熵用于自适应权重
    
    参数:
        img: 输入图像
        window_size: 窗口大小
        
    返回:
        局部熵图
    """
    def local_entropy_func(values):
        hist, _ = np.histogram(values, bins=10, range=(0, 1))
        hist = hist + 1e-10  # 避免 log(0)
        return entropy(hist)
    
    entropy_map = generic_filter(img, local_entropy_func, size=window_size, mode='nearest')
    return entropy_map


def compute_local_gradient(img: np.ndarray, window_size: int = 9) -> np.ndarray:
    """
    计算局部梯度幅度用于自适应权重
    
    参数:
        img: 输入图像
        window_size: 窗口大小
        
    返回:
        局部梯度图
    """
    grad_x = sobel(img, axis=1)
    grad_y = sobel(img, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    local_grad = uniform_filter(grad_mag, size=window_size, mode='nearest')
    return local_grad


def compute_local_entropy(img: np.ndarray, window_size: int = 9) -> np.ndarray:
    """
    使用 skimage 计算真正的局部熵（基于香农熵）
    
    参数:
        img: 输入图像 [0, 1]
        window_size: 窗口大小（用于计算 disk 半径）
        
    返回:
        局部熵图
    """
    # 将窗口大小转换为 disk 半径
    radius = max(1, window_size // 2)
    selem = disk(radius)
    
    # skimage.filters.rank.entropy 需要 uint8 图像
    img_clipped = np.clip(img, 0, 1)
    img_uint8 = img_as_ubyte(img_clipped)
    
    # 计算局部熵
    entropy_map = skimage_entropy(img_uint8, selem)
    
    return entropy_map.astype(np.float64)


# ============================================================================
# Slow Adaptive-HASA 算法
# ============================================================================

def adaptive_hasa_slow(A: np.ndarray, y: np.ndarray, img_shape: Tuple[int, int],
                       n_iterations: int = 50, step_size: float = 0.001,
                       tv_weight_base: float = 0.1, wavelet_threshold_base: float = 0.05,
                       window_size: int = 9, update_weights_every: int = 5,
                       wavelet: str = 'db4', wavelet_level: int = 3) -> np.ndarray:
    """
    Slow Adaptive-HASA 算法（基准方法）
    
    周期性更新自适应权重的 HASA 算法
    
    参数:
        A: 测量矩阵
        y: 带噪声测量值
        img_shape: 目标图像形状
        n_iterations: 迭代次数
        step_size: 梯度下降步长
        tv_weight_base: TV 权重基础值
        wavelet_threshold_base: 小波阈值基础值
        window_size: 局部统计窗口大小
        update_weights_every: 权重更新频率
        wavelet: 小波基
        wavelet_level: 小波分解层数
        
    返回:
        重建图像
    """
    # 初始反投影
    x = initial_backprojection(A, y, img_shape)
    
    tv_weight = tv_weight_base
    wavelet_threshold = wavelet_threshold_base
    
    for it in range(n_iterations):
        # 周期性更新自适应权重
        if it % update_weights_every == 0:
            # 计算局部统计
            local_grad = compute_local_gradient(x, window_size=window_size)
            local_entropy_map = compute_local_entropy(x, window_size=window_size)
            
            # 归一化到 [0, 1]
            grad_norm = (local_grad - local_grad.min()) / (local_grad.max() - local_grad.min() + 1e-10)
            entropy_norm = (local_entropy_map - local_entropy_map.min()) / \
                          (local_entropy_map.max() - local_entropy_map.min() + 1e-10)
            
            # 自适应权重：结构更多的区域给予更高权重
            tv_weight_map = tv_weight_base * (1 + grad_norm)
            wavelet_threshold_map = wavelet_threshold_base * (1 + entropy_norm)
            
            # 使用均值（scikit-image TV 是空间不变的）
            tv_weight = tv_weight_map.mean()
            wavelet_threshold = wavelet_threshold_map.mean()
        
        # 数据保真度梯度: A^T(Ax - y)
        residual = A @ x.flatten() - y
        gradient = A.T @ residual
        gradient = gradient.reshape(img_shape)
        
        # 梯度下降步骤
        x = x - step_size * gradient
        
        # 顺序执行 TV 去噪和小波阈值（顺序很重要！）
        x = denoise_tv_chambolle(x, weight=tv_weight)
        x = wavelet_soft_threshold(x, threshold=wavelet_threshold,
                                   wavelet=wavelet, level=wavelet_level)
        
        # 裁剪到有效范围
        x = np.clip(x, 0, 1)
    
    return x


# ============================================================================
# U-Net 网络定义
# ============================================================================

class UNet(nn.Module):
    """
    轻量级 U-Net 用于预测自适应权重图
    
    输入：单通道反投影图像
    输出：双通道权重图（TV 权重，小波阈值）
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 2, 
                 base_channels: int = 16):
        """
        初始化 U-Net
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            base_channels: 基础通道数
        """
        super(UNet, self).__init__()
        
        # 编码器
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        
        # 解码器
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        
        # 输出层
        self.out = nn.Conv2d(base_channels, out_channels, 1)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """构建卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 编码器
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # 瓶颈
        b = self.bottleneck(p3)
        
        # 解码器（带跳跃连接）
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # 输出
        out = self.out(d1)
        return out


# ============================================================================
# 数据集类
# ============================================================================

class WeightMapDataset(Dataset):
    """自适应权重图数据集"""
    
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        """
        初始化数据集
        
        参数:
            inputs: 输入图像数组 (N, H, W)
            targets: 目标权重图数组 (N, 2, H, W)
        """
        self.inputs = torch.FloatTensor(inputs).unsqueeze(1)  # (N, 1, H, W)
        self.targets = torch.FloatTensor(targets)  # (N, 2, H, W)
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# ============================================================================
# 训练数据生成
# ============================================================================

def generate_training_data(train_images: List[Tuple], config: dict,
                           n_images: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成带数据增强的训练数据
    
    参数:
        train_images: 训练图像列表
        config: 配置参数
        n_images: 使用的图像数量（默认全部）
        
    返回:
        (输入数组, 目标数组)
    """
    if n_images is None:
        n_images = len(train_images)
    else:
        n_images = min(n_images, len(train_images))
    
    print(f"生成增强训练数据集...")
    print(f"使用 {n_images} 张训练图像 x {config['n_augmentations']} 增强")
    
    train_inputs = []
    train_targets = []
    
    for idx in tqdm(range(n_images), desc="处理图像"):
        img, category, fname = train_images[idx]
        
        # 预处理原始图像
        img_preprocessed = preprocess_image(img, config['target_size'])
        
        # 创建增强版本
        for aug_idx in range(config['n_augmentations']):
            aug_seed = idx * 100 + aug_idx
            img_aug = augment_image(
                img_preprocessed, aug_seed=aug_seed,
                rotation_range=config['rotation_range'],
                translation_range=config['translation_range'],
                gamma_range=config['gamma_range']
            )
            
            # 模拟压缩感知采集
            A_aug, y_aug = simulate_cs_acquisition(
                img_aug, sampling_rate=config['sampling_rate'],
                measurement_seed=config['measurement_seed'] + aug_seed,
                noise_seed=config['noise_seed'] + aug_seed
            )
            
            # 初始反投影
            x_init = initial_backprojection(A_aug, y_aug, img_aug.shape)
            
            # 计算目标权重图
            local_grad = compute_local_gradient(x_init, window_size=config['window_size'])
            local_entropy = compute_local_entropy(x_init, window_size=config['window_size'])
            
            # 归一化
            grad_norm = (local_grad - local_grad.min()) / (local_grad.max() - local_grad.min() + 1e-10)
            entropy_norm = (local_entropy - local_entropy.min()) / (local_entropy.max() - local_entropy.min() + 1e-10)
            
            train_inputs.append(x_init)
            train_targets.append(np.stack([grad_norm, entropy_norm], axis=0))
    
    return np.array(train_inputs), np.array(train_targets)


# ============================================================================
# 模型训练
# ============================================================================

def train_unet(model: nn.Module, train_loader: DataLoader, 
               config: dict, device: str = 'cpu') -> List[float]:
    """
    训练 U-Net 模型
    
    参数:
        model: U-Net 模型
        train_loader: 训练数据加载器
        config: 配置参数
        device: 设备（'cpu' 或 'cuda'）
        
    返回:
        损失历史
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    loss_history = []
    
    print(f"\n训练 U-Net ({config['n_epochs']} epochs)...")
    for epoch in range(config['n_epochs']):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['n_epochs']}, Loss: {avg_loss:.6f}")
    
    return loss_history


# ============================================================================
# Fast-Adaptive-HASA 算法
# ============================================================================

def fast_adaptive_hasa(A: np.ndarray, y: np.ndarray, img_shape: Tuple[int, int],
                       model: nn.Module, device: str = 'cpu',
                       n_iterations: int = 50, step_size: float = 0.001,
                       tv_weight_base: float = 0.1, wavelet_threshold_base: float = 0.05,
                       wavelet: str = 'db4', wavelet_level: int = 3) -> np.ndarray:
    """
    Fast-Adaptive-HASA 算法
    
    使用预训练 CNN 预测自适应权重，无需迭代计算局部统计
    
    参数:
        A: 测量矩阵
        y: 带噪声测量值
        img_shape: 目标图像形状
        model: 预训练 U-Net 模型
        device: 设备
        n_iterations: 迭代次数
        step_size: 梯度下降步长
        tv_weight_base: TV 权重基础值
        wavelet_threshold_base: 小波阈值基础值
        wavelet: 小波基
        wavelet_level: 小波分解层数
        
    返回:
        重建图像
    """
    # 初始反投影
    x = initial_backprojection(A, y, img_shape)
    
    # 使用 CNN 预测权重图（只需一次）
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(device)
        weight_maps = model(x_tensor).squeeze(0).cpu().numpy()
    
    grad_weight = weight_maps[0]
    var_weight = weight_maps[1]
    
    # 计算自适应权重
    tv_weight = tv_weight_base * (1 + grad_weight.mean())
    wavelet_threshold = wavelet_threshold_base * (1 + var_weight.mean())
    
    # 迭代重建（权重固定，无需重新计算）
    for it in range(n_iterations):
        # 数据保真度梯度
        residual = A @ x.flatten() - y
        gradient = A.T @ residual
        gradient = gradient.reshape(img_shape)
        
        # 梯度下降
        x = x - step_size * gradient
        
        # TV 去噪和小波阈值
        x = denoise_tv_chambolle(x, weight=tv_weight)
        x = wavelet_soft_threshold(x, threshold=wavelet_threshold,
                                   wavelet=wavelet, level=wavelet_level)
        
        x = np.clip(x, 0, 1)
    
    return x


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_reconstruction(img_true: np.ndarray, img_recon: np.ndarray) -> Dict:
    """
    评估重建质量
    
    参数:
        img_true: 真实图像
        img_recon: 重建图像
        
    返回:
        指标字典
    """
    psnr_val = peak_signal_noise_ratio(img_true, img_recon, data_range=1.0)
    ssim_val = structural_similarity(img_true, img_recon, data_range=1.0)
    
    return {
        'psnr': psnr_val,
        'ssim': ssim_val
    }


# ============================================================================
# 主函数
# ============================================================================

def main(config: dict = None):
    """主函数"""
    if config is None:
        config = CONFIG
    
    print("=" * 70)
    print("域特定数据增强改善 Fast-Adaptive-HASA 重建")
    print("=" * 70)
    
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 检查数据集
    if not os.path.exists(config['dataset_path']):
        print(f"错误: 数据集路径不存在 - {config['dataset_path']}")
        return None
    
    # 加载数据集
    print("\n[Step 1] 加载 BUSI 数据集...")
    all_images = load_busi_images(config['dataset_path'])
    print(f"总图像数: {len(all_images)}")
    
    # 统计类别
    categories = {}
    for img, cat, fname in all_images:
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # 创建训练/测试分割
    print("\n[Step 2] 创建训练/测试分割...")
    train_images, test_images = create_train_test_split(all_images, config)
    print(f"训练集: {len(train_images)} 张图像")
    print(f"测试集: {len(test_images)} 张图像")
    
    # 生成训练数据
    print("\n[Step 3] 生成增强训练数据...")
    n_train_images = min(10, len(train_images))  # 使用10张图像加速
    train_inputs, train_targets = generate_training_data(
        train_images, config, n_images=n_train_images
    )
    print(f"训练输入形状: {train_inputs.shape}")
    print(f"训练目标形状: {train_targets.shape}")
    
    # 创建数据加载器
    train_dataset = WeightMapDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 创建并训练 U-Net
    print("\n[Step 4] 创建并训练 U-Net...")
    model = UNet(in_channels=1, out_channels=2, base_channels=config['unet_base_channels'])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_history = train_unet(model, train_loader, config, device=device)
    
    # 保存模型
    if config['save_model']:
        torch.save(model.state_dict(), config['model_path'])
        print(f"模型已保存至: {config['model_path']}")
    
    # 在测试集上评估
    print("\n[Step 5] 在测试集上评估...")
    results = []
    
    for idx, (img, category, fname) in enumerate(test_images):
        print(f"\n测试图像 {idx+1}/{len(test_images)}: {fname}")
        
        img_preprocessed = preprocess_image(img, config['target_size'])
        A, y = simulate_cs_acquisition(
            img_preprocessed, 
            sampling_rate=config['sampling_rate'],
            measurement_seed=config['measurement_seed'],
            noise_seed=config['noise_seed']
        )
        
        # Slow Adaptive-HASA
        start_time = time.time()
        img_slow = adaptive_hasa_slow(
            A, y, config['target_size'],
            n_iterations=config['n_iterations'],
            step_size=config['step_size'],
            tv_weight_base=config['tv_weight_base'],
            wavelet_threshold_base=config['wavelet_threshold_base'],
            window_size=config['window_size'],
            update_weights_every=config['update_weights_every']
        )
        time_slow = time.time() - start_time
        metrics_slow = evaluate_reconstruction(img_preprocessed, img_slow)
        
        # Fast-Adaptive-HASA
        start_time = time.time()
        img_fast = fast_adaptive_hasa(
            A, y, config['target_size'],
            model=model, device=device,
            n_iterations=config['n_iterations'],
            step_size=config['step_size'],
            tv_weight_base=config['tv_weight_base'],
            wavelet_threshold_base=config['wavelet_threshold_base']
        )
        time_fast = time.time() - start_time
        metrics_fast = evaluate_reconstruction(img_preprocessed, img_fast)
        
        results.append({
            'image_name': fname,
            'category': category,
            'psnr_slow': metrics_slow['psnr'],
            'ssim_slow': metrics_slow['ssim'],
            'time_slow': time_slow,
            'psnr_fast': metrics_fast['psnr'],
            'ssim_fast': metrics_fast['ssim'],
            'time_fast': time_fast,
            'speedup': time_slow / time_fast
        })
        
        print(f"  Slow: PSNR={metrics_slow['psnr']:.2f}dB, SSIM={metrics_slow['ssim']:.4f}, Time={time_slow:.2f}s")
        print(f"  Fast: PSNR={metrics_fast['psnr']:.2f}dB, SSIM={metrics_fast['ssim']:.4f}, Time={time_fast:.2f}s")
        print(f"  加速比: {time_slow/time_fast:.2f}x")
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    avg_psnr_slow = np.mean([r['psnr_slow'] for r in results])
    avg_ssim_slow = np.mean([r['ssim_slow'] for r in results])
    avg_time_slow = np.mean([r['time_slow'] for r in results])
    
    avg_psnr_fast = np.mean([r['psnr_fast'] for r in results])
    avg_ssim_fast = np.mean([r['ssim_fast'] for r in results])
    avg_time_fast = np.mean([r['time_fast'] for r in results])
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    
    print(f"\nSlow Adaptive-HASA:")
    print(f"  平均 PSNR: {avg_psnr_slow:.2f} dB")
    print(f"  平均 SSIM: {avg_ssim_slow:.4f}")
    print(f"  平均时间: {avg_time_slow:.2f} 秒")
    
    print(f"\nFast-Adaptive-HASA:")
    print(f"  平均 PSNR: {avg_psnr_fast:.2f} dB")
    print(f"  平均 SSIM: {avg_ssim_fast:.4f}")
    print(f"  平均时间: {avg_time_fast:.2f} 秒")
    print(f"  平均加速比: {avg_speedup:.2f}x")
    
    # 质量保持率
    psnr_retention = avg_psnr_fast / avg_psnr_slow * 100
    ssim_retention = avg_ssim_fast / avg_ssim_slow * 100
    print(f"\n质量保持率:")
    print(f"  PSNR: {psnr_retention:.1f}%")
    print(f"  SSIM: {ssim_retention:.1f}%")
    
    print("=" * 70)
    
    return results, model, loss_history


if __name__ == '__main__':
    results, model, loss_history = main()
