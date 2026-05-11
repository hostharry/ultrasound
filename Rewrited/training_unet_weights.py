"""
训练轻量级 U-Net 预测自适应权重图
================================

本模块实现了训练轻量级 U-Net 来预测 Adaptive-HASA 算法的自适应权重图，
从而加速重建过程。

核心思想：
    - 传统 Adaptive-HASA 需要在每次迭代中计算局部熵和梯度，非常耗时
    - 训练 U-Net 直接从反投影图像预测权重图
    - 推理时只需一次前向传播，大幅加速

训练数据生成：
    1. 加载原始图像
    2. 模拟压缩感知测量
    3. 生成反投影图像（U-Net 输入）
    4. 运行 Adaptive-HASA 获取最终权重图（U-Net 目标）

实验设置：
    - 图像尺寸：256×256
    - 采样率：15%
    - SNR：25 dB
    - 训练/测试：50/10 张图像

作者: Auto-generated from Jupyter notebook
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import denoise_tv_chambolle
import scipy.ndimage as ndimage
import pywt
from numba import jit
import time
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch 未安装，U-Net 相关功能不可用")


# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 数据集参数
    'data_dir': 'Dataset_BUSI_with_GT',
    'target_size': 256,
    
    # 压缩感知参数
    'sampling_rate': 0.15,              # 采样率 15%
    'snr_db': 25,                       # 信噪比 25 dB
    'meas_seed': 42,
    'noise_seed': 43,
    
    # Adaptive-HASA 参数
    'n_iterations': 10,                 # 迭代次数（减少以加速）
    'step_size': 0.001,
    'base_tv_weight': 0.15,
    'base_wavelet_weight': 0.15,
    
    # 训练参数
    'n_train': 50,                      # 训练图像数
    'n_test': 10,                       # 测试图像数
    'batch_size': 4,
    'epochs': 50,
    'learning_rate': 1e-4,
    
    # 输出参数
    'verbose': True,
    'save_model': True,
}


# ============================================================================
# 数据集加载
# ============================================================================

def load_dataset(base_path: str = 'Dataset_BUSI_with_GT') -> pd.DataFrame:
    """
    加载 BUSI 数据集
    
    参数:
        base_path: 数据集路径
        
    返回:
        DataFrame 包含图像路径和类别信息
    """
    all_images = []
    
    for class_name in ['benign', 'malignant', 'normal']:
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            continue
        
        files = os.listdir(class_path)
        image_files = [f for f in files if not f.endswith('_mask.png') and f.endswith('.png')]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            mask_file = img_file.replace('.png', '_mask.png')
            mask_path = os.path.join(class_path, mask_file)
            
            all_images.append({
                'image_path': img_path,
                'mask_path': mask_path if os.path.exists(mask_path) else None,
                'class': class_name,
                'filename': img_file
            })
    
    return pd.DataFrame(all_images)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理数据集，移除无效条目
    
    参数:
        df: 原始 DataFrame
        
    返回:
        清理后的 DataFrame
    """
    # 移除错误识别的 mask 文件
    df_clean = df[~df['filename'].str.contains('_mask_1')].reset_index(drop=True)
    
    # 只保留有有效 mask 的图像
    def is_valid_image(row):
        if row['class'] == 'normal':
            return True
        return row['mask_path'] is not None and os.path.exists(row['mask_path'])
    
    df_clean = df_clean[df_clean.apply(is_valid_image, axis=1)].reset_index(drop=True)
    
    return df_clean


def split_dataset(df: pd.DataFrame, n_train: int = 50, n_test: int = 10, 
                  seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分层划分数据集
    
    参数:
        df: 数据集 DataFrame
        n_train: 训练集大小
        n_test: 测试集大小
        seed: 随机种子
        
    返回:
        (训练集, 测试集)
    """
    train_df, test_df = train_test_split(
        df,
        train_size=n_train,
        test_size=n_test,
        stratify=df['class'],
        random_state=seed
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================================
# 压缩感知工具
# ============================================================================

def preprocess_image(img_path: str, target_size: int = 256) -> np.ndarray:
    """
    加载并预处理图像
    
    参数:
        img_path: 图像路径
        target_size: 目标尺寸
        
    返回:
        归一化的灰度图像 [0, 1]
    """
    img = Image.open(img_path).convert('L')
    img = img.resize((target_size, target_size), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float64) / 255.0
    return img_array


def generate_measurement_matrix(n_measurements: int, n_pixels: int, 
                                 seed: int = 42) -> np.ndarray:
    """
    生成高斯随机测量矩阵
    
    参数:
        n_measurements: 测量数量
        n_pixels: 像素总数
        seed: 随机种子
        
    返回:
        测量矩阵 (m × n)
    """
    np.random.seed(seed)
    A = np.random.randn(n_measurements, n_pixels) / np.sqrt(n_measurements)
    return A


def add_noise_to_measurements(y_clean: np.ndarray, target_snr_db: float = 25, 
                               seed: int = 43) -> np.ndarray:
    """
    向测量添加高斯噪声以达到目标 SNR
    
    参数:
        y_clean: 原始测量
        target_snr_db: 目标 SNR (dB)
        seed: 随机种子
        
    返回:
        带噪声的测量
    """
    np.random.seed(seed)
    signal_power = np.mean(y_clean ** 2)
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(y_clean))
    return y_clean + noise


def create_cs_measurements(image: np.ndarray, sampling_rate: float = 0.15,
                           target_snr_db: float = 25
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建压缩感知测量
    
    参数:
        image: 原始图像
        sampling_rate: 采样率
        target_snr_db: 目标 SNR
        
    返回:
        (测量矩阵, 测量向量)
    """
    n_pixels = image.size
    n_measurements = int(sampling_rate * n_pixels)
    
    A = generate_measurement_matrix(n_measurements, n_pixels, seed=42)
    
    x = image.flatten()
    y_clean = A @ x
    y = add_noise_to_measurements(y_clean, target_snr_db, seed=43)
    
    return A, y


def backproject(A: np.ndarray, y: np.ndarray, 
                image_shape: Tuple[int, int]) -> np.ndarray:
    """
    简单反投影重建
    
    参数:
        A: 测量矩阵
        y: 测量向量
        image_shape: 图像形状
        
    返回:
        反投影图像
    """
    x_bp = A.T @ y
    return x_bp.reshape(image_shape)


# ============================================================================
# Numba 优化的局部熵计算
# ============================================================================

@jit(nopython=True)
def calculate_entropy_numba(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    使用 Numba JIT 计算局部熵
    
    比 scipy.ndimage.generic_filter 快约 45 倍
    
    参数:
        image: 输入图像
        window_size: 窗口大小
        
    返回:
        局部熵图
    """
    height, width = image.shape
    entropy_map = np.zeros((height, width), dtype=np.float64)
    half_window = window_size // 2
    
    for i in range(height):
        for j in range(width):
            hist = np.zeros(256, dtype=np.int32)
            
            for wi in range(-half_window, half_window + 1):
                for wj in range(-half_window, half_window + 1):
                    # 反射边界条件
                    ii = i + wi
                    jj = j + wj
                    
                    if ii < 0:
                        ii = -ii
                    elif ii >= height:
                        ii = 2 * height - ii - 2
                    
                    if jj < 0:
                        jj = -jj
                    elif jj >= width:
                        jj = 2 * width - jj - 2
                    
                    # 夹紧到有效范围
                    if ii < 0:
                        ii = 0
                    if ii >= height:
                        ii = height - 1
                    if jj < 0:
                        jj = 0
                    if jj >= width:
                        jj = width - 1
                    
                    # 添加到直方图
                    bin_idx = int(image[ii, jj] * 255)
                    if bin_idx > 255:
                        bin_idx = 255
                    if bin_idx < 0:
                        bin_idx = 0
                    hist[bin_idx] += 1
            
            # 计算熵
            total = window_size * window_size
            entropy = 0.0
            for count in hist:
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            
            entropy_map[i, j] = entropy
    
    return entropy_map


def calculate_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    计算梯度幅度（用于自适应 TV 权重）
    
    参数:
        image: 输入图像
        
    返回:
        梯度幅度图
    """
    grad_x = ndimage.sobel(image, axis=1)
    grad_y = ndimage.sobel(image, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return grad_mag


# ============================================================================
# Adaptive-HASA 重建
# ============================================================================

def adaptive_hasa_reconstruction_numba(x_init: np.ndarray, A: np.ndarray, 
                                        y: np.ndarray, image_shape: Tuple[int, int],
                                        n_iterations: int = 50, 
                                        step_size: float = 0.001,
                                        base_tv_weight: float = 0.15,
                                        base_wavelet_weight: float = 0.15,
                                        return_weights: bool = False
                                        ) -> Tuple[np.ndarray, ...]:
    """
    Numba 优化的 Adaptive-HASA 重建
    
    使用梯度幅度进行 TV 权重，使用熵进行小波权重
    
    参数:
        x_init: 初始图像
        A: 测量矩阵
        y: 测量向量
        image_shape: 图像形状
        n_iterations: 迭代次数
        step_size: 步长
        base_tv_weight: TV 基础权重
        base_wavelet_weight: 小波基础权重
        return_weights: 是否返回权重图
        
    返回:
        重建图像，可选地返回权重图
    """
    x = x_init.copy().flatten()
    
    tv_weight_map = None
    wavelet_weight_map = None
    
    for iteration in range(n_iterations):
        x_2d = x.reshape(image_shape)
        
        # 计算自适应权重
        # 梯度幅度用于 TV（边缘需要更多 TV 正则化）
        grad_mag = calculate_gradient_magnitude(x_2d)
        grad_mag_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
        tv_weight_map = base_tv_weight * (1 + grad_mag_norm)
        
        # 局部熵用于小波（纹理区域需要更多小波）
        entropy_map = calculate_entropy_numba(x_2d, window_size=5)
        entropy_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-8)
        wavelet_weight_map = base_wavelet_weight * (1 + entropy_norm)
        
        # 数据保真度梯度
        residual = A @ x - y
        grad_data = A.T @ residual
        
        # 梯度下降步骤
        x = x - step_size * grad_data
        
        # 应用 TV 去噪（使用平均权重）
        x_2d = x.reshape(image_shape)
        tv_weight_avg = tv_weight_map.mean()
        x_tv = denoise_tv_chambolle(x_2d, weight=tv_weight_avg)
        
        # 应用小波软阈值
        wavelet_weight_avg = wavelet_weight_map.mean()
        coeffs = pywt.wavedec2(x_tv, 'db4', level=3)
        coeffs_thresh = [coeffs[0]]
        for detail_level in coeffs[1:]:
            thresh_level = []
            for detail in detail_level:
                threshold = wavelet_weight_avg * np.median(np.abs(detail)) / 0.6745
                detail_thresh = pywt.threshold(detail, threshold, mode='soft')
                thresh_level.append(detail_thresh)
            coeffs_thresh.append(tuple(thresh_level))
        
        x_wavelet = pywt.waverec2(coeffs_thresh, 'db4')
        
        if x_wavelet.shape != image_shape:
            x_wavelet = x_wavelet[:image_shape[0], :image_shape[1]]
        
        x = x_wavelet.flatten()
    
    final_reconstruction = x.reshape(image_shape)
    
    if return_weights:
        return final_reconstruction, tv_weight_map, wavelet_weight_map
    else:
        return final_reconstruction


# ============================================================================
# 训练数据生成
# ============================================================================

def generate_training_data(train_df: pd.DataFrame, config: dict,
                           max_images: int = None) -> List[Dict]:
    """
    生成训练数据（反投影图像和对应的权重图）
    
    参数:
        train_df: 训练集 DataFrame
        config: 配置参数
        max_images: 最大图像数（None 表示全部）
        
    返回:
        训练数据列表
    """
    training_data = []
    image_shape = (config['target_size'], config['target_size'])
    
    n_images = len(train_df) if max_images is None else min(max_images, len(train_df))
    
    print(f"生成训练数据 ({n_images} 张图像)...")
    start_time = time.time()
    
    for idx in range(n_images):
        row = train_df.iloc[idx]
        img_start = time.time()
        
        # 加载图像
        img_orig = preprocess_image(row['image_path'], config['target_size'])
        
        # 创建压缩感知测量
        A, y = create_cs_measurements(img_orig, 
                                      config['sampling_rate'], 
                                      config['snr_db'])
        
        # 生成反投影
        x_bp = backproject(A, y, image_shape)
        
        # 运行 Adaptive-HASA 获取权重图
        x_recon, tv_weights, wavelet_weights = adaptive_hasa_reconstruction_numba(
            x_bp, A, y, image_shape,
            n_iterations=config['n_iterations'],
            step_size=config['step_size'],
            base_tv_weight=config['base_tv_weight'],
            base_wavelet_weight=config['base_wavelet_weight'],
            return_weights=True
        )
        
        training_data.append({
            'index': idx,
            'class': row['class'],
            'filename': row['filename'],
            'image_orig': img_orig,
            'backprojection': x_bp,
            'tv_weights_gt': tv_weights,
            'wavelet_weights_gt': wavelet_weights,
            'A': A,
            'y': y,
            'reconstruction': x_recon
        })
        
        img_time = time.time() - img_start
        if config['verbose']:
            print(f"  图像 {idx+1}/{n_images} ({row['class']}): {img_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"总时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"平均: {total_time/len(training_data):.1f}s/图像")
    
    return training_data


# ============================================================================
# U-Net 模型（如果 PyTorch 可用）
# ============================================================================

if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        """双卷积块"""
        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):
        """下采样块"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.maxpool_conv(x)

    class Up(nn.Module):
        """上采样块"""
        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class LightweightUNet(nn.Module):
        """
        轻量级 U-Net 用于预测自适应权重图
        
        输入: 反投影图像 (1 通道)
        输出: TV 和小波权重图 (2 通道)
        """
        def __init__(self, n_channels=1, n_classes=2, bilinear=True):
            super(LightweightUNet, self).__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear

            # 减少通道数以实现轻量化
            self.inc = DoubleConv(n_channels, 32)
            self.down1 = Down(32, 64)
            self.down2 = Down(64, 128)
            self.down3 = Down(128, 256)
            factor = 2 if bilinear else 1
            self.down4 = Down(256, 512 // factor)
            self.up1 = Up(512, 256 // factor, bilinear)
            self.up2 = Up(256, 128 // factor, bilinear)
            self.up3 = Up(128, 64 // factor, bilinear)
            self.up4 = Up(64, 32, bilinear)
            self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return torch.sigmoid(logits)  # 权重在 [0, 1] 范围

    class WeightMapDataset(Dataset):
        """权重图预测数据集"""
        def __init__(self, training_data):
            self.data = training_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            
            # 输入: 反投影图像
            backproj = torch.FloatTensor(item['backprojection']).unsqueeze(0)
            
            # 目标: TV 和小波权重图
            tv_weights = torch.FloatTensor(item['tv_weights_gt'])
            wav_weights = torch.FloatTensor(item['wavelet_weights_gt'])
            
            # 归一化权重到 [0, 1]
            tv_weights = (tv_weights - tv_weights.min()) / (tv_weights.max() - tv_weights.min() + 1e-8)
            wav_weights = (wav_weights - wav_weights.min()) / (wav_weights.max() - wav_weights.min() + 1e-8)
            
            targets = torch.stack([tv_weights, wav_weights], dim=0)
            
            return backproj, targets


# ============================================================================
# 训练函数
# ============================================================================

def train_unet(training_data: List[Dict], config: dict) -> Optional[nn.Module]:
    """
    训练 U-Net 预测权重图
    
    参数:
        training_data: 训练数据列表
        config: 配置参数
        
    返回:
        训练好的模型
    """
    if not TORCH_AVAILABLE:
        print("错误: PyTorch 未安装")
        return None
    
    print("\n训练轻量级 U-Net...")
    
    # 创建数据集和加载器
    dataset = WeightMapDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightUNet(n_channels=1, n_classes=2).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"设备: {device}")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    model.train()
    losses = []
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Loss = {avg_loss:.6f}")
    
    print("训练完成!")
    
    # 保存模型
    if config['save_model']:
        torch.save(model.state_dict(), 'unet_weight_predictor.pth')
        print("模型已保存: unet_weight_predictor.pth")
    
    return model


# ============================================================================
# 主函数
# ============================================================================

def run_training_pipeline(config: dict = None) -> Dict:
    """
    运行完整训练流程
    
    参数:
        config: 配置参数
        
    返回:
        结果字典
    """
    if config is None:
        config = CONFIG
    
    print("=" * 70)
    print("训练轻量级 U-Net 预测自适应权重图")
    print("=" * 70)
    
    # Step 1: 加载数据集
    print("\n[Step 1] 加载数据集...")
    df = load_dataset(config['data_dir'])
    print(f"总图像数: {len(df)}")
    print(f"类别分布:\n{df['class'].value_counts()}")
    
    # Step 2: 清理数据集
    print("\n[Step 2] 清理数据集...")
    df_clean = clean_dataset(df)
    print(f"清理后图像数: {len(df_clean)}")
    
    # Step 3: 划分数据集
    print("\n[Step 3] 划分数据集...")
    train_df, test_df = split_dataset(df_clean, config['n_train'], config['n_test'])
    print(f"训练集: {len(train_df)}")
    print(f"测试集: {len(test_df)}")
    
    # Step 4: 测试 Numba 编译
    print("\n[Step 4] 测试 Numba 编译...")
    test_img = np.random.rand(config['target_size'], config['target_size'])
    start = time.time()
    _ = calculate_entropy_numba(test_img, window_size=5)
    elapsed = time.time() - start
    print(f"熵计算时间: {elapsed:.2f}s")
    
    # Step 5: 生成训练数据（使用少量图像以节省时间）
    print("\n[Step 5] 生成训练数据...")
    print("注意: 由于计算限制，使用减少的图像数量")
    
    # 减少图像数量以适应计算限制
    max_train_images = min(10, len(train_df))  # 最多 10 张
    training_data = generate_training_data(train_df, config, max_images=max_train_images)
    
    print(f"\n生成的训练样本: {len(training_data)}")
    
    # Step 6: 训练 U-Net（如果有足够数据）
    model = None
    if TORCH_AVAILABLE and len(training_data) >= 5:
        print("\n[Step 6] 训练 U-Net...")
        model = train_unet(training_data, config)
    else:
        print("\n[Step 6] 跳过 U-Net 训练（数据不足或 PyTorch 不可用）")
    
    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"✓ 数据集加载完成: {len(df_clean)} 张图像")
    print(f"✓ 训练数据生成: {len(training_data)} 张图像")
    if model is not None:
        print(f"✓ U-Net 训练完成")
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'training_data': training_data,
        'model': model
    }


def main():
    """主函数"""
    results = run_training_pipeline(CONFIG)
    return results


if __name__ == '__main__':
    results = main()
