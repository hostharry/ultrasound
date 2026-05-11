"""
Fast-Adaptive-HASA 1D 超声压缩感知重建
======================================

针对 dataset_fdbf_energy_mu_8_9_15.npz 数据集

核心思想：
    - Slow Adaptive-HASA：每次迭代计算局部梯度和熵（计算密集）
    - Fast-Adaptive-HASA：使用 1D CNN 直接从反投影预测权重（大幅加速）
    - 数据增强：时移、幅度缩放、加噪声扩展训练数据

网络架构：
    - 1D U-Net：输入单通道信号，输出双通道权重图（TV 权重，小波权重）
"""

import numpy as np
import os
import time
import pywt
from numba import jit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    'npz_path': '../dataset_fdbf_energy_mu_8_9_15.npz',
    'cs_ratio': 9,
    
    # 数据分割
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    
    # 数据增强
    'n_augmentations': 5,           # 每条线的增强数量
    'shift_range': 50,              # 时移范围（样本数）
    'scale_range': (0.8, 1.2),      # 幅度缩放范围
    'noise_snr_range': (20, 40),    # 噪声 SNR 范围 (dB)
    
    # 重建参数
    'n_iterations': 50,
    'lambda_tv': 0.001,
    'lambda_wav': 0.001,
    'wavelet': 'db4',
    'wavelet_level': 4,
    'entropy_window_size': 32,
    'update_weights_every': 5,      # Slow 方法权重更新频率
    
    # U-Net 参数
    'unet_base_channels': 16,
    
    # 训练参数
    'batch_size': 16,
    'learning_rate': 1e-3,
    'n_epochs': 30,
    
    # 测试参数
    'num_test_lines': 30,
    
    # 输出
    'output_dir': 'outputs_fast_hasa_1d',
    'save_model': True,
    'model_path': 'fast_adaptive_hasa_1d.pth',
}


# ============================================================================
# 数据加载
# ============================================================================

def load_ultrasound_dataset(npz_path, cs_ratio=9):
    print(f"Loading dataset from {npz_path} ...")
    data = np.load(npz_path)
    
    if cs_ratio == 8:
        X = data['X8'].astype(np.float64)
        mu = data['mu8']
    elif cs_ratio == 9:
        X = data['X9'].astype(np.float64)
        mu = data['mu9']
    elif cs_ratio == 15:
        X = data['X15'].astype(np.float64)
        mu = data['mu15']
    else:
        raise ValueError(f"Unsupported cs_ratio: {cs_ratio}")
    
    Y = data['Y'].astype(np.float64)
    fs = float(data['fs'])
    fc = float(data['fc'])
    c = float(data['c']) if 'c' in data else 1540.0
    
    L, N = Y.shape
    print(f"  Loaded: L={L} lines, N={N} samples, cs_ratio={cs_ratio}x")
    
    return X, Y, mu, fs, fc, c


# ============================================================================
# 频域测量算子
# ============================================================================

class FrequencyMaskOperator:
    def __init__(self, N, mu):
        self.N = N
        self.mu = mu
        self.rfft_len = N // 2 + 1
        self.mask = np.zeros(self.rfft_len, dtype=np.float64)
        self.mask[mu] = 1.0
    
    def forward(self, x):
        X_freq = np.fft.rfft(x)
        return X_freq[self.mu]
    
    def adjoint(self, y_sub):
        X_freq = np.zeros(self.rfft_len, dtype=np.complex128)
        X_freq[self.mu] = y_sub
        return np.fft.irfft(X_freq, n=self.N)


# ============================================================================
# 1D 正则化
# ============================================================================

def tv_prox_1d(x, weight, n_iter=20):
    if weight <= 0:
        return x.copy()
    n = len(x)
    p = np.zeros(n - 1, dtype=np.float64)
    for _ in range(n_iter):
        div_p = np.zeros(n, dtype=np.float64)
        div_p[0] = p[0]
        div_p[1:-1] = p[1:] - p[:-1]
        div_p[-1] = -p[-1]
        grad_z = x - weight * div_p
        diff = np.diff(grad_z)
        p = (p + (1.0 / (4.0 * weight)) * diff) / (1.0 + np.abs(diff) / (4.0 * weight))
    div_p = np.zeros(n, dtype=np.float64)
    div_p[0] = p[0]
    div_p[1:-1] = p[1:] - p[:-1]
    div_p[-1] = -p[-1]
    return x - weight * div_p


def wavelet_soft_threshold_1d(x, threshold, wavelet='db4', level=4):
    if threshold <= 0:
        return x.copy()
    coeffs = pywt.wavedec(x, wavelet, level=level)
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
    reconstructed = pywt.waverec(coeffs_thresh, wavelet)
    if len(reconstructed) != len(x):
        reconstructed = reconstructed[:len(x)]
    return reconstructed


# ============================================================================
# 1D 局部统计
# ============================================================================

@jit(nopython=True)
def local_gradient_1d(x, window_size=32):
    n = len(x)
    result = np.zeros(n, dtype=np.float64)
    half = window_size // 2
    for i in range(n):
        i_min = max(0, i - half)
        i_max = min(n - 1, i + half)
        local_grad = 0.0
        count = 0
        for j in range(i_min, i_max):
            local_grad += abs(x[j + 1] - x[j])
            count += 1
        result[i] = local_grad / max(1, count)
    return result


@jit(nopython=True)
def local_entropy_1d(x, window_size=32, n_bins=64):
    n = len(x)
    result = np.zeros(n, dtype=np.float64)
    half = window_size // 2
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        return result
    x_norm = ((x - x_min) / (x_max - x_min) * (n_bins - 1)).astype(np.int32)
    x_norm = np.clip(x_norm, 0, n_bins - 1)
    for i in range(n):
        i_min = max(0, i - half)
        i_max = min(n, i + half + 1)
        hist = np.zeros(n_bins, dtype=np.int32)
        for j in range(i_min, i_max):
            hist[x_norm[j]] += 1
        total = i_max - i_min
        entropy = 0.0
        for k in range(n_bins):
            if hist[k] > 0:
                p = hist[k] / total
                entropy -= p * np.log2(p)
        result[i] = entropy
    return result


def compute_weight_maps_1d(x, window_size=32):
    """计算归一化的梯度和熵权重图"""
    grad = local_gradient_1d(x, window_size)
    ent = local_entropy_1d(x, window_size)
    
    g_min, g_max = grad.min(), grad.max()
    if g_max - g_min > 1e-12:
        grad_norm = (grad - g_min) / (g_max - g_min)
    else:
        grad_norm = np.zeros_like(grad)
    
    e_min, e_max = ent.min(), ent.max()
    if e_max - e_min > 1e-12:
        ent_norm = (ent - e_min) / (e_max - e_min)
    else:
        ent_norm = np.zeros_like(ent)
    
    return grad_norm, ent_norm


# ============================================================================
# 数据增强
# ============================================================================

def augment_signal_1d(x, aug_seed, shift_range=50, scale_range=(0.8, 1.2),
                      noise_snr_range=(20, 40)):
    """
    1D 信号数据增强：时移 + 幅度缩放 + 加噪
    """
    np.random.seed(aug_seed)
    
    # 时移
    shift_amount = np.random.randint(-shift_range, shift_range + 1)
    x_aug = np.roll(x, shift_amount)
    
    # 幅度缩放
    scale = np.random.uniform(scale_range[0], scale_range[1])
    x_aug = x_aug * scale
    
    # 加噪
    snr_db = np.random.uniform(noise_snr_range[0], noise_snr_range[1])
    signal_power = np.mean(x_aug ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(x_aug)) * np.sqrt(noise_power)
    x_aug = x_aug + noise
    
    return x_aug


# ============================================================================
# Slow Adaptive-HASA
# ============================================================================

def slow_adaptive_hasa_1d(y_sub, op, lambda_base=0.001, n_iter=50,
                          wavelet='db4', level=4, window_size=32,
                          update_every=5):
    """
    Slow Adaptive-HASA: 周期性重新计算权重
    """
    x = op.adjoint(y_sub)
    z = x.copy()
    t = 1.0
    L = 1.0
    
    lambda_tv = lambda_base
    lambda_wav = lambda_base
    
    for it in range(n_iter):
        # 周期性更新权重
        if it % update_every == 0:
            grad_norm, ent_norm = compute_weight_maps_1d(z, window_size)
            lambda_tv = lambda_base * (0.5 + np.mean(grad_norm))
            lambda_wav = lambda_base * (0.5 + np.mean(ent_norm))
        
        # 梯度下降
        residual = op.forward(z) - y_sub
        grad = op.adjoint(residual)
        x_new = z - (1.0 / L) * grad
        
        # 近端算子
        x_new = tv_prox_1d(x_new, weight=lambda_tv / L)
        x_new = wavelet_soft_threshold_1d(x_new, threshold=lambda_wav / L,
                                          wavelet=wavelet, level=level)
        
        # FISTA 动量
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new
    
    return x


# ============================================================================
# 1D U-Net
# ============================================================================

class UNet1D(nn.Module):
    """
    1D U-Net 用于预测自适应权重
    输入: (B, 1, N) 反投影信号
    输出: (B, 2, N) 权重图 (grad_weight, entropy_weight)
    """
    def __init__(self, in_channels=1, out_channels=2, base_channels=16):
        super(UNet1D, self).__init__()
        
        # 编码器
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool1d(2)
        
        # 瓶颈
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        
        # 解码器
        self.upconv3 = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        
        # 输出
        self.out = nn.Conv1d(base_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # 瓶颈
        b = self.bottleneck(p3)
        
        # 解码（带跳跃连接）
        d3 = self.upconv3(b)
        # 处理尺寸不匹配
        if d3.shape[2] != e3.shape[2]:
            diff = e3.shape[2] - d3.shape[2]
            d3 = nn.functional.pad(d3, (0, diff))
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        if d2.shape[2] != e2.shape[2]:
            diff = e2.shape[2] - d2.shape[2]
            d2 = nn.functional.pad(d2, (0, diff))
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        if d1.shape[2] != e1.shape[2]:
            diff = e1.shape[2] - d1.shape[2]
            d1 = nn.functional.pad(d1, (0, diff))
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out(d1)
        out = self.sigmoid(out)
        return out


# ============================================================================
# 数据集
# ============================================================================

class WeightMapDataset1D(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.FloatTensor(inputs).unsqueeze(1)  # (N, 1, L)
        self.targets = torch.FloatTensor(targets)  # (N, 2, L)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================================
# 训练数据生成
# ============================================================================

def generate_training_data(X, Y, op, config):
    """
    生成带增强的训练数据
    输入: 反投影信号
    输出: 目标权重图 (grad_norm, entropy_norm)
    """
    print("生成训练数据...")
    
    L, N = Y.shape
    n_aug = config['n_augmentations']
    window_size = config['entropy_window_size']
    
    train_inputs = []
    train_targets = []
    
    for i in range(L):
        y_gt = Y[i]
        x_init = X[i]  # 零填充初始化
        
        # 原始样本
        grad_norm, ent_norm = compute_weight_maps_1d(x_init, window_size)
        train_inputs.append(x_init)
        train_targets.append(np.stack([grad_norm, ent_norm], axis=0))
        
        # 增强样本
        for aug_idx in range(n_aug):
            aug_seed = i * 100 + aug_idx
            x_aug = augment_signal_1d(x_init, aug_seed,
                                      shift_range=config['shift_range'],
                                      scale_range=config['scale_range'],
                                      noise_snr_range=config['noise_snr_range'])
            
            grad_norm, ent_norm = compute_weight_maps_1d(x_aug, window_size)
            train_inputs.append(x_aug)
            train_targets.append(np.stack([grad_norm, ent_norm], axis=0))
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{L} lines")
    
    return np.array(train_inputs), np.array(train_targets)


# ============================================================================
# Fast-Adaptive-HASA
# ============================================================================

def fast_adaptive_hasa_1d(y_sub, op, model, device, lambda_base=0.001,
                          n_iter=50, wavelet='db4', level=4):
    """
    Fast-Adaptive-HASA: 使用 CNN 预测权重（仅预测一次）
    """
    # 初始反投影
    x_init = op.adjoint(y_sub)
    
    # 使用 CNN 预测权重（仅一次）
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_init).unsqueeze(0).unsqueeze(0).to(device)
        weight_maps = model(x_tensor).squeeze(0).cpu().numpy()
    
    grad_weight = weight_maps[0]
    ent_weight = weight_maps[1]
    
    # 计算固定权重
    lambda_tv = lambda_base * (0.5 + np.mean(grad_weight))
    lambda_wav = lambda_base * (0.5 + np.mean(ent_weight))
    
    # 迭代重建（权重固定）
    x = x_init.copy()
    z = x.copy()
    t = 1.0
    L = 1.0
    
    for _ in range(n_iter):
        residual = op.forward(z) - y_sub
        grad = op.adjoint(residual)
        x_new = z - (1.0 / L) * grad
        
        x_new = tv_prox_1d(x_new, weight=lambda_tv / L)
        x_new = wavelet_soft_threshold_1d(x_new, threshold=lambda_wav / L,
                                          wavelet=wavelet, level=level)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new
    
    return x


# ============================================================================
# 指标
# ============================================================================

def calc_snr(y_true, y_pred, eps=1e-12):
    signal_power = np.sum(y_true ** 2)
    noise_power = np.sum((y_true - y_pred) ** 2)
    return 10.0 * np.log10((signal_power + eps) / (noise_power + eps))


def calc_psnr(y_true, y_pred, eps=1e-12):
    mse = np.mean((y_true - y_pred) ** 2)
    max_val = np.max(np.abs(y_true))
    return 20.0 * np.log10((max_val + eps) / (np.sqrt(mse) + eps))


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("Fast-Adaptive-HASA 1D 超声压缩感知重建")
    print("=" * 70)
    
    config = CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 1. 加载数据
    print("\n[1] 加载数据集...")
    X, Y, mu, fs, fc, c = load_ultrasound_dataset(config['npz_path'], config['cs_ratio'])
    L, N = Y.shape
    
    # 2. 创建测量算子
    print("\n[2] 创建频域测量算子...")
    op = FrequencyMaskOperator(N, mu)
    
    # 3. 预编译 JIT
    print("\n[3] 预编译 JIT 函数...")
    _ = local_gradient_1d(Y[0], window_size=32)
    _ = local_entropy_1d(Y[0], window_size=32)
    print("  JIT 编译完成")
    
    # 4. 划分训练/测试集
    n_train = int(L * config['train_ratio'])
    train_indices = np.arange(n_train)
    test_indices = np.arange(n_train, L)
    
    print(f"\n[4] 数据划分: 训练 {n_train}, 测试 {L - n_train}")
    
    # 5. 生成训练数据
    print("\n[5] 生成训练数据（带增强）...")
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    train_inputs, train_targets = generate_training_data(X_train, Y_train, op, config)
    print(f"  训练样本数: {len(train_inputs)}")
    
    # 6. 创建数据加载器
    train_dataset = WeightMapDataset1D(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 7. 创建并训练 U-Net
    print(f"\n[6] 创建并训练 1D U-Net ({config['n_epochs']} epochs)...")
    model = UNet1D(in_channels=1, out_channels=2, base_channels=config['unet_base_channels'])
    model = model.to(device)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    loss_history = []
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
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config['n_epochs']}, Loss: {avg_loss:.6f}")
    
    # 保存模型
    os.makedirs(config['output_dir'], exist_ok=True)
    if config['save_model']:
        model_path = os.path.join(config['output_dir'], config['model_path'])
        torch.save(model.state_dict(), model_path)
        print(f"  模型已保存: {model_path}")
    
    # 8. 测试
    print(f"\n[7] 测试比较...")
    num_test = min(config['num_test_lines'], len(test_indices))
    
    snr_init = []
    snr_slow = []
    snr_fast = []
    time_slow = []
    time_fast = []
    
    for i in range(num_test):
        idx = test_indices[i]
        y_gt = Y[idx]
        x_init = X[idx]
        y_sub = op.forward(y_gt)
        
        # Initial
        snr_init.append(calc_snr(y_gt, x_init))
        
        # Slow Adaptive-HASA
        t0 = time.time()
        x_slow = slow_adaptive_hasa_1d(y_sub, op,
                                       lambda_base=config['lambda_tv'],
                                       n_iter=config['n_iterations'],
                                       wavelet=config['wavelet'],
                                       level=config['wavelet_level'],
                                       window_size=config['entropy_window_size'],
                                       update_every=config['update_weights_every'])
        t_slow = time.time() - t0
        snr_slow.append(calc_snr(y_gt, x_slow))
        time_slow.append(t_slow)
        
        # Fast-Adaptive-HASA
        t0 = time.time()
        x_fast = fast_adaptive_hasa_1d(y_sub, op, model, device,
                                       lambda_base=config['lambda_tv'],
                                       n_iter=config['n_iterations'],
                                       wavelet=config['wavelet'],
                                       level=config['wavelet_level'])
        t_fast = time.time() - t0
        snr_fast.append(calc_snr(y_gt, x_fast))
        time_fast.append(t_fast)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{num_test}] Slow: {snr_slow[-1]:.2f} dB ({t_slow:.3f}s) | "
                  f"Fast: {snr_fast[-1]:.2f} dB ({t_fast:.3f}s)")
    
    # 转数组
    snr_init = np.array(snr_init)
    snr_slow = np.array(snr_slow)
    snr_fast = np.array(snr_fast)
    time_slow = np.array(time_slow)
    time_fast = np.array(time_fast)
    
    # 9. 汇总
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    print(f"\nInitial (零填充):")
    print(f"  SNR: {snr_init.mean():.2f} ± {snr_init.std():.2f} dB")
    
    print(f"\nSlow Adaptive-HASA:")
    print(f"  SNR:  {snr_slow.mean():.2f} ± {snr_slow.std():.2f} dB")
    print(f"  Time: {time_slow.mean()*1000:.1f} ± {time_slow.std()*1000:.1f} ms/line")
    
    print(f"\nFast-Adaptive-HASA:")
    print(f"  SNR:  {snr_fast.mean():.2f} ± {snr_fast.std():.2f} dB")
    print(f"  Time: {time_fast.mean()*1000:.1f} ± {time_fast.std()*1000:.1f} ms/line")
    
    avg_speedup = time_slow.mean() / time_fast.mean()
    snr_retention = snr_fast.mean() / snr_slow.mean() * 100
    
    print(f"\n加速比: {avg_speedup:.2f}x")
    print(f"SNR 保持率: {snr_retention:.1f}%")
    
    # 10. 保存结果
    result_file = os.path.join(config['output_dir'], f"results_fast_hasa_ratio{config['cs_ratio']}.txt")
    with open(result_file, 'w') as f:
        f.write("Fast-Adaptive-HASA 1D 超声重建结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"数据集: {config['npz_path']}\n")
        f.write(f"压缩比: {config['cs_ratio']}x\n")
        f.write(f"测试样本: {num_test}\n\n")
        
        f.write(f"Initial:\n")
        f.write(f"  SNR: {snr_init.mean():.2f} ± {snr_init.std():.2f} dB\n\n")
        
        f.write(f"Slow Adaptive-HASA:\n")
        f.write(f"  SNR:  {snr_slow.mean():.2f} ± {snr_slow.std():.2f} dB\n")
        f.write(f"  Time: {time_slow.mean()*1000:.1f} ms/line\n\n")
        
        f.write(f"Fast-Adaptive-HASA:\n")
        f.write(f"  SNR:  {snr_fast.mean():.2f} ± {snr_fast.std():.2f} dB\n")
        f.write(f"  Time: {time_fast.mean()*1000:.1f} ms/line\n\n")
        
        f.write(f"加速比: {avg_speedup:.2f}x\n")
        f.write(f"SNR 保持率: {snr_retention:.1f}%\n")
    
    print(f"\n结果已保存: {result_file}")
    
    # 11. 可视化
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # SNR 比较
        methods = ['Initial', 'Slow', 'Fast']
        snrs = [snr_init.mean(), snr_slow.mean(), snr_fast.mean()]
        stds = [snr_init.std(), snr_slow.std(), snr_fast.std()]
        colors = ['gray', 'coral', 'limegreen']
        
        bars = axes[0].bar(methods, snrs, yerr=stds, capsize=5, color=colors, alpha=0.8)
        axes[0].set_ylabel('SNR (dB)')
        axes[0].set_title('SNR Comparison')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 时间比较
        times = [0, time_slow.mean() * 1000, time_fast.mean() * 1000]
        bars = axes[1].bar(['Initial', 'Slow', 'Fast'], times, color=colors, alpha=0.8)
        axes[1].set_ylabel('Time (ms/line)')
        axes[1].set_title(f'Speed Comparison (Speedup: {avg_speedup:.1f}x)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig_path = os.path.join(config['output_dir'], f'fast_hasa_comparison_ratio{config["cs_ratio"]}.png')
        plt.savefig(fig_path, dpi=150)
        print(f"可视化图已保存: {fig_path}")
        plt.show()
        
    except Exception as e:
        print(f"[warn] 可视化失败: {e}")
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    
    return {
        'snr_init': snr_init,
        'snr_slow': snr_slow,
        'snr_fast': snr_fast,
        'time_slow': time_slow,
        'time_fast': time_fast,
        'speedup': avg_speedup,
        'model': model,
    }


if __name__ == '__main__':
    results = main()
