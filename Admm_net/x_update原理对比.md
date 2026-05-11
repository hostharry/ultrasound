# HASA-ADMM-Net 架构详解

## 1. 优化问题

超声 RF 信号压缩感知重建的数学表述：

$$
\min_x \; \frac{1}{2}\|Ax - y\|^2 + \lambda_1\|Wx\|_1 + \lambda_2\|D \cdot \text{Env}(x)\|_1
$$

| 符号 | 含义 | 实现 |
|---|---|---|
| $x \in \mathbb{R}^N$ | 待恢复的 RF 信号 | 网络输出 |
| $y \in \mathbb{C}^K$ | 频域欠采样观测 ($K \ll N$) | 输入数据 |
| $A$ | 测量算子：FFT + 掩膜采样 | `MaskedRFFT1D` |
| $W$ | 稀疏变换（小波） | `HaarDWT1D` / `LearnableAnalysis1D` |
| $D$ | 有限差分算子 | `FiniteDiff1D`：$[1, -1]$ 卷积 |
| $\text{Env}(\cdot)$ | 希尔伯特包络 $\|x + jH(x)\|$ | `hilbert_envelope`：FFT → 解析信号 → 取模 |

三项含义：**数据保真** + **小波稀疏** + **包络域全变分平滑**。

---

## 2. ADMM 变量分裂

直接优化上述目标困难（两个非光滑正则项耦合）。引入辅助变量 $w, p$ 解耦：

$$
\min_{x,w,p} \; \frac{1}{2}\|Ax - y\|^2 + \lambda_1\|w\|_1 + \lambda_2\|p\|_1 \quad \text{s.t.} \; w = Wx, \; p = D \cdot \text{Env}(x)
$$

增广拉格朗日函数（缩放形式）：

$$
\mathcal{L} = \frac{1}{2}\|Ax - y\|^2 + \lambda_1\|w\|_1 + \lambda_2\|p\|_1 + \frac{\rho_1}{2}\|Wx - w + u_1\|^2 + \frac{\rho_2}{2}\|D \cdot \text{Env}(x) - p + u_2\|^2
$$

其中 $u_1, u_2$ 是缩放对偶变量（拉格朗日乘子 / $\rho$）。

ADMM 的核心思想：**将一个困难的联合优化问题拆成多个简单子问题，交替求解**。

---

## 3. 深度展开：将 ADMM 迭代映射为网络层

将 ADMM 的 $K$ 次迭代展开为 $K$ 层神经网络，每层结构相同但**参数独立可学习**。

### 网络总体流程

```
输入: y (频域观测), op (测量算子)
  │
  ├─ 初始化: x₀ = A^T y (零填充逆 FFT)
  ├─ 幅度归一化: x₀ ← x₀/scale, y ← y/scale
  ├─ 辅助变量初始化: w₀ = W(x₀), p₀ = D·Env(x₀), u₁=u₂=0
  │
  ├─ [ADMM Block 1] → x₁, w₁, p₁, u₁, u₂
  ├─ [ADMM Block 2] → x₂, w₂, p₂, u₁, u₂
  ├─ ...
  ├─ [ADMM Block K] → x_K, w_K, p_K, u₁, u₂
  │
  ├─ 反归一化: x_out = x_K × scale
  │
输出: x_out (重建 RF 信号)
```

### 幅度归一化

RF 信号幅值差异大（不同深度/角度），直接处理会导致 ADMM 步长参数无法适配所有样本。归一化确保所有信号在 $[-1, 1]$ 范围内，ADMM 参数（$\rho, \eta$）与数据尺度解耦。频域观测 $y$ 必须同步缩放。

---

## 4. 单层 ADMM Block 详解

每层包含五个步骤，对应一次完整的 ADMM 迭代：

```
输入: x, w, p, u₁, u₂, y, op
  │
  ├── ① HASA: λ_wav, λ_tv = HASA(x)        [自适应阈值生成]
  ├── ② w-update: 小波域软阈值              [稀疏近端]
  ├── ③ p-update: 包络 TV 域软阈值          [TV 近端]
  ├── ④ x-update: 频域闭式解 + TV 修正      [重建核心]
  ├── ⑤ u-update: 对偶变量更新              [约束执行]
  │
输出: x_new, w_new, p_new, u₁_new, u₂_new
```

### ① HASA 自适应阈值生成

```
x ──→ [Conv1d 3×16] → ReLU → [Conv1d 5×16] → ReLU ──→ feat
                                                         ├── [Conv1d 1×1] → Softplus → λ_wav(x)
                                                         └── [Conv1d 1×1] → Softplus → λ_tv(x)
```

**作用**：根据当前重建 $x$ 的**局部特征**，逐点生成自适应正则化强度。

- 信号边缘/脉冲处 → $\lambda$ 小 → 保留细节
- 噪声/平坦区域 → $\lambda$ 大 → 压制噪声

**与传统 ADMM 的区别**：传统 ADMM 的 $\lambda$ 是标量常数，而 HASA 输出的是与信号等长的向量，实现**空间自适应正则化**。

**初始化**：输出层偏置初始化为 $-4$，使初始 $\lambda \approx \text{softplus}(-4) \approx 0.018$，避免过大阈值导致发散。

### ② w-update：小波域稀疏近端映射

$$
w^{(k+1)} = \text{soft}\Big(Wx^{(k)} + u_1^{(k)}, \; \frac{\lambda_\text{wav}(x)}{\rho_1}\Big)
$$

```python
Wx = W.forward(x)                          # 小波变换
thr = lambda_wav / rho1                     # 自适应阈值
w_new = sign(Wx + u1) * max(|Wx + u1| - thr, 0)  # 软阈值
```

**数学背书**：这是 $\min_w \lambda_1\|w\|_1 + \frac{\rho_1}{2}\|w - (Wx + u_1)\|^2$ 的**精确闭式解**。ADMM 的变量分裂保证了无论 $W$ 是否正交，对 $w$ 的软阈值操作都是准确的 L1 近端算子。

### ③ p-update：包络域 TV 近端映射

$$
p^{(k+1)} = \text{soft}\Big(D \cdot \text{Env}(x^{(k)}) + u_2^{(k)}, \; \frac{\lambda_\text{tv}(x)}{\rho_2}\Big)
$$

```python
env = hilbert_envelope(x)                   # RF → 包络 (FFT 实现)
Denv = D(env)                               # 有限差分 (conv1d [1,-1])
thr = lambda_tv / rho2
p_new = soft_threshold(Denv + u2, thr)
```

**为什么不直接对 RF 信号做 TV**：RF 信号是高频振荡载波，TV 假设"分段恒定"，直接施加会抹平载波。提取包络后，信号变为低频平滑曲线，TV 假设成立。

**希尔伯特包络的可微实现**：

$$
\text{Env}(x) = \sqrt{\text{Re}^2 + \text{Im}^2 + \epsilon}, \quad \text{analytic}(x) = \mathcal{F}^{-1}[H \cdot \mathcal{F}(x)]
$$

其中 $H$ 是单边频谱滤波器。整个过程完全可微（FFT + 逐元素运算）。

### ④ x-update：频域闭式解 + TV 梯度修正

这是本网络与早期版本（梯度步）的**核心区别**。

#### (a) 线性部分：频域精确求解

x-子问题中，数据保真 + 小波约束对 $x$ 是二次的：

$$
\min_x \; \frac{1}{2}\|Ax - y\|^2 + \frac{\rho_1}{2}\|Wx - (w - u_1)\|^2
$$

对正交 $W$（$W^TW = I$），令梯度为零：

$$
(A^TA + \rho_1 I)\,x = A^Ty + \rho_1 W^T(w - u_1)
$$

由于 $A$ 是频域掩膜采样，$A^TA$ 在频域是对角阵 $\text{diag}(M)$：

$$
\hat{X}[\omega] = \frac{Y_\text{full}[\omega] + \rho_1 \cdot \widehat{W^T(w - u_1)}[\omega]}{M[\omega] + \rho_1}
$$

```python
rhs_wav  = W.inverse(w_new - u1)           # W^T(w - u1): 空间域先验
x_linear = IFFT[ (Y_full + ρ1·FFT(rhs)) / (mask + ρ1) ]  # 频域除法
```

| 频点 | 分母 | 行为 |
|---|---|---|
| 被采样 ($M=1$) | $1 + \rho_1$ | 观测与先验的加权平均 |
| 未采样 ($M=0$) | $\rho_1$ | 完全由先验填充 |

这是 **Wiener 滤波器** 结构：有数据信数据，没数据用先验。

#### (b) 非线性部分：TV 梯度修正

包络 TV 项 $\frac{\rho_2}{2}\|D \cdot \text{Env}(x) - p + u_2\|^2$ 含有非线性希尔伯特包络，无法纳入闭式解。在 $x_\text{linear}$ 处做一步梯度修正：

$$
x^{(k+1)} = x_\text{linear} - \eta \cdot \rho_2 \cdot \nabla_x \Big[\frac{1}{2}\|D \cdot \text{Env}(x_\text{linear}) - p + u_2\|^2\Big]
$$

```python
grad_tv = autograd.grad(loss_tv, x_linear, create_graph=True)
x_new = x_linear - eta * rho2 * grad_tv
```

`create_graph=True` 确保 `grad_tv` 对网络参数可微，使 HASA 的 $\lambda_\text{tv}$（通过 $p_\text{new}$）能获得梯度。

#### 与旧方案（梯度步）的对比

| | 梯度下降 x-update | 频域闭式解 x-update |
|---|---|---|
| 方法 | $x - \eta \nabla(\text{全部三项})$ | $\mathcal{F}^{-1}[\ldots]$ 精确解 + TV 修正 |
| 对未观测频率 | **梯度恒为零**，无法恢复 | **先验直接填充** |
| 效果 (零训练) | 3.32 dB | **7.41 dB** |
| 效果 (100步后) | 3.39 dB | **11.68 dB** |

### ⑤ 对偶变量更新

$$
u_1^{(k+1)} = u_1^{(k)} + \gamma^{(k)} (Wx^{(k+1)} - w^{(k+1)})
$$
$$
u_2^{(k+1)} = u_2^{(k)} + \gamma^{(k)} (D \cdot \text{Env}(x^{(k+1)}) - p^{(k+1)})
$$

$\gamma$ 是可学习步长（参照 ADMM-CSNet），控制约束违反的累积惩罚速度。标准 ADMM 中 $\gamma = 1$。

---

## 5. 可学习参数总览

| 参数 | 每层独立 | 初始值 | 通过 Softplus | 作用 |
|---|---|---|---|---|
| $\rho_1$ | ✓ | softplus(-1)≈0.31 | 保证正 | 频域闭式解中数据/先验权衡 |
| $\rho_2$ | ✓ | softplus(-1)≈0.31 | 保证正 | TV 修正力度 |
| $\eta$ | ✓ | softplus(-2)≈0.13 | 保证正 | TV 梯度步长 |
| $\gamma$ | ✓ | softplus(0)≈0.69 | 保证正 | 对偶更新步长 |
| HASA 网络 | ✓ | Kaiming + bias=-4 | — | 空间自适应 $\lambda_\text{wav}, \lambda_\text{tv}$ |
| W (Mode B) | 共享/独立 | Kaiming | — | 可学习稀疏变换基 |

K 层网络共有 $K \times (4 + |\text{HASA}|)$ 个独立可学习参数组。

---

## 6. 与 ADMM-CSNet 的对比

| 设计选择 | ADMM-CSNet (标准) | HASA-ADMM-Net (本网络) |
|---|---|---|
| **应用领域** | 2D MRI 图像 | 1D 超声 RF 信号 |
| **正则化** | 单一 CNN 去噪器 | **双正则化**: 小波 L1 + 包络域 TV |
| **z-update** | Conv + PWL + Conv (学习去噪器) | **精确软阈值** (数学闭式解) |
| **x-update** | 频域闭式解 | **频域闭式解 + TV 梯度修正** |
| **阈值** | 全局标量 / PWL | **HASA 逐点自适应** |
| **正交性要求** | 隐式（CNN 不保证） | **ADMM 分裂消除** (w=Wx 约束) |
| **物理先验** | 无 | **希尔伯特包络 + TV** (超声特有) |

### 本网络的三个创新点

1. **双正则化 ADMM 展开**：同时处理小波稀疏和包络域 TV，通过变量分裂解耦为独立闭式子问题
2. **HASA 自适应阈值**：替代全局常数 $\lambda$，根据信号局部特征逐点生成正则化强度
3. **包络域 TV 的可微实现**：通过 FFT 实现希尔伯特变换 + `torch.autograd.grad` 计算包络 TV 对 RF 信号的梯度

---

## 7. 数据流图（单层）

```
                                ┌──────────────────┐
                    x ─────────►│  HASA 网络        │──► λ_wav, λ_tv
                    │           └──────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
     ┌────▼────┐         ┌───▼────┐
     │ W(x)    │         │Env(x)  │  ← hilbert_envelope
     │小波变换  │         │D·Env(x)│  ← 有限差分
     └────┬────┘         └───┬────┘
          │                   │
    ┌─────▼──────┐     ┌─────▼──────┐
    │ w-update   │     │ p-update   │
    │ soft_thr   │     │ soft_thr   │
    │ (λ_wav/ρ1) │     │ (λ_tv/ρ2)  │
    └─────┬──────┘     └─────┬──────┘
          │ w_new             │ p_new
          │                   │
    ┌─────▼───────────────────▼──────┐
    │         x-update               │
    │  ┌───────────────────────┐     │
    │  │ 频域闭式解 (线性部分)   │     │
    │  │ X = (Y + ρ1·F(W^T(w-u1)))  │
    │  │     / (M + ρ1)        │     │
    │  └───────────┬───────────┘     │
    │              │ x_linear        │
    │  ┌───────────▼───────────┐     │
    │  │ TV 梯度修正 (非线性)   │     │
    │  │ x_new = x_lin         │     │
    │  │   - η·ρ2·∇TV_env     │     │
    │  └───────────┬───────────┘     │
    └──────────────┼─────────────────┘
                   │ x_new
    ┌──────────────▼─────────────────┐
    │         u-update               │
    │  u1 += γ·(W·x_new - w_new)    │
    │  u2 += γ·(D·Env(x_new)-p_new) │
    └──────────────┬─────────────────┘
                   │
            x_new, w_new, p_new, u1, u2
                   ↓ (传入下一层)
```

---

## 附录 A：x-update 从梯度步到闭式解的演进

### 旧方案：单步梯度下降

对增广拉格朗日对 $x$ 求梯度，走一步：

$$
x^{(k+1)} = x^{(k)} - \eta \Big( A^T(Ax^{(k)} - y) + \rho_1 W^T(Wx^{(k)} - w + u_1) + \rho_2 \nabla_x \text{TV}_\text{env} \Big)
$$

**致命缺陷**：

1. **步长-精度矛盾**：$\eta$ 必须小以保证稳定，但太小则每层几乎不动
2. **未利用频域结构**：$A^TA$ 在频域是对角阵，梯度下降完全忽视这一点
3. **未观测频率梯度为零**：$A^T(Ax - y)$ 对未采样频点的贡献恒为零，意味着 87.6% 的频率信息永远无法通过数据保真项恢复

### 新方案：频域闭式解

$$
\hat{X}[\omega] = \frac{Y_\text{full}[\omega] + \rho_1 \cdot \hat{R}[\omega]}{M[\omega] + \rho_1}
$$

- 被采样频点：数据与先验加权平均
- 未采样频点：完全由小波先验 $W^T(w - u_1)$ 填充

**这是 ADMM-CSNet (NeurIPS 2016, TPAMI 2019) 的标准做法**，也是 ADMM 展开网络的核心设计。

### 效果对比（合成稀疏信号 N=256, K=64, 稀疏度 20/256）

| | 梯度步 | 频域闭式解 |
|---|---|---|
| **零训练 SNR** | 3.32 dB | **7.41 dB** |
| **100步训练后** | 3.39 dB | **11.68 dB** |
| **提升** | +0.27 dB | **+8.56 dB** |

频域闭式解在未训练时就比梯度步训练后好 4 dB。

---

## 附录 B：梯度流设计

深度展开网络的端到端训练要求梯度能从最终 loss 流回每一层的所有可学习参数。关键设计：

| 路径 | 实现方式 | 说明 |
|---|---|---|
| loss → $\eta, \rho_1, \rho_2$ | 直接参与 x_new 计算 | 标量参数，梯度自然流通 |
| loss → HASA $\lambda_\text{wav}$ | $\lambda \to \text{thr} \to w_\text{new} \to \text{rhs\_wav} \to x_\text{linear}$ | 不 detach $w_\text{new}$ |
| loss → HASA $\lambda_\text{tv}$ | $\lambda \to \text{thr} \to p_\text{new} \to \text{residual\_tv} \to \text{grad\_tv}$ | `create_graph=True` + 不 detach $p_\text{new}$ |
| loss → $\gamma$ | $\gamma \to u_\text{new} \to$ 下一层 $w/p$ update | 最后一层 $\gamma$ 无梯度（正常） |

`torch.enable_grad()` 包裹 TV 梯度计算，确保在验证的 `no_grad` 上下文中也能正确求导。

---

# HASA-FISTA vs HASA-ADMM：原理与架构对比

## 1. 核心思想差异

两个网络解决的是**同一个优化问题**：

$$
\min_x \; \frac{1}{2}\|Ax - y\|^2 + \lambda_1\|Wx\|_1 + \lambda_2\|D \cdot \text{Env}(x)\|_1
$$

但展开的**经典优化算法不同**，导致网络结构完全不同：

| | HASA-ADMM-Net | HASA-FISTA-Net |
|---|---|---|
| **展开算法** | ADMM（交替方向乘子法） | FISTA（快速迭代收缩阈值算法） |
| **处理多正则化的方式** | 变量分裂：引入 $w, p$ 解耦 | 并行双分支 Prox + 加权融合 |
| **辅助变量** | $w, p, u_1, u_2$（5 状态） | $x_\text{prev}, v$（2 状态，含动量） |
| **x-update** | 频域闭式解 + TV 梯度修正 | 梯度下降步（数据一致性） |
| **近端算子** | 精确软阈值（分裂后闭式） | 学习型 Encoder-Decoder + 软阈值 |
| **加速策略** | 无（依赖闭式解精度） | Nesterov 动量（FISTA 特征） |

---

## 2. 算法推导对比

### 2.1 HASA-ADMM：变量分裂 → 子问题交替求解

ADMM 引入辅助变量将两个非光滑项解耦：

$$
\min_{x,w,p} \; \frac{1}{2}\|Ax - y\|^2 + \lambda_1\|w\|_1 + \lambda_2\|p\|_1 \quad \text{s.t.} \; w = Wx, \; p = D\text{Env}(x)
$$

每层迭代包含**五个串行子步骤**：

$$
\begin{aligned}
w^{(k+1)} &= \text{prox}_{\lambda_1/\rho_1}(Wx^{(k)} + u_1^{(k)}) & \text{(w-update: 闭式软阈值)} \\
p^{(k+1)} &= \text{prox}_{\lambda_2/\rho_2}(D\text{Env}(x^{(k)}) + u_2^{(k)}) & \text{(p-update: 闭式软阈值)} \\
x^{(k+1)} &= (A^TA + \rho_1 I)^{-1}[A^Ty + \rho_1 W^T(w - u_1)] - \eta \rho_2 \nabla_x\text{TV}_\text{env} & \text{(x-update: 频域闭式+TV修正)} \\
u_1^{(k+1)} &= u_1^{(k)} + \gamma(Wx^{(k+1)} - w^{(k+1)}) & \text{(对偶更新)} \\
u_2^{(k+1)} &= u_2^{(k)} + \gamma(D\text{Env}(x^{(k+1)}) - p^{(k+1)}) & \text{(对偶更新)}
\end{aligned}
$$

**关键优势**：x-update 中利用频域对角结构得到**精确闭式解**，未观测频率由先验填充。

### 2.2 HASA-FISTA：近端梯度 + 动量加速

FISTA 不做变量分裂，直接对原问题做近端梯度迭代：

$$
\begin{aligned}
z^{(k)} &= v^{(k)} - \rho^{(k)} A^T(Av^{(k)} - y) & \text{(数据一致性梯度步)} \\
x^{(k+1)} &= \mathcal{D}_\theta(z^{(k)}) & \text{(学习型近端算子)} \\
v^{(k+1)} &= x^{(k+1)} + \beta^{(k)}(x^{(k+1)} - x^{(k)}) & \text{(Nesterov 动量)}
\end{aligned}
$$

其中 $\mathcal{D}_\theta$ 是**双分支学习型去噪器**：

$$
\mathcal{D}_\theta(z) = \alpha(z) \cdot \underbrace{D_\text{tv}(\text{shrink}(E_\text{tv}(z), \tau_\text{tv}))}_{\text{TV 分支}} + (1 - \alpha(z)) \cdot \underbrace{D_\text{wav}(\text{shrink}(E_\text{wav}(z), \tau_\text{wav}))}_{\text{WAV 分支}}
$$

---

## 3. 单层网络结构对比

### 3.1 HASA-ADMM Block

```
输入: x, w, p, u₁, u₂
  │
  ├── ① HASA(x) → λ_wav, λ_tv            [2 输出头]
  ├── ② w = soft_thr(Wx + u₁, λ_wav/ρ₁)   [精确闭式解]
  ├── ③ p = soft_thr(DEnv(x)+u₂, λ_tv/ρ₂) [精确闭式解]
  ├── ④ x = IFFT[...] - η·ρ₂·∇TV_env     [频域闭式+梯度修正]
  ├── ⑤ u₁ += γ(Wx - w), u₂ += γ(DEnv-p)  [对偶更新]
  │
输出: x, w, p, u₁, u₂  (5 个状态变量传递)
```

### 3.2 HASA-FISTA Block

```
输入: x_prev, v
  │
  ├── ① z = v - ρ·A^T(Av - y)             [数据一致性]
  ├── ② HASA(z) → λ_tv, λ_wav, α          [3 输出头]
  ├── ③ 双分支 Prox:
  │     ├─ TV branch:  E_tv(z) → shrink → D_tv → x_tv
  │     ├─ WAV branch: E_wav(z) → shrink → D_wav → x_wav
  │     └─ x = α·x_tv + (1-α)·x_wav       [HASA 融合]
  ├── ④ v = x + β·(x - x_prev)            [FISTA 动量]
  │
输出: x, v  (2 个状态变量传递)
```

---

## 4. 逐模块技术细节对比

### 4.1 HASA 权重网络

| | HASA-ADMM | HASA-FISTA |
|---|---|---|
| **输入** | 当前重建 $x$ | 数据一致性后的 $z$ |
| **输出头** | 2 个：$\lambda_\text{wav}, \lambda_\text{tv}$ | 3 个：$\lambda_\text{tv}, \lambda_\text{wav}, \alpha$ |
| **额外输出** | — | $\alpha \in (0,1)$：双分支融合权重 |
| **$\alpha$ 激活** | — | Sigmoid（逐像素软切换） |
| **结构** | Conv→ReLU→Conv→ReLU→两个 1×1 头 | 相同骨架 + 额外 $\alpha$ 头 |

HASA-FISTA 多了 $\alpha$ 头，因为它需要决定每个空间位置更信任 TV 分支还是 WAV 分支。HASA-ADMM 不需要，因为 ADMM 的变量分裂天然将两个正则化项解耦为独立子问题。

### 4.2 近端算子 / z-update

| | HASA-ADMM | HASA-FISTA |
|---|---|---|
| **小波分支** | $\text{soft}(Wx + u_1, \lambda/\rho_1)$ | $D_\text{wav}(\text{soft}(E_\text{wav}(z), \tau))$ |
| **TV 分支** | $\text{soft}(D\text{Env}(x) + u_2, \lambda/\rho_2)$ | $D_\text{tv}(\text{soft}(E_\text{tv}(z), \tau))$ |
| **变换域** | 固定 Haar 小波 / 固定有限差分 | **可学习** Conv Encoder-Decoder |
| **软阈值作用域** | 变换系数空间 | 学习特征空间 |
| **正交性保证** | ADMM 分裂消除正交性需求 | 无保证（PnP 框架下合理） |
| **理论保证** | L1 近端映射的精确闭式解 | 近似去噪器（Plug-and-Play） |

**关键区别**：
- ADMM 中的软阈值直接作用于**物理意义明确的变换系数**（小波系数、TV 梯度），是精确的 L1 prox
- FISTA 中的软阈值作用于**学习到的特征空间**，Encoder-Decoder 将信号映射到一个"更容易阈值处理"的表示空间

### 4.3 x-update / 数据一致性

| | HASA-ADMM | HASA-FISTA |
|---|---|---|
| **方法** | 频域闭式解 $\hat{X} = \frac{Y + \rho_1 \hat{R}}{M + \rho_1}$ | 梯度步 $z = v - \rho A^T(Av - y)$ |
| **对未采样频率** | 由先验 $W^T(w - u_1)$ 直接填充 | 依赖迭代积累（需要更多层） |
| **计算复杂度** | 2×FFT + 1×IFFT + 逐元素除法 | 1×FFT + 1×IFFT（A 和 A^T 各一次） |
| **非线性项处理** | 额外一步 TV 梯度修正 | 不需要（TV 在 prox 中处理） |

**这是两者最根本的性能差异来源**：ADMM 的闭式解在第一层就能恢复未采样频率，而 FISTA 的梯度步需要多层迭代才能间接恢复。

### 4.4 动量机制

| | HASA-ADMM | HASA-FISTA |
|---|---|---|
| **有无动量** | 无 | 有（Nesterov 加速） |
| **实现** | — | $v = x + \beta(x - x_\text{prev})$ |
| **$\beta$ 范围** | — | $\tanh(\beta_\text{raw}) \in (-1, 1)$ |
| **理论加速** | — | 凸问题下 $O(1/k^2)$ vs $O(1/k)$ |

FISTA 的动量是其名称中"Fast"的来源。每层会记住上一层的输出 $x_\text{prev}$，利用"惯性"加速收敛。ADMM 不使用动量，因为频域闭式解本身精度更高。

---

## 5. 参数量对比

以 K=12 层、HASA hidden=16、inner_ks=5 为例：

### HASA-ADMM (1D, Mode A)

| 组件 | 每层参数 | 共享 | 总计 |
|---|---|---|---|
| HASA (feat_net + 2 heads) | ≈ 1,266 | × | 15,192 |
| $\rho_1, \rho_2, \eta, \gamma$ | 4 | × | 48 |
| HaarDWT1D | 0 (固定) | — | 0 |
| FiniteDiff1D | 0 (固定) | — | 0 |
| **总计** | | | **≈ 15,240** |

### HASA-ADMM (2D, Mode A)

| 组件 | 每层参数 | 共享 | 总计 |
|---|---|---|---|
| HASA 2D (feat_net + 2 heads) | ≈ 6,578 | × | 78,936 |
| $\rho_1, \rho_2, \eta, \gamma$ | 4 | × | 48 |
| HaarDWT2D | 0 (固定) | — | 0 |
| FiniteDiff2D | 0 (固定) | — | 0 |
| **总计** | | | **≈ 78,984** |

### HASA-FISTA (2D, feat_ch=64, prox_k=3)

| 组件 | 每层参数 | 共享 | 总计 |
|---|---|---|---|
| HASA-FISTA (feat_net + 3 heads) | ≈ 6,611 | × | 79,332 |
| ISTAProx2D_Dual (8 层 Conv2d) | ≈ 32,898 | × | 394,776 |
| $\rho, \beta, \text{soft\_thr}$ | 3 | × | 36 |
| **总计** | | | **≈ 474,144** |

**FISTA 参数量约为 ADMM 的 6 倍**，主要来自双分支 Encoder-Decoder（每分支 4 层 Conv2d × 2 分支 = 8 层 Conv2d）。

---

## 6. 理论可解释性对比

| 维度 | HASA-ADMM | HASA-FISTA |
|---|---|---|
| **优化理论对应** | 严格对应 ADMM 子问题求解 | Plug-and-Play 去噪器框架 |
| **w/p-update 正确性** | 精确 L1 prox（无需正交性） | 近似去噪（学习型 prox） |
| **x-update 正确性** | 频域闭式解（精确） | 梯度步（一阶近似） |
| **双分支融合** | 不需要融合（ADMM 分裂解耦） | $\alpha$ 加权融合（无理论保证） |
| **收敛性论述** | 经典 ADMM 收敛定理可引用 | 需要 Lipschitz/压缩映射论证 |
| **审稿友好度** | 高（每步都有闭式公式） | 中（需要 PnP 框架包装） |

**核心区别**：ADMM 的每一步都能写出精确的优化子问题，变量分裂保证了 $w$-update 和 $p$-update 的软阈值是 L1 近端映射的闭式解。FISTA 的双分支并行融合在数学上是 $\text{prox}_{g+h} \neq \text{prox}_g + \text{prox}_h$ 的"错误"操作，但在 Plug-and-Play 框架下可重新解释为**混合注意力去噪器**。

---

## 7. 实验对比价值

在论文中设置 HASA-FISTA 作为 HASA-ADMM 的对比实验（ablation），可以回答以下关键问题：

| 实验问题 | 如何回答 |
|---|---|
| **ADMM 变量分裂 vs FISTA 近端梯度** | 控制 HASA 完全相同，仅改变求解器 |
| **频域闭式解是否关键** | ADMM 有闭式解，FISTA 只有梯度步 |
| **固定算子 vs 可学习 Encoder-Decoder** | ADMM 用固定 Haar+TV，FISTA 用学习型 Conv |
| **参数量与性能的 trade-off** | FISTA 参数量 6× 但不一定更好 |
| **动量加速是否有帮助** | FISTA 有 Nesterov 动量，ADMM 无 |

**预期结论**：在超声 CS 重建任务中，**频域闭式解**的精确频率恢复能力比学习型 Encoder-Decoder 的表达力更重要，因此轻量的 HASA-ADMM 应优于参数量更大的 HASA-FISTA。这也呼应了审稿意见中"ADMM 变量分裂为双分支提供严密数学背书"的建议。

---

## 8. 数据流图对比（单层）

### HASA-ADMM Block

```
      x ──────► HASA ──► λ_wav, λ_tv
      │           │
      ├── W(x) ──┼──► soft_thr → w_new        (小波闭式)
      │           │
      ├── Env(x)─D─┼──► soft_thr → p_new      (TV 闭式)
      │           │         │
      │      ┌────┘    ┌───┘
      │      ▼         ▼
      └──► 频域闭式解(y, w, u₁) ──► TV修正(p, u₂) ──► x_new
                                                    │
                  u₁ += γ(Wx - w), u₂ += γ(DEnv - p)
```

### HASA-FISTA Block

```
      v ──── A^T(Av-y) ──► z = v - ρ·grad     (数据一致性)
                            │
                            ├── HASA(z) → λ_tv, λ_wav, α
                            │
                    ┌───────┴───────┐
                    │               │
              ┌─────▼─────┐  ┌─────▼─────┐
              │ TV branch │  │WAV branch │
              │ E→shrink→D│  │E→shrink→D │
              └─────┬─────┘  └─────┬─────┘
                    │   x_tv       │  x_wav
                    └───────┬──────┘
                            │
                      α·x_tv + (1-α)·x_wav = x_new
                            │
                      v_new = x_new + β·(x_new - x_prev)
                            │                (FISTA 动量)
```

---

## 9. 总结

| 指标 | HASA-ADMM | HASA-FISTA |
|---|---|---|
| **理论严谨性** | 高（严格 ADMM 展开） | 中（PnP 框架） |
| **频率恢复能力** | 强（频域闭式解） | 弱（梯度步） |
| **参数量** | 轻量（~79K for 2D） | 较重（~474K for 2D） |
| **每层计算** | 3×FFT + 1×autograd | 2×FFT + 8×Conv |
| **需要学习的组件** | HASA 权重 + 4 标量 | HASA 权重 + Encoder-Decoder + 3 标量 |
| **对比实验角色** | **主方法** | **消融对照** |

HASA-ADMM 用更少参数、更强理论、更精确的频域求解实现重建；HASA-FISTA 用更多参数、更灵活的学习型 prox 实现重建。二者对比可以验证**"在超声 CS 中，物理结构利用（频域闭式解）比黑盒学习（CNN prox）更重要"**这一核心假说。
