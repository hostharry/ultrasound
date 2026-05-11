# FISTA-DWT-Lite 模型结构说明

本文档说明 `FISTA_DWT_Lite_Net` 的整体数据流、**残差连接**出现在哪里、各支路如何工作，并附一个**小型数值例子**便于对照代码。

---

## 1. 总览：主干在做什么

网络输入为频域观测 `y`（以及前向算子 `op`，如 MaskedRFFT），输出为时域重建 `x̂`。

**主干流程**可以概括为：

1. **初始化**：`x₀ = A†y`（若未显式传入 `x0`）。
2. **幅度归一化**：按当前 batch 内 `x₀` 的**最大绝对值**做缩放，使内部迭代在稳定数值范围内进行；最后**乘回** `scale` 得到物理幅度。
3. **重复 `layer_num` 次**（例如 4 次）**FISTA-DWT-Lite Block**：每块做一次「梯度步 + HASA + 双分支近端 + FISTA 动量」。
4. **DFFM 融合**：把每一层的 1 通道输出堆成多通道，再与**累加的 prox 特征**融合，得到最终 `x̂`。

```
                    ┌─────────────────────────────────────┐
  y  ──► x₀=A†y ──►  scale 归一化  ──►  [Block]×K  ──►  DFFM  ──► ×scale ──► x̂
                    └─────────────────────────────────────┘
```

---

## 2. 有无残差？分别在哪里

| 位置 | 是否有残差 | 形式 |
|------|------------|------|
| **ConvResBlock1D**（TV / DWT 子带内的卷积栈） | **有** | `out = x + Conv(GELU(Conv(x)))`，经典残差块 |
| **DFFM1D** | **有** | `x = DMlp(SE(x))` 后与 `shortcut` 相加：`x = x + shortcut` |
| **单层 Block 的 FISTA 动量** | **有（外推项）** | `v_{k+1} = x_{k+1} + β·(x_{k+1} - x_k)`，不是 CNN 里的 skip，而是 FISTA/Nesterov 型动量 |
| **从 x₀ 直接跳到最终输出** | **无** | 最终输出由多层迭代 + DFFM 决定，没有全局 `x̂ = x₀ + …` 的单一 skip |

**小结**：卷积 prox 内部和 DFFM 里都有显式残差；整体迭代结构是「展开优化 + 末端融合」，而不是 U-Net 式跨层跳连。

---

## 3. 单层 Block：`FISTA_DWT_Lite_Block`

对**已缩放**的变量 `x`、`v` 与观测 `y_scaled`，单步为：

### 3.1 数据一致项（梯度下降步）

\[
r = A(v) - y,\quad g = A^\top r,\quad z = v - \rho \cdot g
\]

其中 `ρ = softplus(rho)` 为可学习步长（实现里 `rho1/rho2` 日志相同标量）。

### 3.2 HASA 支路（全局 Transformer）

以 `z`（形状 `(B,1,L)`）为输入，**HASAWeightTransformer1D** 在全长序列上做自注意力，输出三个与位置对齐的权重图：

- `λ_tv`：TV/空域分支阈值缩放 `(B,1,L)`
- `λ_wav`：小波分支阈值缩放 `(B,1,L)`
- `α`：两分支融合系数 `(B,1,L)`，取值经 Sigmoid 在 (0,1)

实际阈值：

\[
\tau_{tv} = \eta \cdot \lambda_{tv},\quad \tau_{wav} = \eta \cdot \lambda_{wav}
\]

其中 `η = softplus(soft_thr)`。

### 3.3 Lite Prox：`ISTAProxDWT1D_Lite`

并行两路，再在**信号域**和**特征域**用 `α` 加权融合（`α` 小时更信小波支路）：

- **信号**：\(x_{next} = \alpha \odot x_{tv} + (1-\alpha) \odot x_{wav}\)
- **特征**（供 DFFM 累加）：同样按 `α` 对 `feat_tv`、`feat_wav` 逐通道加权

### 3.4 FISTA 动量更新

\[
v_{next} = x_{next} + \tanh(\beta)\cdot(x_{next} - x_{prev})
\]

下一层的「梯度步」用的是 `v_next`，而不是仅用 `x_next`。

---

## 4. TV 支路：`ConvProxTV1D`

**作用**：在**原采样长度 L** 上做局部卷积建模（替代原版的 TV 分支 Transformer）。

数据流：

1. `1×1 Conv`：`1 → d_model`
2. **Encoder**：若干 `ConvResBlock1D`（**带残差**）
3. **Soft-threshold**：对每个通道、每个位置，阈值由 `thr_tv` 广播后与特征同形状
4. **Decoder**：再若干 `ConvResBlock1D`
5. `1×1 Conv`：`d_model → 1` → `x_tv`

同时返回 encoder 末特征 `feat_tv`（形状 `(B,L,d_model)`），供 DFFM 路径使用。

---

## 5. DWT 支路：`DWTConvBranch`

**作用**：先 **Haar 多尺度分解**，再**每个子带独立**卷积编码—处理—解码，最后 **IDWT 合成**回时域（替代原版的「子带拼接后全局 Transformer」）。

要点：

1. **右侧零填充**：使长度可被 `2^J` 整除，IDWT 后**裁回**原始长度 `L_orig`。
2. **子带**：`cA`（最低频近似）+ `J` 个 detail 带 `cD_J,…,cD_1`，共 `J+1` 路。
3. **每路一套** `proj_in → ConvRes 编码 → 处理 → ConvRes 解码 → proj_out`（**编码/解码块内均有残差**）。
4. **近似系数 `cA`**：经 `sigmoid(Conv1d(·))` **门控**（逐通道缩放）。
5. **细节系数**：对 encoder 特征做 **soft_threshold**，阈值由 `thr_wav` **按子带长度下采样**（平均池化）后与该子带空间分辨率对齐，再乘以可学习的 `softplus(subband_scale[j])`。
6. **特征对齐**：各子带 encoder 特征上采样到同一长度后**取平均**，得到与 TV 支路对齐的 `feat_wav`（长度可能含 padding，再裁到 `L` 与 TV 融合）。

---

## 6. 末端融合：DFFM1D

输入：

- `x_stages`：每层 Block 输出的 `x` 去掉通道维后堆叠，形状 `(B, L, K)`，`K = layer_num`。
- `feat_sum`：各层 prox 输出的 `feat_fused` 经 `(B,L,D)→(B,D,L)` 后**逐层相加**。

流程：

1. `Conv1d(K → d_model, k=3)` 把「多阶段标量轨迹」升为 `d_model` 通道。
2. **残差支路**：`shortcut = x`；`x ← DMlp(SE(x))`；`x ← x + shortcut`。
3. `tail`：若干 `Conv1d` 得到 1 通道；并与 `weight * feat_sum` 相加后再过 tail 前的组合（实现为 `tail(x + weight * feat_sum)`）。

`weight` 为**单标量**可学习参数，控制「累加 prox 特征」注入强度。

---

## 7. 小型数值例子（手算友好）

以下与具体 `L、B` 无关，只说明**运算形态**。

### 7.1 梯度步（标量想象）

设某位置标量上：`v=1.0`，`A` 在该步简化成恒等且 `y=0.5`，`ρ=0.4`：

- `r = v - y = 0.5`
- `g = r = 0.5`（仅示意）
- `z = v - ρ·g = 1.0 - 0.2 = 0.8`

### 7.2 Soft-threshold

\[
S_\tau(u) = \mathrm{sign}(u)\cdot\max(|u|-\tau, 0)
\]

若某通道上 `u=0.5`，`τ=0.3`：\(|0.5|-0.3=0.2\) → 输出 **0.2**。  
若 `u=0.2`，`τ=0.3`：输出 **0**（抑制小系数）。

### 7.3 HASA 融合

若某位置 `α=0.7`，`x_tv=1.0`，`x_wav=0.0`：

\[
x_{next} = 0.7\times 1.0 + 0.3\times 0.0 = 0.7
\]

### 7.4 FISTA 动量

`x_prev=0.5`，`x_next=0.7`，`β=0.5`（即 `tanh(β)≈0.46`，此处取 **0.5** 便于算）：

\[
v_{next} = 0.7 + 0.5\times(0.7-0.5) = 0.7 + 0.1 = 0.8
\]

下一轮梯度步用 `v=0.8` 继续下降，起到**加速/平滑**迭代的作用。

### 7.5 多子带长度（帮助建立形状直觉）

取 **L_padded = 8**，**J = 2**（仅作教学，`2^J|L`）：

- 第 2 层近似 `cA` 长度约为 **2**
- 细节带长度按实现依次为更粗到更细（例如 **2** 与 **4** 量级）

每路子带上的 Conv 都在**该子带长度**上卷积，因此总计算量相对全长全局注意力更省；最后再 IDWT 合成回长度 8。

---

## 8. 与「原版 FISTA-Transformer-DWT」的差异（一句话）

- **HASA + DFFM**：思路一致（全局调权 + 多层输出融合）。
- **Prox**：原版在 TV/小波路径上用**全长 Transformer**；Lite 版改为 **Conv 残差块 + DWT 子带内卷积**，以**降低 \(O(L^2)\) 显存**，代价是**参数量往往更大**（多子带、多重复用卷积栈）。

---

## 9. 相关代码文件

| 模块 | 文件 |
|------|------|
| Lite 网络定义 | `FISTA_DWT_Lite.py` |
| HASA | `Admm_net/FISTA_Transformer.py` → `HASAWeightTransformer1D` |
| DFFM | `Admm_net/FISTA_Transformer_DFFM.py` → `DFFM1D` |
| Haar DWT/IDWT | `FISTA_Transformer_DWT.py` → `HaarDWT1d` / `HaarIDWT1d` |

训练/评估入口：`train_lite.py`、`evaluate_lite.py`。
