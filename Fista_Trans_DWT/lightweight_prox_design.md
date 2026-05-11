# FISTA-Trans-DWT 轻量化 Prox 设计说明

## 背景

当前 `FISTA_Transformer_DWT.py` 的核心思路是：

1. `HASA` 权重网络使用 Transformer 预测 `lambda_tv / lambda_wav / alpha`
2. `TV prox` 分支使用全局 Transformer encoder/decoder
3. `DWT prox` 分支先做 DWT，再把所有子带 token 拼接后送入全局 Transformer encoder/decoder
4. 最终通过 `DFFM` 做跨层融合

这条路线在方法上是可行的，但在全量 PICMUS 数据和较长 1D 序列上会暴露两个问题：

1. 显存和计算开销过大，尤其是全局 self-attention 的 `O(L^2)` 复杂度
2. DWT 分支已经引入了明确的多尺度先验，再做全局 attention 会削弱“分而治之”的归纳偏置

因此，更合理的方向不是继续增加 Transformer 深度，而是重新分配“谁负责全局，谁负责局部”。

## 核心判断

### 1. 不是参数量太大，而是全局 attention 太重

当前模型参数量本身并不夸张，但由于：

- 序列长度约为 1527
- `TV prox` 是全局 Transformer
- `DWT prox` 也是全局 Transformer
- 整个结构还被放进 FISTA 展开中重复执行

所以真正的瓶颈是 attention 的显存和时间复杂度，而不是权重数量。

### 2. Prox 主干不一定需要全局 Transformer

`prox` 模块的职责更像：

- 去噪
- 去伪影
- 保留局部结构和边缘
- 注入稀疏先验、多尺度先验、平滑先验

这些任务往往更依赖局部模式，而不是“任意两个深度位置之间的全局关系”。

因此，`Conv1d`、`TCN`、轻量 U-Net、窗口注意力等结构，往往比全局 Transformer 更适合承担 prox 主干。

### 3. DWT 分支尤其不适合再做全局 attention

DWT 本身已经把信号拆成了：

- `cA`：低频、轮廓、主体结构
- `cD`：高频、边缘、突变、细节

这意味着模型已经得到了一个很强的先验：不同频带应该区别处理。

如果再把所有子带 token 拼接起来做 full attention，会出现两个问题：

1. 计算上并不节省，因为 token 总长度仍接近原长度
2. 方法上会削弱 DWT 的多尺度归纳偏置，使模型重新倾向于跨频带自由混合

因此，更自然的设计是：

- 子带内独立处理
- 高频做阈值收缩
- 低频做轻量门控
- 仅在必要位置做少量跨带融合

## 推荐的轻量化分工

### 保留 Transformer 的部分

建议保留 `HASA` 权重网络中的 Transformer。

原因是 `HASA` 的职责是根据整条信号的上下文去预测：

- `lambda_tv`
- `lambda_wav`
- `alpha`

它本质上是一个全局条件建模模块，因此保留 Transformer 是合理的。

### 轻量化 TV prox

建议把 `TV prox` 从全局 Transformer 改成更轻量的局部结构，例如：

- `Conv1d` 残差块
- `TCN`（膨胀卷积）
- 1D 窗口注意力
- 小型 U-Net 风格编码器

理由：

- TV 先验本身偏局部平滑和边缘保持
- 局部建模通常更符合 TV 分支的物理意义
- 可以显著降低显存和计算量

### 轻量化 DWT prox

建议把 `DWT prox` 改成“子带内轻量网络 + 阈值收缩 + 少量跨带融合”。

更具体地说：

1. 对每个子带单独做投影
2. 每个子带内部用轻量网络处理，例如 `Conv1d` 或小残差块
3. `cA` 使用门控或弱调制
4. `cD` 使用 `per-subband x per-position` 阈值收缩
5. 如果需要跨带交互，只在少数层做少量融合，而不是直接 full attention

这样更符合 DWT 的设计初衷，也更容易训练。

## 推荐的新结构

### 方案 A：最推荐

`HASA = Transformer`

`TV prox = Conv1d/TCN`

`DWT prox = 子带内 Conv + gate/shrink + 少量跨带融合`

这个方案兼顾：

- 全局自适应建模
- 多尺度归纳偏置
- 较低显存
- 更强可训练性

### 方案 B：折中

`HASA = Transformer`

`TV prox = 窗口注意力`

`DWT prox = 子带内窗口注意力或 Conv`

这个方案比全局 Transformer 更轻，但仍保留一定注意力建模能力。

### 方案 C：仅在瓶颈位置保留注意力

如果仍希望在 prox 中保留少量 Transformer，可以只在：

- 最低分辨率特征
- 或跨带融合节点

使用少量注意力，而不是在全序列上反复做全局 MHSA。

## 一个可执行的改造版本

下面给出一个更符合当前毕设主线的轻量版本：

### HASA 分支

- 输入 `z`
- 使用 Transformer encoder
- 输出 `lambda_tv / lambda_wav / alpha`

### TV prox 分支

- 输入 `z`
- 1D Conv/TCN encoder
- 局部特征 shrink
- 1D Conv decoder
- 输出 `x_tv`

### DWT prox 分支

- 输入 `z`
- 3-level Haar DWT
- 每个子带分别进入轻量 Conv block
- `cA` 使用 gate
- `cD` 使用 `Theta_s(i)` 做软阈值
- 必要时做一次跨带融合
- IDWT 重建得到 `x_wav`

### 融合

- `x_next = alpha * x_tv + (1 - alpha) * x_wav`
- 展开层间仍使用 `DFFM`

## 为什么这条路线更适合现在的实验

### 1. 更容易在全量 PICMUS 上训练

全局 Transformer 在长序列和全量验证上很容易爆显存，而轻量 prox 更适合大样本训练。

### 2. 更符合超声重建的先验

超声 1D RF 信号的核心问题往往是局部结构恢复和深层弱响应保留，而不是全局 token 关系建模。

### 3. 更有利于论文叙事

这条思路可以形成一个更清晰的方法论：

- 用 Transformer 负责全局条件建模
- 用 DWT 负责多尺度表示
- 用轻量局部网络负责 prox 修正

这样各个模块职责更清楚，也更容易解释为什么这样设计。

## 建议的实验顺序

如果后续继续做轻量化版本，建议按下面顺序推进：

1. 保持 `HASA` 不变，仅把 `TV prox` 改为 Conv/TCN
2. 再把 `DWT prox` 改为子带内轻量网络
3. 最后比较：
   - 原始 FISTA-Trans-DWT
   - Lite-TV
   - Lite-DWT
   - Lite-TV + Lite-DWT

这样可以明确收益到底来自哪一部分。

## 最终结论

当前 `FISTA-Trans-DWT` 的问题不在参数量，而在于全局 Transformer 使用过多。

从方法和工程两方面看，都没有必要让 prox 主干全部使用 Transformer。

更合理的设计应当是：

- `HASA` 保留 Transformer
- `TV prox` 使用轻量局部网络
- `DWT prox` 使用子带内轻量处理和少量跨带融合

这条路线更省显存、更符合 DWT 的归纳偏置，也更适合全量 PICMUS 数据上的稳定训练。
