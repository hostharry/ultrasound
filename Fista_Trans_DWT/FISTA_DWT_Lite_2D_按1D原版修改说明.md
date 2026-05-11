# FISTA_DWT_Lite_2D 按原版 1D Lite 结构的修改说明

## 1. 文档目的

这份文档的目的不是再重新解释一遍 `FISTA_DWT_Lite`，而是把下面这件事讲清楚：

```text
原版 1D Lite 到底保留了哪些核心设计，
当前 2D Lite 应该如何按这个思路修改，
哪些地方已经对齐，哪些地方还没有完全对齐。
```

也就是说，这份文档是一个“结构对齐说明”，方便后面继续改 `FISTA_DWT_Lite_2D.py` 时有明确参照。


## 2. 原版 1D Lite 的核心结构思想

原版 `1D Lite` 的设计，不是简单“把原版 Transformer 砍掉”，而是保留了下面这条主线：

```text
data consistency
-> HASA 权重调度
-> TV prox + DWT prox 双分支
-> alpha 融合
-> FISTA momentum
-> DFFM 跨层融合
```

其中最重要的思想有 4 个：

1. `HASA` 仍然保留强条件建模能力  
   在 1D Lite 里，HASA 没有被砍成卷积，而是保留了 Transformer 版，用来生成逐位置的：
   - `lambda_tv`
   - `lambda_wav`
   - `alpha`

2. `prox` 被轻量化，但“双分支”思想不变  
   也就是：
   - TV 分支负责局部平滑 / 边缘结构
   - DWT 分支负责多尺度稀疏化

3. `block` 输出仍然是标准 FISTA 形式  
   先做 data consistency，再经过 learned prox，再做动量更新。

4. `DFFM` 保留  
   最终输出不是只看最后一层，而是收集每层输出和特征做末端融合。


## 3. 原版 1D Lite 中最关键的模块对应关系

### 3.1 HASA

原版 1D Lite 用的是 `HASAWeightTransformer1D`。

作用：
- 根据 `z` 输出逐位置 `lambda_tv / lambda_wav / alpha`
- 负责“什么时候该更强 TV、什么时候该更强 DWT、什么时候更信哪一支”

### 3.2 TV prox

原版 1D Lite 的 TV 分支是：

```text
1x1 Conv -> ConvRes 编码 -> soft-threshold -> ConvRes 解码 -> 1x1 Conv
```

重点：
- 是 learned prox
- 是局部卷积版
- 输出的是 `x_tv`
- 同时保留编码器特征 `feat_tv`

### 3.3 DWT prox

原版 1D Lite 的 DWT 分支是：

```text
DWT
-> 各子带独立编码
-> cA 门控 / cD shrink
-> 各子带独立解码
-> IDWT
-> 回到时域
```

重点：
- 子带内独立卷积建模
- `cA` 和 `cD` 处理方式不同
- 最终回到主时域输出 `x_wav`
- 同时构造 `feat_wav`

### 3.4 Prox 融合

原版 1D Lite 的 `ISTAProxDWT1D_Lite` 里，最终做的是：

```text
x_next = alpha * x_tv + (1-alpha) * x_wav
feat_fused = alpha * feat_tv + (1-alpha) * feat_wav
```

这是“结构候选 + 多尺度候选”的显式融合。

### 3.5 Block 外层

原版 1D Lite block 的逻辑是：

```text
z = v - rho * A^T(A(v)-y)
lambda_tv, lambda_wav, alpha = HASA(z)
thr_tv = eta * lambda_tv
thr_wav = eta * lambda_wav
x_next = prox(z, thr_tv, thr_wav, alpha)
v_next = x_next + beta * (x_next - x_prev)
```

重点：
- data consistency 在 prox 之前
- `prox` 负责 learned regularization
- FISTA 动量用的是 `x_next` 和 `x_prev`

### 3.6 Net 末端

原版 1D Lite Net 的输出不是最后一层直接出，而是：

```text
收集所有 stage_outputs
累加所有 feat_fused
送入 DFFM
输出最终 x_out
```


## 4. 当前 2D Lite 已经对齐了哪些地方

当前 `FISTA_DWT_Lite_2D.py` 已经有几处和原版 1D Lite 对齐得比较好了。

### 4.1 外层 FISTA 主干已经对齐

当前 2D block 已经是：

```text
z = v - rho * A^T(A(v)-y)
-> HASA
-> prox
-> momentum
```

也就是说，data consistency 这一层已经改正确，不再是之前错误的观测域直接相减。

### 4.2 双分支 prox 结构已经对齐

当前 2D 版也保持了：
- `TV2D` 分支
- `DWT2D` 分支
- `alpha` 融合

这和原版 1D Lite 的基本组织是一致的。

### 4.3 2D 版已经加入了 stage residual

这是最近最重要的一次修改。

当前 `ISTAProxDWT2D_Lite` 已经不是直接输出整幅 `x_next`，而是：

```text
delta_tv
delta_wav
delta = alpha * delta_tv + (1-alpha) * delta_wav
x_next = z + delta
```

这一步虽然不是原版 1D Lite 的字面写法，但从训练稳定性角度，是一个合理增强，且与 HUNet 的“学修正量”思想相近。

### 4.4 末端 DFFM 已经接入

当前 2D 版也已经把 `DFFM2D` 接进来了：
- 收集每层 `x`
- 累加每层 `feat_fused`
- 用 `DFFM2D` 做末端融合

这一点和原版 1D Lite 的整体框架也是一致的。


## 5. 当前 2D Lite 和原版 1D Lite 仍然存在的关键差异

这部分最重要，因为它决定后面改哪里才真正是在“学习原版结构”。

### 5.1 HASA 没有对齐

这是当前最大的结构差异。

原版 1D Lite：
- 用的是 `Transformer` 版 HASA
- 具有全局条件建模能力

当前 2D Lite：
- 用的是 `HASAWeightFISTA2D` 卷积版
- 更偏局部卷积条件建模

这意味着当前 2D Lite 的调度能力，比原版 1D Lite 更弱。

对 `cont` 这种任务，这一点尤其关键，因为它可能更依赖：
- 更大范围的上下文
- 区域统计
- 更复杂的正则强度分配

所以如果说“按原版 1D Lite 结构学习”，那么最值得补的一刀其实不是 DWT，而是：

```text
把 2D 卷积 HASA 往更强条件建模方向推进
```

但这不一定意味着立刻上重型 2D Transformer，也可以先做更强的多尺度卷积 HASA。

### 5.2 DWT 分支的特征融合方式没有完全对齐

原版 1D Lite 在 DWT 分支最后，会把各子带特征对齐回主空间，构造 `feat_wav` 参与后续融合。

当前 2D Lite 这里做的是：

```text
feat_spatial = mean(enc_feats)
再上采样回原空间
```

这只是一个最小实现。

问题在于：
- 它默认所有子带等权
- 没有显式保留 band identity
- 对 `cont` 这种更依赖多尺度统计的任务可能过于粗糙

所以这一点和 1D Lite 的思想只对齐了一半：
- 有“多子带特征回流”这个概念
- 但还没有足够细致的子带融合设计

### 5.3 DFFM 的最终输出仍然是“纯融合输出”

当前 2D 版最后仍然是：

```text
x_out = x_fused * scale
```

这和原版 1D Lite 是一致的，所以它不算错误。  
但如果从稳定性增强的角度看，当前还可以继续考虑：

```text
x_out = (x_last + x_fused) * scale
```

是否会更稳。

注意：这不是“必须改”，只是一个可能的额外增强方向。

### 5.4 2D DWT 目前仍然是单层最小版

原版 1D Lite 是 `J=3`。
当前 2D Lite 为了稳定和最小可跑，只做了：
- `J=1`
- `LL/LH/HL/HH`

这不代表错，而是意味着：
- 当前 2D 版仍是一个验证性 baseline
- 还不是完全等价于原版 1D Lite 的多尺度表达力


## 6. 如果严格按“学习原版 1D Lite 结构”来修改，优先级应该怎么排

### 第一优先级：增强 HASA，而不是先加深 DWT

原因：
- 原版 1D Lite 保留 Transformer HASA，不是偶然
- 它说明作者认为“条件调度能力”很重要
- 当前 2D 版最明显比原版弱的地方就在这

建议方向：
- 从当前 `Conv2d HASA` 升级到多尺度卷积 HASA
- 或轻量 window-attention HASA
- 或显式引入局部统计辅助特征

### 第二优先级：改 DWT 特征融合

当前的“均值上采样融合”太朴素。

建议方向：
- 改成 concat + 1x1 Conv
- 或者 band-attention
- 至少不要默认所有子带等权

### 第三优先级：再考虑扩展 2D DWT 层级

当前 `J=1` 是为了先稳住。
如果前两项做完仍然不够，再考虑：
- `J=2`
- 更复杂的多层子带处理

而不是一开始就照抄 1D 的 `J=3`。

### 第四优先级：再考虑输出端 skip

这不是最像原版的方向，而是更偏工程稳定性增强。
可以作为附加选项，但不应先于 HASA 和 DWT 特征融合。


## 7. 对当前 2D Lite 的修改建议

如果目标是“按原版 1D Lite 结构继续逼近”，我建议把后续修改分成下面三步。

### Step 1. 保留当前 stage residual，不回退

当前：

```text
x_next = z + delta
```

这一步已经让训练更稳，建议保留。

### Step 2. 优先增强 HASA

这是最值得下一步动手的地方。

建议选一个低风险版本：
- 保持 `Conv2d` 主体
- 加一个更大感受野分支
- 或加局部统计输入
- 或改成两层尺度的特征提取后再出 `lambda_tv / lambda_wav / alpha`

目标不是立刻换成重型 Transformer，而是让 2D HASA 更接近原版 1D Lite 的“强调度器”角色。

### Step 3. 把 DWT feature fusion 从 mean 改成 learnable fusion

建议最小改法：

```text
四个子带特征上采样到原空间
-> concat
-> 1x1 Conv
-> feat_wav
```

这样做的收益通常会比继续抠 patch size 更直接。


## 8. 当前版本的定位

当前 `FISTA_DWT_Lite_2D.py` 可以这样定位：

```text
它已经是一个“按 1D Lite 框架迁移到 2D”的可跑版本，
但还不是“真正结构上对齐 1D Lite 精神”的完整版本。
```

它已经对齐的部分：
- FISTA 主干
- 双分支 prox
- DFFM 末端融合
- 当前加入的 stage residual

它还没有完全对齐的部分：
- HASA 的条件建模能力
- DWT 分支的特征融合精细度
- 多层 2D 多尺度表达


## 9. 结论

如果用一句话总结当前最重要的判断，就是：

```text
学习原版 1D Lite 的结构，最该学的不是“把 1D 模块翻成 2D”，
而是保留它的结构分工：
强 HASA 调度 + 轻量 TV prox + 轻量 DWT prox + DFFM 收尾。
```

当前 2D 版已经把：
- 轻量 TV prox
- 轻量 DWT prox
- DFFM
- FISTA 外层
都搭起来了。

下一步最值得补的是：

```text
把 2D HASA 做强，
再把 DWT 特征融合做得更像真正的多尺度融合。
```

这比单纯继续调 patch 大小，或者盲目加深 2D DWT，更符合“学习原版 1D Lite 结构”的方向。
