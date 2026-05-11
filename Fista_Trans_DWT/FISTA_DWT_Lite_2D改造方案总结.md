# FISTA_DWT_Lite 在 `simu_cont` 上的分析与 2D 改造方案总结

## 1. 当前问题的核心结论

`FISTA_DWT_Lite / Hybrid` 在 `simu_cont` 上表现明显弱于 `simu_reso`，主因不是单一训练参数，而是下面几类因素叠加：

1. `simu_cont` 和 `simu_reso` 本身任务属性不同  
   `reso` 更偏点目标、强结构、边界锐利；  
   `cont` 更偏区域统计、speckle 纹理、低对比差异。

2. 当前 `Lite` 结构天然更偏“结构恢复 + 去噪”  
   它对 `reso` 更友好，对 `cont` 这种需要保统计纹理的任务不够友好。

3. 逐线 `1D` 重建会丢失大量横向先验  
   对 `reso` 影响相对可接受；  
   对 `cont`，损失会非常明显，因为很多关键线索来自二维区域统计，而不是单条 RF line。

4. 当前网络没有显式“纹理保持机制”  
   现有双分支更像两种不同风格的 proximal 去噪器，而不是“结构分支 + 纹理保持分支”。


## 2. 当前 `Lite / Hybrid` 结构是怎么工作的

每一层 block 可以概括为：

```text
data consistency -> HASA -> TV prox + DWT prox -> alpha 融合 -> FISTA momentum
```

其中：

- `data consistency`：保证结果不偏离观测
- `HASA`：根据当前中间解 `z` 输出逐位置的 `lambda_tv / lambda_wav / alpha`
- `TV branch`：偏局部平滑、边界保持
- `DWT branch`：偏多尺度稀疏收缩
- `alpha`：控制两个分支的融合比例


## 3. TV 分支本质上在做什么

当前 TV 分支不是传统显式 TV 解析 proximal，而是一个 `TV-like learned prox`：

```text
1x1 Conv -> ConvResBlock 编码 -> soft-threshold -> ConvResBlock 解码 -> 1x1 Conv
```

它的偏好是：

- 抑制局部振荡
- 保主要边界
- 做分段平滑

这对 `reso` 很有帮助，但对 `cont` 的 speckle 纹理并不天然友好，因为很多 speckle 在它看来像“该被压掉的小波动”。


## 4. DWT 分支本质上在做什么

当前 DWT 分支会先把信号分解为多个子带，再分别处理：

- `cA / 低频`：走门控
- `cD / 细节`：走 soft-threshold
- 然后再解码和反变换回时域

它比 TV 更接近 `cont` 所需的“多尺度信息”，但当前实现里 detail 子带仍然强依赖 shrink，所以仍然偏：

- 保强系数
- 压弱系数
- 偏稀疏

而 `cont` 里很多真实纹理恰恰是“低幅但统计上重要”的成分。


## 5. 为什么可以说双分支都更像“去噪器”

更准确地说，两个分支都是先验驱动的 proximal 正则器，但它们都带有很强的去噪器属性：

- `TV`：结构型、平滑型去噪
- `DWT`：多尺度稀疏型去噪

因此，对 `cont` 的难点不是“它们完全没用”，而是：

```text
两个分支都容易把弱纹理当成噪声处理
```

所以问题往往不是“双分支不该工作”，而是“双分支工作得太像去噪器，缺少纹理保持能力”。


## 6. 当前 HASA 是怎么工作的

当前 `Hybrid` 中使用的是 Transformer 版 HASA 权重网络。

它的输入只有当前数据一致性后的中间结果 `z`：

```text
z -> token embedding -> positional encoding -> Transformer -> 
    head_tv, head_wav, head_alpha
```

最终输出：

- `lambda_tv`
- `lambda_wav`
- `alpha`

因此，HASA 当前是在回答：

```text
根据 z 这条 1D 信号本身的形状和上下文，
当前位置应该施加多强的 TV/Wav 正则，以及更信哪一支
```

但它没有显式看到：

- DWT 子带特征
- TV branch 编码特征
- 局部方差 / 局部均值
- 包络统计
- 纹理保护掩码

所以它虽然能从 `z` 中提取一部分纹理线索，但这些线索是：

- 隐式的
- 混合的
- 1D 的
- 没有被整理成明确的统计表示


## 7. 为什么 `1D` 对 `cont` 天然吃亏

逐线重建时，模型只看到一条 RF line，因此仍然能学到一些单线信息：

- 轴向亮暗变化
- 单线局部包络起伏
- 边界穿过该线时的统计突变
- 一维多尺度幅值变化

但它很难学到真正对 `cont` 很重要的二维区域信息：

- 横向 speckle 连续性
- 区域级同质性
- lesion/background 的二维统计差
- 横向边界形状

所以 `1D` 模型更容易退化成：

```text
结构恢复器 + 去噪器
```

而不是：

```text
区域统计恢复器 + 纹理保持器
```


## 8. 是否可以通过增强 HASA 来改善 `cont`

可以，而且这是一个相对稳妥的方向。

因为当前真正的问题更像是：

```text
TV 和 DWT 这两把刀在某些位置下得太重了
```

所以比起重写两条主分支，一个更稳的思路是增强 HASA，让它更会识别：

- 哪些位置更像关键纹理区
- 哪些位置不该强 shrink
- 哪些位置应该减少 TV 平滑
- 哪些位置更该偏向保留 DWT 多尺度成分

但这个方向不能根治 `1D` 的先天缺陷，因为它仍然看不到完整二维区域统计。


## 9. 为什么最终建议改成 2D

如果目标是提升 `simu_cont`，最值得验证的核心假设是：

```text
问题的关键瓶颈并不只是阈值调度不够好，
而是 1D 模型本身丢失了 cont 所需的二维先验
```

因此，最有信息量的下一步不是继续在 1D 上叠加很多局部技巧，而是先做一个 2D 版本，验证：

- 横向相关性是否显著提升 `cont`
- patch/frame 输入是否比逐线更有利于 contrast/speckle 任务


## 10. 推荐的 2D 改造策略

不建议一步把当前 `1D Hybrid` 的所有复杂模块全部搬到 `2D`。  
更稳妥的路线是两阶段。

### 阶段一：先做最小可跑的 `FISTA_DWT_Lite_2D`

保留：

- FISTA 外层
- 2D HASA
- 2D TV prox
- 2D DWT prox
- 2D frame/patch 训练

暂时不保留：

- cross-subband mixer
- learnable band fusion
- DFFM

目标是先回答：

```text
仅仅把 1D 改成 2D，是否就能明显改善 simu_cont
```

### 阶段二：如果 2D baseline 有效，再补 Hybrid 机制

如果最小版 2D 已经明显优于 1D，再逐步加入：

- 2D 跨子带交互
- 2D 可学习子带融合
- 2D DFFM

这样能避免一上来变量过多，导致无法判断到底是哪一块起作用。


## 11. 2D 版本的建议结构

推荐第一版 2D 模型做成：

```text
MaskedRFFT2D
  -> data consistency
  -> HASAWeightFISTA2D
  -> ConvProxTV2D
  -> HaarDWT2DProx
  -> alpha 融合
  -> FISTA momentum
```

### 11.1 data consistency

直接复用现有 2D 测量算子：

- `MaskedRFFT2D`

它已经支持：

- 逐行 `rfft`
- 所有行共享同一采样 mask

这和当前超声帧级输入的测量方式是一致的。

### 11.2 HASA

第一版建议直接复用 2D 卷积版 HASA，而不是上 2D Transformer。

理由：

- 更稳
- 更省显存
- 对 patch 训练更友好
- 足够先验证 2D 局部空间相关性是否有价值

HASA 输出仍然是：

- `lambda_tv (B,1,H,W)`
- `lambda_wav (B,1,H,W)`
- `alpha (B,1,H,W)`

### 11.3 TV 分支

把当前 `ConvProxTV1D` 改成 `ConvProxTV2D`：

```text
Conv2d(1 -> d_model, 1x1)
-> ConvResBlock2D 编码
-> 特征 soft-threshold
-> ConvResBlock2D 解码
-> Conv2d(d_model -> 1, 1x1)
```

这样 TV 分支才能真正利用：

- 轴向局部信息
- 横向局部信息
- 二维边缘结构

### 11.4 DWT 分支

第一版建议采用：

- 单层 2D Haar (`J=1`)
- 4 个子带：`LL / LH / HL / HH`

处理方式：

- `LL`：门控保留
- `LH/HL/HH`：soft-threshold
- 各子带卷积编码/解码后再 IDWT

不要第一版就做多层 `J=3 + mixer`，否则复杂度和调试难度都会明显上升。


## 12. 训练与数据建议

建议直接复用现有 2D 训练管线：

- 数据：`picmus_simu_cont_frames.npz`
- 数据集类：`UltrasoundFrameDataset`
- 训练循环：`run_train_2d`

推荐用 patch 而不是整帧：

- `patch_h=32` 或 `64`
- `patch_stride=16` 或 `32`

原因：

- 完整帧样本数较少
- patch 可以增加训练样本量
- 同时保留一定横向相关性
- 显存更可控


## 13. 2D 改造时的风险控制

### 13.1 为什么不建议一上来就上 2D Transformer

因为这会同时引入：

- 更高显存
- 更长 token
- 更难训练
- 更多不确定性

你当前最想验证的是：

```text
2D 先验是否本身就能改善 cont
```

而不是：

```text
2D Transformer 是否优于 2D Conv
```

### 13.2 为什么不建议第一版就加 Hybrid 全套模块

因为这样很难判断：

- 是 2D 有效
- 还是 mixer 有效
- 还是 feature fusion 有效

先做简单 2D baseline，更利于对比和消融。


## 14. 总结性判断

当前 `simu_cont` 难做，根本原因大概率是下面三者共同作用：

1. `Lite / Hybrid` 双分支都偏“去噪型正则”
2. HASA 虽然能调权，但没有显式纹理统计输入
3. `1D` 逐线重建天然缺失 `cont` 最需要的二维区域先验

因此，最值得尝试的下一步是：

```text
先做一个最小可跑的 2D FISTA_DWT_Lite 版本，
验证 2D 先验能否明显改善 simu_cont；
若有效，再把 Hybrid 机制逐步补回去。
```


## 15. 建议的下一步实施顺序

1. 新建 `FISTA_DWT_Lite_2D.py`
2. 新建 `train_lite_2d.py`
3. 直接接入 `run_train_2d`
4. 用 `picmus_simu_cont_frames.npz` 跑最小版 2D baseline
5. 与当前 `1D Lite / Hybrid` 做公平对比
6. 如果 2D baseline 明显更强，再继续做 `Hybrid_2D`

---

## 16. 已实现文件 (阶段一 baseline)

| 文件 | 说明 |
|------|------|
| `FISTA_DWT_Lite_2D.py` | `FISTA_DWT_Lite_2D_Net`：卷积 HASA + `ConvProxTV2D` + 单层 `HaarDWT2D`（J=1），无 DFFM |
| `train_lite_2d.py` | 调用 `run_train_2d`，默认 `../data/picmus_simu_cont_frames.npz`，`patch_h=32` / `patch_stride=16` |
| `evaluate_lite_2d.py` | 与 `evaluate_2d` 对接，读取 `config.txt` 重建网络 |

**训练示例**（在 `Fista_Trans_DWT` 目录下）:

```bash
python train_lite_2d.py \
  --npz ../data/picmus_simu_cont_frames.npz \
  --cs_ratio 8 --patch_h 32 --patch_stride 16 \
  --layers 4 --d_model 32 --dwt_levels 1 \
  --loss_mode nmse_logenv --gpu 0
```

**评估示例**:

```bash
python evaluate_lite_2d.py --exp_dir model/<实验目录>/ --gpu 0
```
