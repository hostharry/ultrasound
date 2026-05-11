# FISTA-DWT-Lite Hybrid DWT 修改方案

本文档详细说明如何在 **不大改 Lite 主体结构** 的前提下，重点增强 `DWTConvBranch`，目标是：

1. 补足 `simu_cont` 所需要的跨尺度协同与纹理保持能力
2. 尽量保留 `simu_reso` 已经表现不错的局部结构恢复能力
3. 避免直接退回到原版 `FISTA_Transformer_DWT` 的重型全局 Transformer 方案

---

## 1. 修改目标

当前 `FISTA_DWT_Lite` 的整体结构是合理的：

- 外层仍然是 `FISTA` 展开
- `HASAWeightTransformer1D` 仍负责全局调权
- `TV` 分支负责局部结构修正
- `DWT` 分支负责多尺度稀疏先验
- `DFFM` 负责跨层融合

问题主要集中在 `DWTConvBranch`：

1. **各子带独立编码 / 独立收缩 / 独立解码**
2. **子带特征对齐时直接平均**

这两点使得当前 DWT 分支虽然“有多个子带”，但还没有真正形成“多子带协同”。

因此建议的改动原则是：

- **先不动 TV 分支**
- **先不动 FISTA 主干**
- **先不动 HASA 与 DFFM 主体**
- **只增强 DWT 分支内部的信息流**

---

## 2. 当前 DWTConvBranch 的数据流

当前实现位于 [FISTA_DWT_Lite.py](/home/user/毕业设计/Ultrasound/Fista_Trans_DWT/FISTA_DWT_Lite.py) 中的 `DWTConvBranch`。

### 2.1 现有流程

给定输入 `z` 和阈值图 `thr_wav`：

```text
z
 -> pad 到 2^J 整除
 -> Haar DWT
 -> {cA, cD_J, ..., cD_1}
 -> 每个子带各自:
      proj_in
      ConvRes encoder
 -> cA 做 gate
 -> cD 做 soft-threshold
 -> 每个子带各自:
      ConvRes decoder
      proj_out
 -> IDWT
 -> crop 回原长度
 -> x_wav
```

同时，编码器特征会被上采样到同一长度，再取平均，得到：

```text
feat_spatial = mean( Up(feat_0), Up(feat_1), ..., Up(feat_J) )
```

### 2.2 当前实现的两个关键缺口

#### 缺口 A：没有跨子带交互

当前 `enc_feats` 的生成方式是：

```python
enc_feats = []
for band, proj, enc_blk in zip(subbands, self.proj_in, self.enc):
    feat = proj(band)
    feat = enc_blk(feat)
    enc_feats.append(feat)
```

这意味着：

- `cA` 不知道各个 detail 带在看什么
- `cD_1` 不知道当前位置对应的是平坦背景还是结构边界
- 某一子带的收缩强弱，只受自身特征和 `thr_wav` 影响

换句话说，当前结构只有“多子带并行”，还没有“多子带协作”。

#### 缺口 B：最终特征对齐采用简单平均

当前 `_realign_to_spatial()` 做的是：

```python
return torch.stack(aligned, dim=0).mean(dim=0)
```

这会带来两个问题：

1. 默认所有子带同等重要
2. 子带身份被抹平，后续 `DFFM` 无法知道某个信息来自低频还是高频

对 `simu_reso`，这个缺点可能不致命；但对 `simu_cont`，这通常会损失多尺度纹理信息。

---

## 3. 总体修改思路

建议将 DWT 分支改成如下形式：

```text
子带独立编码
 -> 轻量跨子带交互
 -> 子带特定 gate / shrink
 -> 子带独立解码
 -> IDWT 重建 x_wav

并行地:
对齐后的多子带 encoder 特征
 -> 可学习融合
 -> feat_spatial
```

更具体地说：

1. **在 `enc_feats` 之后加入一个轻量的 `CrossSubbandMixer`**
2. **将 `_realign_to_spatial()` 从简单平均改为可学习融合**

这两步分别解决：

- “子带之间不交流”
- “交流后的多尺度信息被平均抹掉”

---

## 4. 修改一：加入轻量跨子带交互模块

### 4.1 修改目标

希望在不引入全序列重型 Transformer 的前提下，让不同子带在同一空间位置上互相参考。

最重要的是让模型学到：

- 当前位置的 detail 是否应被视为真实结构
- 当前位置的低频背景是否应保护某个弱纹理
- 某个尺度上的弱响应，是否能被其他尺度“证实”

### 4.2 推荐插入位置

建议插在：

```text
enc_feats 计算完成之后
gate / shrink 之前
```

即把原来的：

```text
子带编码 -> gate/shrink -> 子带解码
```

改成：

```text
子带编码 -> CrossSubbandMixer -> gate/shrink -> 子带解码
```

### 4.3 推荐模块形式

推荐使用 **band-axis attention**，但注意不是沿时间维度做全局 attention，而是：

- 在每个位置 `t`
- 只让 `J+1` 个子带 token 互相注意

当 `J=3` 时，子带数只有 4 个，因此复杂度非常低。

### 4.4 推荐的数据形状

当前每个子带编码特征为：

```text
f_i: (B, D, s_i)
```

先统一上采样到公共长度 `L_padded`：

```text
u_i = Up(f_i) : (B, D, L_padded)
```

然后重排为：

```text
U: (B, L_padded, n_bands, D)
```

再把 `(B, L_padded)` 合并，得到：

```text
U_flat: (B * L_padded, n_bands, D)
```

此时可以直接沿 `n_bands` 维度做一次多头注意力。

### 4.5 推荐模块定义

建议新增一个很轻的模块：

```python
class CrossSubbandMixer(nn.Module):
    def __init__(self, d_model, n_bands, nhead=4, mlp_ratio=2.0):
        super().__init__()
        self.band_embed = nn.Embedding(n_bands, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, aligned_feats):
        ...
```

### 4.6 为什么 `gamma` 建议零初始化

这是一个很重要的稳定性设计。

令输出采用残差形式：

```python
out = x + torch.tanh(self.gamma) * delta
```

其中：

- `x` 是原始的对齐后子带特征
- `delta` 是 band-mixer 学出的修正量

零初始化时，网络一开始几乎等价于原版 Lite，不会突然改变 `simu_reso` 的已有行为。训练过程中，只有当优化确实发现跨子带交互有价值时，`gamma` 才会逐步放大。

这是“增量式增强”而不是“结构跳变”。

### 4.7 推荐前向流程

假设已有：

```python
enc_feats = [f0, f1, ..., fJ]
```

新的流程建议是：

```python
aligned_feats = self._align_feats(enc_feats, L_padded)
mixed_aligned = self.cross_band_mixer(aligned_feats)
mixed_feats = self._restore_lengths(mixed_aligned, lengths)
```

然后后续的 gate / shrink / decode 都用 `mixed_feats`，而不是原始 `enc_feats`。

### 4.8 推荐的伪代码

```python
enc_feats = []
for band, proj, enc_blk in zip(subbands, self.proj_in, self.enc):
    feat = proj(band)
    feat = enc_blk(feat)
    enc_feats.append(feat)

aligned_feats = self._align_feats(enc_feats, L_padded)       # list[(B, D, Lp)]
mixed_aligned = self.cross_band_mixer(aligned_feats)         # list[(B, D, Lp)]
mixed_feats = self._restore_lengths(mixed_aligned, lengths)  # list[(B, D, s_i)]

processed = []
gated = mixed_feats[0] * self.approx_gate(mixed_feats[0])
processed.append(gated)

for j, (feat_j, thr_j) in enumerate(zip(mixed_feats[1:], thr_per_band[1:])):
    scale_j = F.softplus(self.subband_scale[j])
    thr_expanded = (scale_j * thr_j).expand_as(feat_j)
    processed.append(soft_threshold(feat_j, thr_expanded))
```

### 4.9 预期收益

该模块主要改善：

- `simu_cont` 中弱纹理被误删的问题
- detail 带孤立 shrink 过强的问题
- 低频背景与细节带脱节的问题

它对 `simu_reso` 的风险相对可控，因为：

- TV 支路没动
- DWT 子带内卷积没动
- 重建主路径仍然是 local conv + shrink
- mixer 是残差增量，不会立刻破坏现有行为

---

## 5. 修改二：把平均对齐改成可学习融合

### 5.1 修改目标

当前 `feat_spatial` 用的是简单平均，这对于 `DFFM` 来说太“粗糙”。

更合理的目标是：

- 保留不同子带的身份信息
- 让模型自己学不同子带在不同位置上的重要性

### 5.2 推荐替换方式

当前：

```python
feat_spatial = mean(Up(feat_i))
```

推荐改成：

```python
feat_spatial = FeatureFusion([Up(feat_0), ..., Up(feat_J)])
```

### 5.3 第一版最推荐的实现

最稳、最容易落地的方式是：

1. 将所有对齐后的子带特征在通道维拼接
2. 用一个 `1x1 Conv` 融合回 `d_model`

即：

```text
aligned_feats:
  [(B, D, L), (B, D, L), ..., (B, D, L)]

concat ->
  (B, n_bands * D, L)

1x1 Conv ->
  (B, D, L)
```

### 5.4 推荐模块定义

```python
class LearnableBandFusion(nn.Module):
    def __init__(self, d_model, n_bands):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(n_bands * d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, aligned_feats):
        x = torch.cat(aligned_feats, dim=1)
        return self.fuse(x)
```

### 5.5 推荐接入位置

建议使用 **cross-band mixer 输出后的对齐特征** 做融合：

```python
feat_spatial = self.feature_fuser(mixed_aligned)
return x_wav, feat_spatial.permute(0, 2, 1)
```

这样 `feat_spatial` 既包含：

- 子带内卷积编码信息
- 跨子带交互信息
- 可学习尺度融合信息

### 5.6 为什么比平均更好

简单平均相当于固定假设：

- 所有子带一样重要
- 所有位置都一样重要

而可学习融合允许模型学到：

- 在点目标附近，更依赖高频细节子带
- 在平坦背景处，更依赖低频 `cA`
- 在边界过渡区，需要多尺度混合

这对于 `simu_cont` 尤其重要，因为 contrast 任务常常需要“多尺度共同确认”。

### 5.7 进阶版本

如果第一版 `concat + 1x1 conv` 有效果，可以进一步尝试显式 band attention：

```python
weights = softmax(gate_net(concat_feats), dim=band)
fused = sum_i weights_i * aligned_feat_i
fused = proj(fused)
```

但建议先从 `concat + 1x1 conv` 开始，因为：

- 更稳
- 参数少
- 实现更简单
- 已足够验证“平均是不是瓶颈”

---

## 6. 推荐最终数据流

修改后的 `DWTConvBranch` 推荐数据流如下：

```text
z
 -> pad
 -> DWT
 -> 子带独立编码
 -> 对齐到公共长度
 -> CrossSubbandMixer
 -> 恢复各子带原长度
 -> cA gate / cD shrink
 -> 子带独立解码
 -> IDWT
 -> x_wav

同时:
对齐后的 mixed_aligned
 -> LearnableBandFusion
 -> feat_spatial
```

这比当前版本多出的部分只有两块：

1. `CrossSubbandMixer`
2. `LearnableBandFusion`

TV 分支、HASA、FISTA 外层、DFFM 都不需要先动。

---

## 7. 建议的代码修改点

建议只修改 [FISTA_DWT_Lite.py](/home/user/毕业设计/Ultrasound/Fista_Trans_DWT/FISTA_DWT_Lite.py)。

### 7.1 新增模块

在 `ConvResBlock1D` 之后、`DWTConvBranch` 之前，新增：

- `CrossSubbandMixer`
- `LearnableBandFusion`

### 7.2 在 `DWTConvBranch.__init__` 中新增成员

建议新增：

```python
self.cross_band_mixer = CrossSubbandMixer(
    d_model=d_model,
    n_bands=self.n_bands,
    nhead=4,
)

self.feature_fuser = LearnableBandFusion(
    d_model=d_model,
    n_bands=self.n_bands,
)
```

### 7.3 新增辅助函数

建议增加两个 helper：

```python
def _align_feats(self, feats, L_target):
    ...

def _restore_lengths(self, aligned_feats, lengths):
    ...
```

目的：

- `_align_feats`: 把不同子带特征上采样到统一长度
- `_restore_lengths`: 把 mixer 后的统一长度特征恢复到各自子带长度

### 7.4 修改 `forward()`

重点改四处：

1. `enc_feats` 生成后，不再直接进入 `processed`
2. 先走 `align -> mix -> restore`
3. gate / shrink 改用 `mixed_feats`
4. `feat_spatial` 改用 `feature_fuser(mixed_aligned)`

---

## 8. 推荐的第一版实验设置

为了控制风险，第一版不要同时大改很多超参数。

建议先：

- 保持 `TV` 分支完全不变
- 保持 `HASA` 不变
- 保持 `DFFM` 不变
- `CrossSubbandMixer` 只用 **1 个 block**
- `gamma` 零初始化
- `LearnableBandFusion` 先用最简单的 `concat + 1x1 conv`

也就是说，第一版是：

```text
Lite baseline
+ cross-band mixer
+ learnable feature fusion
```

不要一上来再额外：

- 增加更多 Transformer block
- 改动 TV 分支
- 改动 `alpha` 生成逻辑
- 改动外层 FISTA

否则会很难判断效果到底来自哪一部分。

---

## 9. 推荐的消融顺序

建议按以下顺序做实验：

### A0: 原版 Lite

当前基线。

### A1: 只改 LearnableBandFusion

目的：

- 验证“平均融合是不是瓶颈”

如果 A1 就显著提升 `simu_cont`，说明问题的一部分确实在于特征对齐过于粗糙。

### A2: 只加 CrossSubbandMixer

目的：

- 验证“缺少跨子带交互是不是更核心的瓶颈”

如果 A2 提升比 A1 更明显，说明协同建模比简单融合更关键。

### A3: CrossSubbandMixer + LearnableBandFusion

这是推荐的完整第一版。

### A4: 再考虑更进一步的修改

例如：

- 显式 band attention 融合
- `alpha` 依赖两支特征而不只依赖 `z`
- DWT 分支加第二个 mixer block

---

## 10. 风险与收益评估

### 10.1 预期收益

对 `simu_cont`：

- 更可能保住弱纹理
- 更可能减轻过度 shrink
- 更可能让多尺度信息共同参与判断

对 `simu_reso`：

- 理论上风险可控
- 因为局部卷积主干仍在
- TV 分支未改
- mixer 为残差增量，不会立刻推翻现有行为

### 10.2 主要风险

1. 子带交互过强，可能让高频细节被低频背景“拉平”
2. learnable fusion 若过度偏向低频，可能损伤 `simu_reso`
3. 若不做残差门控，结构变化会过于激进

### 10.3 降风险建议

建议同时做三件事：

1. `CrossSubbandMixer` 使用 `gamma=0` 初始化
2. `LearnableBandFusion` 先用轻量 `1x1 conv`
3. 第一版只在 DWT 分支内改，不碰 TV 分支

---

## 11. 一句话结论

如果只允许对 Lite 做一轮“最值当、最稳”的结构增强，那么最推荐的是：

1. **在 DWT 分支中加入轻量的跨子带交互模块**
2. **把子带对齐从简单平均改成可学习融合**

这两步不会破坏 Lite 的局部卷积主干，但能补回当前最缺的“跨尺度协同”和“尺度自适应融合”能力，是最适合优先尝试的 Hybrid 方案。

