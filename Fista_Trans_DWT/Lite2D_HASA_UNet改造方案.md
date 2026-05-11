# Lite2D HASA U-Net 改造方案

这份文档用于说明：如何把当前 `FISTA_DWT_Lite_2D` 中偏弱的 2D HASA，替换成一个更适合 `simu_cont` 的 mini-U-Net 权重网络。

目标不是推翻现有 `Lite2D`，而是在不改 FISTA 外壳、不改 TV/DWT 双分支主体的前提下，增强 HASA 的区域理解能力。

## 1. 为什么要改 HASA

当前 `Lite2D` 的主干结构是：

- FISTA unfolding skeleton
- data consistency
- HASA 权重预测
- TV / DWT 双分支 prox
- stage residual + momentum

这里最容易成为瓶颈的部分之一，是 HASA。

当前 HASA 的问题主要是：

- 更像局部卷积调度器
- 能看局部边界，但区域理解不够
- 对 `simu_cont` 这种“contrast + speckle + 区域统计”任务支持不足

所以这次改造的核心目标是：

**把 HASA 从局部权重预测器，升级成一个多尺度 dense map predictor。**

## 2. 改造原则

这次改造遵守三个原则：

1. 不引入过重的手工统计特征
2. 不改 FISTA 主干
3. 不改 TV / DWT prox 的接口

因此采用的思路是：

**用一个轻量 U-Net 直接从 `z` 预测 `lambda_tv / lambda_wav / alpha`。**

也就是说：

\[
z \rightarrow HASA_{UNet} \rightarrow \lambda_{tv}, \lambda_{wav}, \alpha
\]

## 3. 当前 HASA 的接口

当前 HASA 在每个 stage 中承担的角色是：

\[
z \rightarrow \lambda_{tv},\ \lambda_{wav},\ \alpha
\]

其中：

- `lambda_tv`：控制 TV 分支阈值强度
- `lambda_wav`：控制 DWT 分支阈值强度
- `alpha`：控制两支融合比例

因此新 HASA 必须保持完全相同的输出接口。

## 4. 新 HASA 的总体结构

建议采用 **mini-U-Net HASA**，输入仍然是：

\[
z \in \mathbb{R}^{B \times 1 \times H \times W}
\]

输出仍然是三张逐位置图：

\[
\lambda_{tv}, \lambda_{wav}, \alpha \in \mathbb{R}^{B \times 1 \times H \times W}
\]

总体结构：

```text
z
 -> Encoder level 1
 -> Downsample
 -> Encoder level 2
 -> Downsample
 -> Bottleneck
 -> Upsample + skip
 -> Decoder level 1
 -> Upsample + skip
 -> Decoder level 2
 -> 3 个输出 head
```

## 5. 推荐网络配置

### 5.1 通道数

为了不让 HASA 过重，建议从非常轻的配置开始：

- `base_ch = 16`

则各层通道数为：

- encoder1: 16
- encoder2: 32
- bottleneck: 64
- decoder1: 32
- decoder2: 16

如果显存压力仍大，可以进一步试：

- `base_ch = 8`

### 5.2 深度

建议只做 **两次下采样**。

原因：

- 你已经是 patch 训练，不需要特别深的 U-Net
- HASA 是权重网络，不是主重建 backbone
- 太深会导致参数和显存不必要上升

### 5.3 基础 block

推荐使用简单残差卷积块：

```text
Conv2d -> GELU -> Conv2d -> GELU -> residual
```

不要一上来加复杂 attention。

## 6. 编码器设计

编码器作用是逐渐扩大感受野，获取区域上下文。

### Level 1

输入：

\[
(B,1,H,W)
\]

输出：

\[
(B,16,H,W)
\]

### Downsample 1

输出：

\[
(B,32,H/2,W/2)
\]

### Downsample 2

输出：

\[
(B,64,H/4,W/4)
\]

这里建议使用：

- `3x3 stride=2 Conv`

而不是 maxpool。

理由是：

- 学习式下采样更灵活
- 对权重图预测更友好

## 7. Bottleneck 设计

bottleneck 的目标是整合更大范围上下文。

推荐：

- 两个轻量 ConvBlock
- 不加 attention
- 不加额外复杂模块

原因：

- 当前 first priority 是把 HASA 从“局部卷积”升级到“多尺度区域理解”
- U-Net 本身已经提供多尺度能力

## 8. 解码器设计

解码器的目标是把高层区域上下文重新映射回逐位置权重图。

推荐：

- `bilinear upsample + 3x3 Conv`
- 与 encoder 做 skip connection

不建议第一版用反卷积，先以稳定为主。

## 9. 输出头设计

共享 decoder feature 后，接三个 head：

### TV head

\[
\lambda_{tv} = softplus(head_{tv}(f))
\]

### Wav head

\[
\lambda_{wav} = softplus(head_{wav}(f))
\]

### Alpha head

\[
\alpha = sigmoid(head_{alpha}(f))
\]

这样可以确保：

- `lambda_tv > 0`
- `lambda_wav > 0`
- `alpha \in (0,1)`

与当前逻辑完全兼容。

## 10. 为什么这种结构适合 `simu_cont`

### 10.1 更强的区域建模

`simu_cont` 关注的不是单点，而是：

- 区域均值差异
- 区域纹理差异
- 区域边界过渡

U-Net 的 encoder-decoder 比单层卷积更适合这种 dense region reasoning。

### 10.2 更符合深度学习原则

这版改造仍然只输入 `z`，没有显式引入太多手工统计特征。

也就是说：

- 不依赖手工 mean/var/grad 输入
- 多尺度信息由网络自己学
- 更符合“让网络自己学习表征”的原则

### 10.3 工程风险适中

相比：

- full Transformer HASA
- 显式统计量特征工程
- 重写 prox 主干

mini-U-Net HASA 属于一个很稳的中间方案。

## 11. 它在整个网络中的位置

替换前：

```text
z -> current HASA -> lambda_tv, lambda_wav, alpha
```

替换后：

```text
z -> MiniUNetHASA2D -> lambda_tv, lambda_wav, alpha
```

因此：

- FISTA 外壳不改
- TV / DWT 双分支不改
- prox 接口不改
- 只改 HASA 模块实现

## 12. 与 HUNet 的关系

这个改造**不是**把整个网络改成 HUNet。

区别要分清：

- HUNet：U-Net-like 主干本身就是 stage 修正器
- 这里：U-Net 只用于 HASA，充当权重预测器

所以这是一个温和改法，不会破坏当前 FISTA-DWT-Lite 的整体框架。

## 13. 推荐伪代码

```python
class MiniUNetHASA2D(nn.Module):
    def __init__(self, base_ch=16):
        super().__init__()
        self.enc1 = ConvBlock(1, base_ch)
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 4)

        self.up1 = UpBlock(base_ch * 4, base_ch * 2)
        self.up2 = UpBlock(base_ch * 2, base_ch)

        self.head_tv = nn.Conv2d(base_ch, 1, 1)
        self.head_wav = nn.Conv2d(base_ch, 1, 1)
        self.head_alpha = nn.Conv2d(base_ch, 1, 1)

    def forward(self, z):
        f1 = self.enc1(z)
        f2 = self.down1(f1)
        f3 = self.down2(f2)

        b = self.bottleneck(f3)

        u2 = self.up1(b, f2)
        u1 = self.up2(u2, f1)

        lambda_tv = F.softplus(self.head_tv(u1))
        lambda_wav = F.softplus(self.head_wav(u1))
        alpha = torch.sigmoid(self.head_alpha(u1))
        return lambda_tv, lambda_wav, alpha
```

## 14. 推荐实验顺序

建议不要和 loss 改造一起上。

### 实验 A

只替换 HASA：

- 保持当前 loss 不变
- 保持 prox 不变
- 观察 `cont` 是否提升

### 实验 B

如果 A 有提升，再叠加你前面设计的 loss：

- `nmse_logenv + grad`

### 实验 C

再进一步加：

- `local statistics loss`

这样变量最少，最容易判断增益来源。

## 15. 实现建议

建议在 [FISTA_DWT_Lite_2D.py](/home/user/毕业设计/Ultrasound/Fista_Trans_DWT/FISTA_DWT_Lite_2D.py) 中：

1. 新增 `MiniUNetHASA2D`
2. 保留当前旧 HASA 类，便于做 ablation
3. 在 `train_lite_2d.py` 增加一个参数，比如：
   - `--hasa_type conv`
   - `--hasa_type unet`

这样可以方便做对照实验。

## 16. 一句话结论

对当前 `Lite2D + simu_cont`，如果想增强 HASA，又不想走太多手工特征工程，**最合理的方向就是把 HASA 改成一个轻量 mini-U-Net 权重网络。**

这会比当前卷积 HASA 更有 2D 区域理解能力，同时又不至于像 full Transformer 那样太重。
