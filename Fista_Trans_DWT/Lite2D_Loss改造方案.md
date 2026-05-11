# Lite2D Loss 改造方案

这份文档专门针对当前 `FISTA_DWT_Lite_2D` 在 `simu_cont` 上的表现，说明为什么要改 loss、建议改成什么、以及怎么一步一步落地。

## 目标

当前 `Lite2D` 的主要问题不是完全学不动，而是：

- 基础重建能学到一些
- 结构边界容易变钝
- contrast 区域的局部统计和 speckle 纹理保持不够

所以 loss 的目标不应该只是继续压低 RF 误差，而应该同时约束三件事：

1. 保基础重建正确性
2. 保边界和结构过渡
3. 保局部对比度和统计纹理

## 总体思路

建议把当前 2D loss 改成下面这个组合：

\[
\mathcal{L}
=
\mathcal{L}_{nmse\_logenv}
+
\lambda_g \mathcal{L}_{grad}
+
\lambda_c \mathcal{L}_{stat}
\]

其中：

- `L_nmse_logenv`：基础项，保证 RF 与包络域重建
- `L_grad`：结构项，保证边界与过渡
- `L_stat`：局部统计项，保证 contrast/speckle 的局部统计

## 1. 基础项：NMSE + Log-Envelope

### 1.1 RF-NMSE

\[
\mathcal{L}_{nmse}
=
\frac{\|\hat{x}-x\|_2^2}{\|x\|_2^2+\epsilon}
\]

这里：

- `x`：GT RF
- `x_hat`：重建 RF

这一项继续保留，因为它仍然是主任务约束。

### 1.2 Log-Envelope L1

先定义包络：

\[
e(\hat{x}) = Env(\hat{x}), \qquad e(x) = Env(x)
\]

然后定义：

\[
\mathcal{L}_{logenv}
=
\frac{1}{N}
\left\|
\log(e(\hat{x})+\epsilon) - \log(e(x)+\epsilon)
\right\|_1
\]

最终基础项：

\[
\mathcal{L}_{nmse\_logenv}
=
\mathcal{L}_{nmse}
+
\gamma_{env}\mathcal{L}_{logenv}
\]

### 1.3 为什么保留它

- `NMSE` 保证 RF 重建不跑偏
- `logenv` 更接近超声成像可见的强弱动态范围
- 对 `cont` 来说，单纯 RF-MSE 不够，`logenv` 是必须保留的底座

## 2. 结构项：Gradient Consistency

这一项建议加在 `log-envelope` 图上，而不是原始 RF 上。

先定义：

\[
u_{\hat{x}} = \log(e(\hat{x})+\epsilon), \qquad
u_x = \log(e(x)+\epsilon)
\]

用 Sobel 或 Scharr 计算 2D 梯度：

\[
\nabla u = (\partial_h u, \partial_w u)
\]

定义梯度一致性：

\[
\mathcal{L}_{grad}
=
\frac{1}{N}
\left(
\|\partial_h u_{\hat{x}} - \partial_h u_x\|_1
+
\|\partial_w u_{\hat{x}} - \partial_w u_x\|_1
\right)
\]

也可以写成：

\[
\mathcal{L}_{grad}
=
\frac{1}{N}\|\nabla u_{\hat{x}}-\nabla u_x\|_1
\]

### 2.1 它解决什么问题

这一项主要补结构信息：

- 保 lesion 边界
- 保 contrast 区域过渡
- 防止 envelope 图整体被抹糊

### 2.2 为什么不用更强 TV 代替

因为当前网络本身已经带有明显的 TV-like bias。再上强 TV，风险是：

- 结构也许更干净
- 但 `cont` 的局部纹理更容易被抹平

`gradient consistency` 更像是“保边界”，不是“强行平滑”。

## 3. 局部统计项：Local Contrast / Statistics

这是这次 loss 改造里最贴 `simu_cont` 的部分。

建议第一版不要直接上复杂的 Nakagami 参数拟合，而是先做局部均值与方差匹配。

继续使用：

\[
u_{\hat{x}} = \log(e(\hat{x})+\epsilon), \qquad
u_x = \log(e(x)+\epsilon)
\]

用局部平均池化定义窗口统计。

### 3.1 局部均值

\[
\mu_{\hat{x}} = AvgPool(u_{\hat{x}}), \qquad
\mu_x = AvgPool(u_x)
\]

### 3.2 局部方差

\[
\sigma_{\hat{x}}^2 = AvgPool(u_{\hat{x}}^2) - \mu_{\hat{x}}^2
\]

\[
\sigma_x^2 = AvgPool(u_x^2) - \mu_x^2
\]

### 3.3 定义局部统计损失

\[
\mathcal{L}_{stat}
=
\frac{1}{K}\sum_k
\left(
|\mu_{\hat{x}}(k)-\mu_x(k)|
+
\alpha_{var}
\left|
\log(\sigma_{\hat{x}}^2(k)+\epsilon)
-
\log(\sigma_x^2(k)+\epsilon)
\right|
\right)
\]

这里：

- 第一项保局部亮度和 contrast
- 第二项保局部 speckle 粗糙度和纹理强弱

### 3.4 为什么适合 `simu_cont`

`cont` 的核心不只是边界，而是：

- 某个区域比背景更暗或更亮
- 局部统计分布不同
- speckle 不能被抹成塑料感

这一项就是在显式告诉网络：

“局部对比度和局部方差本身是目标，不是噪声。”

## 4. 最终推荐公式

\[
\mathcal{L}
=
\underbrace{
\frac{\|\hat{x}-x\|_2^2}{\|x\|_2^2+\epsilon}
+
\gamma_{env}
\cdot
\frac{1}{N}
\left\|
\log(e(\hat{x})+\epsilon)-\log(e(x)+\epsilon)
\right\|_1
}_{\mathcal{L}_{nmse\_logenv}}
\]

\[
\quad
+
\lambda_g
\cdot
\underbrace{
\frac{1}{N}
\|\nabla \log(e(\hat{x})+\epsilon)-\nabla \log(e(x)+\epsilon)\|_1
}_{\mathcal{L}_{grad}}
\]

\[
\quad
+
\lambda_c
\cdot
\underbrace{
\frac{1}{K}\sum_k
\left(
|\mu_{\hat{x}}(k)-\mu_x(k)|
+
\alpha_{var}
\left|
\log(\sigma_{\hat{x}}^2(k)+\epsilon)-\log(\sigma_x^2(k)+\epsilon)
\right|
\right)
}_{\mathcal{L}_{stat}}
\]

## 5. 推荐初始权重

第一版建议用轻权重起步：

- `gamma_env = 0.3`
- `lambda_g = 0.05`
- `lambda_c = 0.03`
- `alpha_var = 0.5`
- `epsilon = 1e-6`

局部统计窗口建议：

- `7x7` 作为第一版
- 如果效果太局部，可以试 `9x9`

## 6. 落地实现建议

### 6.1 改动位置

建议在 [loss.py](/home/user/毕业设计/Ultrasound/Utils/loss.py) 中扩展，不要单独新建太多训练入口。

建议新增：

- `gradient_consistency_2d(...)`
- `local_stat_loss_2d(...)`

然后在 `CombinedLoss` 中加入两个新权重：

- `gamma_grad`
- `gamma_stat`

### 6.2 推荐接口

可以把 `CombinedLoss.__init__()` 扩成：

```python
def __init__(self,
             gamma_env=0.1,
             gamma_constraint=0.01,
             gamma_msle=0.0,
             gamma_grad=0.0,
             gamma_stat=0.0,
             stat_var_weight=0.5,
             stat_win=7,
             ...):
```

其中：

- `gamma_grad` 控制 `L_grad`
- `gamma_stat` 控制 `L_stat`
- `stat_var_weight` 对应上面的 `alpha_var`
- `stat_win` 控制局部窗口大小

### 6.3 只在 2D 时启用

建议做成：

- `x.ndim == 4` 时启用 `grad/stat`
- `1D` 保持原逻辑不变

这样不影响现有 `Lite1D / HUNet1D` 训练。

## 7. 实验顺序

不要三项一上来全大权重一起开，建议按下面顺序做。

### 实验 A

只开：

\[
\mathcal{L}_{nmse\_logenv}
+
\lambda_g \mathcal{L}_{grad}
\]

目标：

- 先看结构边界是否改善
- 观察训练是否稳定

### 实验 B

再开：

\[
\mathcal{L}_{nmse\_logenv}
+
\lambda_c \mathcal{L}_{stat}
\]

目标：

- 单独判断统计项是否能改善 `cont`

### 实验 C

最后再上完整组合：

\[
\mathcal{L}_{nmse\_logenv}
+
\lambda_g \mathcal{L}_{grad}
+
\lambda_c \mathcal{L}_{stat}
\]

### 不建议

不建议一开始同时再叠：

- 强 TV
- 强 wavelet sparsity
- 复杂 Nakagami loss

否则变量太多，难判断是谁起作用。

## 8. 预期效果

如果方向对，通常会看到：

- `cont` 的 lesion/background 更可分
- 局部区域不像被抹平
- 边界更稳
- RF-NMSE 可能不一定显著大涨，但 envelope/visual quality 会更合理

如果方向不对，常见现象会是：

- `grad` 太大导致噪声边缘也被强化
- `stat` 太大导致整体亮度分布被硬拉，训练变慢

## 9. 一句话结论

对当前 `Lite2D + simu_cont`，最推荐先试的 loss 组合是：

- `nmse_logenv` 作为底座
- `gradient consistency` 保结构
- `local mean/variance statistics` 保 contrast 与 speckle 统计

这是当前最平衡、最容易落地、也最适合做第一轮验证的一套组合。
