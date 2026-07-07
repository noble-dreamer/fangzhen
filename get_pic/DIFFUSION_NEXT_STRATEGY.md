# Diffusion 精细缺陷反演下一步策略

本文档基于当前 `simple/f_domain` 频域仿真和 `simple/get_pic` 粗图管线，回答下一步是否继续递进 V2/V3 粗图、最终进入 diffusion 的数据形式、模型结构和参数规模建议。


## 1. 当前判断

结论先写清楚：

1. 下一步粗图主线应继续使用当前 V1 ray-tube 反投影。
2. V2 相干散射可以做成少量辅助通道，但不建议把它作为主线阻塞 diffusion。
3. V3 线性/TV/FISTA 粗反演暂时不建议继续投入大量实现。
4. 现在最重要的是形成 `pic(x), x -> label` 的训练闭环，验证 diffusion 是否能从粗图和原始频域测量中恢复更精细缺陷。

原因是：当前粗图在整个系统中的角色不是最终反演结果，而是物理先验。它只需要提供“缺陷大概在哪里、哪些区域观测可靠、哪些 tx-rx-frequency 发生异常”。最终边界、深度和形状细化应由 diffusion 从 label 监督中学习。


## 2. V1 是否足够作为粗图

当前 V1 已经具备做 `pic(x)` 的关键条件：

- 使用健康/损伤复频响差分，而不是单独 `abs(Hd)`。
- 使用展开管壁上的 tx-rx 螺旋 ray-tube。
- 支持 `helical_orders = -1, 0, 1`。
- 输出多通道粗图：
  - `ray_log_amp_loss`
  - `ray_relative_delta`
  - `ray_phase_change`
  - `ray_delta_abs`
  - `low_frequency_band_map`
  - `mid_frequency_band_map`
  - `high_frequency_band_map`
  - `path_coverage`
  - `valid_case_count`
  - `reliability_mask`
- 同时输出 `x_matrix`，保留原始 tx-rx-frequency 结构。
- label 只用于评价，不进入粗图生成。

这已经满足 diffusion 条件输入的最低物理要求。

V1 的缺点是粗、糊、条带状，形状不会像真实腐蚀 label。这不是致命问题。对 diffusion 来说，粗图如果过于“像 label”，反而可能诱导网络只学粗图后处理；V1 的模糊和条带能迫使模型同时利用 `x_matrix`。


## 3. 是否继续推进 V2

建议推进，但只做 V2-lite，不做完整 Born 反演。

### 3.1 推荐做的 V2-lite

只增加 2 到 3 个辅助通道：

```text
born_coherent_abs
born_coherent_real_positive
phase_consistency
```

其中 `c_phase(f)` 先用常数或低/中/高三段常数，不追求精确色散建模。

V2-lite 的作用不是替代 V1，而是给 diffusion 一个额外提示：哪些像素位置在多频、多路径相位补偿后更容易相干增强。

### 3.2 不建议现在做的 V2

暂时不要做：

- 完整 Green function Born kernel；
- 多模态色散分解；
- 模态转换建模；
- 依赖精确相速度的高置信相干反投影；
- 把 V2 指标作为唯一粗图评价目标。

当前频域模型、真实 PZT 片激励和实际管道实验之间仍有差异。V2 相干项如果校准不足，可能在 label 指标上不稳定，甚至误导网络。

### 3.3 V2 是否可能对最终 diffusion 帮助不大

有可能。

如果 diffusion 已经能看到：

```text
V1 多通道 pic(x)
x_matrix 中的 phase_cos / phase_sin / log_abs_reldelta
```

那么 V2 的相干通道本质上只是把 `x_matrix` 中的相位信息提前做了一次人工投影。若投影核不准，收益可能很小，甚至负收益。

因此 V2 应作为可消融通道：

```text
Model A: V1 pic + x_matrix
Model B: V1+V2-lite pic + x_matrix
```

只有当 Model B 在验证集上稳定提升，才保留 V2 通道。


## 4. 是否继续推进 V3

当前不建议把 V3 作为下一步重点。

V3 线性反演粗图，例如 SIRT/LSQR/TV/FISTA，理论上会比 V1 更像反演结果，但它有三个实际风险：

1. V3 会引入新的正则、迭代次数、非负约束、TV 权重等超参数。
2. V3 如果使用 label 调参，很容易变成间接 label 泄漏。
3. V3 可能把物理模型误差固化成“过于自信”的错误图，diffusion 反而更难纠正。

从信息论角度看，如果最终模型输入还包含 `x_matrix`，那么 V3 提供的很多信息已经存在于 `x_matrix` 中。V3 只是一个更强的人工压缩。人工压缩越强，错误先验越难被网络推翻。

因此 V3 当前更适合放到两个后续位置：

- 作为论文中的对照方法；
- 作为 diffusion 的可选物理一致性 forward operator，而不是作为主输入粗图。


## 5. 推荐下一步粗图策略

### 5.1 主线粗图

主线保持：

```text
V1 ray-tube backprojection
```

保留当前通道：

```text
ray_log_amp_loss
ray_relative_delta
ray_phase_change
ray_delta_abs
low_frequency_band_map
mid_frequency_band_map
high_frequency_band_map
path_coverage
valid_case_count
reliability_mask
```

其中进入 diffusion 的必要通道建议缩减为：

```text
ray_log_amp_loss
ray_relative_delta
ray_phase_change
low_frequency_band_map
mid_frequency_band_map
high_frequency_band_map
path_coverage
reliability_mask
```

`ray_delta_abs` 可以保留做消融。它容易被高幅值路径主导，不建议作为核心定位通道。

### 5.2 辅助粗图

只做 V2-lite：

```text
born_coherent_abs
phase_consistency
```

如果短期实现成本高，可以先不做 V2，直接训练 V1+`x_matrix` diffusion。

### 5.3 label 的使用

label 可以用于：

- 粗图评价；
- 选择粗图参数；
- 比较 V1、V1+V2-lite、V3；
- diffusion 监督训练目标。

label 不应用于：

- 生成粗图；
- 选频主指标；
- 针对单一样本调整粗图权重。

建议所有粗图评价都写成后验报告，和粗图生成脚本分离。


## 6. 最终进入 diffusion 的数据

建议每个样本包含三类输入。

### 6.1 图像条件 `pic`

来自：

```text
simple/get_pic/output2/coarse_maps/<sample>_coarse_maps.npz
```

字段：

```text
pic: (C_pic, 256, 256)
```

建议使用通道：

```text
0 ray_log_amp_loss
1 ray_relative_delta
2 ray_phase_change
3 low_frequency_band_map
4 mid_frequency_band_map
5 high_frequency_band_map
6 path_coverage
7 reliability_mask
```

如果 V2-lite 完成，再追加：

```text
8 born_coherent_abs
9 phase_consistency
```

当前粗图默认直接生成：

```text
256 x 256
```

如果验证有效，再用 `--grid-size 512` 重新生成高分辨率粗图并微调。

### 6.2 原始测量条件 `x_matrix`

来自：

```text
simple/get_pic/output2/x_matrix/<sample>_x_matrix.npz
```

字段：

```text
x: (C_x, F, TX, RX)
```

当前通道：

```text
log_abs_delta
log_abs_reldelta
phase_cos
phase_sin
healthy_log_abs
damaged_log_abs
valid_mask
```

建议保留全部。特别是：

- `log_abs_reldelta`：缺陷敏感；
- `phase_cos/phase_sin`：保留相位信息；
- `valid_mask`：避免网络把未完成频点当真实 0。

### 6.3 目标 `y`

来自：

```text
simple/f_domain/output2/streaming_dataset_a_frequency_shell/labels/<sample>_defect_depth_norm.npy
```

形状：

```text
y: 原始通常为 (512, 512)，进入训练时统一为 (256, 256)
```

当前 `evaluate_coarse_maps.py` 会在评价时把 label 最近邻下采样到粗图尺寸。训练数据加载器中也应保持同样规则：

```text
256 x 256
```

输出仍建议为连续厚度损失图，而不是二值 mask。


## 7. 推荐 diffusion 结构

### 7.1 不建议一开始用大模型

不建议使用 Stable Diffusion 级别的大 U-Net，也不建议引入大型 Transformer backbone。你的任务不是自然图像生成，而是单通道科学图像反演。数据量也不会像自然图像那样大。

可用算力：

- 1 张 A6000；
- 或 3 张 A5000。

这足够训练中小型 conditional diffusion，但不适合盲目上几亿参数模型。

### 7.2 推荐模型：Conditional DDPM / DDIM + 小型 U-Net

主模型：

```text
epsilon = UNet(
    noisy_y,
    timestep,
    condition_image = pic,
    condition_vector_or_tokens = encoder(x_matrix)
)
```

结构：

```text
Input:
  noisy_y: 1 channel
  pic:     8-10 channels

Image condition:
  concat(noisy_y, pic) at input

x_matrix condition:
  small DataEncoder -> embedding
  inject by FiLM / scale-shift in each UNet block
```

先不要上 cross-attention。FiLM 更稳、更省显存。

### 7.3 U-Net 推荐配置

第一版建议：

```text
image_size: 256
in_channels: 1 + C_pic
out_channels: 1
base_channels: 64
channel_mult: [1, 2, 4, 4]
num_res_blocks: 2
attention_resolutions: [32]
dropout: 0.05
time_embedding_dim: 256
data_embedding_dim: 256
diffusion_steps: 1000 train, 50-100 DDIM sample
prediction_type: epsilon 或 v_prediction
```

参数量大致：

```text
30M - 55M
```

这是一张 A6000 可以稳训的规模，A5000 单卡也能训练。三张 A5000 可用 DDP 加速，但不必为了并行而扩大网络。

如果数据量少于几百个样本，建议进一步缩小：

```text
base_channels: 48
channel_mult: [1, 2, 4, 4]
参数量约 18M - 35M
```

### 7.4 DataEncoder 推荐配置

`x_matrix` 形状大概率为：

```text
(7, F, 16, 16)
```

不要把它展平成超大向量直接接全连接。推荐轻量 3D/2D 编码：

```text
Input: (C_x, F, TX, RX)

Conv3D/Conv2D encoder:
  7 -> 32
  32 -> 64
  64 -> 128
global average pooling
MLP -> 256 dim embedding
```

等价实现可以把 `F` 当作深度，用 3D CNN；也可以 reshape 为：

```text
(C_x * F, TX, RX)
```

再用 2D CNN 编码。若频点数量固定且不大，2D CNN 更简单。

推荐第一版：

```text
x_reshape: (C_x * F, 16, 16)
Conv2D channels: 64, 128, 256
global pooling
MLP 256
```

参数量：

```text
1M - 3M
```

注入方式：

```text
每个 ResBlock:
  scale, shift = Linear(data_embedding)
  h = h * (1 + scale) + shift
```

也就是 FiLM 条件。


## 8. 训练阶段建议

### Stage A: 先不用 diffusion，做确定性 baseline

先训练一个小 U-Net：

```text
y_pred = UNetRegressor(pic, x_matrix)
loss = L1 + Dice/BCE optional
```

目的：

- 检查数据加载；
- 检查 `pic/x/y` 对齐；
- 检查模型是否能 overfit 10 个样本；
- 评估 V1 粗图是否足以提供定位。

参数量：

```text
10M - 25M
```

如果这个 baseline 都不能 overfit，直接上 diffusion 没意义。

### Stage B: 训练小 diffusion

使用：

```text
image_size = 256
base_channels = 48 或 64
batch_size = 4-16
mixed precision = fp16/bf16
EMA = true
```

损失：

```text
L_noise = MSE(eps_pred, eps)
L_aux = 0.1 * L1(x0_pred, y)
```

不要第一版就加复杂物理一致性损失。先让模型学会稳定重建 label。

### Stage C: 消融实验

必须做以下消融：

```text
1. diffusion(pic only)
2. diffusion(x_matrix only)
3. diffusion(V1 pic + x_matrix)
4. diffusion(V1+V2-lite pic + x_matrix)
5. deterministic UNet(V1 pic + x_matrix)
```

如果第 3 项比第 1/2 项都好，说明 `pic(x), x` 组合有效。

如果第 4 项没有明显提升，就不要继续加 V2/V3。

### Stage D: 再考虑 512

只有当 256 结果稳定后，再做：

```text
image_size = 512
base_channels = 48
gradient_checkpointing = true
batch_size = 1-4
```

512 的主要价值是边界更细，但训练成本明显提高。不要一开始就上 512。


## 9. 参数量与算力建议

### 9.1 A6000 单卡

推荐：

```text
image_size = 256
base_channels = 64
params = 30M - 55M
batch_size = 8 - 16, 视显存和 x_encoder 而定
```

512 微调：

```text
base_channels = 48
params = 20M - 40M
batch_size = 2 - 4
gradient_checkpointing = true
```

### 9.2 三张 A5000

推荐：

```text
DDP, 每卡 batch_size = 4 - 8
global batch_size = 12 - 24
base_channels = 64
```

不建议因为有 3 张卡就把模型扩大到 100M 以上。数据量和物理泛化比模型大小更关键。

### 9.3 参数量上限

建议上限：

```text
第一阶段: 20M - 50M
第二阶段: 50M - 80M
不建议超过: 100M
```

超过 100M 的风险：

- 对当前仿真样本量容易过拟合；
- 训练和调参周期明显变长；
- 对 V1 粗图误差的鲁棒性不一定更好；
- 真实管道域迁移更难。


## 10. 数据量建议

粗略建议：

```text
debug/overfit: 10 - 20 samples
first train:   100 - 300 samples
usable model:  500 - 2000 samples
```

如果频域样本生成成本仍高，优先扩大缺陷位置、大小、深度和噪声扰动，而不是盲目增加频点。

数据增强建议：

```text
theta circular shift
small z shift
amplitude scale jitter
phase noise jitter
frequency dropout
tx/rx dropout
Gaussian measurement noise
```

注意：`z` shift 不应越过 tx/rx ring 和管端边界；`theta` shift 可以严格周期平移。


## 11. 训练评价指标

不要只看 MSE。

建议保存：

```text
NRMSE
Pearson
mask IoU
top5_hit_rate
prediction_mass_in_label
centroid_error_mm
area_error
max_depth_error
```

对 diffusion 还要看：

```text
sample variance
mean prediction
uncertainty map
```

如果多次采样结果位置很不稳定，说明条件不足或模型过大。


## 12. 推荐执行顺序

### Step 1: 固化 V1 数据集

使用当前 `generate_coarse_maps.py` 输出：

```text
coarse_maps/<sample>_coarse_maps.npz
x_matrix/<sample>_x_matrix.npz
labels/<sample>_defect_depth_norm.npy
```

先统一下采样到 `256 x 256`。

### Step 2: 做 deterministic U-Net baseline

目标：

```text
UNet(pic + x_embedding) -> y
```

必须能 overfit 10 个样本。

### Step 3: 做 30M 级 diffusion

输入：

```text
noisy_y + V1 pic
x_matrix embedding by FiLM
```

输出：

```text
defect_depth_norm
```

### Step 4: 加 V2-lite 通道做消融

只比较是否提升验证集。

### Step 5: 再考虑 V3

只有在以下情况才推进 V3：

- V1+`x_matrix` 的定位误差很大；
- V2-lite 没有改善；
- 粗图评价显示 V1 的 `prediction_mass_in_label` 很低；
- 但原始 `x_matrix` 中确实存在强缺陷信息。

否则不要先做 V3。


## 13. 最终建议

当前最稳妥路线是：

```text
V1 ray-tube 粗图 + x_matrix
-> deterministic U-Net baseline
-> 30M-55M conditional diffusion
-> V2-lite 通道消融
-> 仅在必要时做 V3
```

不要把下一步主要工作放在 V3 粗图上。V3 可能让粗图更像一个传统反演结果，但对最终 diffusion 未必有明显收益。当前更有价值的是让模型同时看到：

```text
pic(x): 物理可解释空间先验
x:      未压缩的多频 tx-rx 复频响
y:      defect_depth_norm label
```

然后用消融实验判断 V2/V3 是否真的贡献增益。
