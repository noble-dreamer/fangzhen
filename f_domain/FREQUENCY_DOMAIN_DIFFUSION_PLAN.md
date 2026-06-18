# Frequency-Domain Coarse Map and Diffusion Plan

本文件记录 `simple` 壳模型后续频域数据生成、展开平板粗图、时域校准和 diffusion 细化缺陷图的规划。目标不是替代全部时域 COMSOL，而是把计算量更低的频域样本用于扩大训练集，同时保留少量时域样本作为物理时间特征的校准和验证。

## 1. 总体目标

目标输出是管外表面展开后的缺陷厚度损失图：

```text
target y = labels/<sample>_defect_depth_norm.npy
shape    = (z_index, theta_index)
value    = local thickness loss normalized by (h0 - h_min)
```

频域分支生成两类条件输入：

```text
x      = 原始多频复频响特征，保留 tx-rx-frequency 物理数据
pic(x) = 根据 x 反投影到 (z, theta) 展开平板上的粗图
```

训练形式应接近：

```text
fine_defect = diffusion(condition_image=pic(x), condition_data=x)
```

其中 `pic(x)` 给模型空间定位先验，`x` 给模型保留未被粗图压缩掉的多频、多通道物理信息。

## 2. 频域仿真数据 x

### 2.1 频域 COMSOL 物理设置

频域模型沿用当前 `simple_shell_common.py` 的几何、材料、壳厚缺陷、等效 PZT 面载荷和接收 patch 加权平均：

```text
receiver = intop_shell(w_rx * u_r) / intop_shell(w_rx)
u_r      = cos(theta_rx) * u + sin(theta_rx) * v
```

与时域不同的是，载荷不使用 `pztpulse(t)`，而是单位谐波等效面力：

```text
F(tx, f, theta, z) = F0 / pzt_A * window_tx(theta, z)
```

每个频点得到一个复响应：

```text
H(sample, tx, rx, f) = complex radial receiver response
```

输出建议保存为压缩数组，而不是只保存 CSV：

```text
frequency_response/<sample>_H_complex.npz
  H_real: shape (n_tx, n_rx, n_f)
  H_imag: shape (n_tx, n_rx, n_f)
  tx_indices
  rx_indices
  frequencies_hz
  metadata_json
```

### 2.2 频点选择

不要只选 base 时域信号 FFT 幅值最大的频点。最大幅值通常代表激励强或传播损耗小，不一定代表缺陷敏感。

先扫 25-90 kHz，每 2.5 或 5 kHz 一个，再按 sensitivity 排序选 6-10 个

第一阶段建议固定候选频点：

```text
30, 35, 40, 45, 50, 55, 60, 65, 70 kHz
```

后续根据少量健康/缺陷配对样本计算缺陷敏感度：

```text
sensitivity(f) = mean_samples ||H_damaged(f) - H_healthy(f)|| / (||H_healthy(f)|| + eps)
```

保留同时满足以下条件的频点：

- 健康响应不太弱，避免除噪声；
- 健康-损伤差分明显；
- 在多个缺陷样本上稳定；
- 不只依赖某一个异常 tx-rx 通道。

### 2.3 原始频域特征 x

对于每个样本，健康基准为：

```text
H0(tx, rx, f)
```

缺陷样本为：

```text
Hd(tx, rx, f)
```

建议构造以下特征：

```text
DeltaH    = Hd - H0
RelDeltaH = (Hd - H0) / (abs(H0) + eps)
PhaseDiff = angle(Hd * conj(H0))
```

`x` 可以包含：

```text
log_abs_delta     = log(abs(DeltaH) + eps)
log_abs_reldelta  = log(abs(RelDeltaH) + eps)
phase_cos         = cos(PhaseDiff)
phase_sin         = sin(PhaseDiff)
healthy_log_abs   = log(abs(H0) + eps)
damaged_log_abs   = log(abs(Hd) + eps)
```

数组形状建议统一为：

```text
x_matrix: (feature_channel, frequency, tx, rx)
```

这部分不要丢掉。即使 `pic(x)` 已经是空间粗图，`x_matrix` 仍然包含完整 tx-rx-frequency 结构，可作为 diffusion 的额外条件。

## 3. 管道展开平板粗图 pic(x)

### 3.1 坐标约定

与 `labels/` 保持一致，把管外表面展开为二维网格：

```text
horizontal axis = theta, periodic, 0 <= theta < 360 deg
vertical axis   = z, 0 <= z <= L_pipe
array shape     = (z_index, theta_index)
```

后续所有粗图、细图、label、评价都必须使用同一 `(z, theta)` 网格。

### 3.2 基础反投影

对每个候选像素 `p = (theta, z)`、每个 `tx-rx-f`，计算发射端到像素、像素到接收端的展开面路径长度：

```text
L_tx_p = sqrt((z - z_tx)^2 + (Rm * wrap(theta - theta_tx))^2)
L_p_rx = sqrt((z_rx - z)^2 + (Rm * wrap(theta_rx - theta))^2)
L_path = L_tx_p + L_p_rx
```

最简单的幅值反投影：

```text
pic_amp[p] += W(tx, rx, f, p) * abs(DeltaH(tx, rx, f))
```

其中权重可以先用：

```text
W = 1 / (L_tx_p * L_p_rx + eps)
```

并屏蔽离发射端和接收端过近的像素，避免局部奇异。

### 3.3 相位补偿反投影

如果有可靠的频散/波数模型，可以用相位补偿提高定位：

```text
k(f) = 2*pi*f / c_phase(f)
pic_complex[p] += W * DeltaH(tx, rx, f) * exp(1j * k(f) * L_path)
pic_phase[p] = abs(pic_complex[p])
```

第一阶段可以用常数速度近似：

```text
c_phase(f) ~= c_group ~= 2522 m/s
```

更好做法是用时域健康样本或 COMSOL/Safe 色散结果校准 `c_phase(f)` 和 `c_group(f)`。

### 3.4 多通道粗图

不要只生成一张单通道粗图。建议 `pic(x)` 至少包含：

```text
pic_amp_delta       = backprojection(log_abs_delta)
pic_amp_reldelta    = backprojection(log_abs_reldelta)
pic_phase_coherent  = abs(coherent phase-compensated backprojection)
pic_phase_diff_var  = phase-difference consistency map
pic_path_coverage   = number/weight sum of paths crossing each pixel
```

如果频率较多，也可以按低/中/高频分组：

```text
low:  30-40 kHz
mid:  45-55 kHz
high: 60-70 kHz
```

形成：

```text
pic(x): (coarse_channel, z, theta)
```

粗图不必非常清晰。它的作用是给 diffusion 一个物理上合理的空间先验，而不是直接作为最终结果。

## 4. 时域数据的作用

稀疏频域扫描没有时间轴，不能直接得到 Γ0、Γ1、Γ-1 的到达时间。因此时域数据主要用于校准、验证和少量高保真参考。

### 4.1 校准速度和路径窗

使用 `csv/tomography_features/*_helical_order_projections.csv`：

```text
predicted_arrival_s
order_peak_time_s
order_peak_amplitude_m
helical_order = -1, 0, 1
```

调节：

```text
group_velocity
window_us
c_phase(f)
c_group(f)
```

目标是让健康样本和少量缺陷样本中 Γ0/Γ±1 窗口峰值稳定落在预测窗口内。

### 4.2 校准频域粗图参数

用少量同一缺陷样本的时域与频域配对数据，选择能让粗图更接近真实标签的参数：

```text
frequency set
path weighting W
phase velocity c_phase(f)
path family selection: direct / wrap +1 / wrap -1
normalization method
```

评价指标直接对 `pic(x)` 和 label 计算：

```text
Pearson correlation
NRMSE
mask IoU after thresholding
top-k localization distance
```

### 4.3 验证频域样本是否物理可信

对少量样本同时运行时域和频域：

- 频域粗图高响应区域应与时域健康-损伤差分路径异常一致；
- 频域预测的缺陷位置应与 Γ0/Γ±1 窗口峰值变化相关；
- 若频域图只在发射端/接收端附近发亮，而 label 在中部缺陷处，应调整归一化、路径权重或频点。

## 5. Diffusion 训练规划

### 5.1 数据项

每个训练样本包含：

```text
y        = labels/<sample>_defect_depth_norm.npy
pic      = coarse_maps/<sample>_coarse_maps.npz
x_matrix = frequency_response/<sample>_features.npz
meta     = metadata/<sample>.json
```

建议训练前把所有输入归一化：

```text
y:       [0, 1]
pic:     per-channel robust normalization, e.g. percentile 1-99
x_matrix: per-feature z-score or robust log scaling
```

### 5.2 模型形式

推荐条件扩散：

```text
noise target:
  y_t = sqrt(alpha_t) * y + sqrt(1-alpha_t) * noise

model:
  eps_pred = UNet(y_t, t, image_condition=pic, data_condition=x_matrix)
```

`pic` 的注入方式：

- 直接与 `y_t` 在通道维拼接；
- 或通过 ControlNet/adapter 分支提取多尺度图像条件。

`x_matrix` 的注入方式：

- 用小型 CNN/Transformer/MLP 编码 `x_matrix`；
- 得到全局 tokens 或 embedding；
- 通过 cross-attention 或 FiLM/scale-shift 注入 UNet 各层。

概念上就是：

```text
z_img  = ImageEncoder(pic)
z_data = DataEncoder(x_matrix)
eps    = DenoisingUNet(y_t, t, z_img, z_data)
```

### 5.3 为什么同时输入 pic(x) 和 x

`pic(x)` 已经做过物理反投影，空间结构强，适合告诉模型“缺陷可能在哪里”。

`x` 保留完整测量矩阵，包含：

- 哪些 tx-rx 对异常；
- 哪些频率异常；
- 幅值和相位差是否一致；
- 粗图反投影时可能被平均掉的方向性信息。

因此训练目标不是：

```text
y = diffusion(pic)
```

而是：

```text
y = diffusion(pic(x), x)
```

这能减少粗图算法错误造成的信息损失。

### 5.4 损失函数和评价

基础 diffusion loss：

```text
L_noise = MSE(eps_pred, eps)
```

可选辅助损失，在预测干净图 `y_pred` 上计算：

```text
L_l1      = mean(abs(y_pred - y))
L_mask    = BCE(y_pred > threshold, y_mask)
L_grad    = edge/gradient consistency
L_physics = compare coarse forward projection of y_pred with observed x
```

第一阶段可以只用 diffusion loss + L1。等输入输出跑通后，再加物理一致性约束。

评价指标：

```text
Pearson correlation
NRMSE
mask IoU
defect centroid error in (z, theta)
area error
max depth error
```

## 6. 建议实施阶段

### Stage 0: 频域单样本验证

- 健康管 `tx=1, f=50 kHz` 已验证可跑通。
- 下一步扩展到 `tx=1..16, f=30..70 kHz`。
- 输出 `H_real/H_imag`，确认 16 receiver 全部 finite/nonzero。

### Stage 1: 健康-缺陷配对频域数据

- 生成 1 个健康基准 `H0`。
- 生成 5-10 个缺陷样本 `Hd`。
- 计算 `DeltaH/RelDeltaH/PhaseDiff`。
- 保存 `x_matrix`。

### Stage 2: 展开平板粗图

- 实现基础幅值反投影。
- 实现相位补偿反投影。
- 与 `labels/*_defect_depth_norm.npy` 对齐网格。
- 用 Pearson/NRMSE/IoU 筛选频点和权重。

### Stage 3: 时域校准

- 对少量样本运行当前时域流式脚本。
- 使用 `helical_order_projections.csv` 校准 `group_velocity/window_us`。
- 用时域异常路径验证频域粗图是否物理合理。

### Stage 4: Diffusion 训练

- 数据输入：`pic(x)` 多通道粗图 + `x_matrix` 多频复响应特征。
- 目标：`labels/*_defect_depth_norm.npy`。
- 先训练小模型验证 overfit 10 个样本。
- 再扩大频域样本量训练。
- 最后用少量时域样本和 label 做独立验证。

## 7. 风险和注意事项

- 单频 `abs(H)` 很容易被入射场主导，不建议作为粗图。
- 频域粗图应优先使用健康-损伤差分或相对差分。
- 若 Dataset B 有材料、位置、幅值扰动，健康基准必须与扰动配置匹配；不能随意用 Dataset A 健康基准相减。
- 绝对幅值不可靠时，优先使用归一化差分、相位差和通道内标准化。
- 粗图坐标必须与 `labels/` 完全一致，否则 diffusion 会学习错误映射。
- 时域 Γ0/Γ±1 是校准和解释工具，不是稀疏频域扫描能直接输出的量。
