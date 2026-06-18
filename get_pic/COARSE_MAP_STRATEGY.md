# Coarse Map Strategy for Physics-Guided Diffusion

本文档记录 `get_pic/` 后续生成粗图 `pic(x)` 的策略。目标不是把 `tx-rx-frequency` 数据简单插值成图，而是先用导波传播几何、健康-损伤频响差分、相位一致性和路径覆盖度生成一个物理可解释的粗图，再把粗图和原始测量矩阵一起输入 diffusion。

## 0. 2026-06-07 审查修订结论

当前策略的大方向合理：频域批量数据应以 `H(tx, rx, f)` 的健康-损伤复差分为核心，用展开管壁路径反投影生成多通道粗图，并把粗图和原始 `x_matrix` 同时作为 diffusion 条件输入。

需要修正和固化的点：

- 粗图生成必须以 `completed_mask` 为有效工况掩码。频域 `.npz` 可能预分配完整频率轴，但未完成频点中 `H_real/H_imag` 为 `NaN`；健康和损伤样本都完成且有限的 `tx-frequency` 才能参与投影。
- 第一版不要把 `abs(Hd)` 或单频幅值图当作粗图。必须使用 `DeltaH`、`RelDeltaH`、`LogRatio`、`PhaseDiff` 这类健康-损伤配对特征。
- V1 先落地为覆盖度归一化的 ray-tube 反投影；V2 相干散射只作为附加通道，不能作为唯一定位依据，因为当前相速度/波数尚未精确校准。
- Dataset A 可按规则 16 等分阵列重建几何；Dataset B 或实验数据必须从 metadata/外部通道表读取实际 PZT 位置和通道增益，不能默认套用 Dataset A 理想阵列。
- `path_coverage`、`valid_mask`、`reliability_mask` 是必需输出通道，diffusion 需要区分“不可观测/低覆盖”和“确定无缺陷”。
- 标签只能用于评价和调参，不能参与粗图生成。输出坐标必须严格对齐 `labels/*_defect_depth_norm.npy` 的 `(z_index, theta_index)`、`theta` 周期和 `z` 方向。
- 时域数据的主要作用是校准 `group_velocity`、路径窗、相位展开和有效 `c_phase(f)`，不是替代大规模频域粗图输入。

因此 `get_pic` 第一阶段应先实现一个可复现的离线 Python 管线，输入健康/损伤频域 `.npz` 与 metadata，输出 `coarse_maps/*.npz` 和 `x_matrix/*.npz`；MATLAB 暂时用于交叉可视化和人工 sanity check，不作为主生产路线。

推荐训练形式：

```text
fine_defect = diffusion(pic(x), x)
```

其中：

```text
pic(x): 物理反投影后的多通道展开图，shape = (coarse_channel, z, theta)
x:      原始或轻度预处理后的 tx-rx-frequency 复频响特征
y:      labels/<sample>_defect_depth_norm.npy，shape = (z_index, theta_index)
```

## 1. 设计原则

粗图应保留尽可能多的物理含义：

- 使用健康-损伤差分，而不是单个缺陷样本的 `abs(H)`。
- 使用复数信息，包括幅值差、相位差和相干叠加，不只用幅值。
- 使用展开管壁上的传播路径和螺旋绕行阶次，不把接收矩阵当普通图像。
- 每张粗图都输出覆盖度/可靠性通道，避免 diffusion 把低覆盖区域误学成缺陷。
- 粗图只负责给出缺陷空间先验，不要求直接达到标签精度；细节由 diffusion 学习，但 diffusion 要能看到原始 `x_matrix`。

## 2. 需要的数据

### 2.1 频域数据，生成粗图的核心输入

来自：

```text
simple/f_domain/output/streaming_dataset_a_frequency_shell/frequency_response/<sample>_H_complex.npz
```

字段：

```text
H_real: shape (n_tx, n_rx, n_f)
H_imag: shape (n_tx, n_rx, n_f)
completed_mask: shape (n_tx, n_f)
tx_indices
rx_indices
frequencies_hz
sample_id
dataset
defect_state
```

需要同时有：

```text
H0 = healthy baseline
Hd = damaged sample
```

然后构造：

```text
DeltaH       = Hd - H0
RelDeltaH    = (Hd - H0) / (abs(H0) + eps)
ComplexRatio = Hd / (H0 + eps_complex)
LogRatio     = log(abs(Hd) + eps) - log(abs(H0) + eps)
PhaseDiff    = angle(Hd * conj(H0))
```

`ComplexRatio/PhaseDiff` 用于相位一致性，`LogRatio/RelDeltaH` 用于幅值和衰减型投影。

### 2.2 几何数据

来自当前 `simple_shell_common.py` 和每个样本 metadata：

```text
pipe.length_mm        = 1000
pipe.mid_radius_mm    = 155
transducer.tx_z_mm    = 100
transducer.rx_z_mm    = 900
transducer.count      = 16 per ring
tx indices            = 1..16
rx indices            = 17..32
```

发射/接收位置由代码中的 `transducer_positions()`、`transmitter_positions()`、`receiver_positions()` 定义。Dataset B 如果有位置扰动或幅值扰动，必须从对应 metadata 读取实际位置，不能直接套 Dataset A 的理想位置。

### 2.3 标签数据，只用于训练、评价和参数校准

来自：

```text
labels/<sample>_defect_depth_mm.npy
labels/<sample>_defect_depth_norm.npy
labels/<sample>_defect_mask.npy
labels/<sample>_defect_label_metadata.json
```

标签坐标：

```text
coordinate = unfolded outer pipe surface, theta-z
shape      = (z_index, theta_index)
theta      = 0 <= theta < 360 deg, periodic
z          = 0 <= z <= L_pipe
default    = 512 x 512
```

生成粗图时不能使用标签；标签只用于：

- 选择粗图算法参数；
- 评价 Pearson/NRMSE/IoU；
- diffusion 监督训练目标。

### 2.4 时域数据，用于校准速度和物理解释

来自：

```text
csv/tomography_features/<sample>_helical_order_projections.csv
csv/tomography_features/<sample>_tomography_features.csv
csv/waveforms/<sample>_txXX_fYYYYYHz_waveforms.csv
```

当前时域特征包含：

```text
tof_first_s
hilbert_peak_amplitude_m
hilbert_peak_time_s
fft_amplitude_m
fft_phase_rad
helical_order = -1, 0, 1
predicted_arrival_s
order_peak_amplitude_m
order_peak_time_s
```

这些数据不直接参与大规模频域粗图生成，但用于校准：

```text
c_group
c_phase(f)
helical order path family
phase unwrap rule
frequency weighting
```

## 3. 坐标与路径模型

把圆柱壳展开成平板：

```text
s = Rm * theta_rad
p = (theta, z)
```

任意两个点 `a, b` 在螺旋阶次 `m` 下的展开路径长度：

```text
dtheta_m = wrap(theta_b - theta_a + 360*m)
L_m(a,b) = sqrt((z_b - z_a)^2 + (Rm * dtheta_m_rad)^2)
```

对每个候选像素 `p`、发射端 `tx`、接收端 `rx`、频率 `f`：

```text
L_tx_p = L_m1(tx, p)
L_p_rx = L_m2(p, rx)
L_scat = L_tx_p + L_p_rx
L_ray  = L_m(tx, rx)
```

螺旋阶次建议先使用：

```text
m = -1, 0, 1
```

这和当前时域 helical order 特征一致。后续如果发现绕行信号有效，再扩展到 `m = -2, 2`。

## 4. 推荐算法

### 4.1 V1: 覆盖度归一化的螺旋路径反投影

这是第一版最稳妥的粗图算法，计算量低，物理含义清楚。

对每个 `tx-rx-f-order`，把观测异常投影到该路径附近的像素。路径核使用展开平板上的 tube kernel：

```text
K_ray(p | tx, rx, m) = exp(-d_perp(p, ray_m)^2 / (2*sigma_ray^2)) * aperture / (L_ray + eps)
```

其中：

```text
d_perp: 像素到 tx-rx 螺旋直线的垂直距离
sigma_ray: 路径宽度，建议 0.4~0.8 个波长或 PZT patch 尺寸上限
aperture: 屏蔽离 tx/rx 过近、超出检测区域或低信噪比的点
```

观测量：

```text
g_amp   = log(abs(H0) + eps) - log(abs(Hd) + eps)
g_rel   = abs(Hd - H0) / (abs(H0) + eps)
g_phase = unwrap(angle(Hd * conj(H0)))
```

投影：

```text
M_amp[p]   += w_f * w_trx * K_ray[p] * positive(g_amp)
M_rel[p]   += w_f * w_trx * K_ray[p] * g_rel
M_phase[p] += w_f * w_trx * K_ray[p] * positive(abs(g_phase))
C[p]       += w_f * w_trx * K_ray[p]
```

归一化：

```text
pic_amp   = M_amp / (C + eps)
pic_rel   = M_rel / (C + eps)
pic_phase = M_phase / (C + eps)
coverage  = C / percentile(C, 99)
```

物理意义：

- 腐蚀减薄会改变导波传播速度、散射和透射幅值；
- `tx-rx` 异常应该主要沿对应传播路径附近贡献；
- 多角度、多频率、多发射接收组合叠加后，真实缺陷区域会被多条异常路径交叉增强。

V1 的优点是稳健；缺点是分辨率受 ray tube 宽度限制，容易把缺陷拉成长条。

### 4.2 V2: Born/Rytov 风格的相干散射反投影

为了让粗图更物理、更适合 diffusion 加约束，第二版应加入相位补偿的散射核。

使用弱散射近似，把健康-损伤复差分看作缺陷散射贡献：

```text
DeltaH(tx,rx,f) ~= integral K_scat(tx,rx,f,p) * q(p) dp
```

其中 `q(p)` 是与厚度损失、局部波速扰动或散射强度相关的未知场。

散射路径：

```text
L_scat(p) = L(tx,p) + L(p,rx)
k(f)      = 2*pi*f / c_phase(f)
```

相干反投影：

```text
B_complex[p] += w_f * w_trx * DeltaH(tx,rx,f) * exp(1j*k(f)*L_scat(p)) * A(p)
C_complex[p] += w_f * w_trx * abs(A(p))
pic_coherent[p] = abs(B_complex[p]) / (C_complex[p] + eps)
```

`A(p)` 是几何扩散和接收可靠性权重：

```text
A(p) = 1 / sqrt((L_tx_p + eps) * (L_p_rx + eps))
```

如果直接用 `DeltaH` 相位不稳定，可以改用归一化复差分：

```text
DeltaH_norm = (Hd - H0) / (abs(H0) + eps)
```

或 Rytov 型观测：

```text
g_rytov = log(Hd / (H0 + eps_complex))
```

V2 的关键不是“相位一定完全正确”，而是让正确位置在多频、多 tx-rx 下相位更一致，错误位置相互抵消。

### 4.3 V3: 有约束的线性反演粗图

当 V1/V2 能稳定运行后，建议把粗图生成升级为显式反问题：

```text
g = A q + noise
```

其中：

```text
g: 由 tx-rx-f 观测组成的向量，可包含幅值衰减、相位延迟、复差分实部/虚部
A: 路径核或散射核矩阵
q: 展开图上的非负缺陷强度/厚度损失代理
```

优化问题：

```text
min_q ||W(Aq - g)||_2^2 + lambda_tv * TV(q) + lambda_l2 * ||q||_2^2
s.t.  q >= 0
```

可选解法：

```text
SIRT / ART:      易实现，适合稀疏路径层析粗图
LSQR + Tikhonov: 稳定，适合较大线性系统
FISTA + TV:      可加入边缘平滑和非负约束
```

V3 适合作为 diffusion 前的“物理先验粗反演”。它不需要达到 FWI 精度，但能提供更明确的物理一致性项：

```text
L_forward = ||A * y_pred - g||_1 or ||A * y_pred - g||_2
```

这就是后续在 diffusion 阶段引入物理制约的接口。

## 5. 输出粗图通道

建议 `get_pic` 最终输出：

```text
coarse_maps/<sample>_coarse_maps.npz
```

字段：

```text
pic: shape (channel, z, theta)
channel_names
theta_deg
z_mm
coverage
frequency_hz
tx_indices
rx_indices
algorithm_config_json
source_healthy_npz
source_damaged_npz
```

第一阶段通道建议：

```text
0  ray_log_amp_loss
1  ray_relative_delta
2  ray_phase_change
3  coherent_scatter_abs
4  coherent_scatter_real_positive
5  phase_consistency
6  low_frequency_band_map
7  mid_frequency_band_map
8  high_frequency_band_map
9  path_coverage
10 reliability_mask
```

`path_coverage` 和 `reliability_mask` 必须保留。diffusion 需要知道哪些区域是“没有足够路径照射”，而不是“确定没有缺陷”。

同时保存原始数据条件：

```text
x_matrix/<sample>_x_matrix.npz
```

建议字段：

```text
x: shape (feature_channel, frequency, tx, rx)
feature_names:
  log_abs_delta
  log_abs_reldelta
  phase_cos
  phase_sin
  healthy_log_abs
  damaged_log_abs
  completed_mask
```

## 6. 频率和权重

频点选择使用 `f_domain/select_sensitive_frequencies.py` 输出的 top-N 频点：

```text
frequency_sensitivity_top15_frequencies.txt
```

粗图中每个频率的权重：

```text
w_f = sensitivity(f) * healthy_reliability(f) * band_balance(f)
```

其中：

```text
sensitivity(f): 来自健康/缺陷样本的平均相对差分
healthy_reliability(f): 健康响应幅值不能过低
band_balance(f): 避免某一频段权重过大
```

如果还没有 sensitivity 文件，初始建议：

```text
20-100 kHz, step=5 kHz 粗筛
按 sensitivity 选 15 个
必要时对高 sensitivity 区间 step=2.5 kHz 加密
```

## 7. 速度和相位校准

V1 可以不依赖精确相速度，只需要路径几何。

V2/V3 需要 `c_phase(f)` 或有效波数 `k(f)`。初始可用：

```text
c_phase(f) ~= c_group ~= 2522 m/s
```

这是当前时域特征默认使用的 group velocity，只能作为初始近似。

更推荐用已有健康时域/频域数据校准：

1. 对每个高信噪比 `tx-rx-order`，用几何路径长度 `L_order` 和健康频响相位 `angle(H0(f))`。
2. 对频率做相位展开 `unwrap(angle(H0(f)))`。
3. 拟合：

```text
phase(f) ~= -2*pi*f*L_order/c_phase(f_band) + phase0
```

4. 分低/中/高频段估计有效 `c_phase`。
5. 用时域 `helical_order_projections.csv` 检查对应 order 的预测到达时间是否与窗口峰值一致。

如果相位展开不稳定，V2 的相干通道只作为辅助，不作为唯一粗图。

## 8. Diffusion 阶段如何引入物理制约

粗图生成阶段要保存 `A` 的隐式配置或可重建配置：

```text
tx/rx positions
frequency_hz
kernel type
sigma_ray
c_phase(f)
helical_orders
normalization
```

训练 diffusion 时可以加入：

### 8.1 输入制约

```text
condition_image = pic(x)
condition_data  = x_matrix
```

让模型同时看到空间粗图和未压缩的物理测量。

### 8.2 输出形状先验

腐蚀标签本身是平滑厚度损失场，可加：

```text
L_tv     = TV(y_pred)
L_nonneg = penalty(y_pred < 0)
L_range  = penalty(y_pred > 1)
```

### 8.3 前向一致性

把 diffusion 输出的 `y_pred` 通过同一个路径核投影回观测空间：

```text
g_pred = A * y_pred
L_phys = ||normalize(g_pred) - normalize(g_obs)||_1
```

第一阶段可以只对 V1 的 ray attenuation 做前向一致性；后续再加入 V2 的复数相干项。

### 8.4 覆盖度加权损失

对粗图覆盖度低的区域降低强约束，避免模型被不可观测区域误导：

```text
L_image = coverage_weight * |y_pred - y|
```

## 9. 评价指标

粗图本身要先评价，不要直接进入 diffusion：

```text
Pearson(pic_channel, label)
NRMSE(pic_channel, label)
mask IoU
top-k localization distance
centroid error in z/theta
coverage-weighted false positive rate
```

调参顺序：

1. 固定频点，调 `sigma_ray` 和归一化。
2. 固定 V1，加入 helical order `-1,0,1`。
3. 加入 sensitivity 频率权重。
4. 加入 V2 相干散射通道。
5. 如果粗图稳定，再做 V3 SIRT/TV。

## 10. 来源依据

### 10.1 当前代码依据

- `simple_shell_common.py`
  - 管道为圆柱壳中面：`Rm = 155 mm`。
  - PZT 被等效为 shell face-load window。
  - receiver 为 `intop_shell(w_rx*u_r)/intop_shell(w_rx)` 小面积加权平均。
  - tx/rx 位置由 `transducer_positions()` 生成，tx 在 `z=100 mm`，rx 在 `z=900 mm`，每圈 16 个。

- `f_domain/frequency_domain_common.py`
  - 频域导出的是复响应 `H(tx, rx, f)`。
  - NPZ 保存 `H_real/H_imag/completed_mask/tx_indices/rx_indices/frequencies_hz`。
  - README 已定义 `DeltaH`、`RelDeltaH`、`PhaseDiff` 作为频域粗图输入。

- `defect_label_common.py`
  - label 是展开外表面 `theta-z` 图。
  - 默认 shape 为 `512 x 512`。
  - 缺陷标签是厚度损失 `depth_mm/depth_norm`，适合作为 diffusion 目标。

- `streaming_export_common.py`
  - 当前时域 helical order 使用 `(-1, 0, 1)`。
  - 默认 group velocity 为 `2522 m/s`。
  - 已导出 `helical_order_projections.csv`，可用于校准路径速度和窗口。

### 10.2 文献依据

- Huthwaite 和 Simonetti 的 guided-wave tomography 工作指出，可利用 Lamb/guided wave 的色散关系，把壁厚变化转换为速度变化，再进行厚度图重建；其 high-resolution 方法结合 travel-time 和 diffraction tomography，支持本文 V1/V2 的分层设计。
- Huthwaite 的 improved scattering model 说明，简单声学速度扰动模型不足以完全描述 guided wave scattering；因此这里不把 V1 ray map 当最终真值，而是保留 V2/V3 和 diffusion 的物理一致性接口。
- pipe helical-mode tomography 研究说明，管道中利用绕行螺旋模式可以增加入射角覆盖，提高成像分辨率；这支持在展开管壁上使用 `m=-1,0,1` 路径族。
- RAPID/概率重建类方法说明，多 transducer、多频率 guided-wave 数据可以形成缺陷概率/严重度图；这支持 V1 的 coverage-normalized multi-path backprojection 作为稳健初版。
- FWI 类 guided-wave tomography 表明，全波形反演可以更准确重建厚度，但计算和建模成本更高；本文把 V3 只作为粗图反演和 diffusion 物理损失接口，而不一开始就做完整 FWI。

参考链接：

- High-resolution guided wave tomography: https://doi.org/10.1016/j.wavemoti.2013.04.004
- Guided wave tomography with an improved scattering model: https://pmc.ncbi.nlm.nih.gov/articles/PMC5134316/
- Guided wave tomography of pipes with high-order helical modes: https://doi.org/10.1016/j.ndteint.2014.03.010
- Guided-wave tomographic imaging of defects in pipe using RAPID: https://pure.psu.edu/en/publications/guided-wave-tomographic-imaging-of-defects-in-pipe-using-a-probab
- Full waveform inversion guided wave tomography accuracy: https://doi.org/10.1016/j.jsv.2017.04.017

## 11. 第一版落地建议

先实现 V1，不要直接上复杂反演：

```text
input:
  healthy H0 NPZ
  damaged Hd NPZ
  selected frequencies
  metadata/geometry

output:
  pic channels:
    ray_log_amp_loss
    ray_relative_delta
    ray_phase_change
    path_coverage
  x_matrix:
    log_abs_delta
    log_abs_reldelta
    phase_cos
    phase_sin
```

等 V1 能稳定生成位置大致正确的粗图后，再加入：

```text
V2 coherent_scatter_abs
V3 SIRT/TV coarse inversion
diffusion forward consistency loss
```
