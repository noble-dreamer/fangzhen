# get_pic 行动目标与代码路线

本文件是对 `simple/`、`simple/f_domain/`、现有输出和各个 Markdown 的审查后形成的落地目标。当前阶段只规划 `get_pic`，不修改任何 COMSOL 仿真代码。

## 1. 审查结论

`COARSE_MAP_STRATEGY.md` 的总体思路合理，但需要更工程化地约束第一版实现。

合理部分：

- 频域数据确实应作为大规模数据集主来源。`f_domain` 输出的 `H(tx, rx, f)` 是紧凑复响应，比保存时域全场或大量时域 CSV 更适合批量生成。
- 粗图不应直接由 `abs(H)` 插值得到，而应使用健康-损伤配对差分：`DeltaH`、`RelDeltaH`、`LogRatio`、`PhaseDiff`。
- 展开坐标采用和标签一致的 `(z_index, theta_index)` 是正确的；标签 `*_defect_depth_norm.npy` 可直接作为 diffusion 目标。
- 同时输入 `pic(x)` 和 `x_matrix` 是必要的。`pic(x)` 提供空间先验，`x_matrix` 保留 tx-rx-frequency 中的方向性、多频和相位信息。

必须修正/明确的部分：

- `.npz` 中的 `completed_mask` 必须参与所有计算。当前健康样本的频率轴可有 33 个点，但只完成的工况有效，未完成项为 `NaN`。
- Dataset A 可重建理想 16 等分几何；Dataset B/实验数据必须读取实际通道位置或扰动参数，不能默认理想阵列。
- 第一版相干反投影不能强依赖未校准的相速度。先把 V1 ray-tube 覆盖度归一化做好，再把 V2 coherent map 作为辅助通道。
- 粗图必须输出覆盖度/可靠性通道，避免模型把低覆盖区域误判为无缺陷。
- 标签只能用于评价和参数选择，不能在粗图生成中泄漏。

## 2. 第一阶段目标

先完成一个纯离线 Python 管线：

```text
input:
  healthy_H_complex.npz
  damaged_H_complex.npz
  healthy metadata json
  damaged metadata json
  optional selected frequency txt/csv

output:
  coarse_maps/<sample>_coarse_maps.npz
  x_matrix/<sample>_x_matrix.npz
  previews/<sample>_coarse_preview.png
  reports/<sample>_coarse_metrics.json  # 只有传入 label 时生成
```

核心验收标准：

- 输入频域样本缺健康、缺损伤、频率/tx/rx 不匹配、有效工况为空时，脚本直接报错，不输出伪结果。
- 输出 `pic` 的空间 shape 与 label 完全一致，默认 `(channel, 512, 512)`。
- `path_coverage` 非零且无 `NaN`；所有图像通道有限。
- 在有标签的少量样本上，能计算 Pearson、NRMSE、mask IoU 和缺陷质心误差。

## 3. 推荐文件结构

后续只在 `simple/get_pic/` 下新增代码：

```text
simple/get_pic/
  ACTION_PLAN.md
  COARSE_MAP_STRATEGY.md
  coarse_map_common.py          # 读写、几何、特征构造、归一化
  generate_coarse_maps.py       # 单样本/批量入口
  evaluate_coarse_maps.py       # 粗图与 label 指标
  preview_coarse_maps.py        # PNG 可视化
  configs/
    dataset_a_v1.json
```

第一版先不引入训练代码，也不调用 COMSOL。

## 4. V1 粗图算法

V1 使用覆盖度归一化 ray-tube 反投影。

数据特征：

```text
H0 = healthy H
Hd = damaged H
valid = completed0 & completedd & finite(H0) & finite(Hd)

DeltaH    = Hd - H0
RelDeltaH = abs(DeltaH) / (abs(H0) + eps)
LogRatio  = log(abs(H0) + eps) - log(abs(Hd) + eps)
PhaseDiff = angle(Hd * conj(H0))
```

空间投影：

```text
for tx, rx, frequency, helical_order in valid cases:
    ray = unfolded line from tx to rx with order m
    K   = exp(-d_perp^2 / (2*sigma_ray^2)) / (L_ray + eps)
    map += feature(tx, rx, f) * K
    coverage += K

pic = map / (coverage + eps)
```

默认 helical order：

```text
[-1, 0, 1]
```

第一版输出通道：

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

## 5. V2/V3 后续目标

V2：相干散射反投影。

- 输入 `DeltaH` 或 Rytov 型 `log(Hd / H0)`。
- 使用 `k(f)=2*pi*f/c_phase(f)` 做路径相位补偿。
- `c_phase(f)` 先用常数 2522 m/s 起步，只作为辅助通道。
- 等时域/实验校准完成后，再按频段更新 `c_phase(f)`。

V3：线性反演粗图。

- 构造 `g = A q` 的路径核。
- 先实现 SIRT/ART 或 LSQR + Tikhonov。
- TV/FISTA 和 diffusion 物理一致性损失放到后续阶段。

## 6. Python 与 MATLAB 分工

主生产路线选 Python。

原因：

- 当前频域输出是 `.npz/.npy/.json/.csv`，Python 直接读取最自然。
- 已可用 `numpy/scipy/matplotlib/skimage/pandas`，足够完成 V1/V2、评价和预览。
- diffusion 训练大概率也在 Python/PyTorch 中完成，数据格式连续。

MATLAB 的角色：

- 快速画图、检查粗图和旧 `generate_defect/show_defect.png` 风格是否一致。
- 用矩阵脚本验证某些反投影公式。
- 处理实验示波器数据时可作为可视化和信号处理辅助，但最终应导出为 Python 管线可读的 `.npz/.csv`。

当前建议环境：

```text
conda env: get_pic
Python: 3.10
required: numpy, scipy, matplotlib, pandas
optional: scikit-image
COMSOL/mph: get_pic 不需要
MATLAB: 可选验证，不作为主依赖
```

## 7. 时域数据如何用于精确校准

真实实验中示波器和信号发生器首先得到的是时域信号，所以时域数据应作为校准层，而不是被忽略。

### 7.1 从时域得到可校准频响

对实验或时域仿真波形：

```text
s_ij(t) -> 加窗/去基线/同步 -> FFT -> H_ij(f)
```

建议步骤：

- 用发射 trigger 或健康直达波做时间零点校正。
- 对每个 `tx-rx` 做相同窗函数和频率采样。
- 取和频域仿真一致的频点，例如 30-70 kHz 或 sensitivity 选出的频点。
- 用健康-损伤差分或比值进入和频域仿真相同的 `get_pic` 管线。

### 7.2 校准群速度和路径窗

用 `csv/tomography_features/*_helical_order_projections.csv`：

```text
predicted_arrival_s
order_peak_time_s
order_peak_amplitude_m
helical_order
```

调节：

```text
group_velocity
window_us
helical_order set
```

目标是健康样本中 Γ0/Γ±1 的峰值稳定落在预测窗口内；缺陷样本中异常路径和缺陷位置有对应关系。

### 7.3 校准相速度/相位展开

对健康时域 FFT 或频域健康响应，选高信噪比路径：

```text
unwrap(angle(H0_ij(f))) ≈ -2*pi*f*L_order/c_phase + phase0
```

按低/中/高频段拟合有效 `c_phase(f)`，供 V2 相干反投影使用。若相位展开不稳定，V2 只作为辅助通道，不参与主要定位评价。

### 7.4 实验数据和仿真数据对齐

真实实验建议保留：

```text
healthy_time_csv
damaged_time_csv
fft_config
trigger_offset
window_config
channel_gain
tx/rx actual positions
temperature/material notes
```

实验健康 base 可以构造混合频域数据：

```text
R_sim(f) = Hd_sim(f) / (H0_sim(f) + eps)
Hd_hybrid(f) = H0_exp(f) * R_sim(f)
```

这能让真实系统误差由实验健康信号承载，缺陷扰动由仿真提供。但低信噪比频点必须屏蔽，否则比值会放大伪影。

## 8. 建议执行顺序

1. 生成至少 1 个健康 + 3 到 5 个缺陷频域样本，频点先用 30-70 kHz 或 20-100 kHz 粗扫。
2. 跑 `select_sensitive_frequencies.py` 选 10-15 个敏感频点。
3. 在 `get_pic` 实现 V1 粗图，先只支持 Dataset A 理想几何。
4. 用标签评估 V1 的单通道和多通道结果，调 `sigma_ray`、频率权重、归一化。
5. 加入 V2 相干通道，并用时域健康样本校准 `c_phase(f)`。
6. 扩展到 Dataset B/实验：读取实际通道位置、增益和健康 base，避免错用理想几何。
7. 再进入 diffusion 数据集封装和训练。

› 目前已经计算出来几个缺陷样本的sample1-12,14-23的npz文件，同时也有对应的label文件,设计一下评价指标，用来完善select_sensitive_frequencies，并且用依据该评价指标使用v1方式成
  像，在对比所有已计算的频点成像，看看两者evaluate后差距有多大，可以采取sample14作为示例运行计算，同时也能让我看到结果，同时V1的粗图算法并没有请在get_pic中添加其解释物理机
  制的md，同时完善v2的目标，目前想用born近似来做，也添加相应的物理解释机制md，不要修改别的仿真代码，仅修改成像和选择频点的代码。你可以通过运行结果来看看成像效果与选择频点
  是否合理判断是否使用当前评价指标选择频点


conda run -n get_pic python simple/get_pic/batch_compare_frequency_selection.py --physics-selected simple/f_domain/output/
  │ frequency_selection_physics_tomography_tuned/physics_tomography_tuned_top15_frequencies.txt --relative-selected simple/f_domain/output/frequency_selection/
  │ frequency_sensitivity_top15_frequencies.txt --output-root simple/get_pic/output/batch_frequency_selection_compare --preview-first 20
