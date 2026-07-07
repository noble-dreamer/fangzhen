# Born-Fisher 与高频配额物理选频指标

当前 `physics_tomography` 已从经验乘子版改成 **ray-Born/Rytov 线性化 + Fisher information + greedy D-optimal** 的无 label 选频方法。实现位置：

- [select_sensitive_frequencies.py](/D:/lab_ultr/fz/simple/f_domain/select_sensitive_frequencies.py:613)

当前另有下一版无 label 方法：

```text
physics_highfreq_quota
```

它针对当前 `output2` 后验评价中“高频段更有利于 V1 粗图定位”的现象，在物理打分中加入高频分辨率权重和频段配额，同时保留 Rytov 可见性、路径覆盖平衡和低分辨率 V1 无标签聚焦度。实现位置：

- [select_sensitive_frequencies.py](/D:/lab_ultr/fz/simple/f_domain/select_sensitive_frequencies.py:536)

它的目标不是直接学习缺陷位置，而是在频点预算有限时，选择一组对 V1 粗图成像最有信息量、且彼此互补的频点。

## 1. 文献依据

### 多频成像

Park 的 multi-frequency subspace migration 工作说明，多频信息可以提升裂纹/散射体成像质量，频率加权也会影响成像性能。这支持“不要只用单频或固定经验频点，而要设计频率集合”的思路。

- Won-Kwang Park, *Multi-frequency subspace migration for imaging of perfectly conducting, arc-like cracks*, 2013.
  https://arxiv.org/abs/1306.0265

### 频率 continuation

Borges 和 Rachh 的多频 inverse scattering 工作使用 recursive linearization，核心思想是从低频到高频逐步推进，因为低频目标函数更平滑，高频提供更高分辨率但也更容易陷入局部极小。这支持在选频中保留低频/中频/高频信息，而不是只选择最高响应的频点。

- Carlos Borges, Manas Rachh, *Multifrequency inverse obstacle scattering with unknown impedance boundary conditions using recursive linearization*, 2021.
  https://arxiv.org/abs/2104.13489

### Fisher information 与最优实验设计

Ruthotto、Chung 等人的 constrained inverse problem OED 框架把“选哪些测量”写成最优实验设计问题。对于本文任务，测量预算就是可选频点数，设计变量就是频点集合。

- Lars Ruthotto, Julianne Chung, Matthias Chung, *Optimal Experimental Design for Constrained Inverse Problems*, 2017.
  https://arxiv.org/abs/1708.04740

Nordebo 等人的 Fisher information 工作提供了逆问题信息量分析的理论背景。对线性化模型而言，Fisher information 可以写成 Jacobian 的加权 Gram 矩阵。

- Sven Nordebo et al., *Fisher Information for Inverse Problems and Trace Class Operators*, 2012.
  https://arxiv.org/abs/1203.5397

### 频域 FRF 与导波物理

Malladi 等人的工作说明可以直接从 steady-state frequency response 中估计实验 dispersion curves。因此，频域 FRF 本身是合理的数据入口，不一定要先回到时域。

- Malladi et al., *Estimating Experimental Dispersion Curves from Steady-State Frequency Response Measurements*, 2021.
  https://arxiv.org/abs/2101.00155

Haywood-Alexander 等人的 Lamb wave 损伤定位工作强调，导波损伤特征应与 guided-wave propagation physics 相关。这支持使用传播路径、相位、幅值扰动、Fisher 信息等物理量，而不是纯标签相关性指标。

- Haywood-Alexander et al., *Informative Bayesian Tools for Damage Localisation by Decomposition of Lamb Wave Signals*, 2022.
  https://arxiv.org/abs/2205.12161

## 2. 线性化观测模型

对每个 tx-rx 路径和频点，健康/损伤复频响为：

```text
H0_p(f), Hd_p(f)
```

使用 Rytov 形式的复对数扰动：

```text
y_p(f) = log(Hd_p(f) / H0_p(f))
```

其中：

```text
Re(y_p) = log(|Hd_p| / |H0_p|)
Im(y_p) = angle(Hd_p * conj(H0_p))
```

在弱散射/Born 近似下，可以把它近似写成线性模型：

```text
y_f = J_f m + noise
```

`m` 是待估计的缺陷/材料扰动场，`J_f` 是该频点下的灵敏度矩阵。当前 V1 还不是完整波动方程反演，所以这里使用与 V1 粗图一致的 **ray-tube Born Jacobian**：

```text
J_p(x) = normalized ray_kernel_p(x)
```

也就是说，每条 tx-rx-order 螺旋路径是一行 Jacobian，图像网格上的每个像素是一列。

## 3. 经验观测能量

代码中从已计算样本估计每条路径在某个频点下的扰动能量：

```text
E_p(f) =
  mean_samples[
    log(|Hd_p(f)| / |H0_p(f)|)^2
    + alpha * angle(Hd_p(f) * conj(H0_p(f)))^2
  ]
```

当前默认：

```text
alpha = 0.5
```

原因：

- 幅值对数项对应传播衰减/散射强度变化；
- 相位项对应传播常数、有效路径长度、模态耦合等变化；
- 相位在未做精确频散校准前更敏感，所以权重低于幅值项。

为了避免少数异常路径主导，代码会对 `E_p(f)` 做上分位裁剪：

```text
E_p(f) <- clip(E_p(f), 0, quantile_0.99(E(f)))
```

并使用全体观测能量的低分位作为经验噪声底：

```text
sigma_n^2 = quantile_0.10(E)
```

路径权重为：

```text
w_p(f) = E_p(f) / (sigma_n^2 + eps)
```

## 4. Fisher information

对线性高斯近似：

```text
y_f = J_f m + noise
```

频点 `f` 的 Fisher information 可写成：

```text
F_f = J_f^T W_f J_f
```

其中 `W_f = diag(w_p(f))`。当前实现中，为了速度，不直接构造像素级大矩阵，而是先对 ray-Born Jacobian 做低秩压缩：

```text
J J^T ~= A A^T
```

实际计算为：

```text
F_f ~= A^T W_f A
```

默认低秩维度：

```text
rank = 32
```

这样每个频点只需要处理一个小矩阵，100 个样本时也能快速完成。

## 5. 单频得分

单个频点的基础得分使用 D-optimal 设计里的 logdet 信息量：

```text
score(f) = log det(I + beta * F_f)
```

当前默认：

```text
beta = 1.0
```

`logdet` 的意义是最大化参数空间里的信息体积。相比简单的 `relative_l2`，它不仅看扰动强弱，还会通过 `J_f` 的路径几何结构考虑该频点对成像自由度的约束能力。

输出表中对应字段：

- `physics_fisher_logdet`
- `physics_fisher_trace`
- `physics_fisher_min_eig`
- `physics_effective_rank`
- `physics_observation_energy_mean`
- `physics_path_participation_mean`
- `physics_path_contrast_mean`
- `physics_tx_balance_mean`
- `physics_rx_balance_mean`

## 6. 频点集合选择

最终不是简单取单频 `score(f)` 最高的 top-N，而是默认使用 greedy D-optimal：

```text
S_0 = empty
for k = 1...N:
    f_k = argmax_f [
        log det(I + beta * (sum_{g in S_{k-1}} F_g + F_f))
        - log det(I + beta * sum_{g in S_{k-1}} F_g)
    ]
```

这样能减少选出很多“信息重复”的相近频点。输出表中：

- `greedy_step` 表示第几步被选中；
- `greedy_logdet_gain` 表示该频点带来的新增信息量；
- `greedy_cumulative_logdet` 表示累计信息量。

## 7. 与旧经验版的区别

旧版 `physics_tomography` 是：

```text
robust_relative_change
* phase factor
* path contrast factor
* participation factor
* tx/rx balance factor
```

它的每个因子都有物理解释，但系数仍然偏经验。

新版把这些思想压到一个更正式的框架里：

- `J` 来自 V1 的传播路径几何；
- `W_f` 来自健康/损伤复频响的 Rytov 扰动能量；
- `F_f = J^T W_f J` 是线性化逆问题的信息矩阵；
- `logdet` 与 greedy D-optimal 来自最优实验设计。

因此新版更适合论文表述，也更容易扩展到后续 Born 近似或完整波动反演。

## 8. 与 label 的关系

`physics_tomography` 本身不读取 label。

label 只用于后验评价：

1. 用 `physics_tomography` 选频；
2. 用选出的频点生成 V1 粗图；
3. 用 [evaluate_coarse_maps.py](/D:/lab_ultr/fz/simple/get_pic/evaluate_coarse_maps.py:26) 与 label 比较；
4. 与全频点、`relative_l2`、旧经验物理指标等方法做对照。

这样可以保证论文方法不是“偷看缺陷真值”的 label-guided 方案。

## 9. 运行方式

示例：

```powershell
conda run -n get_pic python simple\f_domain\select_sensitive_frequencies.py `
  --metric physics_tomography `
  --sample-ids 1-100 `
  --frequency-min-khz 20 `
  --frequency-max-khz 100 `
  --top-n 15 `
  --prefix physics_tomography_born_fim `
  --output-root simple\f_domain\output\frequency_selection_physics_tomography_born_fim `
  --jobs 0 `
  --born-grid-size 128 `
  --born-rank 32 `
  --born-sigma-ray-mm 25 `
  --selection-strategy greedy_d_optimal
```

`--jobs 0` 表示使用当前机器 CPU 核心数并行加载样本；主要数值计算已经向量化到 NumPy。

## 10. 当前 22 个样本的试跑结果

当前已用 `sample1-12,14-23` 试跑 Born-FIM 版本，输出目录：

- [frequency_selection_physics_tomography_born_fim](/D:/lab_ultr/fz/simple/f_domain/output/frequency_selection_physics_tomography_born_fim)

选出的 15 个频点为：

```text
22.5, 25.0, 75.0, 45.0, 37.5,
72.5, 47.5, 92.5, 80.0, 90.0,
42.5, 55.0, 57.5, 20.0, 87.5 kHz
```

选频耗时约 8.5 秒。100 个样本时主要额外成本是读取 NPZ 和向量化统计，预计仍明显快于逐样本逐频点 Python 循环。

## 11. 下一版：physics_highfreq_quota

### 11.1 设计动机

在 `output2` 的 10 个样本初步比较中，旧 `physics_tomography` 比 `relative_l2` 略好，但没有超过 `all_completed`。后验 `v1_label_guided` 选出的频点明显偏高频，说明当前 V1 粗图定位更依赖高频段的空间分辨率。

因此新增：

```text
--metric physics_highfreq_quota
```

它仍然是不读取 label 的物理选频方法，目标是避免旧 Fisher/logdet 指标过度偏向低/中频稳定响应。

### 11.2 打分规则

每个频点的得分为：

```text
score(f) =
  visibility(f)
  * focus_factor(f)
  * contrast_factor(f)
  * coverage_balance(f)
  * resolution_weight(f)
  * health_weight(f)
```

其中：

```text
visibility(f)
```

来自 Rytov 复扰动：

```text
g_p(f) = log(Hd_p(f) / H0_p(f))
signal_p(f) = |Re(g_p)| + alpha * |sin(Im(g_p))|
```

当前默认：

```text
alpha = 0.35
```

`focus_factor(f)` 来自低分辨率 V1 无标签反投影图：

```text
B_f(theta, z) = sum_p signal_p(f) * ray_kernel_p(theta, z)
```

再计算：

```text
top5_energy / top20_energy
peak_to_median
Gini
entropy_score
```

组合成 `highfreq_focus_score`。它不使用 label，只判断该频点的反投影扰动是否足够局部、非均匀、可聚焦。

`coverage_balance(f)` 使用 participation ratio 的中间最优规则：

```text
participation = (sum w)^2 / (N * sum w^2)
coverage_balance = exp(-((participation - p0) / sigma_p)^2)
```

当前默认：

```text
p0 = 0.35
sigma_p = 0.25
```

含义：

- participation 太低：少数路径异常；
- participation 太高：全局变化，不利于定位；
- 中等 participation：多条路径共同携带缺陷信息。

`resolution_weight(f)` 显式偏向高频：

```text
resolution_weight(f) = clip((f / f_ref)^gamma, low, high)
```

当前默认：

```text
f_ref = 50 kHz
gamma = 0.5
low = 0.70
high = 1.60
```

`health_weight(f)` 用健康响应幅值做稳定性约束，避免健康响应过弱的频点被高频权重误选。

### 11.3 频段配额

最终不再直接取总分 top15，而是使用频段配额：

```text
low:  20-40 kHz   3 个
mid:  40-65 kHz   4 个
high: 65-100 kHz  8 个
```

默认参数：

```text
--highfreq-low-max-khz 40
--highfreq-mid-max-khz 65
--highfreq-low-quota 3
--highfreq-mid-quota 4
--highfreq-high-quota 8
```

这样做的物理规则是：

1. 低频保留穿透和大尺度稳定响应；
2. 中频保留过渡带；
3. 高频提供缺陷边界和局部定位分辨率；
4. 如果高频健康响应过弱，会被 `healthy_response_below_floor` 或 `health_weight` 抑制。

## 12. output2 样本 1-10、21-31 测试结果

测试样本：

```text
1-10,21-31
```

共 21 个 damage response，均来自：

```text
simple/f_domain/output2/streaming_dataset_a_frequency_shell/
```

新方法输出：

```text
simple/f_domain/output2/frequency_selection_physics_highfreq_quota_21samples/
```

选出的 15 个频点：

```text
40.0, 32.5, 20.0,
42.5, 50.0, 47.5, 52.5,
70.0, 72.5, 67.5, 75.0, 80.0, 82.5, 77.5, 95.0 kHz
```

对应频段配额：

```text
low:  40.0, 32.5, 20.0
mid:  42.5, 50.0, 47.5, 52.5
high: 70.0, 72.5, 67.5, 75.0, 80.0, 82.5, 77.5, 95.0
```

后验评价输出：

```text
simple/get_pic/output2/frequency_selection_compare_21samples/five_method_all_channels_summary.csv
```

`ray_relative_delta` 通道平均结果：

| method                 | Pearson | top5 hit | mass in label | centroid error | NRMSE |
| ---------------------- | ------: | -------: | ------------: | -------------: | ----: |
| v1_label_guided        |  0.1657 |   0.2543 |        0.0564 |       479.2 mm | 2.146 |
| all_completed          |  0.1026 |   0.1521 |        0.0514 |       472.3 mm | 2.428 |
| physics_highfreq_quota |  0.0975 |   0.1256 |        0.0511 |       471.6 mm | 2.376 |
| physics_tomography     |  0.0834 |   0.1039 |        0.0501 |       463.1 mm | 2.430 |
| relative_l2            |  0.0785 |   0.0956 |        0.0497 |       454.7 mm | 2.460 |

各方法最好信息通道均为 `high_frequency_band_map`，结果为：

| method                 | Pearson | top5 hit | mass in label | NRMSE |
| ---------------------- | ------: | -------: | ------------: | ----: |
| v1_label_guided        |  0.1772 |   0.2743 |        0.0575 | 2.123 |
| all_completed          |  0.1742 |   0.2807 |        0.0572 | 2.113 |
| physics_highfreq_quota |  0.1669 |   0.2678 |        0.0567 | 2.094 |
| relative_l2            |  0.1587 |   0.2027 |        0.0561 | 2.117 |
| physics_tomography     |  0.1487 |   0.2171 |        0.0554 | 2.033 |

结论：

```text
physics_highfreq_quota > physics_tomography > relative_l2
```

在最有用的 `high_frequency_band_map` 通道上，新方法已经接近 `all_completed` 和 `v1_label_guided`。因此下一步如果必须压缩频点数量，建议优先使用 `physics_highfreq_quota`，而不是旧的 `physics_tomography` 或 `relative_l2`。

注意：`v1_label_guided` 使用 label，只能作为后验上限，不可作为真实实验前无标签选频方法。

## 13. 服务器运行指令

以下命令假设在项目根目录 `fz` 下运行。

### 13.1 生成 physics_highfreq_quota 选频

Linux/bash：

```bash
conda run -n get_pic python simple/f_domain/select_sensitive_frequencies.py \
  --sample-ids 1-10,21-31 \
  --metric physics_highfreq_quota \
  --top-n 15 \
  --output-root simple/f_domain/output2/frequency_selection_physics_highfreq_quota_21samples \
  --prefix physics_highfreq_quota \
  --frequency-min-khz 20 \
  --frequency-max-khz 100 \
  --jobs 0 \
  --highfreq-grid-size 96 \
  --highfreq-low-quota 3 \
  --highfreq-mid-quota 4 \
  --highfreq-high-quota 8 \
  --highfreq-low-max-khz 40 \
  --highfreq-mid-max-khz 65
```

Windows PowerShell：

```powershell
conda run -n get_pic python simple\f_domain\select_sensitive_frequencies.py `
  --sample-ids 1-10,21-31 `
  --metric physics_highfreq_quota `
  --top-n 15 `
  --output-root simple\f_domain\output2\frequency_selection_physics_highfreq_quota_21samples `
  --prefix physics_highfreq_quota `
  --frequency-min-khz 20 `
  --frequency-max-khz 100 `
  --jobs 0 `
  --highfreq-grid-size 96 `
  --highfreq-low-quota 3 `
  --highfreq-mid-quota 4 `
  --highfreq-high-quota 8 `
  --highfreq-low-max-khz 40 `
  --highfreq-mid-max-khz 65
```

### 13.2 用选出的频点生成 V1 粗图

Linux/bash：

```bash
conda run -n get_pic python simple/get_pic/generate_coarse_maps.py \
  --sample-ids 1-10,21-31 \
  --selected-frequencies simple/f_domain/output2/frequency_selection_physics_highfreq_quota_21samples/physics_highfreq_quota_top15_frequencies.txt \
  --output-root simple/get_pic/output2/physics_highfreq_quota_21samples \
  --preview
```

Windows PowerShell：

```powershell
conda run -n get_pic python simple\get_pic\generate_coarse_maps.py `
  --sample-ids 1-10,21-31 `
  --selected-frequencies simple\f_domain\output2\frequency_selection_physics_highfreq_quota_21samples\physics_highfreq_quota_top15_frequencies.txt `
  --output-root simple\get_pic\output2\physics_highfreq_quota_21samples `
  --preview
```

### 13.3 后验评价粗图

Linux/bash 中如果 shell 支持展开，可运行：

```bash
conda run -n get_pic python simple/get_pic/evaluate_coarse_maps.py \
  --coarse simple/get_pic/output2/physics_highfreq_quota_21samples/coarse_maps/*_coarse_maps.npz \
  --output-dir simple/get_pic/output2/physics_highfreq_quota_21samples/reports
```

PowerShell：

```powershell
$coarse = Get-ChildItem simple\get_pic\output2\physics_highfreq_quota_21samples\coarse_maps\*_coarse_maps.npz | ForEach-Object { $_.FullName }
conda run -n get_pic python simple\get_pic\evaluate_coarse_maps.py `
  --coarse $coarse `
  --output-dir simple\get_pic\output2\physics_highfreq_quota_21samples\reports
```
