# Born-Fisher 频点选择方法推导与实验规范

本文档用于说明 `physics_tomography` 当前版本的物理选频方法。该方法面向频域导波数据集，使用健康/损伤复频响构造无 label 的频点信息量指标，并通过 greedy D-optimal 选择一组互补频点。

对应实现：

- [select_sensitive_frequencies.py](/D:/lab_ultr/fz/simple/f_domain/select_sensitive_frequencies.py:613)

当前方法可以写入论文的 “Frequency Selection Method” 或 “Physics-guided Frequency Selection” 小节。它不是直接调经验系数，而是将频点选择写成线性化逆问题中的最优实验设计问题。

## 1. 研究问题定义

设管道上有一组发射换能器和接收换能器。对每个发射-接收路径 `p=(tx,rx)` 和频率 `f`，频域仿真或实验可得到健康状态复频响：

```text
H0_p(f) in C
```

以及损伤状态复频响：

```text
Hd_p(f) in C
```

目标是在候选频率集合：

```text
F = {f_1, f_2, ..., f_M}
```

中选择一个固定预算的子集：

```text
S subset F,  |S| = N
```

使得使用 `S` 生成的 V1 粗图尽量保留全频点成像的信息量。

本文方法有两个约束：

1. 选频阶段不使用缺陷 label；
2. 选频指标必须能从频域复频响 `H0/Hd` 和已知几何中计算。

因此，label 只用于后验评价，不进入选频公式。

## 2. 为什么使用频域复频响

真实实验中示波器采集的是时域信号，但经过同步、截窗和 Fourier transform 后，仍可得到频域传递函数。当前数据集中已经直接保存每个 tx-rx 路径的复频响，所以选频以频域 `H(f)` 为主。

复频响比单纯幅值多包含一类信息：

```text
H_p(f) = |H_p(f)| exp(i phi_p(f))
```

其中：

- `|H_p(f)|` 反映传播衰减、散射损失、能量耦合强弱；
- `phi_p(f)` 反映传播相位、等效路径长度、相速度/群速度变化、模态耦合等。

因此频点选择不应只看：

```text
|Hd_p(f) - H0_p(f)|
```

而应同时考虑幅值扰动、相位扰动、路径几何和频点集合之间的信息互补性。

## 3. Rytov 形式的复扰动观测量

对每个样本 `s`、路径 `p`、频点 `f`，定义复频响比值：

```text
R_{s,p}(f) = Hd_{s,p}(f) / (H0_p(f) + eps)
```

直接使用差分：

```text
Hd - H0
```

会受到健康响应幅值大小的强烈影响。为了把扰动写成相对传播变化，采用 Rytov 形式的复对数观测：

```text
y_{s,p}(f) = log R_{s,p}(f)
```

将其实部和虚部分开：

```text
Re[y_{s,p}(f)] = log(|Hd_{s,p}(f)| / (|H0_p(f)| + eps))
Im[y_{s,p}(f)] = angle(Hd_{s,p}(f) * conj(H0_p(f)))
```

其中 `angle(...)` 使用包裹相位，范围为 `[-pi, pi]`。

物理含义：

- 实部是对数幅值扰动，近似反映传播衰减、散射损失或能量重新分配；
- 虚部是相位扰动，近似反映传播时延、等效波数变化和模态相位变化。

在弱散射条件下，复对数扰动比直接幅值差更接近线性化传播扰动。

## 4. Born/Rytov 线性化模型

设缺陷或材料扰动场为：

```text
m(x)
```

其中 `x` 表示展开后的管道表面或管壁中面坐标。导波传播的完整模型是非线性的，但在小扰动、弱散射条件下，可以用 Born 或 Rytov 近似线性化：

```text
y_p(f) ~= integral K_p(x, f) m(x) dx + n_p(f)
```

离散化后：

```text
y_f = J_f m + n_f
```

其中：

- `y_f` 是所有 tx-rx 路径在频点 `f` 下的观测扰动向量；
- `m` 是离散图像网格上的缺陷扰动；
- `J_f` 是频点 `f` 下的灵敏度矩阵；
- `n_f` 是噪声或模型误差。

完整导波 Born Jacobian 应由波动方程 Green function、模态传播常数和边界条件推导得到。当前 V1 还不是完整波动方程反演，因此使用与 V1 粗图一致的 ray-tube 近似：

```text
J_p(x) = normalized ray_kernel_p(x)
```

也就是把每条 tx-rx 螺旋传播路径视为一条对缺陷敏感的射线管。这样做的优点是：

- 与现有 V1 粗图生成物理机制一致；
- 不需要重新求解 COMSOL 或计算全波 Jacobian；
- 可以直接用几何路径构造频点选择所需的信息矩阵；
- 后续若实现完整 Born 波动核，可替换 `J_p(x)`，而 Fisher/D-optimal 框架不变。

## 5. 管道展开坐标与射线核

当前 V1 在管道中面上成像。设管道中面半径为 `R`，展开坐标为：

```text
x = R * theta
z = axial coordinate
```

对发射端 `tx` 和接收端 `rx`，其角向位置和轴向位置分别为：

```text
(theta_tx, z_tx), (theta_rx, z_rx)
```

由于管道周向周期性，允许螺旋阶次：

```text
o in {-1, 0, 1}
```

对应的展开角向位移为：

```text
Delta theta_o = wrap(theta_rx - theta_tx) + 2*pi*o
```

展开平面中的路径终点横向距离为：

```text
x_2 = R * Delta theta_o
```

路径长度为：

```text
L_p = sqrt(x_2^2 + (z_rx - z_tx)^2)
```

对图像网格中任一像素 `x_i`，计算其到该展开线段的最短距离：

```text
d_{p,i} = distance(pixel_i, line_segment_p)
```

射线管核定义为：

```text
K_{p,i} = exp(-0.5 * (d_{p,i}/sigma)^2) / max(L_p, eps)
```

并使用截断条件：

```text
d_{p,i} <= c_sigma * sigma
```

当前默认：

```text
sigma = 25 mm
c_sigma = 3
helical_orders = {-1, 0, 1}
```

最后对每条射线核做归一化：

```text
J_{p,:} = K_{p,:} / ||K_{p,:}||_2
```

这样每条路径在 Fisher 信息中的作用主要来自其观测扰动强弱，而不是来自射线长度或网格离散尺度。

## 6. 从复频响估计路径观测能量

对每个频点 `f` 和路径 `p`，使用所有已计算损伤样本估计平均扰动能量：

```text
E_p(f) =
  mean_s [
    log(|Hd_{s,p}(f)| / (|H0_p(f)| + eps))^2
    + alpha * angle(Hd_{s,p}(f) * conj(H0_p(f)))^2
  ]
```

当前默认：

```text
alpha = 0.5
```

`alpha` 不是缺陷 label 调出来的系数，而是一个保守的相位权重。原因是：

- 相位项对传播速度、路径长度和模态变化敏感；
- 但在没有完整频散校准、相位 unwrap 和模态分离之前，相位更容易受到包裹和多模态干涉影响；
- 因此相位作为重要信息源保留，但权重低于幅值对数项。

如果后续完成时域校准或 dispersion-based phase correction，可以提高 `alpha`，或把不同模态的相位项分开建模。

## 7. 噪声底与异常路径裁剪

直接使用 `E_p(f)` 会遇到两个问题：

1. 某些路径可能因为数值异常或接收幅值过低而产生极大扰动；
2. 不同频点的整体扰动尺度不同，不能直接比较。

因此先对单个频点内部的路径能量做上分位裁剪：

```text
E_p(f) <- clip(E_p(f), 0, Q_0.99({E_p(f)}_p))
```

然后用所有样本、路径、频点上的扰动能量低分位估计经验噪声底：

```text
sigma_n^2 = Q_0.10({E_p(f)}_{p,f})
```

路径权重定义为：

```text
w_p(f) = E_p(f) / (sigma_n^2 + eps)
```

物理含义：

- `w_p(f)` 越大，说明该路径在该频点下对损伤扰动更敏感；
- 低分位噪声底把小扰动路径视为近似噪声水平；
- 上分位裁剪抑制少数异常路径对 Fisher 矩阵的支配。

## 8. Fisher information 矩阵

在线性高斯观测模型：

```text
y_f = J_f m + n_f
n_f ~ N(0, Sigma_f)
```

下，参数 `m` 的 Fisher information 为：

```text
F_f = J_f^T Sigma_f^{-1} J_f
```

若假设各路径噪声独立，并用路径权重近似信噪比：

```text
Sigma_f^{-1} ~= W_f = diag(w_p(f))
```

则：

```text
F_f = J^T W_f J
```

这里 `J` 是当前 V1 ray-Born Jacobian。它在不同频点之间几何结构相同，频率差异主要进入路径权重 `W_f`。这与当前 V1 的物理假设一致：V1 使用相同几何射线路径，不同频点通过路径标量扰动和频点权重影响成像。

`F_f` 的物理意义：

- 若某频点只让少数路径有大权重，`F_f` 的有效秩较低；
- 若某频点让多个几何方向的路径同时有可靠扰动，`F_f` 的谱分布更充分；
- 因此 Fisher 矩阵不仅度量扰动强度，也度量这些扰动是否能约束图像空间中的多个方向。

## 9. 低秩压缩计算

完整图像网格可能有：

```text
128 * 128 = 16384
```

个像素。如果直接构造：

```text
J in R^{P x 16384}
F_f in R^{16384 x 16384}
```

计算会很慢，也没有必要。

当前实现使用路径空间 Gram 矩阵：

```text
G = J J^T in R^{P x P}
```

其中 `P` 是射线路径数。当前：

```text
tx = 16
rx = 16
helical_orders = 3
P = 16 * 16 * 3 = 768
```

对 `G` 做特征分解：

```text
G = U Lambda U^T
```

保留前 `r` 个主方向：

```text
A = U_r Lambda_r^{1/2}
```

于是：

```text
J J^T ~= A A^T
```

在压缩空间中，频点 Fisher 信息近似为：

```text
F_f^r = A^T W_f A
```

当前默认：

```text
r = 32
```

这样每个频点只需求一个 `32 x 32` 矩阵，适合 100 个样本以上的批量选频。

## 10. 单频信息量得分

采用 D-optimal design 中常用的 log-determinant criterion：

```text
I(f) = log det(I + beta * F_f^r)
```

当前默认：

```text
beta = 1.0
```

解释：

- `trace(F_f)` 更偏向总能量，容易选择强扰动但信息重复的频点；
- `lambda_min(F_f)` 更偏向最弱方向，但对噪声很敏感；
- `log det(I + beta F_f)` 同时奖励多个特征方向上的信息增益，是比较稳健的体积型信息量指标。

若 `F_f` 的特征值为：

```text
lambda_1, lambda_2, ..., lambda_r
```

则：

```text
I(f) = sum_{k=1}^{r} log(1 + beta * lambda_k)
```

因此一个频点只有单一强方向时，得分不会无限制增长；多个独立方向都有信息时，得分更高。

## 11. 频点集合的 greedy D-optimal 选择

单频得分最高的频点可能高度相似。例如两个相邻频点可能激活相似的路径集合，导致信息重复。为此最终选择不直接取 top-N，而是做贪心集合优化。

初始化：

```text
S_0 = empty
F_{S_0} = 0
```

第 `k` 步选择：

```text
f_k =
argmax_{f notin S_{k-1}}
[
  log det(I + beta * (F_{S_{k-1}} + F_f))
  -
  log det(I + beta * F_{S_{k-1}})
]
```

并更新：

```text
S_k = S_{k-1} union {f_k}
F_{S_k} = F_{S_{k-1}} + F_{f_k}
```

该过程的物理意义是：

- 第一个频点通常选择总体信息量最强者；
- 后续频点优先补充当前已选集合缺少的独立信息方向；
- 因此选出的频点集合更偏向互补，而不是简单重复高能频段。

输出字段：

```text
greedy_step
greedy_logdet_gain
greedy_cumulative_logdet
```

分别表示选中顺序、新增信息量和累计信息量。

## 12. 输出物理量解释

`physics_tomography_born_fim_ranked.csv` 中关键字段如下。

### 频点基础信息

```text
frequency_hz
frequency_khz
sample_count
valid_tx_count_min
valid_ray_count_min
healthy_mean_abs
```

解释：

- `sample_count`：参与统计的损伤样本数；
- `valid_tx_count_min`：所有样本中该频点最少有效发射端数量；
- `valid_ray_count_min`：该频点有效射线路径数量；
- `healthy_mean_abs`：健康基线该频点的平均响应幅值。

### Fisher 信息量

```text
physics_fisher_logdet
physics_fisher_trace
physics_fisher_min_eig
physics_effective_rank
```

解释：

- `physics_fisher_logdet`：D-optimal 单频信息量；
- `physics_fisher_trace`：Fisher 总能量；
- `physics_fisher_min_eig`：压缩子空间中最弱信息方向；
- `physics_effective_rank`：Fisher 谱的有效秩，越高表示信息方向越分散。

有效秩定义为：

```text
effective_rank =
  (sum_k lambda_k)^2 / (sum_k lambda_k^2 + eps)
```

### 观测扰动与路径分布

```text
physics_observation_energy_mean
physics_observation_energy_median
physics_path_participation_mean
physics_path_contrast_mean
physics_tx_balance_mean
physics_rx_balance_mean
```

解释：

- `physics_observation_energy_mean`：路径 Rytov 扰动能量均值；
- `physics_observation_energy_median`：路径 Rytov 扰动能量中位数；
- `physics_path_participation_mean`：路径参与度；
- `physics_path_contrast_mean`：路径差异度；
- `physics_tx_balance_mean`：发射端覆盖均衡度；
- `physics_rx_balance_mean`：接收端覆盖均衡度。

其中 participation ratio 为：

```text
PR(a) = (sum_i a_i)^2 / (N * sum_i a_i^2 + eps)
```

`PR` 越接近 1，表示能量分布越均衡；越接近 0，表示少数路径或少数端口主导。

这些字段不是最终选频的经验乘子，而是用于解释 Fisher 信息的物理来源。

## 13. 当前默认参数

当前推荐参数：

```text
metric = physics_tomography
selection_strategy = greedy_d_optimal
born_grid_size = 128
born_rank = 32
born_sigma_ray_mm = 25
born_info_scale = 1.0
born_phase_weight = 0.5
born_noise_quantile = 0.10
born_weight_clip_quantile = 0.99
jobs = 0
```

参数含义：

- `born_grid_size`：构造 ray-Born Jacobian 的低分辨率网格；
- `born_rank`：Fisher 信息压缩维度；
- `born_sigma_ray_mm`：射线管宽度；
- `born_info_scale`：logdet 中的信息尺度参数；
- `born_phase_weight`：相位扰动能量权重；
- `born_noise_quantile`：估计经验噪声底的低分位；
- `born_weight_clip_quantile`：路径能量异常值裁剪上分位；
- `jobs=0`：使用 CPU 核心数并行加载样本。

这些参数中，`born_sigma_ray_mm` 与 V1 粗图物理分辨率相关；`born_rank` 与计算速度和信息维度有关；`born_phase_weight` 与相位校准可信度有关。

## 14. 实验流程

### Step 1: 准备频域数据

需要健康样本：

```text
dataset_a_frequency_healthy_H_complex.npz
```

以及损伤样本：

```text
dataset_a_frequency_sample_XXXX_H_complex.npz
```

每个 `.npz` 至少包含：

```text
H_real
H_imag
tx_indices
rx_indices
frequencies_hz
completed_mask
```

### Step 2: 运行无 label 选频

100 个样本推荐命令：

```powershell
conda run -n get_pic python simple\f_domain\select_sensitive_frequencies.py `
  --metric physics_tomography `
  --sample-ids 1-100 `
  --frequency-min-khz 20 `
  --frequency-max-khz 100 `
  --top-n 15 `
  --prefix physics_tomography_born_fim `
  --output-root simple\f_domain\output\frequency_selection_physics_tomography_born_fim_100 `
  --jobs 0 `
  --born-grid-size 128 `
  --born-rank 32 `
  --born-sigma-ray-mm 25 `
  --selection-strategy greedy_d_optimal
```

该步骤不读取 label。

### Step 3: 用选中频点生成 V1 粗图

用 `get_pic` 中的 V1 粗图脚本读取 `*_top15_frequencies.txt`，生成 coarse map。

### Step 4: 后验评价

用 label 评价：

```text
pearson
nrmse
top5_hit_rate
prediction_mass_in_label
centroid_error_mm
```

评价阶段可以比较：

```text
Born-Fisher selected frequencies
relative_l2 selected frequencies
all_completed frequencies
```

### Step 5: 报告统计

建议报告：

```text
mean +/- std
better_count / worse_count
per-sample paired difference
```

这样可以证明选频指标不是对单一样本偶然有效。

## 15. 当前 22 样本示例结果

已在 `sample1-12,14-23` 上试跑：

```text
Selected frequencies:
22.5, 25.0, 75.0, 45.0, 37.5,
72.5, 47.5, 92.5, 80.0, 90.0,
42.5, 55.0, 57.5, 20.0, 87.5 kHz
```

输出目录：

```text
simple/f_domain/output/frequency_selection_physics_tomography_born_fim
```

首个被选中的频点 `22.5 kHz` 的示例输出：

```text
physics_fisher_logdet = 196.9412
physics_fisher_trace = 19724.6253
physics_effective_rank = 15.4152
physics_observation_energy_mean = 1.7727
physics_path_participation_mean = 0.8989
physics_tx_balance_mean = 0.9988
physics_rx_balance_mean = 0.9981
```

解释：

- `logdet` 高，说明总体信息体积大；
- `effective_rank` 接近压缩秩的一半，说明不是单一方向主导；
- `path_participation` 较高，说明多路径参与；
- `tx/rx_balance` 接近 1，说明发射端和接收端覆盖均衡。

## 16. 与旧经验指标的区别

旧版 `physics_tomography` 使用：

```text
robust_relative_change
* phase factor
* path_contrast factor
* path_participation factor
* tx/rx_balance factor
```

这套写法虽然有物理解释，但仍需要人为设定多个乘法系数。

新版改为：

```text
Rytov observation -> path weights -> Fisher information -> D-optimal subset
```

因此主要依据来自：

- Born/Rytov 线性化；
- Fisher information；
- optimal experimental design；
- V1 射线路径几何。

这比继续手动调 `0.55/0.25/0.60` 之类的系数更适合论文表达。

## 17. 论文中可使用的表述

可以在论文中写成：

```text
For each candidate frequency, a ray-Born sensitivity matrix was constructed
using the same helical ray-tube kernels as the V1 backprojection model.
The complex frequency response perturbation was represented by a Rytov-type
logarithmic ratio between damaged and healthy transfer functions. The
sample-averaged perturbation energy was used to define a diagonal path
weighting matrix, yielding a compressed Fisher information matrix for each
frequency. A greedy D-optimal criterion was then applied to select a fixed
number of frequencies that maximized the incremental log-determinant of the
accumulated Fisher information matrix.
```

中文表述：

```text
对每个候选频点，本文采用与 V1 反投影一致的螺旋射线管核构造
ray-Born 灵敏度矩阵。健康/损伤复频响之间的扰动用 Rytov 型复对数比表示，
并由多损伤样本平均扰动能量形成路径权重矩阵。进一步构造每个频点的压缩
Fisher 信息矩阵，并通过 greedy D-optimal 准则选择累计 Fisher 信息
log-determinant 增益最大的频点集合。
```

## 18. 局限性与后续扩展

当前方法仍是 V1 一致的近似方法，不是完整全波 Born 反演。主要局限：

1. `J` 使用几何射线管核，没有显式包含导波模态色散；
2. 相位项使用 wrapped phase，尚未做 mode-specific unwrapping；
3. 路径权重由样本平均扰动估计，依赖当前已生成数据的缺陷分布；
4. 低秩压缩会舍弃一部分细节方向。

后续可扩展方向：

- 用频散曲线估计不同模态的相速度/群速度；
- 将 `J_p(x)` 从几何射线管替换为 mode-dependent Born kernel；
- 对相位做时域校准和频散校正；
- 使用 A-optimal 或 E-optimal 指标作为对照；
- 在 100 个或更多样本上报告频点稳定性。

## 19. 参考文献

1. Won-Kwang Park, *Multi-frequency subspace migration for imaging of perfectly conducting, arc-like cracks*, 2013.  
   https://arxiv.org/abs/1306.0265

2. Carlos Borges and Manas Rachh, *Multifrequency inverse obstacle scattering with unknown impedance boundary conditions using recursive linearization*, 2021.  
   https://arxiv.org/abs/2104.13489

3. Lars Ruthotto, Julianne Chung, Matthias Chung, *Optimal Experimental Design for Constrained Inverse Problems*, 2017.  
   https://arxiv.org/abs/1708.04740

4. Sven Nordebo et al., *Fisher Information for Inverse Problems and Trace Class Operators*, 2012.  
   https://arxiv.org/abs/1203.5397

5. Malladi et al., *Estimating Experimental Dispersion Curves from Steady-State Frequency Response Measurements*, 2021.  
   https://arxiv.org/abs/2101.00155

6. Haywood-Alexander et al., *Informative Bayesian Tools for Damage Localisation by Decomposition of Lamb Wave Signals*, 2022.  
   https://arxiv.org/abs/2205.12161
