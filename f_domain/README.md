# Frequency-domain Dataset A shell pipeline

`f_domain/` 是 `simple/` 壳模型的频域版本。它复用 `simple_shell_common.py` 的几何、材料、外表面腐蚀厚度减薄、Dataset A 低反射吸收层、等效 PZT 面载荷窗口和 `intop_shell` 接收 patch 加权平均，只把求解从时域 `Transient` 改为频域 `Frequency`。

## 与时域模型的对应关系

- 管道：同一个圆柱壳中面模型。
- 缺陷：同一个壳厚度局部减薄模型；随机样本默认是 1-3 个平滑外表面腐蚀缺陷，采用小/中/大尺寸混合，多缺陷时优先大+中/小组合，少量浅 lobe，局部累计最大厚度损失限制为 5 mm。
- Dataset A 低反射边界：同一个两端轴向渐变 Rayleigh damping absorbing layer。
- 发射端：同一个 PZT window 等效面载荷。
- 接收端：同一个 `intop_shell(w_rx*u_r)/intop_shell(w_rx)` 小面积加权平均。
- 求解：频域 `Frequency` study，频率由全局参数 `pzt_fc` 控制。

频域载荷不再包含 `pztpulse(t)`。它是单位谐波幅值：

```text
F(tx, f, theta, z) = F0/pzt_A * window_tx(theta, z)
```

因此导出的结果是复位移幅值 `H(tx, rx, f)`，不是时间波形，不能直接提取 TOF 或 Γ0/Γ±1 到达时间。

## 文件结构

- `frequency_domain_common.py`: 频域建模、求解、导出公共逻辑。
- `build_dataset_a_frequency_healthy.py`: 构建健康频域 MPH，不求解，用于 COMSOL 模型树检查。
- `solve_export_dataset_a_frequency.py`: 流式频域求解与导出脚本。
- `select_sensitive_frequencies.py`: 根据健康/缺陷频响计算频点 sensitivity 并推荐频点。
- `FREQUENCY_DOMAIN_DIFFUSION_PLAN.md`: 频域粗图、时域校准和 diffusion 训练规划。
- `output/`: 频域脚本默认输出目录。

## 构建可检查模型

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/build_dataset_a_frequency_healthy.py
```

输出：

```text
simple/f_domain/output/dataset_a_frequency_shell/pipe_shell_frequency_healthy.mph
simple/f_domain/output/dataset_a_frequency_shell/metadata/pipe_shell_frequency_healthy.json
simple/f_domain/output/dataset_a_frequency_shell/dataset_a_frequency_shell_build_log.md
```

在 COMSOL Model Builder 中重点检查：

- `Study > simple shell displacement frequency domain`
- `Component 1 > Shell Mechanics > equivalent transducer face load`
- `Results > Derived Values > receiver patch weighted average radial displacement`
- `Global Definitions > Parameters > tx`
- `Global Definitions > Parameters > pzt_fc`

## 流式频域求解

最小健康单工况检查：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/solve_export_dataset_a_frequency.py --only-healthy --tx 1 --frequencies 50000 --linear-solver pardiso --heartbeat-s 20
```

如果服务器在 `sample_start 0/N` 后直接退出，说明还没有进入 COMSOL 建模/求解阶段。当前脚本会继续打印 `label_start`、`label_done` 和 `sample_model_build_start`，可用来定位早退位置。服务器环境下也可以先跳过 label 预览 PNG：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/solve_export_dataset_a_frequency.py --only-healthy --tx 1 --frequencies 50000 --linear-solver pardiso --heartbeat-s 20 --skip-label-preview
```

默认频率为：

```text
30, 35, 40, 45, 50, 55, 60, 65, 70 kHz
```

也可以自动生成扫频范围。下面命令扫 `20-100 kHz`，步长 `5 kHz`：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/solve_export_dataset_a_frequency.py --include-healthy --samples 3 --frequency-start-khz 20 --frequency-stop-khz 100 --frequency-step-khz 5 --linear-solver pardiso --heartbeat-s 30
```

如果希望更密集，可以把步长改成 `2.5`：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/solve_export_dataset_a_frequency.py --include-healthy --samples 3 --frequency-start-khz 20 --frequency-stop-khz 100 --frequency-step-khz 2.5 --linear-solver pardiso --heartbeat-s 30
```

默认发射端为 `1..16`。例如完整健康基准：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/solve_export_dataset_a_frequency.py --only-healthy --tx 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --frequencies 30000,35000,40000,45000,50000,55000,60000,65000,70000 --linear-solver pardiso --heartbeat-s 30
```

生成健康加随机缺陷样本：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/solve_export_dataset_a_frequency.py --include-healthy --samples 1 --linear-solver pardiso --heartbeat-s 30
```

### 样本命名和续跑

缺陷样本命名为：

```text
dataset_a_frequency_sample_0001
dataset_a_frequency_sample_0002
...
```

现在默认不会从 `0001` 盲目重跑。若不传 `--start-id`，脚本会扫描当前 `output-root` 下已有的 `frequency_response/`、`metadata/`、`labels/`、`progress/` 和单工况 CSV，自动选择第一个不重叠的连续编号段。例如已有 `0001..0020`，再运行 `--samples 5` 会自动计划 `0021..0025`。

如果希望手动指定起点：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/solve_export_dataset_a_frequency.py --samples 10 --start-id 101 --linear-solver pardiso --heartbeat-s 30
```

若手动指定的编号已经存在，默认会直接报错，避免覆盖旧样本。只有明确要重算覆盖时才使用：

```powershell
--overwrite-existing
```

`--include-healthy` 现在也不会无条件重算健康基准。脚本会检查：

```text
frequency_response/<healthy-sample-id>_H_complex.npz
```

如果已有健康基准的 `tx_indices`、`frequencies_hz`、`completed_mask` 和矩阵 shape 都与本次请求一致，会跳过健康求解，只记录一行 `skipped_existing` 到 `manifest.csv`。需要强制重算健康基准时使用：

conda run --no-capture-output -n comsol python -u simple/f_domain/solve_export_dataset_a_frequency.py --include-healthy --samples 3 --frequency-start-khz 20 --frequency-stop-khz 100 --frequency-step-khz 2.5 --linear-solver pardiso --heartbeat-s 30

```powershell
--force-healthy
```

如果同一输出目录里需要保存不同频点/不同 tx 集合的健康基准，建议使用不同名字：

```powershell
--healthy-sample-id dataset_a_frequency_healthy_20_100k_step5
```

`manifest.csv` 会按 `sample_id` 合并历史记录，不再每次运行只保留本次样本。

和时域流式脚本一致，默认每个样本复用一个 COMSOL 模型/网格，逐工况修改 `tx` 和 `pzt_fc`，求解后立即导出并 `clearSolutionData()`。如果遇到 COMSOL 重复求解相关问题，可加 `--rebuild-each-case` 回退到每工况重建模型。

## 输出

默认输出目录：

```text
simple/f_domain/output/streaming_dataset_a_frequency_shell/
```

主要文件：

- `csv/frequency_response/*_txXX_fYYYYYHz_frequency_response.csv`: 每个工况一个 CSV，16 行接收通道复响应。
- `csv/frequency_response/<sample>_frequency_response.csv`: 当前样本累计 CSV；每个工况完成后都会重写更新。
- `frequency_response/<sample>_H_complex.npz`: 当前样本累计复频响矩阵；每个工况完成后都会重写更新。
- `metadata/<sample>.json`: 样本、模型、case problems、输出路径和 label 路径。
- `labels/`: 与时域脚本同坐标的缺陷厚度损失标签。
- `progress/<sample>_progress.jsonl`: 心跳进度日志。
- `manifest.csv`: 历史样本索引；多次运行会按 `sample_id` 合并更新。

单工况 CSV 字段：

```text
sample_id,dataset,defect_state,tx,frequency_hz,rx_channel,rx_pzt,
theta_deg,x_mm,y_mm,z_mm,real_ur_m,imag_ur_m,abs_ur_m,phase_rad
```

NPZ 字段：

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

这些频域输出后续用于构造：

```text
DeltaH    = H_damaged - H_healthy
RelDeltaH = DeltaH / (abs(H_healthy) + eps)
PhaseDiff = angle(H_damaged * conj(H_healthy))
```

再通过展开平板反投影生成 `pic(x)` 粗图，并与原始 `x_matrix` 一起作为 diffusion 条件输入。

## 频点 sensitivity 选择

先用同一组 `tx/frequency` 生成一个健康基准和若干缺陷样本。然后运行：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/select_sensitive_frequencies.py --healthy simple/f_domain/output/streaming_dataset_a_frequency_shell/frequency_response/dataset_a_frequency_healthy_H_complex.npz --damaged simple/f_domain/output/streaming_dataset_a_frequency_shell/frequency_response/dataset_a_frequency_sample_0001_H_complex.npz simple/f_domain/output/streaming_dataset_a_frequency_shell/frequency_response/dataset_a_frequency_sample_0002_H_complex.npz simple/f_domain/output/streaming_dataset_a_frequency_shell/frequency_response/dataset_a_frequency_sample_0003_H_complex.npz --top-n 15
```

如果使用默认输出目录和标准样本名，可以直接按样本编号选择参与 sensitivity 计算的缺陷样本：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/select_sensitive_frequencies.py --sample-ids 1-20 --top-n 15
```

也可以只用某些样本，或者用 glob 选择一批文件：

```powershell
conda run --no-capture-output -n comsol python -u simple/f_domain/select_sensitive_frequencies.py --sample-ids 1,3,5-8 --top-n 15
conda run --no-capture-output -n comsol python -u simple/f_domain/select_sensitive_frequencies.py --damaged-glob "dataset_a_frequency_sample_00*_H_complex.npz" --top-n 15
```

默认指标是：

```text
sensitivity(f) = mean_samples ||H_damaged(f) - H_healthy(f)||_2 / (||H_healthy(f)||_2 + eps)
```

### 推荐的频点选择指标

不要只按时域健康信号 FFT 幅值最大来选频点。健康信号能量大只说明该频率激励/传播强，不等价于缺陷敏感。对频域粗图和 diffusion 来说，更合理的选择目标是：

```text
频点应该同时满足：
1. 健康响应足够强，避免低信噪比；
2. 健康-损伤差分足够大；
3. 差分在多个缺陷样本上稳定；
4. 异常不是只来自单个 tx-rx 通道；
5. 相位/复数差分包含可用于定位的散射信息；
6. 最终频点集合覆盖低/中/高频，不全部挤在一个窄频段。
```

建议把每个频率 `f` 的候选评分拆成以下几个量。

#### 1. 健康响应可靠性

```text
R0(f) = median_{tx,rx} |H0(tx,rx,f)|
```

低于健康响应分布某个分位数的频点应剔除：

```text
R0(f) < percentile_f(R0, 5%)  => reject
```

物理解释：如果健康管在某个频点本来就几乎没有响应，`Hd-H0` 很容易被数值误差、边界残余反射或求解噪声主导。当前 `select_sensitive_frequencies.py` 中的 `--min-healthy-abs-percentile` 就是这个筛选。

#### 2. 相对缺陷敏感度

```text
S_rel_l2(f) = mean_samples ||Hd(:,:,f) - H0(:,:,f)||_2 / (||H0(:,:,f)||_2 + eps)
```

这是当前默认指标 `relative_l2`。它衡量该频点下“缺陷造成的复频响变化”相对于健康传播背景有多大。

物理解释：腐蚀减薄会改变局部导波波速、阻抗、散射和透射幅值。频域响应中 `Hd-H0` 可以近似看作缺陷引起的散射/扰动项；用健康响应归一化，可以减少某些频点因绝对幅值大而被错误偏好。

#### 3. L1 鲁棒敏感度

```text
S_rel_l1(f) = mean_samples sum_{tx,rx} |Hd-H0| / (sum_{tx,rx} |H0| + eps)
```

这个指标对应脚本里的 `relative_l1`。相比 L2，它不容易被单个特别大的通道支配。

物理解释：粗图反投影依赖多条传播路径共同约束缺陷位置。如果一个频点只在极少数通道上异常很大，L2 会给它很高分，但层析定位可能不稳；L1 更偏向“总体路径异常”。

#### 4. 样本稳定性

先对每个缺陷样本计算：

```text
s_i(f) = ||Hd_i(:,:,f) - H0(:,:,f)||_2 / (||H0(:,:,f)||_2 + eps)
```

再计算：

```text
S_stable(f) = median_i s_i(f) / (MAD_i(s_i(f)) + eps)
```

或使用更保守的：

```text
S_robust(f) = mean_i s_i(f) / (1 + std_i(s_i(f)) / (mean_i s_i(f) + eps))
```

物理解释：用于大规模生成粗图的频点应该对不同位置、不同尺寸、不同形状的腐蚀都有响应，而不是只对某一个偶然样本高敏感。这个指标能避免 diffusion 后续过拟合到少/数缺陷几何。

#### 5. tx-rx 路径参与度

令：

```text
a_j(f) = mean_samples |Hd_j(f) - H0_j(f)| / (|H0_j(f)| + eps)
j      = one tx-rx channel
```

用 participation ratio 估计有多少通道真正参与：

```text
P(f) = (sum_j a_j)^2 / (N * sum_j a_j^2 + eps)
```

其中 `N = n_tx * n_rx`，`P(f)` 越接近 1，说明异常分布在更多传播路径上；越接近 0，说明主要由少数通道贡献。

物理解释：管道层析需要不同入射角和绕行路径共同约束缺陷位置。只由一两条路径主导的频点可能适合做异常检测，但不适合生成稳定粗图。对于当前 `get_pic` 的展开平板反投影，建议优先选择 `S_rel_l2` 高且 `P(f)` 不太低的频点。

#### 6. 相位/复数敏感度

当前脚本里的 `phase_weighted` 近似为：

```text
PhaseDiff(tx,rx,f) = angle(Hd(tx,rx,f) * conj(H0(tx,rx,f)))

S_phase(f) = mean_{samples,tx,rx}
             |Hd-H0|/(|H0|+eps) * (1 + |sin(PhaseDiff)|)
```

也可以后续扩展为相位一致性：

```text
C_phase(f) = |mean_{samples,tx,rx} exp(i * PhaseDiff)|
```

物理解释：壁厚减薄不只改变幅值，也会改变导波相速度和相位累积。频域粗图如果要做相干反投影或 Born/Rytov 风格散射成像，复数相位是很重要的信息。但相位容易 wrap，早期应把 `sin/cos(PhaseDiff)` 或 `phase_weighted` 作为辅助指标，不要单独按相位排序。

#### 7. 频率多样性和波长尺度

选出的 10-15 个频点不应全部集中在一个窄频段。可使用：

```text
lambda(f) = c_eff / f
```

若取当前时域默认速度 `c_eff = 2522 m/s`，则：

```text
20 kHz  => lambda ~= 126 mm
50 kHz  => lambda ~= 50 mm
100 kHz => lambda ~= 25 mm
```

物理解释：低频波长长、传播更稳、对大缺陷和整体厚度变化更敏感，但空间分辨率较低；高频波长短、对小缺陷和边缘更敏感，但更容易受网格、阻尼、散射和模式复杂性影响。用于 diffusion 的粗图应保留低/中/高频互补信息。

实践上建议：

```text
先按 S_rel_l2 或 phase_weighted 排序；
剔除 R0 太低的频点；
剔除 P(f) 过低或样本稳定性太差的频点；
再从 20-100 kHz 中分频段选 10-15 个：
  low:  20-40 kHz
  mid:  40-70 kHz
  high: 70-100 kHz
```

### 推荐综合评分

如果后续把这些指标写进代码，建议使用下面的综合分数：

```text
Score(f) =
  zlog(S_rel_l2(f))
  + 0.5 * zlog(S_phase(f))
  + 0.5 * z(P(f))
  + 0.5 * z(S_stable(f))
  - penalty_low_R0(f)
  - penalty_band_crowding(f)
```

其中 `z()` 表示在候选频点之间做 robust z-score，`zlog()` 表示先 `log(x+eps)` 再做 robust z-score。`penalty_band_crowding` 用于避免最终频点都集中在同一小段频率。

当前最推荐的人工流程是：

```text
1. 20-100 kHz, step=5 kHz 粗扫；
2. 用 relative_l2 和 phase_weighted 分别排序；
3. 排除 healthy_response_below_floor；
4. 从排序靠前频点所在区域用 step=2.5 kHz 加密；
5. 最终选 10-15 个频点，保证低/中/高频都有覆盖；
6. 用 get_pic 粗图与 labels 的 Pearson/NRMSE/IoU 验证频点集合。
```

### 文献依据

- Huthwaite 和 Simonetti 的 high-resolution guided wave tomography 使用导波色散关系把速度/传播变化与厚度图联系起来，并结合 travel-time 与 diffraction tomography 思想。这支持“不能只看幅值，要同时关注相位/传播速度/散射信息”的频点选择。DOI: https://doi.org/10.1016/j.wavemoti.2013.04.004
- Huthwaite 的 improved scattering model 指出传统只依赖速度图或简单声学散射模型会遗漏 guided wave 与腐蚀缺陷之间的复杂散射，完整散射信息能提高分辨率。因此频点选择应重视复数差分 `Hd-H0`、相位变化和多路径一致性，而不是只选健康幅值最大的频点。论文页面: https://pmc.ncbi.nlm.nih.gov/articles/PMC5134316/
- pipe high-order helical modes tomography 说明管道层析精度依赖可用入射角范围，高阶螺旋路径能增加角度覆盖。因此频点选择要考虑 tx-rx 路径参与度 `P(f)`，避免只由少数路径贡献。DOI: https://doi.org/10.1016/j.ndteint.2014.03.010
- Rao、Ratassepp 和 Fan 的 guided-wave FWI 工作在频域离散频率上做 waveform misfit 优化，并从低频逐步到高频；这支持“频点集合应覆盖低/中/高频，低频提供稳定背景，高频提供细节”的策略。PubMed: https://pubmed.ncbi.nlm.nih.gov/26955027/
- FWI 重建精度分析表明，使用完整波形信息可以更准确重建腐蚀厚度图，但计算成本更高；因此本文把多频复响应用于粗图和 diffusion 条件，而不是直接用单一频点做最终图。DOI: https://doi.org/10.1016/j.jsv.2017.04.017

输出目录默认为：

```text
simple/f_domain/output/frequency_selection/
```

主要文件：

- `frequency_sensitivity_ranked.csv`: 所有频点按 sensitivity 排序。
- `frequency_sensitivity_top15.csv`: 推荐的前 15 个频点。
- `frequency_sensitivity_top15_frequencies.txt`: 可直接复制给 `--frequencies` 的 Hz 列表。
- `frequency_sensitivity_summary.json`: 输入样本、参数和推荐频点记录。

可选指标：

```text
--metric relative_l2
--metric relative_l1
--metric absolute_l2
--metric phase_weighted
```

建议先用 `20-100 kHz, step=5 kHz` 做粗筛，再对高 sensitivity 区间用 `step=2.5 kHz` 加密，最后选 10-15 个频点用于大规模频域样本和粗图生成。
