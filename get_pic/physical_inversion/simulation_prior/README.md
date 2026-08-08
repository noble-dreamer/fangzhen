# v2 分布匹配 COMSOL 复 Rytov 语料

本目录只保留 Dataset A 的 v2 仿真派生物理先验。它用与正式缺陷分布一致、但随机种子完全独立的
COMSOL 频域语料，建立固定螺旋射线核和复频响到路径毫米深度的映射。它不修改 `f_domain` 正式
数据、`coarse_map`、`x_matrix` 或 EDM。

## 目标与边界

正式反演读取冻结健康复频响 `H0` 与每个正式样本的损伤复频响 `Hd`，输出毫米减薄图：

```text
prediction_mm   (256, 256), unit: mm
prediction_norm = prediction_mm / 9
```

正式 40 个样本的标签不会参与 `alpha`、通道截距、通道斜率、通道权重或路径观测的拟合。标签只在
反演结束后用于预览和描述性评价。

当前语料与先验已经完成：48 个独立 COMSOL 样本中，36 个用于拟合，12 个完全留出。留出路径验证为
RMSE `0.0453 mm`、Pearson `0.9037`，通过门限后得到：

```text
alpha[-1, 0, +1] = [0.35, 0.30, 0.35]
```

## v2 语料

```text
48 个独立随机 COMSOL 样本
36 fit + 12 held-out
TX=1, 16 RX, 正式 top15 频率
每例 1-3 个缺陷，位置范围 50-240 mm，减薄深度 0.8-4.2 mm
椭圆缺陷、随机长宽比与不规则 lobe
```

缺陷生成规则与正式 Dataset A 一致，但语料种子 `930001...930048` 与正式样本种子
`710001...710040` 完全隔离。只仿真 `TX=1` 是利用圆管和等间距环阵的旋转对称性：它的 16 个 RX
覆盖全部 16 种相对环偏移，随后映射到正式的 16 TX x 16 RX 共 256 条路径。

## 物理先验

在 64x64 展开管壁网格上，第 `o in {-1,0,+1}` 阶螺旋路径对应归一化几何核
`K_o(path, r)`。语料拟合得到非负且和为一的全局权重：

```text
K_fixed(path, r) = sum_o alpha_o K_o(path, r)
u_sim(path)      = sum_r K_fixed(path, r) d_sim(r)    [mm]
```

`K_fixed` 对所有正式样本相同；样本间变化的是损伤复频响 `Hd`、反演得到的路径深度 `u_hat` 和最终
图像。COMSOL 输出的是总复频响，不是三个螺旋阶次各自的直接测量，因此 `alpha` 是由独立仿真语料
验证的低参数物理先验，而不是三个模式能量的直接观测。

对健康响应 `H0` 与损伤响应 `Hd`，每个频点构造正则化复 Rytov 特征：

```text
epsilon_f = quantile_10%(abs(H0[:,:,f]))
z         = 1 + (Hd-H0) conj(H0) / (abs(H0)^2 + epsilon_f^2)
a         = log(abs(z))
phi       = angle(z)
```

按 `(relative_offset, frequency)` 在独立 COMSOL 语料上拟合 Huber 仿射模型：

```text
a   = b_amp   + m_amp   u + e_amp
phi = b_phase + m_phase u + e_phase
```

运行时，幅值和相位分别反解为毫米路径量，按相关性和路径残差加权融合；相位分支污染过高的通道被
剔除。最终使用固定核进行非负 SIRT 与周期 TV 求解：

```text
min_(0 <= d <= 5 mm) ||W^(1/2)(K_fixed d - u_hat)||^2 + TV(d)
```

固定核的有限路径覆盖会产生峰值低估、边界背景和空间模糊。这些是传统射线层析的分辨率限制，不能
与 EDM 的数据驱动后验重建能力混为一谈。

## 目录与产物

```text
configs/dataset_a_channel_prior.json
build_channel_corpus_plan.py
solve_channel_corpus.py
fit_channel_prior.py
channel_common.py
channel_prior.py
smoke_channel_prior.py
run_local_matched_channel_prior.ps1
finish_local_matched_channel_prior.ps1

output_matched_corpus/
  channel_corpus/                         # 48 个独立 COMSOL 频响和仿真毫米标签
  channel_corpus_plan.json
  channel_prior.json
  channel_prior.npz

output_dataset_matched_corpus/
  samples/<formal-id>/
    <id>_simulation_channel_prior_mm.npy
    <id>_simulation_channel_prior_norm.npy
    <id>_simulation_channel_prior_raw_grid.npy
    <id>_path_observation.npz
    <id>_diagnostics.json
    <id>_simulation_channel_prior_preview.png
  manifest.csv
  run_contract.json
  summary.json
```

## 本机工作流

所有命令从 `D:\lab_ultr\fz` 执行。配置使用 `simple/...` 相对路径，因此不要从
`D:\lab_ultr\fz\simple` 执行。

现有语料和先验已完成时，验证或重新生成正式输出只需要 `get_pic` 环境：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\physical_inversion\simulation_prior\smoke_channel_prior.py

conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\physical_inversion\run_simulation_channel_prior_inversion.py

conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\physical_inversion\validate_outputs.py
```

若语料已经完成但需要重新拟合先验，不启动 COMSOL：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\physical_inversion\simulation_prior\fit_channel_prior.py `
  --config simple\get_pic\physical_inversion\simulation_prior\configs\dataset_a_channel_prior.json `
  --force
```

只有缺少或明确要重建独立语料时，才使用 `comsol` 环境启动完整本机流程：

```powershell
powershell -ExecutionPolicy Bypass -File `
  simple\get_pic\physical_inversion\simulation_prior\run_local_matched_channel_prior.ps1 `
  -Cores 16 -HeartbeatSeconds 120
```

只核对语料计划而不启动 COMSOL：

```powershell
powershell -ExecutionPolicy Bypass -File `
  simple\get_pic\physical_inversion\simulation_prior\run_local_matched_channel_prior.ps1 `
  -DryRun
```

## 正式输出与 EDM 比较

当前 40 个正式样本的描述性结果为平均 Pearson `0.566`、Dice `0.289`、RMSE `0.338 mm`。这些指标
读取标签仅用于后置评价，不会影响预测。将 EDM 输出转换到相同毫米尺度后，可运行：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\physical_inversion\compare_with_edm.py `
  --edm-pred-dir simple\diffusion_EDM\runs\<formal-run>\evaluation
```

本目录不包含其他历史反演路线、历史输出或相应的运行入口。
