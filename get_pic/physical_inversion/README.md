# v2 分布匹配复 Rytov 物理反演

本目录只保留 Dataset A 的 v2 传统物理反演基线。它以独立 COMSOL 语料标定固定螺旋射线核和
复频响到路径毫米深度的关系，不修改 `f_domain` 正式数据、`coarse_map`、`x_matrix` 或 EDM。

语料构建、先验拟合和本机运行入口见
[`simulation_prior/README.md`](simulation_prior/README.md)；该子目录同样只记录当前 v2 流程。

当前复 Rytov 射线层析与频域 FWI 的严格区别、FWI 的可实施层级，以及与 EDM 的论文比较逻辑见
[`FWI_COMPARISON_AND_WRITING_GUIDE.md`](FWI_COMPARISON_AND_WRITING_GUIDE.md)。

## 数据流

```text
独立 COMSOL 随机缺陷语料
  48 samples, 36 fit + 12 held-out
  TX=1, 16 RX, top15 frequency response
              |
              v
固定 alpha 和 complex-Rytov channel prior
              |
正式 H_complex[TX,RX,F] + healthy H0
              |
              v
256 条路径的毫米观测 + K_fixed SIRT/TV
              |
              v
prediction_mm[256,256], prediction_norm=prediction_mm/9
```

正式 40 个样本的标签不会参与 `alpha`、通道斜率、截距或路径观测的拟合。仅在反演结束后，标签被
用于生成预览和描述性指标。

## 物理模型

第 `o in {-1,0,+1}` 阶螺旋路径在展开管壁网格上的归一化核为 `K_o(path,r)`。COMSOL 语料通过
六折交叉验证选择非负、和为一的固定权重：

```text
K_fixed(path,r) = sum_o alpha_o K_o(path,r)
u_sim(path)     = sum_r K_fixed(path,r) d_sim(r)    [mm]
```

当前已验证权重为：

```text
alpha[-1,0,+1] = [0.35, 0.30, 0.35]
```

对健康响应 `H0` 和损伤响应 `Hd`，每个频率使用正则化复 Rytov 特征：

```text
epsilon_f = quantile_10%(abs(H0[:,:,f]))
z = 1 + (Hd-H0) conj(H0) / (abs(H0)^2 + epsilon_f^2)
a   = log(abs(z))
phi = angle(z)
```

按相对环偏移和频率，在独立 COMSOL 语料上拟合 Huber 仿射关系：

```text
a   = b_amp   + m_amp   u + e_amp
phi = b_phase + m_phase u + e_phase
```

运行时，幅值和相位分别反解成路径毫米量，以相关性和毫米残差加权融合。相位分支污染严重的通道被
剔除。最终求解：

```text
min_(0<=d<=5 mm) ||W^(1/2)(K_fixed d-u_hat)||^2 + TV(d)
```

`K_fixed` 对所有正式样本相同；随样本变化的是 `Hd`、`u_hat` 和最终图像。固定核的有限路径覆盖会
导致峰值低估、边界背景和空间模糊，这是该传统射线基线的已知分辨率限制，不应与 EDM 的后验能力
混淆。

## 已验证产物

```text
simulation_prior/output_matched_corpus/
  channel_corpus/                 # 48 个独立 COMSOL 响应和仿真毫米标签
  channel_corpus_plan.json
  channel_prior.json/.npz

simulation_prior/output_dataset_matched_corpus/
  samples/<formal-id>/            # 40 个正式样本的 mm/norm/raw/observation/preview
  manifest.csv
  summary.json
```

独立留出语料路径验证：RMSE `0.0453 mm`，Pearson `0.9037`。

正式 40 样本仅作描述性评价：平均 Pearson `0.566`，Dice `0.289`，RMSE `0.338 mm`。该评价读取了
正式标签，但不会改变任何预测。

## 运行

从 `D:\lab_ultr\fz` 执行：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\physical_inversion\run_simulation_channel_prior_inversion.py
```

验证已保存输出：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\physical_inversion\validate_outputs.py
```

和 EDM 做统一毫米尺度比较：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\physical_inversion\compare_with_edm.py `
  --edm-pred-dir simple\diffusion_EDM\runs\<formal-run>\evaluation
```
