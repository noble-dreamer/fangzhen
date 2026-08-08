# 全波线性化复 Rytov 反演

本目录实现一个与现有代码完全隔离的传统物理基线：在健康管道背景处，用 COMSOL 弱壁损扰动得到
多频复频响 Jacobian，并将该 Jacobian 冻结后反演正式样本。所有新增代码和生成产物都位于
`simple/get_pic/rytov/`；流程只读正式 `f_domain/output_dataset`，不修改 `physical_inversion`、
`coarse_map`、`x_matrix`、EDM 或原始 COMSOL 数据。

准确名称是：

```text
COMSOL-derived full-wave linearized complex-Rytov inversion
COMSOL 派生的全波线性化复 Rytov 反演
```

它比固定射线层析保留更多波动物理，但仍不是 Full-Waveform Inversion（FWI）。

## 1. 输入和坐标

正式输入位于：

```text
simple/f_domain/output_dataset/streaming_dataset_a_frequency_shell/
```

每个健康/损伤响应保存为：

```text
H = H_real + i H_imag
shape = (tx, rx, frequency) = (16, 16, 15)
TX = 1..16
RX = 17..32
```

15 个频率按敏感度排名保存，并非升序。代码始终用 Hz 数值对齐轴，不依赖数组位置代表频率大小。
每个正式样本提供 `3840` 个复观测，即 `7680` 个实数观测分量。

输出图像坐标与正式标签一致：

```text
image[z_index, theta_index]
theta = 0..360 deg，周期边界
z     = 0..1000 mm
```

输出 `prediction_mm` 是壁厚损失毫米图；`prediction_norm` 严格等于
`prediction_mm / 9.0`。当前物理上限为 `0..5 mm`。

## 2. 复 Rytov 观测

记冻结健康响应为 `H0`，损伤响应为 `Hd`。为避免健康响应接近零时直接除法发散，按频率定义：

```text
epsilon_f = quantile_10%(abs(H0[:,:,f]))

R_epsilon = 1 + (Hd-H0) conj(H0) / (abs(H0)^2 + epsilon_f^2)
y          = log(abs(R_epsilon)) + i angle(R_epsilon)
```

`Re(y)` 是正则化对数幅值扰动，`Im(y)` 是主值相位扰动。健康幅值很低、比值接近零或相位接近
`+-pi` 分支切口的观测会降低权重或被剔除。该处理不能消除强缺陷下的相位跳变，只是让健康背景
附近的线性化更稳定。

## 3. COMSOL 全波 Jacobian

在展开管壁上定义周期超高斯基函数 `b_j(theta,z)`。第 `j` 个训练模型施加已知小壁损：

```text
d_j(theta,z) = delta_mm * b_j(theta,z)
```

COMSOL 仍求解完整频域壳体波动方程，包括当前模型中的频散、模态混合、干涉、吸收层、激励窗口和
接收 patch 加权。对训练健康响应 `Htrain0`、弱扰动响应 `Htrain_j(delta)` 和配置中的一个或多个
弱扰动深度，先拟合通过健康原点的原始复响应斜率：

```text
S_H[:,j] = sum_delta delta * (Htrain_j(delta)-Htrain0) / sum_delta delta^2
```

再使用稳定 Rytov 变换在正式健康响应处的解析一阶导数：

```text
J[:,j] = conj(H0) / (abs(H0)^2 + epsilon_f^2) * S_H[:,j]
```

因此 `J` 的每一行仍对应一个具体的 `tx-rx-frequency` 复观测，而不是几何射线长度。当前默认基函数
网格为 `16 x 8 = 128` 个系数，列顺序为：

```text
coefficient_index = z_index * theta_basis_count + theta_index
```

拟合算子前必须通过三项检查：

1. 训练健康响应与冻结正式健康响应的复数相对误差；
2. 训练响应的轴、完整性、有限值、扰动深度和模型指纹；
3. 原始 `dH` 斜率及其 Rytov 一阶预测对实际弱扰动响应的相对线性误差。

毫米灵敏度来自已知 COMSOL 弱扰动深度 `delta_mm`，不是从正式标签拟合出来的。

## 4. 冻结算子反演

正式样本只执行一次复 Rytov 变换，然后求解冻结线性模型：

```text
y_obs ~= J c
d_hat(theta,z) = sum_j c_j b_j(theta,z)
```

系数通过非负、有上界的迭代加权正则最小二乘得到：

```text
min_(0 <= c <= 5 mm)
    ||W_data^(1/2) (Jc-y_obs)||_2^2
  + lambda_ridge ||c||_2^2
  + lambda_tv TV_periodic_theta(c)
```

复方程在求解时将实部和虚部堆叠。周向差分采用周期边界，轴向不首尾相接。最后将基函数系数渲染到
`256 x 256` 全管展开图，并裁剪到物理范围。推理阶段不启动 COMSOL。

## 5. 与当前射线层析及 FWI 的区别

| 项目 | 当前 `physical_inversion` | 本目录全波线性化 Rytov | 频域 FWI |
| --- | --- | --- | --- |
| 数据进入图像求解前 | 15 个复频点压成 256 个路径毫米量 | 保留 3840 个复 `tx-rx-f` 观测 | 保留原始复观测 |
| 空间核 | 几何螺旋射线 `K_fixed` | COMSOL 弱扰动得到的复数、频率相关 `J` | 当前模型处的波动方程导数 |
| 推理中重算波场 | 否 | 否 | 每次模型更新都要前向/伴随求解 |
| 模型关系 | 路径积分线性近似 | 健康背景附近一阶线性化 | 非线性迭代 |
| 多模态和干涉 | 未显式进入空间核 | 在训练 Jacobian 中一阶保留 | 随每次更新重新计算 |
| 主要限制 | 路径压缩、有限孔径 | 弱散射范围、低维基、冻结背景 | cycle skipping、模型失配、计算量和梯度实现 |

本方法不能写成 FWI。FWI 的必要特征是：壁厚图每次更新后重新求解全波场，并由当前复波形残差计算
梯度。本方法只离线计算一次 `J`，正式反演是冻结算子上的凸/近凸正则问题。

## 6. 默认旋转近似和严格 all-TX

### `rotational_tx1`，默认可运行版本

默认配置只对 `TX=1` 逐基函数做 COMSOL 训练，再利用理想等间距圆环的旋转对称性组装其余 TX：

```text
J[t,r,f,z,q]
  = J_ref[(r-t) mod 16, f, z, (q-t) mod 16]
```

这里不做复共轭。该方式将默认训练量从约 `30960` 个频域工况降为约 `1935` 个工况。它保留了
COMSOL 全波弱扰动响应，但“虚拟 TX”是旋转对称近似，不是 16 个 TX 全部独立求解。
组装时会用正式健康响应的通道幅值比校正旋转后的原始 `dH/dd`，随后再应用每个正式通道自己的
Rytov 解析导数；该校正不能替代严格 all-TX 探测。

当前正式健康响应相对理想旋转重排的整体复数相对 L2 误差约为 `3.38%`，元素误差的 95 分位约为
`5.97%`，最大约为 `6.28%`。算子元数据会保存实际检查结果，并在超过配置阈值时拒绝组装。

### `all_tx`，严格训练版本

`dataset_a_fullwave_rytov_strict_all_tx.json` 对每个弱扰动基函数求解所有 16 个 TX，不使用旋转扩展。
它更适合作为最终严格物理基线，但 COMSOL 工况数约为默认版本的 16 倍。两种算子不能混用，输出目录
和指纹相互独立。

## 7. 环境和完整运行命令

所有命令从 `D:\lab_ultr\fz` 执行。计划、算子拟合、反演和验证使用 `get_pic` 环境；只有 COMSOL
求解使用 `comsol` 环境。

### 7.1 默认旋转版本

先生成不可变训练计划：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\rytov\build_training_plan.py
```

先检查计划和 COMSOL 工况，不启动客户端：

```powershell
conda run --no-capture-output -n comsol python -u `
  simple\get_pic\rytov\solve_training_corpus.py --dry-run
```

正式求解训练健康模型和所有弱扰动模型：

```powershell
conda run --no-capture-output -n comsol python -u `
  simple\get_pic\rytov\solve_training_corpus.py
```

拟合并冻结复 Jacobian：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\rytov\fit_operator.py
```

检查算子契约和数值质量：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\rytov\validate_operator.py
```

不读取正式标签地反演 40 个样本：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\rytov\run_inversion.py
```

若需要论文评价和三联预览，必须显式开启标签后处理：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\rytov\run_inversion.py --evaluate-labels
```

验证已保存的正式结果：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\rytov\validate_outputs.py
```

也可以使用封装脚本依次运行；默认不会执行正式标签评价：

```powershell
& simple\get_pic\rytov\run_local_pipeline.ps1
```

### 7.2 严格 all-TX 版本

上述每条 Python 命令追加：

```text
--config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json
```

先用 `solve_training_corpus.py --dry-run` 确认工况数和输出根目录，再决定是否启动长时间 COMSOL 求解。

### 7.3 Linux 服务器运行

以下命令从服务器的**项目根目录**执行，即当前目录下应存在 `simple/`。这与已有数据集生成命令的
目录约定一致。服务器需要有 `comsol_lzx` 与 `get_pic` 两个 conda 环境；COMSOL 路径沿用已有服务器的
`$HOME/comsol/comsol64/multiphysics`。

先生成默认旋转版本的计划。这一步不启动 COMSOL：

```bash
conda run --no-capture-output -n get_pic python -u \
  simple/get_pic/rytov/build_training_plan.py
```

先检查默认 `rotational_tx1` 的 1935 个工况，不启动 COMSOL client：

```bash
conda run --no-capture-output -n comsol_lzx bash -lc '
export COMSOL_ROOT="$HOME/comsol/comsol64/multiphysics"
export PATH="$COMSOL_ROOT/bin:$PATH"
python -u simple/get_pic/rytov/solve_training_corpus.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov.json \
  --cores 16 \
  --linear-solver pardiso \
  --heartbeat-s 300 \
  --dry-run
'
```

确认 dry-run 显示 `Pending COMSOL cases: 1935` 后，启动默认旋转一致性训练。中断后重复同一条命令，
`--resume-incomplete` 会只重算不完整或不兼容的训练样本：

```bash
conda run --no-capture-output -n comsol_lzx bash -lc '
export COMSOL_ROOT="$HOME/comsol/comsol64/multiphysics"
export PATH="$COMSOL_ROOT/bin:$PATH"
python -u simple/get_pic/rytov/solve_training_corpus.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov.json \
  --cores 16 \
  --linear-solver pardiso \
  --heartbeat-s 300 \
  --resume-incomplete
'
```

严格 `all_tx` 版本使用独立计划和输出根 `output_strict_all_tx/`。先生成严格计划并核对其约
30960 个工况：

```bash
conda run --no-capture-output -n get_pic python -u \
  simple/get_pic/rytov/build_training_plan.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json

conda run --no-capture-output -n comsol_lzx bash -lc '
export COMSOL_ROOT="$HOME/comsol/comsol64/multiphysics"
export PATH="$COMSOL_ROOT/bin:$PATH"
python -u simple/get_pic/rytov/solve_training_corpus.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json \
  --cores 16 \
  --linear-solver pardiso \
  --heartbeat-s 300 \
  --dry-run
'
```

严格版本正式求解命令如下。它不会使用旋转展开，运行时间显著长于 1935 工况版本：

```bash
conda run --no-capture-output -n comsol_lzx bash -lc '
export COMSOL_ROOT="$HOME/comsol/comsol64/multiphysics"
export PATH="$COMSOL_ROOT/bin:$PATH"
python -u simple/get_pic/rytov/solve_training_corpus.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json \
  --cores 16 \
  --linear-solver pardiso \
  --heartbeat-s 300 \
  --resume-incomplete
'
```

严格版本的训练计划有 `128` 个弱基函数，`basis_index` 是计划中的**零基**编号 `0..127`；它既不是
`f_domain/output_dataset` 的正式样本 ID，也不是最终反演时选择的 `--start-id/--end-id`。参数
`--start-basis-index` 和 `--end-basis-index` 是包含端点的基函数范围，范围内每个基函数的所有
`perturbation_depths_mm` 都会一起选择。当前严格配置只有一个 `0.25 mm` 深度，因此每 8 个基函数对应
`8 x 16 x 15 = 1920` 个 probe 工况；若配置多个弱扰动深度，工况数还要乘以深度数量。

训练健康响应 `rytov_training_healthy` 是整个 `output_strict_all_tx/` 共享的一份基线，不属于任何
`basis_index`。默认 `--healthy-baseline auto` 会校验并复用已有基线，缺失时在当前批次开始前求解一次，
所以第一批若尚未有基线会额外求解 `16 x 15 = 240` 个健康工况。后续批次可加
`--healthy-baseline require-existing`，这样基线缺失时会在启动 COMSOL 前直接报错，不会意外重复调度健康
工况。所有批次必须顺序使用同一个输出根目录，不能并行写同一份 manifest。

```bash
# batch 1: plan basis_index 0..7 (inclusive); auto-schedule shared healthy baseline
conda run --no-capture-output -n comsol_lzx bash -lc '
export COMSOL_ROOT="$HOME/comsol/comsol64/multiphysics"
export PATH="$COMSOL_ROOT/bin:$PATH"
python -u simple/get_pic/rytov/solve_training_corpus.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json \
  --start-basis-index 0 \
  --end-basis-index 7 \
  --healthy-baseline auto \
  --cores 16 \
  --linear-solver pardiso \
  --heartbeat-s 300 \
  --resume-incomplete
'

# batch 2: plan basis_index 8..15 (inclusive); require the shared baseline from batch 1
conda run --no-capture-output -n comsol_lzx bash -lc '
export COMSOL_ROOT="$HOME/comsol/comsol64/multiphysics"
export PATH="$COMSOL_ROOT/bin:$PATH"
python -u simple/get_pic/rytov/solve_training_corpus.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json \
  --start-basis-index 8 \
  --end-basis-index 15 \
  --healthy-baseline require-existing \
  --cores 16 \
  --linear-solver pardiso \
  --heartbeat-s 300 \
  --resume-incomplete
'
```

后续继续使用不重叠的 `16..23`、`24..31`，直到 `120..127`。`--probe-sample-id`（兼容别名
`--sample-id`）只接受训练计划内部的 `rytov_basis_...` 文件 ID；它也不是正式样本 ID。旧版
`--start-id/--end-id` 仍可使用，但仅作为一基的 Rytov 基函数序号兼容别名，并会打印弃用警告；新任务应始终
使用带 `basis` 名称的参数。训练语料脚本不会枚举或读取正式 `f_domain/output_dataset` 的大量样本。每批完成后
可用同一范围追加 `--dry-run --healthy-baseline require-existing`，确认该批 `pending=0`；若范围内的所有
probe 已完整且基线有效，脚本不会启动 COMSOL。

训练语料完整后，在同一项目根目录执行严格版本的算子拟合、验证和正式反演。下面无范围参数的
`run_inversion.py` 保持原有兼容行为：只处理配置中声明的正式样本（当前为 `1..40`）：

```bash
conda run --no-capture-output -n get_pic python -u \
  simple/get_pic/rytov/fit_operator.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json

conda run --no-capture-output -n get_pic python -u \
  simple/get_pic/rytov/validate_operator.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json

conda run --no-capture-output -n get_pic python -u \
  simple/get_pic/rytov/run_inversion.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json \
  --evaluate-labels

conda run --no-capture-output -n get_pic python -u \
  simple/get_pic/rytov/validate_outputs.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json
```

服务器的 `f_domain/output_dataset/.../frequency_response/` 若包含大量正式样本，应按**正式响应文件的数值
ID** 分批反演，而不是一次处理整个目录。以下示例只处理实际存在且连续的
`dataset_a_frequency_sample_0641_H_complex.npz` 到
`dataset_a_frequency_sample_0840_H_complex.npz`，端点均包含：

```bash
# formal response-ID batch 641..840; no COMSOL is started here
conda run --no-capture-output -n get_pic python -u \
  simple/get_pic/rytov/run_inversion.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json \
  --start-id 641 \
  --end-id 840 \
  --no-preview

# validate exactly the same saved batch without reading labels
conda run --no-capture-output -n get_pic python -u \
  simple/get_pic/rytov/validate_outputs.py \
  --config simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json \
  --start-id 641 \
  --end-id 840
```

`run_inversion.py --start-id/--end-id` 的语义与训练语料脚本不同：这里是
`f_domain/output_dataset` 中 `dataset_a_frequency_sample_<ID>_H_complex.npz` 的真实数值样本 ID，不是
`basis_index`、不是 `probe-sample-id`，也不是弱扰动 COMSOL 工况编号。它会先核对整个闭区间中的每个
响应文件都存在；任何缺失 ID 都会在反演前报错，避免静默得到不完整批次。显式范围允许处理服务器上存在而
本机配置 `formal_sample_ids` 尚未列出的 ID。

每个范围在未传 `--output-root` 时自动写入独立目录，例如：

```text
simple/get_pic/rytov/output_strict_all_tx/output_dataset/
  batches/ids_000641_000840/
    run_contract.json
    manifest.csv
    summary.json
    samples/...
```

下一批可使用 `--start-id 841 --end-id 1040`，其 manifest 不会覆盖前一批。运行契约记录精确的 ID
集合和包含端点；如果手动给 `--output-root`，必须为每个批次指定独立目录，且已有不同选择的契约会被拒绝，
不会静默覆盖。范围批次不需要修改严格配置中的 `formal_sample_ids`。只有需要指标和三联图时，才在相应批次
追加 `--evaluate-labels` 并去掉 `--no-preview`。

`--evaluate-labels` 只在冻结预测完成后读取标签以写指标和预览，不会参与 Jacobian、正则参数或反演
系数的拟合。若服务器只生成语料、拟合与评价在本机进行，可在训练完成后完整取回
`simple/get_pic/rytov/output_strict_all_tx/`；不要只取回 NPZ 响应文件，因为 operator 契约还会验证
训练健康响应、计划和 metadata。

## 8. 生成产物

默认根目录：

```text
simple/get_pic/rytov/output/
  training_plan.json
  training_corpus/
    frequency_response/
    metadata/
    progress/
    labels/
  fullwave_rytov_operator.npz
  fullwave_rytov_operator.json
  output_dataset/
    run_contract.json
    manifest.csv
    summary.json
    batches/ids_<start>_<end>/   # automatic only for formal --start-id/--end-id batches
      run_contract.json
      manifest.csv
      summary.json
      samples/<sample-id>/...
    samples/<sample-id>/
      <sample-id>_rytov_coefficients_mm.npy
      <sample-id>_rytov_prediction_mm.npy
      <sample-id>_rytov_prediction_norm.npy
      <sample-id>_rytov_data_fit.npz
      <sample-id>_rytov_diagnostics.json
      <sample-id>_rytov_preview.png
```

`data_fit.npz` 保存观测、线性预测、残差和权重，可用于报告数据空间误差。`diagnostics.json` 明确记录
算子模式、有效观测比例、优化状态，以及 `formal_labels_used_for_inversion=false`。

所有写路径都必须位于 `simple/get_pic/rytov/`。配置会拒绝指向 `physical_inversion`、`f_domain` 或 EDM
目录的输出路径。已有不兼容训练/算子产物不会被静默覆盖。

## 9. 正式标签使用边界

训练计划中的弱扰动深度和位置是本目录自行定义的 COMSOL 输入，不是正式数据标签。算子、数据权重、
正则参数和停止准则均不得读取正式标签；配置中的 `formal_sample_ids` 只定义无参数兼容运行的默认集合，
不限制显式 `--start-id/--end-id` 批次的服务器样本 ID。

`run_inversion.py` 默认不读取 `labels/`。只有显式传入 `--evaluate-labels` 后，程序才在预测已经完成后
加载标签，用于：

- Pearson、RMSE、SSIM、Dice、IoU；
- 峰值和周期表面位置误差；
- 三联预览图；
- 描述性汇总。

该分支不能回写算子或改变预测。论文中应将这些量表述为后验评价，而不是标定结果。

## 10. 验证和烟雾测试

不需要 COMSOL 的代数烟雾测试：

```powershell
conda run --no-capture-output -n get_pic python -u `
  simple\get_pic\rytov\smoke_test.py
```

它检查复 Rytov 零扰动、旋转索引、周期差分、非负有界反演、毫米归一化和写路径隔离。算子验证还会
检查：

- `J.shape == (16*16*15, 16*8)`；
- 复 Jacobian、权重、健康响应和轴均有限且一致；
- 正式健康文件和配置指纹匹配；
- 数值秩、条件信息和非零列；
- 训练健康闭合与旋转近似没有超过阈值。

## 11. 客观限制

即使核来自 COMSOL，本方法仍有以下限制：

1. Jacobian 只描述健康背景附近的一阶变化。正式缺陷深度可达数毫米，强散射、多重散射和模态转换的
   非线性部分不会被冻结 `J` 更新。
2. 主值复对数在相位接近 `+-pi` 时不连续；剔除分支污染观测会减少有效信息。
3. `128` 个低维基函数不能恢复任意锐利边界。输出到 `256 x 256` 是物理场渲染，不代表存在
   `65536` 个独立可辨识参数。
4. 多缺陷的复散射并不严格等于单缺陷响应相加，因此多个缺陷仍可能合并、遗漏或产生假阳性。
5. 默认虚拟 TX 额外包含旋转近似误差；严格 all-TX 可以移除该项，但不能移除一阶线性化误差。
6. COMSOL 与未来真实实验之间的材料、阻尼、换能器耦合和边界失配可能被错误解释为壁损。

因此公平结论应是：比较 EDM 与“冻结 COMSOL 全波线性化 Rytov 基线”的形态恢复和数据一致性差异。
除非另行实现每轮重算全波场和伴随梯度的非线性循环，不能宣称 EDM 已经与完整 FWI 比较。
