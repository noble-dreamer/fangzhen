# Diffusion 代码使用说明

本目录代码只实现 diffusion 侧的数据读取、训练、采样和评价；没有修改 `simple/f_domain` 的仿真代码，也没有修改 `simple/get_pic` 的粗图生成代码。

## 1. 代码结构

```text
simple/diffusion/
  README.md                         # 建模方案说明，偏设计文档
  CODE_README.md                    # 当前文件，偏代码检查和运行说明
  task.md                           # 本次实现任务要求
  environment.yml                   # diffusion conda 环境

  configs/
    dataset_a_256_debug.yaml        # 当前 40 样本调试配置，默认入口
    dataset_a_256_base48.yaml       # 256x256、base=48 的正式小模型配置
    dataset_a_256_base64.yaml       # A6000 上可用的 base=64 配置
    dataset_a_2080ti_debug.yaml     # 小显存 debug 配置
    local_two_sample_smoke.yaml     # 本地两样本 smoke test 配置

  data/
    dataset.py                      # 发现样本、读取 pic/x_matrix/label、构建 DataLoader
    transforms.py                   # label resize、x_matrix 归一化、周期 roll 增强

  models/
    x_encoder.py                    # 把 x_matrix 编码为全局条件 embedding
    unet.py                         # FiLM 条件 U-Net 主干
    regressor.py                    # 确定性 baseline: y = UNet(pic, x)
    diffusion.py                    # 条件 DDPM/DDIM: y = diffusion(pic, x)

  physics/
    losses.py                       # range、TV、coverage-weighted L1 等输出物理先验
    ray_operator.py                 # 可微 V1 ray forward consistency 代理算子

  utils/
    config.py                       # YAML/JSON 配置读取、输出目录创建
    checkpoint.py                   # 保存/加载断点、latest checkpoint 查找
    ema.py                          # EMA 模型
    logging.py                      # CSV/JSONL 训练日志
    metrics.py                      # MAE/RMSE/SSIM/IoU/Pearson 等指标
    reproducibility.py              # 随机种子
    training.py                     # device、AMP、optimizer、scheduler、batch 搬运

  inspect_dataset.py                # 只检查数据读取，不训练
  train_regressor.py                # 训练确定性 baseline
  train_diffusion.py                # 训练 conditional diffusion
  sample_diffusion.py               # 用 diffusion checkpoint 采样并保存预测图
  evaluate.py                       # 评价 checkpoint 或已有 prediction.npy
```

`runs/`、`__pycache__/`、`checkpoints/`、`samples/`、`eval/` 是运行后生成的结果目录，不属于核心源码。检查代码时可以先忽略这些目录。

## 2. 每个核心文件的作用

### 2.1 配置文件

`configs/*.yaml` 控制所有路径、模型规模、训练超参数和 loss 权重。检查时先看：

```yaml
data:
  coarse_dir
  x_dir
  label_dir
  sample_ids
  pic_channels

model:
  x_channels
  frequency_count
  base_channels
  channel_mult

diffusion:
  timesteps
  prediction_type

loss:
  lambda_x0_l1
  lambda_tv
  lambda_range
  lambda_phys_v1
```

当前默认数据流假设：

```text
pic:      (8, 256, 256)
x_matrix: (7, 15, 16, 16)
label:    (1, 256, 256)
```

如果换成全频 33 个频点的 `x_matrix`，必须把配置里的 `model.frequency_count` 改成 `33`，否则 `XMatrixEncoder` 会主动报错。

### 2.2 数据读取

`data/dataset.py` 是检查数据流的第一重点文件。主要逻辑：

- `parse_sample_id_ranges()`：把 `1-40`、`1,3,5-8` 转成标准样本名。
- `UltrasonicDiffusionDataset._discover()`：匹配三类文件：
  - `coarse_maps/<sample>_coarse_maps.npz`
  - `x_matrix/<sample>_x_matrix.npz`
  - `labels/<sample>_defect_depth_norm.npy`
- `_load_pic()`：从 coarse npz 读取 `pic` 或 `pic_raw`，按 `pic_channels` 选择通道。
- `_load_x()`：读取 `x`，并调用 `normalize_x_matrix()` 做 per-sample robust z-score。
- `__getitem__()`：把 label 下采样到 256x256，返回 PyTorch tensor。
- `build_dataloaders()`：按配置构建训练和验证 DataLoader。

`data/transforms.py` 中最需要检查的是：

- `resize_label_nearest()`：与 `get_pic/evaluate_coarse_maps.py` 使用同样的最近邻下采样规则。
- `normalize_x_matrix()`：最后一个通道 `valid_mask` 保持 0/1，其它通道做 robust z-score。
- `RandomCircularRoll()`：可选增强。默认关闭；打开时会同时 roll `pic`、`label` 和 tx/rx 维度。

### 2.3 模型

`models/x_encoder.py`：

- 输入 `x_matrix: (B, Cx, F, TX, RX)`。
- reshape 为 `(B, Cx*F, TX, RX)`，例如 `(B, 105, 16, 16)`。
- 用小 CNN + global pooling 输出 `data_embedding`。

`models/unet.py`：

- `ConditionalUNet` 是图像主干。
- `ResBlock` 通过 FiLM 注入条件：`scale, shift = Linear(data_embedding + time_embedding)`。
- diffusion 模式下输入通道是 `noisy_y_t + pic`。
- regressor 模式下输入通道只有 `pic`。

`models/regressor.py`：

- 确定性 baseline。
- 形式是：

```text
embedding = XMatrixEncoder(x_matrix)
pred = sigmoid(UNet(pic, embedding))
```

它用于先验证数据是否对齐。如果 regressor 都不能 overfit 小样本，就不应继续看 diffusion 结果。

`models/diffusion.py`：

- `GaussianDiffusion` 实现 DDPM 训练和 DDIM/DDPM 采样。
- `training_losses()`：
  - 从真实 label `y` 加噪得到 `y_t`。
  - 拼接 `y_t` 和 `pic`。
  - 用 `x_matrix` embedding 做 FiLM 条件。
  - 默认训练 `v_prediction`。
- `sample()`：
  - `steps < timesteps` 时走 DDIM。
  - 否则走完整 DDPM reverse loop。

### 2.4 物理约束

`physics/losses.py`：

- `range_penalty()`：惩罚输出小于 0 或大于 1。
- `total_variation()`：周向带周期边界的 TV 平滑。
- `coverage_weighted_l1()`：用 `path_coverage` 让高覆盖区域的监督更重。
- `output_prior_losses()`：训练脚本统一调用的输出先验 loss。

`physics/ray_operator.py`：

- 构造标准 16 发 16 收、helical orders `(-1,0,1)` 的可微 ray kernel。
- `forward(image)` 把预测缺陷图投影成 ray/path 标量。
- `observed_from_x()` 从 `x_matrix` 中取观测侧标量，目前默认使用 `log_abs_reldelta`。
- `consistency_loss()` 对预测投影和观测投影做 robust normalization 后计算 smooth L1。

默认配置里 `lambda_phys_v1: 0.0`，因此这个约束默认不参与训练。先跑通基础训练后再打开。

### 2.5 训练入口

`train_regressor.py`：

- 读取 config。
- 构建 DataLoader。
- 构建 `ConditionalRegressor`。
- 训练 loss 默认是：

```text
L = lambda_l1 * L1
  + lambda_ssim * (1 - SSIM)
  + output_prior_losses
  + lambda_phys_v1 * V1_ray_consistency
```

- 写出：
  - `loss_history.csv`
  - `metrics.jsonl`
  - `checkpoints/last.pt`
  - `checkpoints/best.pt`

`train_diffusion.py`：

- 读取 config。
- 构建 `GaussianDiffusion`。
- 每个 batch 随机采样 timestep。
- 训练 loss 默认是：

```text
L = L_diffusion
  + lambda_x0_l1 * L1(x0_pred, y)
  + lambda_tv * TV(x0_pred)
  + lambda_range * range_penalty(x0_pred)
  + lambda_phys_v1 * V1_ray_consistency
```

- 支持：
  - `tqdm` 进度条
  - AMP
  - EMA
  - gradient clipping
  - cosine warmup scheduler
  - `--resume latest` 断点续跑

### 2.6 采样和评价

`sample_diffusion.py`：

- 加载 diffusion checkpoint。
- 可用 `--use-ema` 选择 EMA 权重。
- 对每个样本输出：
  - `<sample>_prediction.npy`
  - `<sample>_preview.png`
  - `manifest.csv`

`evaluate.py`：

- 可以评价已有 `prediction.npy`。
- 也可以直接加载 checkpoint 采样后评价。
- 输出：
  - `metrics.csv`
  - `summary.json`
- 指标包括：
  - MAE
  - RMSE
  - NRMSE
  - Pearson
  - volume error
  - IoU/Dice at 0.1/0.2/0.3
  - SSIM

`inspect_dataset.py`：

- 不训练，只读取数据并输出 shape、finite fraction、min/mean/max/percentile。
- 这是检查数据路径、通道和 label resize 是否正确的最快入口。

## 3. 推荐阅读顺序

按下面顺序检查，最不容易迷路：

1. 先读 `README.md` 和 `task.md`，确认模型目标是 `diffusion(pic(x), x)`，不是单独从粗图恢复 label。
2. 读 `configs/dataset_a_256_debug.yaml`，确认当前实验读哪些路径、哪些样本、哪些通道。
3. 读 `data/dataset.py` 的 `UltrasonicDiffusionDataset.__getitem__()`，确认一个样本如何从文件变成 `pic/x_matrix/target`。
4. 读 `data/transforms.py`，重点确认 label 下采样和 `x_matrix` 归一化。
5. 先读 `models/regressor.py`，它是最简单的数据到图像路径。
6. 再读 `models/x_encoder.py` 和 `models/unet.py`，确认 `x_matrix` 如何通过 FiLM 注入 U-Net。
7. 读 `train_regressor.py`，检查 baseline 的训练 loop、loss 记录、checkpoint 保存。
8. 读 `models/diffusion.py`，重点看 `training_losses()`、`predict_start_from_v()`、`sample()`。
9. 读 `train_diffusion.py`，确认 diffusion loss、x0 辅助 loss、EMA 和断点续跑。
10. 最后读 `physics/losses.py` 和 `physics/ray_operator.py`，因为物理约束默认关闭，不影响基础数据流。
11. 读 `sample_diffusion.py` 和 `evaluate.py`，确认训练后如何生成图和算指标。

检查时建议先回答这几个问题：

```text
1. 配置中的 coarse_dir/x_dir/label_dir 是否指向同一批 sample_id？
2. pic_channels 是否和 coarse_maps.npz 中 channel_names 一致？
3. x_matrix 的频点数是否等于 model.frequency_count？
4. label resize 后方向是否仍是 (z, theta)？
5. regressor 能否 overfit 10 个样本？
6. diffusion 的 x0_pred 辅助 MAE 是否随训练下降？
7. 采样图是否在 [0,1]，是否和 label 坐标方向一致？
```

## 4. 先生成 V1 粗图和 x_matrix

如果 `simple/get_pic/output_dataset` 不存在，先运行：

```powershell
conda run -n get_pic python simple/get_pic/generate_coarse_maps.py --sample-ids 1-40 --output-root simple/get_pic/output_dataset
```

生成后应存在：

```text
simple/get_pic/output_dataset/coarse_maps/<sample>_coarse_maps.npz
simple/get_pic/output_dataset/x_matrix/<sample>_x_matrix.npz
simple/f_domain/output_dataset/streaming_dataset_a_frequency_shell/labels/<sample>_defect_depth_norm.npy
```

## 5. 检查数据管线

```powershell
conda run -n diffusion python simple/diffusion/inspect_dataset.py --config simple/diffusion/configs/dataset_a_256_debug.yaml --max-samples 3
```

输出：

```text
simple/diffusion/runs/dataset_a_256_debug/dataset_summary.json
```

## 6. 训练确定性 baseline

先用 10-40 个样本验证能否 overfit：

```powershell
conda run -n diffusion python simple/diffusion/train_regressor.py --config simple/diffusion/configs/dataset_a_256_debug.yaml
```

断点续跑：

```powershell
conda run -n diffusion python simple/diffusion/train_regressor.py --config simple/diffusion/configs/dataset_a_256_debug.yaml --resume latest
```

## 7. 训练 conditional diffusion

```powershell
conda run -n diffusion python simple/diffusion/train_diffusion.py --config simple/diffusion/configs/dataset_a_256_debug.yaml
```

断点续跑：

```powershell
conda run -n diffusion python simple/diffusion/train_diffusion.py --config simple/diffusion/configs/dataset_a_256_debug.yaml --resume latest
```

训练日志：

```text
loss_history.csv
metrics.jsonl
checkpoints/last.pt
checkpoints/best.pt
```

## 8. 采样与评价

```powershell
conda run -n diffusion python simple/diffusion/sample_diffusion.py `
  --config simple/diffusion/configs/dataset_a_256_debug.yaml `
  --checkpoint simple/diffusion/runs/dataset_a_256_debug/checkpoints/last.pt `
  --use-ema `
  --max-samples 5
```

评价已有采样结果：

```powershell
conda run -n diffusion python simple/diffusion/evaluate.py `
  --config simple/diffusion/configs/dataset_a_256_debug.yaml `
  --pred-dir simple/diffusion/runs/dataset_a_256_debug/samples
```

或直接评价 checkpoint：

```powershell
conda run -n diffusion python simple/diffusion/evaluate.py `
  --config simple/diffusion/configs/dataset_a_256_debug.yaml `
  --checkpoint simple/diffusion/runs/dataset_a_256_debug/checkpoints/last.pt `
  --model-type diffusion `
  --use-ema `
  --max-samples 5
```

## 9. 物理约束

默认配置中 `lambda_phys_v1: 0.0`，先跑通基础数据流和训练。需要启用 V1 ray forward consistency 时，在配置中设置：

```yaml
loss:
  lambda_phys_v1: 0.01
  phys_start_step: 20000
  phys_warmup_steps: 60000
```

该约束使用 diffusion 目录内的可微低阶 ray 代理算子，不调用 COMSOL。
