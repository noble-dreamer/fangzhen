# Diffusion 精细缺陷图像反演方案

本文档给出 `diffusion(pic(x), x)` 的可落地建模路线。目标是把当前频域仿真得到的 V1 粗图 `pic(x)` 和原始频域测量矩阵 `x` 作为条件，生成更细致的管道外表面腐蚀厚度损失图。

核心结论：

1. 第一版应使用 256 x 256 条件 DDPM/DDIM，不要直接上 Stable Diffusion 级大模型。
2. 主输入保持 `pic(x), x`：`pic(x)` 给空间物理先验，`x_matrix` 保留 tx-rx-frequency 频域复响应信息。
3. 类 PINN 结构不建议把 COMSOL 壳模型嵌入训练循环；应使用可微的低阶物理代理算子做约束。
4. 现阶段 V1 ray-tube 粗图足够进入 diffusion；V2-lite 相干通道可做消融，V3 线性/TV/FISTA 粗反演暂不作为主输入。
5. 当前本地可见 `output_dataset` 只有 40 个损伤样本，只够验证数据管线和 overfit。真正训练 diffusion 至少建议 300-1000 个样本，更理想是 1000+。

## 1. 当前数据

频域数据根目录：

```text
simple/f_domain/output_dataset/streaming_dataset_a_frequency_shell/
```

已选频点为 `physics_highfreq_quota` top15：

```text
40000,32500,20000,42500,50000,47500,52500,70000,72500,67500,75000,80000,82500,77500,95000
```

每个频域响应文件：

```text
frequency_response/<sample>_H_complex.npz

H_real:         (16, 16, 15)
H_imag:         (16, 16, 15)
completed_mask: (16, 15)
tx_indices
rx_indices
frequencies_hz
```

粗图和原始条件由 `get_pic` 生成：

```text
simple/get_pic/output_dataset/coarse_maps/<sample>_coarse_maps.npz
simple/get_pic/output_dataset/x_matrix/<sample>_x_matrix.npz
```

当前 V1 粗图默认尺寸：

```text
pic: (C_pic, 256, 256)
```

当前 V1 通道：

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

建议第一版进入 diffusion 的 `pic` 通道：

```text
ray_log_amp_loss
ray_relative_delta
ray_phase_change
low_frequency_band_map
mid_frequency_band_map
high_frequency_band_map
path_coverage
reliability_mask
```

`ray_delta_abs` 和 `valid_case_count` 可以保留做消融，但不建议作为核心输入。`ray_delta_abs` 容易被高幅值路径主导；`valid_case_count` 与 `path_coverage/reliability_mask` 信息有重叠。

`x_matrix` 当前结构：

```text
x: (C_x, F, TX, RX) = (7, 15, 16, 16)
```

通道：

```text
log_abs_delta
log_abs_reldelta
phase_cos
phase_sin
healthy_log_abs
damaged_log_abs
valid_mask
```

监督目标：

```text
labels/<sample>_defect_depth_norm.npy
```

本地 label 当前为 512 x 512 float32。训练时统一下采样到 256 x 256。厚度损失是连续值，输出应做连续回归，不应只做二值 mask。

## 2. 总体模型

最终形式：

```text
y = diffusion(pic(x), x)
```

训练时的扩散变量是目标缺陷图 `y`，不是粗图。条件包括两个分支：

```text
image condition: pic(x)
data condition:  x_matrix
```

推荐第一版网络：

```text
epsilon_or_v = ConditionalUNet(
    noisy_y_t,
    timestep=t,
    pic_condition=pic,
    data_embedding=Encoder(x_matrix)
)
```

输入拼接：

```text
UNet input channels = noisy_y_t(1) + pic_condition(C_pic)
```

`x_matrix` 不直接展平成大向量。先经过轻量编码器：

```text
x_matrix: (7, 15, 16, 16)
reshape:  (7 * 15, 16, 16)
Conv2D encoder -> global pooling -> MLP -> data_embedding(256)
```

`data_embedding` 通过 FiLM 注入每个 ResBlock：

```text
scale, shift = Linear(data_embedding)
h = h * (1 + scale) + shift
```

先不用 cross-attention。FiLM 更省显存、更稳，也更适合当前小数据规模。

## 2.1 Conda 环境

服务器环境文件：

```text
simple/diffusion/environment.yml
```

创建环境：

```bash
conda env create -f simple/diffusion/environment.yml
```

激活：

```bash
conda activate diffusion
```

验证 PyTorch 和 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

环境文件采用清华 conda 源解析常规依赖，并用 conda 安装 GPU 版 PyTorch：

```text
conda channels:
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

pytorch-gpu=2.5.1
torchvision=0.20.*
torchaudio=2.5.*
cuda-version=12.6
```

注意：conda 的 GPU PyTorch 不能做到“完全不下载 CUDA 相关库”。它不会安装系统级 CUDA Toolkit，也不会覆盖机器已有 CUDA 12.4；但会在 conda 环境内部安装 PyTorch 运行所需的 CUDA runtime、cuDNN、cuBLAS 等依赖。

当前推荐 `cuda-version=12.6`，而不是 12.4。原因是你当前 conda 源中的 `pytorch-gpu` 包没有可安装的 CUDA 12.4 组合；求解器提示 `pytorch-gpu=2.5.1` 对应的可用 GPU 构建需要 `cuda-version >=12.6,<13`。系统已安装 CUDA 12.4 不影响使用 PyTorch CUDA 12.6 runtime；PyTorch 运行时主要依赖 NVIDIA driver 向后兼容，而不是必须匹配系统 CUDA Toolkit 小版本。

同时包含：

```text
diffusers, accelerate, transformers, safetensors
matplotlib, seaborn, plotly, scikit-image, opencv, tifffile
tensorboard, jupyterlab
```

这些包足够完成训练、采样、指标统计和论文出图。

## 2.2 Diffusion 模型选择

推荐使用 Hugging Face `diffusers` 的组件作为第一版基础，而不是从零手写完整 DDPM 训练框架。

首选：

```text
diffusers.UNet2DModel + DDPMScheduler/DDIMScheduler
```

需要自己扩展的部分：

```text
1. 输入通道改为 noisy_y + pic_condition。
2. 增加 x_matrix encoder。
3. 在 UNet block 中用 FiLM 或 class embedding 注入 x_matrix embedding。
4. 输出 1 通道厚度图噪声或 v_prediction。
```

关于“使用预训练模型”的判断：

1. 不建议直接使用 Stable Diffusion 的预训练 U-Net 权重。它是 RGB 自然图像 latent diffusion，输入空间、通道语义、VAE latent 和本项目单通道厚度图差异太大，强行迁移收益不稳定。
2. 可以使用 `diffusers` 预训练/成熟代码模块，而不是预训练自然图像权重。这样能减少工程风险，但模型仍应在本项目数据上训练。
3. 若一定要利用预训练权重，建议只尝试 encoder/backbone 的部分迁移，例如用 ImageNet 预训练 ResNet/TIMM 编码 `pic` 或 `x_matrix`，不要把自然图像 diffusion 权重作为主模型。
4. 真正能减少训练 epoch 的方式是先训练 deterministic U-Net baseline，再用其权重初始化 diffusion U-Net 的 encoder/decoder 主干；这比迁移 Stable Diffusion 更符合本任务。

推荐实施顺序：

```text
Phase 1: diffusers UNet2DModel from scratch, base=48, overfit 10 samples
Phase 2: deterministic U-Net 预训练 pic+x -> y
Phase 3: 用 deterministic U-Net 的主干权重初始化 conditional diffusion
Phase 4: 加入 V1 ray forward consistency
```

如果后续样本达到 1000+，可以再考虑更复杂的预训练策略。

## 3. 推荐网络规模

### 3.1 主模型：256 x 256 Conditional DDPM/DDIM

默认配置：

```text
image_size: 256
target_channels: 1
pic_channels: 8
x_channels: 7
frequency_count: 15
tx_count: 16
rx_count: 16

unet_base_channels: 64
channel_mult: [1, 2, 4, 4]
num_res_blocks: 2
attention_resolutions: [32]
dropout: 0.05
time_embedding_dim: 256
data_embedding_dim: 256
prediction_type: v_prediction
train_diffusion_steps: 1000
sample_steps_ddim: 50-100
```

参数量预估：

```text
UNet:      30M-55M
x_encoder: 1M-3M
total:     35M-60M
```

如果训练样本少于 300，建议缩小：

```text
unet_base_channels: 48
channel_mult: [1, 2, 4, 4]
attention_resolutions: [32]
total: 20M-35M
```

如果只有几十个样本，不建议正式训练 diffusion，只做 overfit 和数据检查。

### 3.2 不建议的结构

暂不建议：

```text
Stable Diffusion / latent diffusion 大模型
DiT / 大型 Transformer backbone
完整 ControlNet 双分支大模型
直接把 x_matrix flatten 后全连接注入
512 x 512 从零训练
```

原因是你的任务是单通道科学图像反演，不是自然图像生成；数据量和显存预算都不支持盲目扩大模型。

## 4. 显卡资源建议

按常见显存配置估算：

```text
A6000:   48 GB
A5000:   24 GB each
2080 Ti: 11 GB each
```

推荐优先级：

1. 一张 A6000：最适合开发和主训练，单卡避免 DDP 调试成本。
2. 三张 A5000：适合稳定后用 DDP 提速，单卡显存仍可训练 256 中型模型。
3. 三张 2080Ti：只能做小模型或 debug，不建议做 512。

配置建议：

| 资源 | 推荐模型 | batch | 备注 |
| --- | --- | --- | --- |
| 1 x A6000 | base=64, 35M-60M | 8-12 | 首选；可开 EMA、AMP、较大 batch |
| 3 x A5000 | base=64, 35M-60M | 4/GPU | DDP 后 effective batch 12；适合正式训练 |
| 3 x 2080Ti | base=32 或 48, 12M-35M | 1-2/GPU | 必须 AMP + gradient checkpointing |

2080Ti 没有 bf16 优势，建议用 fp16。A6000/A5000 可优先尝试 fp16；如果框架和驱动支持 bf16，也可以测试 bf16 稳定性。

## 5. 训练路线

### Stage 0: 生成训练输入

先生成粗图和 x_matrix：

```bash
conda run -n get_pic python simple/get_pic/generate_coarse_maps.py \
  --sample-ids 1-40 \
  --output-root simple/get_pic/output_dataset
```

当服务器继续生成样本后，把 `--sample-ids` 扩展到完整范围。

### Stage 1: 确定性 baseline

先训练非 diffusion 的确定性 U-Net：

```text
y_pred = UNetRegressor(pic, x_matrix)
```

损失：

```text
L_det = L1(y_pred, y) + 0.5 * SSIM_or_MS_SSIM + 0.1 * BCE(mask_pred, mask)
```

目的：

1. 检查 `pic/x/y` 是否对齐。
2. 检查 label 下采样是否正确。
3. 检查模型是否能 overfit 10 个样本。
4. 建立 diffusion 的最低对照。

如果确定性 U-Net 都不能 overfit，说明数据读取、坐标、通道归一化或标签对齐有问题，不应继续训练 diffusion。

### Stage 2: 基础 conditional diffusion

训练目标：

```text
y_t = sqrt(alpha_bar_t) * y + sqrt(1 - alpha_bar_t) * noise
v_pred = UNet(y_t, t, pic, Encoder(x_matrix))
L_diff = MSE(v_pred, v_target)
```

第一版损失：

```text
L = L_diff
  + 0.05 * L1(x0_pred, y)
  + 0.001 * TV(x0_pred)
  + 0.01 * range_penalty(x0_pred)
```

其中：

```text
x0_pred = predict_x0_from_v_or_eps(y_t, v_pred, t)
range_penalty = mean(relu(-x0_pred)^2 + relu(x0_pred - 1)^2)
```

训练设置：

```text
optimizer: AdamW
lr: 1e-4 for base=48, 5e-5 for base=64
weight_decay: 1e-4
lr_schedule: cosine with warmup
ema_decay: 0.999 or 0.9999
amp: true
gradient_clip: 1.0
epochs: 按样本数定，先以 100k-300k steps 为量级
```

### Stage 3: 加入 PINN-like 物理约束

这里的 PINN-like 不是传统连续 PDE PINN。原因是：

1. 完整 COMSOL shell model 不可微，也不适合每个 batch 调用。
2. 当前可用数据是频域边界响应 `H(tx,rx,f)` 和缺陷厚度图。
3. 更合理的是构造可微低阶物理代理算子，把输出缺陷图重新投影到观测空间。

推荐物理约束分三层。

#### 3.1 输出物理先验

腐蚀厚度损失应满足：

```text
0 <= y <= 1
空间上相对平滑
缺陷区域局部连通
低覆盖区域不做过强监督
```

损失：

```text
L_range = mean(relu(-x0_pred)^2 + relu(x0_pred - 1)^2)
L_tv    = total_variation(x0_pred)
L_cov   = mean(path_coverage * abs(x0_pred - y))
```

建议权重：

```text
lambda_range = 0.01
lambda_tv    = 1e-4 到 1e-3
lambda_cov   = 0.05
```

#### 3.2 V1 ray 前向一致性

构造一个稀疏 ray operator：

```text
A_v1: y(theta,z) -> g_pred(tx,rx,f)
```

近似：

```text
g_pred[m] = sum_p A_v1[m,p] * y[p]
```

观测目标从 `x_matrix` 取：

```text
g_obs = log_abs_reldelta 或 ray_relative_delta 对应的路径标量
```

损失：

```text
L_phys_v1 = mean_valid( huber( normalize(g_pred) - normalize(g_obs) ) )
```

关键要求：

1. `A_v1` 使用与 `get_pic` 相同的 tx/rx 位置、helical_orders、sigma_ray、频点列表。
2. 对每个 batch 不重新生成全矩阵；预先缓存稀疏索引或 ray mask。
3. 对 `g_obs` 做 per-sample robust normalization，避免幅值尺度主导。
4. 只在中后期加入，避免训练早期被粗物理算子锁死。

建议权重日程：

```text
step 0-20k:       lambda_phys = 0
step 20k-80k:     lambda_phys 从 0 线性升到 0.01
step 80k 之后:    lambda_phys = 0.01 到 0.05
```

#### 3.3 V2-lite 相干一致性

如果后续实现了 V2-lite：

```text
born_coherent_abs
phase_consistency
```

则可加入：

```text
A_born: y(theta,z) -> complex_deltaH_pred(tx,rx,f)
L_phys_born = huber(real/imag or abs/phase normalized residual)
```

但 V2-lite 只作为可选项。若相位补偿不稳定，它可能降低效果。

### Stage 4: 采样期物理引导

除训练损失外，采样时也可以做类似 Diffusion Posterior Sampling 的物理引导。

每个 DDIM step 得到 `x0_pred` 后，计算：

```text
E_phys = L_phys_v1(x0_pred, x_matrix)
grad = dE_phys / d y_t
y_t = y_t - eta_phys * grad
```

建议：

```text
eta_phys: 0.01 到 0.1 之间网格搜索
只在最后 50%-70% denoising steps 开启
先只用于验证，不进入第一版训练闭环
```

采样期引导的优点是可以不改变训练模型，便于做消融。

## 6. 数据增强

管道周向是周期坐标，可以做物理一致增强。

推荐：

```text
theta circular roll
z small crop/shift only if不破坏 tx/rx 几何
轻微高斯噪声加到 x_matrix 的 log/phase 通道
随机丢弃少量频点或 rx 通道，用 valid_mask 标记
```

周向 roll 必须同时处理：

```text
pic:      theta 维 circular roll
label:    theta 维 circular roll
x_matrix: tx/rx 维 circular roll 相同 offset
```

如果暂时不实现 tx/rx 同步 roll，则不要只 roll 图像，否则 `pic/y` 与 `x_matrix` 的物理方位会不一致。

不建议：

```text
上下翻转 z
左右镜像 theta
随机旋转普通图像角度
强颜色增强
```

这些操作会破坏 tx/rx 几何或管道坐标含义。

## 7. 消融实验

至少做以下模型：

```text
A0: deterministic UNet(pic only)
A1: deterministic UNet(pic + x_matrix)
B0: diffusion(pic only)
B1: diffusion(x_matrix only)
B2: diffusion(V1 pic + x_matrix)
B3: diffusion(V1 pic + x_matrix + output priors)
B4: diffusion(V1 pic + x_matrix + output priors + V1 forward consistency)
B5: diffusion(V1+V2-lite pic + x_matrix + physics)
```

判断逻辑：

1. `A1 > A0`：说明 `x_matrix` 提供了粗图之外的信息。
2. `B2 > B0/B1`：说明 `pic(x), x` 组合有效。
3. `B4 > B2`：说明 PINN-like 物理一致性有帮助。
4. `B5 > B4`：才保留 V2-lite；否则不要继续投入 V2/V3。

## 8. 评价指标

连续厚度图指标：

```text
MAE
RMSE
NRMSE
Pearson correlation
SSIM / MS-SSIM
volume error = abs(sum(pred)-sum(label)) / sum(label)
```

缺陷区域指标：

```text
IoU at thresholds: 0.1, 0.2, 0.3
Dice
precision/recall
centroid error in theta-z
top-k localization distance
```

物理一致性指标：

```text
V1 forward residual
coverage-weighted false positive rate
low-coverage hallucination rate
frequency/path residual distribution
```

论文中建议同时报告：

```text
image quality: MAE/RMSE/SSIM
localization: centroid error/top-k distance/IoU
physics: forward residual
ablation: no physics vs physics
```

## 9. 推荐代码结构

后续实现建议：

```text
simple/diffusion/
  README.md
  configs/
    dataset_a_256_base48.yaml
    dataset_a_256_base64.yaml
  data/
    dataset.py
    transforms.py
  models/
    unet.py
    x_encoder.py
    regressor.py
    diffusion.py
  physics/
    ray_operator.py
    losses.py
  train_regressor.py
  train_diffusion.py
  sample_diffusion.py
  evaluate.py
```

第一批代码优先级：

1. `dataset.py`：读取 `coarse_maps/x_matrix/labels`，统一 256 x 256。
2. `train_regressor.py`：确定性 U-Net baseline。
3. `unet.py + x_encoder.py`：条件 diffusion 主体。
4. `train_diffusion.py`：基础 DDPM/DDIM 训练，不加物理前向。
5. `physics/ray_operator.py`：复用 `get_pic` 的 ray 几何，加入 V1 前向一致性。

## 10. 推荐实验顺序

### 10.1 当前 40 样本

只做：

```text
1. 生成粗图。
2. 训练 deterministic UNet overfit 10 samples。
3. 训练小 diffusion overfit 10 samples。
4. 检查生成图和 label 是否坐标一致。
```

不要用 40 样本报告最终泛化结论。

### 10.2 200-500 样本

可以做：

```text
base_channels = 48
256 x 256
deterministic baseline
diffusion(V1 pic + x_matrix)
简单 output priors
```

### 10.3 1000+ 样本

可以做正式实验：

```text
base_channels = 64
DDIM 50/100 step sampling
V1 forward consistency
V2-lite ablation
多随机种子
论文主表格
```

### 10.4 512 x 512

只有 256 稳定后再做：

```text
base_channels = 48
batch = 1-4
gradient_checkpointing = true
先加载 256 模型权重做微调
```

512 不应作为第一版主线。

## 11. 参考依据

本方案参考了以下方向：

1. DDPM：Ho, Jain, Abbeel, Denoising Diffusion Probabilistic Models. https://arxiv.org/abs/2006.11239
2. DDIM：Song, Meng, Ermon, Denoising Diffusion Implicit Models. https://arxiv.org/abs/2010.02502
3. ControlNet：Zhang, Rao, Agrawala, Adding Conditional Control to Text-to-Image Diffusion Models. https://arxiv.org/abs/2302.05543
4. DPS：Chung et al., Diffusion Posterior Sampling for General Noisy Inverse Problems. https://arxiv.org/abs/2209.14687
5. PINN：Raissi, Perdikaris, Karniadakis, Physics-informed neural networks. https://doi.org/10.1016/j.jcp.2018.10.045

对应到本项目：

```text
DDPM/DDIM -> 训练和快速采样框架
ControlNet -> 空间条件 pic(x) 的思想，但本项目使用轻量 concat/FiLM 而非大 ControlNet
DPS -> 采样期物理一致性引导
PINN -> 物理残差作为训练正则；本项目用低阶可微 guided-wave 代理算子替代完整 COMSOL PDE
```

## 12. 最终建议

短期最优路线：

```text
V1 pic + x_matrix
256 x 256
base_channels 48/64
Conditional DDPM/DDIM
先做 deterministic baseline
再做 diffusion
最后加入 V1 ray forward consistency
```

不要现在把时间投入到完整 V3 粗反演或大型 diffusion backbone。你的物理信息应主要通过三件事进入模型：

```text
1. physics_highfreq_quota 选频后的 x_matrix
2. V1/V2-lite pic(x) 空间条件
3. 可微 ray/Born 代理算子的物理一致性损失
```

这条路线最符合当前数据、计算资源和论文可解释性的平衡。
