# Diffusion 代码使用说明

本目录只实现 diffusion 侧的数据读取、训练、采样和评价；没有修改 `simple/f_domain` 的仿真代码，也没有修改 `simple/get_pic` 的粗图生成代码。

当前代码已经把 `new_plan.md` 中适合现阶段落地的结构作为唯一主线，不再保留旧版全局 embedding-only、频率 mean 的 RayOperator、旧 debug 配置等兼容分支。

## 1. 当前主线结构

```text
simple/diffusion/
  configs/
    dataset_a_256_base48.yaml       # 主推荐配置，256x256，base=48
    dataset_a_256_base64.yaml       # A6000/更大显存配置，256x256，base=64

  data/
    dataset.py                      # 匹配并读取 coarse_maps/x_matrix/labels
    transforms.py                   # label resize、x_matrix robust normalization、可选 roll

  models/
    x_encoder.py                    # x_matrix -> global FiLM embedding + TX-RX tokens
    unet.py                         # PicAdapter + FiLM + x-token cross-attention U-Net
    regressor.py                    # 确定性 baseline，使用同一新版条件结构
    diffusion.py                    # DDPM/DDIM diffusion，含 self-conditioning

  physics/
    losses.py                       # TV/range/coverage 等输出先验
    ray_operator.py                 # frequency-aware V1 ray consistency

  train_regressor.py                # 训练 deterministic baseline
  train_diffusion.py                # 训练 conditional diffusion
  sample_diffusion.py               # DDIM/DDPM 采样，可选物理引导
  evaluate.py                       # 评价 prediction.npy 或 checkpoint
  inspect_dataset.py                # 检查数据 shape 和数值范围
```

`runs/`、`__pycache__/`、`checkpoints/`、`samples/`、`eval/` 是运行生成物，不属于核心源码。

## 2. 新版网络路径

每个样本输入：

```text
pic:      [B,8,256,256]
x_matrix: [B,7,15,16,16]
target:   [B,1,256,256]
```

新版条件注入是固定路径：

```text
x_matrix
  -> reshape [B,105,16,16]
  -> CNN stem
  -> global embedding [B,cond_dim]      # 给 FiLM 使用
  -> TX-RX tokens [B,256,x_token_dim]   # 给 cross-attention 使用

pic
  -> PicEncoder
  -> multi-scale zero-conv features
  -> 注入 U-Net 各尺度

diffusion image path
  -> concat [y_t, self_condition, pic]
  -> Conditional U-Net
  -> predicted v / epsilon
```

注意：这里没有“只把 x_matrix 压成全局 embedding”的旧路径。全局 embedding 仍然保留，但只作为 FiLM 条件的一部分；TX-RX tokens 始终进入 cross-attention。


这是最关键的。

当前代码里的 token 不是文本 token，也不是图像 patch token，而是：
TX-RX 测量位置 token

也就是说，一个 token 大体对应一个：

$(tx,rx)$
发射-接收对。

因为原始 x_matrixx\_matrix**x**_**ma**t**r**i**x** 的最后两个维度是：

$16\times16$
你最后 flatten 出来的 token 数量也是：

$16\times16=256$
所以第 ii**i** 个 token 可以对应某一个：

$(tx_i,rx_i)$

一个 token = 一个 TX-RX 位置的多频多通道测量摘要

U-Net 某个图像位置可以根据自己的当前特征，选择性关注某些 TX-RX 测量 token。

QKV，Q是图片信息，KV都是norm以后的token

其中：

$Q\in\mathbb{R}^{B\times1024\times256}$

$K\in\mathbb{R}^{B\times256\times256}$

$V\in\mathbb{R}^{B\times256\times256}$

$A=QK^\top\in\mathbb{R}^{B\times1024\times256}$



这个矩阵含义非常重要：

$A_{b,i,j}$

第 bb**b** 个样本中，U-Net 第 ii**i** 个图像空间位置，对第 jj**j** 个 TX-RX token 的关注程度。

这一步非常关键：cross-attention 不改变 U-Net 的主干形状，只是向里面加一个从 x tokens 查询到的物理信息残差。

$Q=HW\text{ image tokens}$

$K=HW\text{ image tokens}$

$V=HW\text{ image tokens}$

self-attention图像内部空间位置互相交流。

cross-attention我这个位置的缺陷判断，更应该参考哪些 TX-RX 测量对。

## 3. 物理一致性

`RayOperator.consistency_loss()` 当前是 frequency-aware 实现：

```text
image -> ray projection [B, rays]
x_matrix -> observed frequency paths [B,F,rays]
loss = smooth_l1(robust_norm(pred expanded over F), robust_norm(obs))
```

旧的 `x_matrix[:, feature].mean(dim=frequency)` 路径已经删除。

训练中是否使用物理 loss 由：

```yaml
loss:
  lambda_phys_v1: 0.0
```

控制。默认仍为 0，先保证 diffusion 主训练稳定。需要打开时把它设为 `0.01` 左右，并保留 warmup。

采样期物理引导使用：

```powershell
--physics-guidance-scale 0.01
```

该引导只在 DDIM steps 下启用。

## 4. 推荐阅读顺序

1. `configs/dataset_a_256_base48.yaml`：确认路径、通道、模型规模。
2. `data/dataset.py`：确认一个样本如何变成 `pic/x_matrix/target`。
3. `models/x_encoder.py`：确认 `x_matrix` 如何生成 embedding 和 tokens。
4. `models/unet.py`：确认 PicAdapter、FiLM、cross-attention 如何堆叠。
5. `models/diffusion.py`：确认 noising、v-prediction、self-conditioning、sampling。
6. `physics/ray_operator.py`：确认 frequency-aware ray consistency。
7. `train_diffusion.py`：确认 loss、EMA、checkpoint 和断点续跑。
8. `sample_diffusion.py` 和 `evaluate.py`：确认采样和评价入口。

## 5. 数据准备

如果 `simple/get_pic/output_dataset` 不存在，先生成 V1 粗图和 `x_matrix`：

```powershell
conda run -n get_pic python simple/get_pic/generate_coarse_maps.py --sample-ids 1-40 --output-root simple/get_pic/output_dataset
```

应存在：

```text
simple/get_pic/output_dataset/coarse_maps/<sample>_coarse_maps.npz
simple/get_pic/output_dataset/x_matrix/<sample>_x_matrix.npz
simple/f_domain/output_dataset/streaming_dataset_a_frequency_shell/labels/<sample>_defect_depth_norm.npy
```

## 6. 检查数据

```powershell
conda run -n diffusion python simple/diffusion/inspect_dataset.py --config simple/diffusion/configs/dataset_a_256_base48.yaml --max-samples 3
```

## 7. 训练

确定性 baseline：

```powershell
conda run -n diffusion python simple/diffusion/train_regressor.py --config simple/diffusion/configs/dataset_a_256_base48.yaml
```

Diffusion：

```powershell
conda run -n diffusion python simple/diffusion/train_diffusion.py --config simple/diffusion/configs/dataset_a_256_base48.yaml
```

断点续跑：

```powershell
--resume latest
```

训练输出：

```text
loss_history.csv
metrics.jsonl
checkpoints/last.pt
checkpoints/best.pt
```

## 8. 采样和评价

普通采样：

```powershell
conda run -n diffusion python simple/diffusion/sample_diffusion.py `
  --config simple/diffusion/configs/dataset_a_256_base48.yaml `
  --checkpoint simple/diffusion/runs/dataset_a_256_base48/checkpoints/last.pt `
  --use-ema `
  --max-samples 5
```

带采样期物理引导：

```powershell
conda run -n diffusion python simple/diffusion/sample_diffusion.py `
  --config simple/diffusion/configs/dataset_a_256_base48.yaml `
  --checkpoint simple/diffusion/runs/dataset_a_256_base48/checkpoints/last.pt `
  --use-ema `
  --steps 50 `
  --physics-guidance-scale 0.01 `
  --max-samples 5
```

评价已有采样：

```powershell
conda run -n diffusion python simple/diffusion/evaluate.py `
  --config simple/diffusion/configs/dataset_a_256_base48.yaml `
  --pred-dir simple/diffusion/runs/dataset_a_256_base48/samples
```

直接评价 checkpoint：

```powershell
conda run -n diffusion python simple/diffusion/evaluate.py `
  --config simple/diffusion/configs/dataset_a_256_base48.yaml `
  --checkpoint simple/diffusion/runs/dataset_a_256_base48/checkpoints/last.pt `
  --model-type diffusion `
  --use-ema `
  --max-samples 5
```
