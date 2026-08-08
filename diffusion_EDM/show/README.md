# EDM 网络结构可视化说明

本目录用于展示融合 SFAFNet 思想后的超声缺陷图 EDM 网络结构。

主要文件：

```text
network_architecture_edm_base48.tex
network_architecture_edm_base48.pdf
```

结构图对应配置：

```text
simple/diffusion_EDM/configs/dataset_a_256_base48_edm.yaml
```

TikZ 中的连接线只使用水平线、垂直线和直角折线，没有斜线或曲线。长距离反馈路径从模块外侧绕行，避免箭头覆盖文字。

## 1. 如何阅读六页 PDF

- 第 1 页是完整主流程：`pic + x_matrix + noisy target -> EDM denoise -> output`。检查条件信息是否真正到达最终缺陷图时，先看这一页。
- 第 2 页单独展开 EDM 数学过程，包括加噪、预条件系数、U-Net 原始输出、去噪结果和加权 MSE。
- 第 3 页专门展开物理坐标编码：真实 Hz 排序与频率编码、TX/RX 几何特征、位置向量如何加入 low/high token。
- 第 4 页展开条件主干，包括 PicAdapter 四尺度特征、动态低频和高频 token、U-Net block 数量、skip connection 和 5 个融合位置。
- 第 5 页展示单条 EDM 轨迹中的 Karras sigma、Euler/Heun 更新、self-conditioning 和可选 physics guidance。
- 第 6 页展示固定同一组 `pic+x_matrix` 条件和同一个 EMA 模型时，K 个独立初始噪声如何形成后验均值、uncertainty、分位区间、缺陷概率、熵和共识预测。

建议先从第 1 页建立全局概念，再用第 3 页检查物理编码，用第 4 页核对 block 数量和张量尺寸，接着阅读第 2、5 页确认训练与单轨迹采样公式，最后用第 6 页检查默认预测和不确定性输出。

## 2. 相对 `simple/diffusion` 保留了什么

当前 EDM 保留以下条件主干：

- `XMatrixEncoder` 接收 `x_matrix [B,7,15,16,16]`。
- 数据集同时提供 `frequency_hz [B,15]`、`tx_indices [B,16]` 和 `rx_indices [B,16]`。
- 全局 embedding 与 EDM 噪声 embedding 相加，通过 FiLM 进入每个 ResBlock。
- `PicEncoder` 将 `pic [B,8,256,256]` 编码为四尺度 PicAdapter 特征：
  - `p256 [B,48,256,256]`
  - `p128 [B,96,128,128]`
  - `p64 [B,192,64,64]`
  - `p32 [B,192,32,32]`
- ConditionalUNet 的 base48 尺度保持不变：Down 为 `256 -> 128 -> 64 -> 32`，Up 为 `32 -> 64 -> 128 -> 256`。
- 每个 ResBlock 都接收 FiLM，并叠加对应尺度的 PicAdapter 特征。
- 可选 physics loss/guidance 仍使用 `RayOperator.consistency_loss()`。

## 3. 图中新增的 SFAFNet 融合路径

- 内容自适应的 5-tap 低通核沿物理频率轴分解数据，不会把 15 个频点求均值丢弃。
- 分解前先按真实 Hz 排序；频率 Fourier/MLP 编码在频率轴折叠前加入每个切片。
- `x_low` 和 `x_high = x_matrix - x_low` 使用共享 XMatrixEncoder stem 编码。
- 编码器输出低频和高频 TX-RX token：`T_x^L,T_x^H`，每个形状为 `[B,256,256]`。
- 6 维最小 TX-RX 几何特征经无偏置线性层得到 `[B,256,256]` 位置编码，并同时加入 low/high token。
- mean/STD GATE 分别重标定空间、低频和高频特征。
- U-Net 空间位置在 32×32 层和 Middle 查询两组 token。
- 输出端使用较小的 spectral magnitude loss 抑制没有标签依据的高频杂点。

第 1 页中，绿色 x 路径分成两部分：

```text
原始 x_matrix -> global embedding -> FiLM bus
Hz/几何坐标 + 动态低/高频分解 -> T_low/T_high -> GATE + CAM bus
```

FiLM 提供全局样本条件；GATE + CAM 提供位置相关的低频和高频测量证据。两者不是重复路径。

## 4. 单模型多样本后验如何阅读

第 5 页只表示一次随机轨迹。实际 `sample_edm.py` 在条件不变、网络权重不变的前提下，顺序执行 K 次：

```text
z_k -> 同一个 EDM sampler(pic, x_matrix) -> y_k
stack(y_1,...,y_K) [K,B,1,256,256]
```

第 6 页中的三个输出分支含义不同：

- `posterior mean` 保存为 `*_prediction.npy`，是默认连续预测。
- `posterior std` 保存为 `*_uncertainty.npy`，表示 K 个条件样本在该像素上的幅值分歧。
- 分位数给出经验可信区间，`defect_probability` 表示超过缺陷阈值的采样比例，`defect_entropy` 表示缺陷存在性判断的分歧。
- `consensus_prediction` 只保留缺陷概率达到共识阈值的位置，用于抑制只在少数采样中出现的孤立亮点；它不会替换默认后验均值。

图中没有把 `1-path_coverage` 直接定义为 uncertainty。低覆盖区是否更不确定，需要通过 uncertainty 与 coverage、reliability 和真实误差的相关性来验证。若所有 K 个样本都产生相同假缺陷，单模型采样可能同时给出低 std 和高缺陷概率，因此共识规则不能解决系统性幻觉。

## 5. 与 `simple/diffusion` 的主要区别

原 `simple/diffusion` 使用 DDPM/DDIM 形式：

```text
y_t = sqrt(alpha_bar_t) * y_0 + sqrt(1 - alpha_bar_t) * epsilon
网络预测 v 或 epsilon
再根据 y_t、t 和网络输出恢复 x0
```

当前 EDM 使用连续噪声尺度 `sigma`：

```text
x = x0 + sigma * epsilon
D_theta(x, sigma) = c_skip * x + c_out * F_theta(c_in * x, sigma, pic, x_matrix)
loss = weight(sigma) * ||D_theta(x, sigma) - x0||^2
```

因此存在以下差异：

- EDM 没有 beta schedule。
- EDM 没有 `v_prediction` 等离散时间步训练目标。
- U-Net 原始输出经过 `c_skip/c_out` 预条件后，直接构成干净缺陷图预测。
- 噪声 embedding 使用 `log(sigma)/4`，而不是整数时间步 `t`。
- 采样从 `N(0,sigma_max^2 I)` 开始，而不是固定马尔可夫链末端噪声。
- 采样使用 Karras sigma 和 Euler/Heun 更新，不使用 DDPM posterior 或 DDIM alpha 公式。

## 6. 代码级对照

| 对比项 | `simple/diffusion` | `simple/diffusion_EDM` |
| --- | --- | --- |
| 扩散包装器 | `models/diffusion.py` 根据 beta/alpha schedule 组织训练与采样 | `models/edm.py` 实现 EDM 预条件、sigma 采样、加权去噪损失和 Karras sampler |
| 训练入口 | `train_diffusion.py` 训练离散时间步目标 | `train_edm.py` 直接用干净缺陷图监督 `D_theta(x,sigma)` |
| 采样入口 | `sample_diffusion.py` 使用 DDPM/DDIM 公式 | `sample_edm.py` 对 K 个独立噪声执行连续 sigma 与 Euler/Heun 采样，并输出后验统计 |
| 噪声变量 | 整数时间步 `t` | 连续 `sigma`，embedding 输入为 `log(sigma)/4` |
| 网络输入 | noisy label、self-condition 和 pic，共 `[B,10,256,256]` | 第一通道改为 `c_in*x`，总形状仍为 `[B,10,256,256]` |
| 网络输出 | 解释为 `v` 或 `epsilon`，再恢复 `x0` | `F_theta` 通过 `c_skip*x+c_out*F_theta` 直接构成去噪输出 |
| 条件主干 | PicAdapter、FiLM、单路 x-token cross-attention | 保留 PicAdapter/FiLM，增加物理频率 FDGM 和 gated low/high token fusion |
| 物理约束 | RayOperator 可作为训练损失或采样 guidance | 在 EDM 去噪输出上计算一致性，并可在形成 ODE derivative 前执行 guidance |
| 后验汇总 | 单次预测文件 | `uncertainty.py` 计算 mean/std/quantile/defect probability/entropy/consensus 和校准指标 |
| 主配置 | `configs/dataset_a_256_base48.yaml` | `configs/dataset_a_256_base48_edm.yaml` |

EDM 负责改变 U-Net 外部的概率建模方式；SFAF 融合负责增强 U-Net 内部的条件信息利用方式。二者相互独立：即使关闭 SFAF 融合，EDM 公式仍成立；即使更换 diffusion sampler，低频和高频条件融合仍可作为网络结构存在。

## 7. 对预测杂点问题的意义

缺陷标签通常只有局部厚度减薄，大部分背景接近零。EDM 和 SFAF 融合都不能单独保证输出没有幻觉，但新结构提供了两类约束：

- 低频 token 提供稳定结构证据，减少背景随高频噪声随机变化。
- 高频 token 必须通过 GATE 和空间 query 选择后才能进入 U-Net，避免全局注入所有高频响应。
- `lambda_spectral=0.01` 对 prediction 中额外的高频能量施加惩罚。
- TV、range、physics loss 和可靠 coarse map 仍然是必要补充。
- K 次后验样本提供像素级标准差和缺陷概率；低共识孤立点可在 `consensus_prediction` 中被清零。

正式训练时应观察 prediction 的背景标准差、阈值面积、IoU/Dice 和 physics consistency，而不能只看总 loss 是否下降。

## 8. 编译方法

在本目录执行：

```powershell
pdflatex -interaction=nonstopmode -halt-on-error network_architecture_edm_base48.tex
```

在项目根目录执行：

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory simple/diffusion_EDM/show simple/diffusion_EDM/show/network_architecture_edm_base48.tex
```
