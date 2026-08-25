# diffusion_EDM：融合 SFAFNet 思想的超声缺陷图 EDM 模型

本文档说明当前模型如何将 SFAFNet 的空间域与频域融合思想适配到超声缺陷反演任务。

本目录实现一个面向超声管道缺陷反演的条件 EDM（Elucidated Diffusion Model）。模型根据两类条件信息生成厚度减薄缺陷图：

- 空间域条件 `pic [B,8,256,256]`：由射线粗成像得到的多通道空间先验。
- 物理频域条件 `x_matrix [B,7,15,16,16]`：保留 15 个真实扫频点以及 16×16 TX-RX 阵列响应。

当前版本在原有 EDM、XMatrixEncoder、PicAdapter、FiLM、self-conditioning 和 RayOperator 基础上，引入论文 `2502.14209v1.pdf` 中 SFAFNet 的核心思想：

1. 使用动态低通滤波器分解低频和高频信息，对应论文的 FDGM。
2. 使用均值和标准差生成通道门控，对应论文的 GATE。
3. 使用空间位置查询低频和高频 token，对应论文的 CAM。
4. 使用轻量输出频谱损失约束预测图中的高频杂点。

这里没有完整复制 SFAFNet。SFAFNet 原本处理自然图像去模糊，而本项目的频率维度是具有实际物理意义的超声扫频轴。实现重点是保留论文的“动态分解、门控选择、跨域注意力”思想，同时让每个模块符合当前数据含义。

## 1. 模型整体数据流

```text
物理频域分支
x_matrix [B,7,15,16,16]
frequency_hz [B,15]；tx_indices/rx_indices [B,16]
  ├─ 按 frequency_hz 排序频率轴
  ├─ PhysicalFrequencyEmbedding -> E_f [B,7,15,1,1]
  ├─ 排序后的原始 x_matrix + E_f -> 全局 embedding e_x [B,256]
  │                                      └─ 与 EDM 噪声 embedding 相加后进入 FiLM
  │
  └─ DynamicFrequencyDecomposer
       ├─ x_low  [B,7,15,16,16]
       └─ x_high [B,7,15,16,16] = x_matrix - x_low
             └─ 分别加 E_f，再经过共享 stem 和 token projection
                  ├─ content_low/high [B,256,256]
                  └─ 加同一份 TX-RX physical position [B,256,256]
                       ├─ T_low  [B,256,256]
                       └─ T_high [B,256,256]

空间域分支
pic [B,8,256,256]
  └─ PicEncoder/PicAdapter
       ├─ p256 [B,48,256,256]
       ├─ p128 [B,96,128,128]
       ├─ p64  [B,192,64,64]
       └─ p32  [B,192,32,32]

EDM 去噪分支
[c_in * noisy, self_condition, pic] [B,10,256,256]
  └─ ConditionalUNet
       ├─ ResBlock + FiLM + PicAdapter
       ├─ 32×32 层：GATE + 空间-频域 Cross-Attention
       ├─ Middle：GATE + 空间-频域 Cross-Attention
       └─ raw residual F_theta [B,1,256,256]

EDM 输出
D_theta(x,sigma) = c_skip * x + c_out * F_theta
  └─ 缺陷图 [B,1,256,256]
```

### 1.1 单模型多样本后验与默认预测

反问题中的同一个 `pic + x_matrix` 可能对应多个合理缺陷图，因此采样阶段不再把某一次随机采样当作最终答案。当前 `sample_edm.py` 对同一条件顺序执行 `K` 次 EDM 采样：

```text
同一条件 c = (pic, x_matrix, frequency_hz, tx_indices, rx_indices)
  ├─ z_1 -> EDM sampler -> y_1 [1,1,256,256]
  ├─ z_2 -> EDM sampler -> y_2 [1,1,256,256]
  ├─ ...
  └─ z_K -> EDM sampler -> y_K [1,1,256,256]
                  |
                  v
      posterior mean / std / quantile
      defect probability / entropy / consensus
```

默认 `*_prediction.npy` 已定义为后验均值，而不是第一个样本：

```text
prediction = mean(y_1, ..., y_K)                  [256,256]
uncertainty = population_std(y_1, ..., y_K)       [256,256]
```

这样同一个模型即可同时给出稳定的默认预测和像素级样本分歧。全部 `K` 个条件张量仍按顺序采样，不会一次性复制成大 batch，因此显存主要由单次 `256×256` EDM 采样决定；运行时间近似随 `K` 线性增加。

## 2. 两种“频域”必须区分

当前模型中存在两种不同概念的频域，不能混为一谈。

| 名称         | 数据                        | 物理含义                         | 使用位置                                                          |
| ------------ | --------------------------- | -------------------------------- | ----------------------------------------------------------------- |
| 超声物理频域 | `x_matrix [B,7,15,16,16]` | 15 个真实激励频点上的 TX-RX 响应 | DynamicFrequencyDecomposer、XMatrixEncoder、GATE、Cross-Attention |
| 输出图像频谱 | `RFFT2(prediction)`       | 预测缺陷图的二维空间频谱         | `lambda_spectral` 正则损失                                      |

SFAFNet 原论文主要对图像特征做频率子带分解。本项目没有把缺陷图 FFT 当成超声测量，也没有用 FFT/IFFT 替代 `x_matrix`。真正用于条件融合的是 `x_matrix` 的 15 个物理频点；输出图像 FFT 只用于抑制 prediction 中没有标签依据的高频 speckle。

## 3. 从 SFAFNet 提取了什么

论文中的基本单元是 GSFFBlock，由三个核心部分组成。

| SFAFNet 模块   | 论文中的作用                               | 当前项目中的适配                                            |
| -------------- | ------------------------------------------ | ----------------------------------------------------------- |
| FDGM           | 用动态低通核把图像特征分成低频和高频       | 沿`x_matrix` 的 15 个真实频点生成动态低通核               |
| GATE           | 根据均值和标准差重标定空间、低频、高频特征 | 分别门控 U-Net 空间特征、低频 TX-RX token、高频 TX-RX token |
| CAM            | 通过跨域注意力交换空间域和频域信息         | 每个 U-Net 空间位置查询低频和高频测量 token                 |
| Frequency loss | 约束恢复图像的频率分布                     | 用较小权重抑制预测缺陷图中的额外高频杂点                    |

论文消融结果说明，简单拼接空间和频率特征并不是最优选择。动态分解、GATE 和 CAM 联合使用，才能让网络根据输入内容选择互补信息。因此当前实现没有采用 `torch.cat(pic_feature, x_feature)` 后直接卷积的方式。

## 4. FDGM 如何改造成物理频率分解

### 4.1 输入与输出

当前 `DynamicFrequencyDecomposer` 接收：

```text
x_matrix [B,C,F,TX,RX]
C = 7, F = 15, TX = RX = 16
```

输出：

```text
x_low   [B,7,15,16,16]
x_high  [B,7,15,16,16]
weights [B,7,5]
```

默认滤波核长度 `K=5`。每个样本、每个测量通道都有一个独立的 5-tap 低通核。

### 4.2 动态滤波核如何生成

首先从输入中提取随频率变化的统计量：

```text
frequency_mean = mean(x_matrix, dim=(C,TX,RX))  -> [B,F]
frequency_std  = std(x_matrix, dim=(C,TX,RX))   -> [B,F]
statistics = stack(mean, std)                    -> [B,2,F]
```

这里仅对 `C/TX/RX` 做统计，频率轴 `F=15` 始终保留。代码没有把 15 个频点平均成一个值。

统计量经过一维卷积和池化后，为每个测量通道生成低通核：

```text
[B,2,15]
  -> Conv1d -> SiLU -> AdaptiveAvgPool1d(1) -> Conv1d
  -> [B,7,5]
  -> Softmax over K
```

Softmax 保证 `W >= 0` 且 `sum(W, dim=K)=1`，因此每个动态核具有加权平均的基本低通性质。filter generator 最后一层采用零初始化，训练开始时对应均匀低通核，避免随机初始化立即破坏频率结构。

### 4.3 低频和高频如何计算

```text
x_low  = LPF_W(x_matrix)
x_high = x_matrix - x_low
```

所以严格满足：

```text
x_low + x_high = x_matrix
```

这意味着频率分解在进入编码器前是信息守恒的。低频分支表示随频率缓慢变化的传播衰减和稳定背景，高频分支表示由散射、干涉和缺陷引起的频率敏感残差。最终物理含义仍由训练数据学习，而不是人为指定某个频点一定对应某类缺陷。

### 4.4 为什么读取数据时必须同时取得并处理真实频率

`x_matrix` 的 `F` 轴不能只用数组下标 `0...14` 表示。当前 15 个频点来自灵敏度选频结果，NPZ 保存的是选频顺序，不是物理频率递增顺序。一个真实样本中的原始顺序为：

```text
[40, 32.5, 20, 42.5, 50, 47.5, 52.5,
 70, 72.5, 67.5, 75, 80, 82.5, 77.5, 95] kHz
```

如果直接沿这个数组顺序执行 1D 卷积，长度为 5 的窗口会错误地把 `40 kHz -> 32.5 kHz -> 20 kHz` 当成连续邻频。这样会产生三个问题：

1. 动态低通核处理的是“选频排名邻居”，而不是“物理频率邻居”。
2. `x_high = x - x_low` 会混入由错误频率顺序造成的人工跳变。
3. 不同数据文件只要选频保存顺序不同，同一卷积通道位置就会代表不同 Hz，模型难以复用权重。

因此数据集读取阶段不仅返回 `x_matrix`，还必须返回与其 `F` 轴严格对齐的 `frequency_hz`。当前实现的职责划分是：

```text
NPZ 文件
  ├─ x              [7,15,16,16]
  ├─ frequency_hz   [15]
  ├─ tx_indices     [16]
  └─ rx_indices     [16]
        |
        v
Dataset：读取、检查形状、检查频率为正且不重复，保留原始对应关系
        |
        v
XMatrixEncoder：对每个 batch 样本同步排序 frequency_hz 和 x_matrix 的 F 轴
```

同步排序过程为：

```text
order = argsort(frequency_hz)
sorted_frequency_hz = gather(frequency_hz, order)
sorted_x_matrix     = gather(x_matrix, dim=F, order)

结果：
[20, 32.5, 40, 42.5, 47.5, 50, 52.5,
 67.5, 70, 72.5, 75, 77.5, 80, 82.5, 95] kHz
```

必须使用同一个 `order` 同时重排频率值和 `x_matrix`，不能只对 `frequency_hz` 调用 `sort()`，否则每个响应切片会被标上错误的频率。排序在 X 编码器内部对副本执行，batch 中供 RayOperator 等其他路径使用的原始 `x_matrix` 不会被原地修改。

这里的排序只改变顺序，不执行插值，也不会把 15 个频点变成等间距网格。例如 `20 -> 32.5 kHz` 与 `40 -> 42.5 kHz` 的间隔仍然不同；这个真实间隔信息由下面的 `PhysicalFrequencyEmbedding` 显式提供。

### 4.5 PhysicalFrequencyEmbedding 如何表示真实 Hz

对于每个真实频率 `f`，首先使用配置中的参考频率 `f_ref=100000 Hz` 得到无量纲坐标：

```text
u = f / f_ref
```

然后构造 8 维确定性频率特征：

```text
phi(f) = [
  u,
  log(1 + u),
  sin(2*pi*u), cos(2*pi*u),
  sin(4*pi*u), cos(4*pi*u),
  sin(8*pi*u), cos(8*pi*u)
]
```

其中 1、2、4 阶谐波同时提供平滑的全局频率位置和不同尺度的频率变化。完整张量变化为：

```text
frequency_hz                     [B,15]
  -> phi(f)                      [B,15,8]
  -> Linear(8,28) + SiLU
  -> Linear(28,7)                [B,15,7]
  -> permute + broadcast         [B,7,15,1,1]
                                  = E_f
```

输出维度为 7，是因为 `x_matrix` 有 7 个测量特征通道。`E_f` 在 TX-RX 平面上广播，并在频率轴折叠前分别进入三条共享编码路径：

```text
raw_input  = sorted_x + E_f
low_input  = x_low    + E_f
high_input = x_high   + E_f

[B,7,15,16,16] -> reshape -> [B,105,16,16] -> shared stem
```

这样做有四个原因：

1. 模型知道当前切片是 `20 kHz` 还是 `95 kHz`，而不只知道它位于排序后的第几个槽位。
2. 当未来样本使用不同选频集合时，相同 Hz 仍具有一致的坐标表示。
3. 频率间隔不均匀时，MLP 可以根据真实 `u` 区分大间隔和小间隔。
4. 位置编码只增加 455 个参数，不需要把 token 数扩展为 `15*16*16=3840`。

动态分解仍只作用于排序后的测量值，所以始终满足 `x_low + x_high = sorted_x`；`E_f` 是进入共享 CNN 的条件坐标，不参与这条守恒等式。

对应代码：[models/physical_encoding.py](models/physical_encoding.py) 和 [models/x_encoder.py](models/x_encoder.py)。

## 5. 共享编码器与 TX-RX 物理位置

### 5.1 一个 token 到底代表什么

当前每个 low/high token 对应一个二维矩阵格 `(tx_row, rx_column)`，共有 `16×16=256` 个 token。它聚合了该 TX-RX 对的 15 个频点，因此不能解释成“某一个频率的 token”；频率位置已由上一节的 `E_f` 在聚合前注入。

`PhysicalTxRxPositionEmbedding` 现在只为每个矩阵格构造 6 维最小物理特征。先定义：

```text
i_tx, i_rx     TX/RX 在各自圆环内的 0-based 序号，范围 0...15
theta_tx       2*pi*i_tx/16
theta_rx       2*pi*i_rx/16
delta          wrap(theta_rx-theta_tx) 到 [-pi, pi)
L              管长，当前为 1000 mm
R              管中面半径，当前为 155 mm
dz             z_rx-z_tx，当前为 800 mm
```

6 维向量严格按照以下顺序堆叠：

| 维度 | 特征               | 公式                         | 提供给模型的信息                     |
| ---: | ------------------ | ---------------------------- | ------------------------------------ |
|    1 | TX 角度正弦        | `sin(theta_tx)`            | TX 圆周绝对位置，跨越 0/360 度时连续 |
|    2 | TX 角度余弦        | `cos(theta_tx)`            | 与正弦共同唯一标识 TX                |
|    3 | RX 角度正弦        | `sin(theta_rx)`            | RX 圆周绝对位置                      |
|    4 | RX 角度余弦        | `cos(theta_rx)`            | 与正弦共同唯一标识 RX                |
|    5 | 归一化有符号角差   | `delta/pi`                 | 最短圆周方向和角距离，范围`[-1,1)` |
|    6 | 归一化表面路径长度 | `sqrt((R*delta)^2+dz^2)/L` | 圆柱展开面上 TX-RX 的最短名义距离    |

这 6 维分别回答三个必要问题：前四维说明“是哪一个 TX 和 RX”，第 5 维说明“从 TX 到 RX 的圆周方向和角差”，第 6 维说明“两个端点之间的物理路径有多长”。

例如 TX1 与 RX21 的环内序号分别为 `0` 和 `4`，对应 `0` 度和 `90` 度。其 6 维输入近似为：

```text
[0.0000, 1.0000,
 1.0000, 0.0000,
 0.5000, 0.8362]
```

这 6 个值先组成 `[B,16,16,6]`，再按与 CNN token 相同的 TX-major、RX-minor 顺序展平。由于必要的非线性几何量已经显式算出，不再使用带隐藏层的 MLP，只做一次无偏置线性投影：

```text
minimal physical features       [B,16,16,6]
  -> flatten TX-RX              [B,256,6]
  -> Linear(6,256,bias=False)   [B,256,256]
                                  = P_txrx
```

相较原 17 维版本，已删除矩阵行列标量、角差正余弦、路径中点角、三个固定轴向常量、有符号圆周距离和弦长。删除依据是：TX/RX 角度正余弦已经唯一标识矩阵行列；其余量要么可以由保留特征推导，要么在固定 Dataset A 几何中对所有 token 都相同。这样避免重复表达同一物理关系，也降低小数据集记忆硬件编号的风险。

正式 `D=256` 时，该位置投影只有 `6*256=1,536` 个参数；原 `17->128->256` MLP 有 35,328 个参数。精简后参数减少 33,792 个，并且每个输入维度都可以直接解释。

这里的 `delta` 使用 `[-pi,pi)` 内的最短名义角差，因此第 6 维描述基础最短表面路径。RayOperator 中 `-1/0/1` 圈螺旋路径仍由物理算子单独处理，位置 embedding 不替代 RayOperator。

频率不属于这 6 维。每个 `[B,256,D]` token 表示一个 TX-RX 对并聚合全部 15 个频率，真实 Hz 已由上一节的 `PhysicalFrequencyEmbedding` 在聚合前注入。

最终位置编码加入 low/high 内容 token：

```text
P_txrx [B,256,256]
T_low  = content_low  + P_txrx
T_high = content_high + P_txrx
```

low/high 使用完全相同的物理位置编码。二者的频域身份仍由融合块中的两组 `domain_embedding` 区分，物理位置和频域类型不会混成同一个含义。

保留 256 个 TX-RX token 而不展开成 `15×16×16=3840` 个 token，是为了控制 32×32 Cross-Attention 的显存与计算量。真实 Hz 并未被丢弃，而是在 token 化前进入每个频率切片。

### 5.2 为什么低频和高频使用共享编码器

低频与高频张量不会分别使用两套独立的大编码器，而是共用 `XMatrixEncoder.stem` 和 `token_proj`：

```text
(x_low + E_f) / (x_high + E_f) [B,7,15,16,16]
  -> reshape [B,105,16,16]
  -> shared stem [B,64,16,16]
  -> shared token projection [B,256,16,16]
  -> flatten TX-RX [B,256,256]
```

共享权重有三个目的：

1. 让低频和高频 token 位于相同特征空间，Cross-Attention 可以直接比较。
2. 限制新增参数量，避免为两个频率分支复制整个 encoder。
3. 防止小数据集上两路表示独立漂移，最终无法有效融合。

原始 `x_matrix` 仍单独经过同一个 stem 和 `global_net` 得到 `e_x [B,256]`。这个全局 embedding 继续通过 FiLM 向所有 ResBlock 提供样本级条件，不会被新的低频和高频 token 路径替代。

## 6. GATE 如何适配当前模型

SFAFNet 的 GATE 使用全局均值和标准差判断不同特征通道的重要性。当前实现分别为三类特征生成门控：

- U-Net 空间特征 `H [B,C,H,W]`。
- 低频 token `T_low [B,N,D]`。
- 高频 token `T_high [B,N,D]`。

空间特征统计：

```text
mu_s  = mean(H, dim=(H,W))
std_s = std(H, dim=(H,W))
g_s   = sigmoid((MLP_mean(mu_s) + MLP_std(std_s)) / 2)
```

频率 token 统计：

```text
mu_l/std_l = statistics over N TX-RX tokens
mu_h/std_h = statistics over N TX-RX tokens

g_l = GATE(mu_l, std_l)
g_h = GATE(mu_h, std_h)
```

同时使用 mean 和 STD 的原因：

- mean 描述一个通道整体响应是否较强。
- STD 描述它在空间位置或 TX-RX 对之间是否包含有区分度的变化。
- 只有均值容易偏向全局幅值较大的通道；加入 STD 后，可以保留平均值不大但局部变化明显的缺陷证据。

GATE 不直接输出缺陷图，也不硬编码低频或高频谁更重要。它只对三路特征进行自适应重标定，后续由 Cross-Attention 在每个空间位置完成选择。

> 根据当前 U-Net 特征的全局均值和波动情况，判断哪些通道应该更强地参与后续 query。

## 7. CAM 如何改造成空间-物理频域融合

论文 CAM 的目标是让不同域的特征交换信息。当前模型将其改造成以下形式：

```text
Query:
  gated spatial feature
  [B,C,H,W] -> [B,H*W,D]

Key / Value:
  concat(gated T_low, gated T_high)
  [B,256,D] + [B,256,D] -> [B,512,D]

Output:
  MultiheadAttention(Query, Key, Value)
  -> [B,H*W,D]
  -> project back to [B,C,H,W]
  -> residual add to original spatial feature
```

低频和高频 token 已带有共享的 TX-RX 物理位置编码，并分别叠加可学习 domain embedding，使注意力层同时知道“测量来自哪个 TX-RX 配对”以及“它属于低频还是高频分支”。

这一设计比直接拼接更适合当前任务：

- 每个缺陷图像素可以查询不同的 TX-RX 组合。
- 某些位置可以更依赖低频结构，某些位置可以更依赖高频局部变化。
- 没有测量支持的位置可以降低两类频率 token 的影响。
- 空间关系来自 `pic + noisy target`，频率证据来自 `x_matrix`，两个域的职责保持清晰。

融合输出投影采用零初始化。模型开始训练时融合块近似恒等映射，先保持原 U-Net 行为；随着训练进行，融合残差逐步学习进入主干，从而减少训练初期的数值扰动。

对应代码：[models/unet.py](models/unet.py)。

## 8. 融合发生在哪些 U-Net 层

base48/base64 正式配置使用：

```yaml
spatial_frequency_fusion:
  resolutions: [32]
  heads: 4
  filter_kernel_size: 5
```

在 `num_res_blocks=2` 的正式 U-Net 中，共有 5 个融合位置：

1. Down D3 的第 1 个 ResBlock 后，分辨率 32×32。
2. Down D3 的第 2 个 ResBlock 后，分辨率 32×32。
3. Middle 的 Self-Attention 后。
4. Up U3 的第 1 个 ResBlock 后，分辨率 32×32。
5. Up U3 的第 2 个 ResBlock 后，分辨率 32×32。

没有默认放在 256×256、128×128 或 64×64，原因是 Cross-Attention 的计算量与空间 query 数量相关。32×32 能保留缺陷位置关系，同时把 `[B,512,D]` 的低频和高频 token 融合成本控制在合理范围内。

空间 query 在进入融合块前已经经过：

```text
EDM noisy state
  -> ResBlock
  -> FiLM(e_x + e_sigma)
  -> PicAdapter feature add
  -> GATE + spatial-frequency attention
```

因此融合块看到的不是孤立 U-Net 特征，而是已经包含 coarse pic、当前去噪状态和全局测量 embedding 的空间表示。

由于粗物理图本身已经由输入派生，我们没有采用复杂的双向交互，而是通过显式物理位置编码增强频域 token 的物理约束，使空间特征能够选择性读取具有物理索引的频域表示。

## 9. SFAFNet 思想与现有条件路径如何分工

| 条件路径        | 信息粒度             | 进入模型的方式                   | 主要作用                             |
| --------------- | -------------------- | -------------------------------- | ------------------------------------ |
| PicAdapter      | 多尺度空间特征       | 每个 ResBlock 后相加             | 提供缺陷大致位置、路径覆盖和粗图结构 |
| FiLM            | 样本级全局向量       | 调制每个 ResBlock 的 scale/shift | 提供整体测量状态与 EDM 噪声等级      |
| SFAF 低频 token | TX-RX 级低频表示     | 32×32 与 Middle Cross-Attention | 约束大尺度结构和稳定背景             |
| SFAF 高频 token | TX-RX 级高频表示     | 32×32 与 Middle Cross-Attention | 提供局部变化和边缘证据               |
| RayOperator     | 输出到测量的物理约束 | 训练 loss 或采样 guidance        | 检查预测是否与频率响应一致           |

这些路径不是重复关系。PicAdapter 解决“空间上可能在哪里”，SFAF 融合解决“哪些物理频率证据支持该位置”，FiLM 提供全局条件，RayOperator 在输出层面检查物理一致性。

## 10. 输出频谱损失与杂点抑制

论文除了结构融合，还使用 frequency-domain loss。当前实现采用较保守的 log-magnitude 频谱损失：

```text
L_spectral = L1(
  log(1 + |RFFT2(prediction)|),
  log(1 + |RFFT2(target)|)
)
```

对应代码：[train_edm.py](train_edm.py)。

它不替代像素损失，而是补充约束 prediction 的频谱分布：

- 随机 speckle 会在高频区域增加额外能量。
- 标签背景大部分平滑且为零，其高频能量较低。
- log magnitude 可以降低少数大幅值频点对损失的支配。
- 空间 L1/EDM MSE 负责位置和数值，频谱损失负责额外杂点的整体频率形态。

默认权重：

```yaml
lambda_spectral: 0.01
```

这个权重低于论文图像去模糊任务常用的频率损失权重，因为当前标签稀疏、数据量较小。过大的频谱权重可能导致缺陷边缘被过度平滑。

## 11. EDM 数学定义保持不变

SFAFNet 融合只改变条件特征进入 U-Net 的方式，不改变 EDM 数学定义。

训练加噪：

```text
x = x0 + sigma * epsilon
```

EDM 预条件：

```text
c_in   = 1 / sqrt(sigma^2 + sigma_data^2)
c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
```

去噪输出：

```text
D_theta(x,sigma) = c_skip * x + c_out * F_theta(c_in*x, conditions)
```

训练主损失：

```text
L_edm = w(sigma) * ||D_theta(x,sigma) - x0||^2
```

采样仍使用 Karras sigma schedule 和 Euler/Heun 更新。SFAF 模块只帮助 `F_theta` 更有效地利用物理频域和空间域条件。

## 12. 配置参数与消融建议

唯一的融合参数入口位于 `model.spatial_frequency_fusion`：

```yaml
model:
  x_token_dim: 256
  spatial_frequency_fusion:
    resolutions: [32]
    heads: 4
    filter_kernel_size: 5
    physical_position:
      tx_count: 16
      rx_count: 16
      pipe_length_mm: 1000.0
      mid_radius_mm: 155.0
      tx_z_mm: 100.0
      rx_z_mm: 900.0
      frequency_reference_hz: 100000.0

loss:
  lambda_spectral: 0.01
```

| 参数                   |     当前值 | 说明                                                          |
| ---------------------- | ---------: | ------------------------------------------------------------- |
| `resolutions`        |   `[32]` | 在哪些 U-Net 空间分辨率执行融合；Middle 始终执行一次          |
| `heads`              |      `4` | 空间-频域 MultiheadAttention 的 head 数量                     |
| `filter_kernel_size` |      `5` | 沿 15 个物理频点执行动态低通的核长度，必须为大于等于 3 的奇数 |
| `physical_position`  | 见上方配置 | TX/RX 阵列、管道几何和频率归一化参考值；应与数据生成几何一致  |
| `lambda_spectral`    |   `0.01` | 输出图频谱正则权重，设为 0 可关闭                             |

建议按以下顺序做消融，而不是一次修改多个因素：

1. `lambda_spectral=0`，只测试结构融合。
2. `filter_kernel_size=3/5/7`，比较频率分解尺度。
3. 关闭动态分解、仅使用原始 token，判断 FDGM 的独立贡献。
4. 保留分解但去掉 mean/STD GATE，判断门控贡献。
5. 完整 FDGM + GATE + CAM + spectral loss。

不要一开始把 `resolutions` 扩展到 `[64,32]` 或 `[128,64,32]`。这会显著增加 attention 的显存和计算量，也更容易让小数据集过拟合频率噪声。

## 13. 与旧 checkpoint 的兼容性

旧参数：

```text
cross_attention_resolutions
cross_attention_heads
```

已从 EDM 正式配置和模型构造路径移除，统一替换为：

```text
model.spatial_frequency_fusion
```

更早的 checkpoint 不包含 PhysicalFrequencyEmbedding 和 PhysicalTxRxPositionEmbedding；本次精简前的 checkpoint 则使用 `17->128->256` 位置 MLP。两者都与当前无偏置 `6->256` 投影的参数名或形状不一致，不能使用 `strict=True` 直接恢复训练。新结构应从头训练。当前 `runs/local_sfaf_fusion_smoke` 属于旧结构历史目录，不应继续作为新版 smoke 输出目录。运行新版 smoke 时显式指定：

```text
simple/diffusion_EDM/runs/local_two_sample_smoke
```

## 14. 主要文件及阅读顺序

```text
diffusion_EDM/
├─ 2502.14209v1.pdf
├─ models/
│  ├─ x_encoder.py       # 动态物理频率分解、共享编码器、低/高频 token
│  ├─ physical_encoding.py # Hz 排序/编码、TX-RX 几何位置编码
│  ├─ unet.py            # PicAdapter、FiLM、GATE、空间-频域 Cross-Attention
│  └─ edm.py             # EDM 预条件、训练加噪、Karras/Heun 采样
├─ configs/
│  ├─ dataset_a_256_base48_edm.yaml
│  ├─ dataset_a_256_base64_edm.yaml
│  ├─ local_two_sample_smoke.yaml
│  └─ splits/             # 固定 1200 样本 train/val ID 清单
├─ train_edm.py          # EDM loss、输出先验、spectral loss、physics loss
├─ uncertainty.py        # 多样本后验统计、保存/加载、不确定性校准指标
├─ sample_edm.py         # K 次 EDM 采样、后验均值与 uncertainty 输出
├─ evaluate_edm.py       # 均值/共识精度、区间覆盖率、CRPS 等评估
├─ smoke_test_physical_encoding.py # tiny EDM 前向/反向/两步采样检查
├─ smoke_test_uncertainty.py # seed、后验统计、共识过滤、持久化检查
└─ show/
   ├─ network_architecture_edm_base48.tex
   ├─ network_architecture_edm_base48.pdf
   └─ README.md
```

推荐阅读顺序：

1. 先看 `show/network_architecture_edm_base48.pdf`，建立完整数据流概念。
2. 阅读 `models/physical_encoding.py`，确认 Hz、TX、RX、角度和距离如何编码。
3. 阅读 `models/x_encoder.py`，确认排序后 15 个频点如何分成低频和高频并形成 token。
4. 阅读 `models/unet.py` 中 `MeanStdGate` 和 `GatedSpatialFrequencyFusionBlock`。
5. 阅读 `models/edm.py`，确认融合模块没有改变 EDM 公式。
6. 阅读 `uncertainty.py`，确认 K 次样本如何形成默认预测、uncertainty 和共识图。
7. 阅读 `train_edm.py`，检查 spectral loss 与 physics loss 的权重，最后核对 YAML。

`data`、`utils` 和 `physics` 继续复用 `simple/diffusion` 中的实现，因此数据读取、checkpoint、指标和 RayOperator 与原模型保持一致。

## 15. 已完成的验证

当前实现已完成以下检查：

- Python 静态编译通过。
- 未排序的真实频率可正确重排为严格递增 Hz，`x_matrix` 使用同一 gather order。
- 最小物理特征形状为 `[B,TX×RX,6]`，线性投影后为 `[B,TX×RX,D]`，不同 TX/RX 矩阵格得到不同向量。
- low/high token 加入同一位置编码，融合块的两组 domain embedding 仍不同。
- tiny EDM 的完整 forward、backward 和两步 sampling 在 CUDA 上通过，输出全为有限值。
- tiny EDM 的 K=3 多样本后验测试通过：同 seed 逐元素一致、不同 seed 输出不同，所有统计量形状正确且为有限值。
- 构造的“仅 1/4 样本出现孤立缺陷”在后验均值中非零，但被 `consensus_prediction` 清零。
- 不保存原始样本时，posterior summary 保存/加载后仍保留正确的 `sample_count`；保存原始样本时 CRPS 可正常计算。
- 一个真实样本的数据读取与 X 编码通过，输出 `e_x [1,64]`、low/high `[1,256,64]`。
- 一个真实 `256×256` 样本的当前 local smoke 模型 FP16 loss/backward 通过，240 组梯度均为有限值；本次运行峰值已分配显存约 300.8 MiB。
- 正式 base48 参数量为 `18,451,227`，base64 为 `30,136,203`；其中频率与 TX-RX 位置编码共 `1,991` 个参数。

smoke test 只能验证代码链路和数值稳定性，不能证明融合一定改善缺陷定位。正式结论仍应通过固定数据划分、固定随机种子和上述消融实验获得。

## 16. 运行命令

### 正式 1200 样本的固定 train/val 划分

`dataset_a_256_base48_edm.yaml` 与 `dataset_a_256_base64_edm.yaml` 不再使用运行时的 `val_fraction` 随机划分，而是共用一对显式 ID 清单：

```text
configs/splits/dataset_a_1200_train_ids.txt  # 1080 个训练样本
configs/splits/dataset_a_1200_val_ids.txt    # 120 个验证样本
```

清单覆盖 `dataset_a_frequency_sample_0001` 至 `dataset_a_frequency_sample_1200`，由固定 seed `20260708` 生成一次后写死。训练加载器会拒绝 train/val 重叠；验证集单独构造，始终关闭圆周 roll 与输入噪声。

每次训练启动后，正式 run 根目录还会写出：

```text
data_split.json                 # train/val 的完整 ID、数量、seed 与划分模式
validation_sample_ids.txt       # 只含 120 个 validation sample ID，便于人工检查
```

直接打开 `configs/splits/dataset_a_1200_val_ids.txt` 或运行目录中的 `validation_sample_ids.txt`，即可确认哪一些 sample 用于 validation。本阶段按要求没有额外规划 test 集；validation 仅用于选择 checkpoint，不能在论文中表述为独立最终测试集。

### 物理位置编码最小 smoke test

```powershell
conda run -n diffusion python simple/diffusion_EDM/smoke_test_physical_encoding.py
```

### 多样本后验 uncertainty smoke test

```powershell
conda run -n diffusion python simple/diffusion_EDM/smoke_test_uncertainty.py
```

### 两样本训练 smoke test

```powershell
conda run -n diffusion python simple/diffusion_EDM/train_edm.py `
  --config simple/diffusion_EDM/configs/local_two_sample_smoke.yaml `
  --run-dir simple/diffusion_EDM/runs/local_two_sample_smoke
```

### smoke checkpoint 采样

```powershell
conda run -n diffusion python simple/diffusion_EDM/sample_edm.py `
  --config simple/diffusion_EDM/configs/local_two_sample_smoke.yaml `
  --checkpoint simple/diffusion_EDM/runs/local_two_sample_smoke/checkpoints/last.pt `
  --output-dir simple/diffusion_EDM/runs/local_two_sample_smoke/samples `
  --max-samples 1
```

### base48 正式训练

```powershell
conda run -n diffusion python simple/diffusion_EDM/train_edm.py `
  --config simple/diffusion_EDM/configs/dataset_a_256_base48_edm.yaml
```

### 三卡 FP16 训练（优先 DataParallel）

`dataset_a_256_base48_edm_3gpu_fp16.yaml` 固定每卡 batch 为 3，三卡 global batch 为 9；1080 个 train sample 对应每 epoch 120 step。普通 Python 启动会使用 DataParallel，因此当前 Windows A5000 和 Linux 2080 Ti 都优先使用以下命令。可见 GPU 少于 3 张时会直接失败，不会退化为单卡。

Windows PowerShell：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1,2"
conda run --no-capture-output -n diffusion_cuda128 python simple/diffusion_EDM/train_edm.py `
  --config simple/diffusion_EDM/configs/dataset_a_256_base48_edm_3gpu_fp16.yaml
```

Linux：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 conda run --no-capture-output -n diffusion_cuda128 \
  python simple/diffusion_EDM/train_edm.py \
  --config simple/diffusion_EDM/configs/dataset_a_256_base48_edm_3gpu_fp16.yaml
```

统一保持 `amp_dtype: float16`。A5000 支持 BF16，但这里不切换，以保持 2080 Ti、A5000 和后续单卡 A6000 的 checkpoint、optimizer state 和训练动态可比。

### Linux 可选 NCCL DDP

只有显式使用 `torchrun`（`WORLD_SIZE > 1`）时，入口才进入 DDP；每个 rank 读取不重叠的训练 sample，rank 0 负责验证、日志和 checkpoint。若需要评估 DDP 吞吐，可在 Linux 使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  simple/diffusion_EDM/train_edm.py \
  --config simple/diffusion_EDM/configs/dataset_a_256_base48_edm_3gpu_fp16.yaml
```

Windows 原生 PyTorch 不提供 NCCL；该配置在 Windows 的 `torchrun` 下会明确报错，请使用上面的 DataParallel 命令，不将 Gloo DDP 作为正式训练路径。

### Checkpoint 续训与迁移

checkpoint 始终保存原始 EDM 模型和 EMA，不包含 `module.` 前缀，可严格加载到 single、DataParallel 或 DDP。`--resume` 同时恢复 model、optimizer、scheduler、scaler、EMA、epoch 和 step，仅用于 world size 与 global batch 都不变的续训：

```powershell
conda run --no-capture-output -n diffusion_cuda128 python simple/diffusion_EDM/train_edm.py `
  --config simple/diffusion_EDM/configs/dataset_a_256_base48_edm_3gpu_fp16.yaml `
  --resume simple/diffusion_EDM/runs/dataset_a_256_base48_edm_3gpu_fp16/checkpoints/last.pt
```

从旧单卡 batch 8 切换到三卡 global batch 9 时，使用 `--init-checkpoint`，它只严格加载模型权重并重置 optimizer、scheduler、scaler、EMA、epoch 和 step：

```powershell
conda run --no-capture-output -n diffusion_cuda128 python simple/diffusion_EDM/train_edm.py `
  --config simple/diffusion_EDM/configs/dataset_a_256_base48_edm_3gpu_fp16.yaml `
  --init-checkpoint simple/diffusion_EDM/runs/dataset_a_256_base48_fp16_22g/checkpoints/last.pt
```

三卡 checkpoint 可直接在单卡 A6000 上采样或评估，使用结构匹配的单卡配置和 `--use-ema`；三卡专用配置因 `require_device_count: 3` 不应用于该单卡训练入口。

### base48 EDM 采样

```powershell
conda run -n diffusion python simple/diffusion_EDM/sample_edm.py `
  --config simple/diffusion_EDM/configs/dataset_a_256_base48_edm.yaml `
  --checkpoint simple/diffusion_EDM/runs/dataset_a_256_base48_edm/checkpoints/last.pt `
  --use-ema `
  --steps 32 `
  --num-posterior-samples 16
```

### 可选 physics-guided sampling

```powershell
conda run -n diffusion python simple/diffusion_EDM/sample_edm.py `
  --config simple/diffusion_EDM/configs/dataset_a_256_base48_edm.yaml `
  --checkpoint simple/diffusion_EDM/runs/dataset_a_256_base48_edm/checkpoints/last.pt `
  --use-ema `
  --physics-guidance-scale 0.05 `
  --physics-guidance-start-fraction 0.5
```

`dataset_a_256_base64_edm.yaml` 是更大的通道配置，A6000 可以训练；建议先用 base48 完成消融和损失权重确认，再决定是否扩大到 base64。

## 17. Uncertainty map、后验均值与稀疏幻觉处理

### 17.1 当前统计量的定义

设固定条件下的 K 个 EDM 输出为 `y^(1)...y^(K)`，每个像素分别计算：

```text
posterior_mean = (1/K) * sum_k y^(k)
posterior_std  = sqrt((1/K) * sum_k (y^(k) - posterior_mean)^2)
lower/upper   = empirical_quantile(y, q_lower/q_upper)
P_defect      = (1/K) * sum_k I[y^(k) >= defect_threshold]
entropy       = -P_defect*log2(P_defect) - (1-P_defect)*log2(1-P_defect)
```

这里的标准差使用总体标准差 `correction=0`，所以 `K=1` 时严格为 0，不会产生 NaN。默认分位区间为 `[q_0.05,q_0.95]`，名义覆盖率为 90%。

默认连续预测和保守共识预测分别是：

```text
default prediction = posterior_mean
consensus_prediction[i,j] = posterior_mean[i,j],
                            if P_defect[i,j] >= consensus_probability_threshold
                          = 0, otherwise
```

`prediction` 必须保持为后验均值，便于 MAE、RMSE 和体积误差等连续指标稳定比较。`consensus_prediction` 是额外的保守结果，适合观察稀疏缺陷和计算阈值分割指标，不会覆盖默认 prediction。

### 17.2 为什么共识概率比直接乘 uncertainty 更适合去除孤立杂点

若某个孤立亮点只在少数随机轨迹中出现，它通常具有低 `P_defect`。共识规则会将该位置清零，同时保留在多数采样中反复出现的缺陷。直接使用 `mean * (1-std)` 没有概率含义，而且可能错误压低物理欠约束区中的真实缺陷，因此当前实现没有这样处理。

三张图应联合阅读：

| 图                     | 应回答的问题                               | 对稀疏幻觉的意义                   |
| ---------------------- | ------------------------------------------ | ---------------------------------- |
| `uncertainty`        | K 个连续厚度预测相差多大                   | 高值表示幅值或边界不稳定           |
| `defect_probability` | 有多少采样认为该像素超过缺陷阈值           | 低概率孤立点可由共识规则过滤       |
| `defect_entropy`     | 是否接近“一半认为有缺陷、一半认为无缺陷” | 接近 1 表示存在/不存在判断最不确定 |

这个方法只能抑制“不同随机轨迹不一致”的幻觉。如果所有 K 个样本都稳定地产生同一个错误缺陷，则该处可能同时表现为低标准差、高缺陷概率和低熵，即模型会自信地出错。此类系统性偏差仍需依靠更多健康/困难负样本、physics loss、可靠的 coarse 条件、阈值校准和独立测试集发现，不能由单模型采样自动解决。

### 17.3 输出文件

`sample_edm.py` 在 samples 根目录保留一个总 `manifest.csv`，并从原始 sample ID 的尾号生成子目录。例如：

```text
dataset_a_frequency_sample_0001 -> samples/sample0001/
dataset_a_frequency_sample_0002 -> samples/sample0002/
```

每个 `sampleNNNN` 子目录只保存该样本的文件。对样本 `<id>`，子目录内包含：

| 文件                                 | 内容                                                                |
| ------------------------------------ | ------------------------------------------------------------------- |
| `<id>_prediction.npy`              | 后验均值，当前默认预测                                              |
| `<id>_uncertainty.npy`             | 像素级后验总体标准差                                                |
| `<id>_posterior_median.npy`        | 后验中位数                                                          |
| `<id>_posterior_lower.npy`         | 下分位数图，默认 q=0.05                                             |
| `<id>_posterior_upper.npy`         | 上分位数图，默认 q=0.95                                             |
| `<id>_defect_probability.npy`      | 超过`defect_threshold` 的样本比例                                 |
| `<id>_defect_entropy.npy`          | 二元缺陷概率熵，范围`[0,1]`                                       |
| `<id>_consensus_prediction.npy`    | 按样本共识过滤后的后验均值                                          |
| `<id>_posterior_summary.npz`       | 上述数组、阈值、分位数和真实 sample count                           |
| `<id>_posterior_samples.npz`       | 可选的全部 K 个样本；仅在`save_all_samples=true` 时保存           |
| `<id>_prediction_mm.npy`           | 反归一化并裁剪到物理上限后的后验均值，单位 mm                       |
| `<id>_uncertainty_mm.npy`          | 每个后验样本映射到 0-5 mm 后重新计算的像素标准差                    |
| `<id>_posterior_*_mm.npy`          | 中位数、上下分位数和区间宽度的毫米版本                              |
| `<id>_consensus_prediction_mm.npy` | 共识预测的 0-5 mm 物理版本                                          |
| `<id>_preview.png`                 | mean、label、coarse、consensus、std、区间宽度、概率和熵的 2×4 预览 |

`samples/manifest.csv` 的每一行记录 `sample_directory` 和该样本全部输出路径。`posterior_summary.npz` 始终记录 `sample_count`。因此即使为节省磁盘不保存全部样本，`evaluate_edm.py --pred-dir` 仍能报告实际 K；评估器同时兼容新子目录结构和修改前的旧平铺目录。离线 CRPS 需要原始样本，所以只在保存了 `<id>_posterior_samples.npz` 时可重新计算。

### 17.4 归一化值、毫米值和 preview 色阶

当前 label package 同时包含：

```text
*_defect_depth_mm.npy    真实壁厚损失，物理范围 0-5 mm
*_defect_depth_norm.npy  训练标签
*_defect_label_metadata.json
```

需要特别注意：当前数据生成器的归一化分母不是 5 mm，而是：

```text
normalization_denominator_mm
  = wall_thickness_mm - h_min_mm
  = 10 mm - 1 mm
  = 9 mm

depth_norm = depth_mm / 9 mm
```

缺陷生成仍被 `depth_limit_mm=5 mm` 截断，所以有效训练标签范围实际是 `[0,5/9]`，不是完整 `[0,1]`。样本 0001 的最大缺陷为 `2.323703 mm`，对应归一化值约 `0.258189`。旧 preview 固定使用 `vmin=0,vmax=1`，因此该缺陷只使用约四分之一色阶，看起来不明显。

当前后处理不再直接显示归一化数组，而是从每个 label metadata 读取分母和物理范围：

```text
sample_mm^(k) = clip(sample_norm^(k) * normalization_denominator_mm,
                     0, depth_limit_mm)

prediction_mm = mean_k(sample_mm^(k))
uncertainty_mm = std_k(sample_mm^(k))
```

preview 使用与 dataset label PNG 相同的物理展示原则：

- 横轴为 `theta=0...360 deg`，纵轴为 `z=0...1000 mm`。
- prediction、label 和 consensus 共用标签的 `preview_max_mm`，保证颜色可以直接比较。
- 样本 0001 的共同色阶为 `0...2.323703 mm`，因此 label 外观与 dataset label PNG 一致。
- 该色阶是逐样本自适应的，适合检查同一样本的 prediction/label；不同样本之间不能只按颜色深浅比较损失大小，应读取毫米色条或 `*_mm.npy`。跨样本统一展示时应固定使用 `0...5 mm`。
- 每个 K 样本先映射并裁剪到 0-5 mm，再重新计算 mean/std/quantile/interval width，因此物理后验内部保持一致；原始归一化统计仍保留用于诊断超限。
- coarse map、缺陷概率和熵仍是无量纲的 `[0,1]` 图。

`manifest.csv` 会记录 `normalization_denominator_mm`、`physical_depth_limit_mm`、`preview_vmax_mm`、`defect_threshold_mm`、`prediction_above_physical_limit_fraction` 和 `posterior_sample_above_physical_limit_fraction`。后两项使用未裁剪的归一化结果计算，用于暴露模型超出 5 mm 物理范围的程度。`evaluate_edm.py` 额外输出 `mae_mm` 与 `rmse_mm`。

若离线目录没有保存 `<id>_posterior_samples.npz`，则无法从 mean/std/quantile 唯一重建“逐样本裁剪后”的物理标准差。正式采样时毫米数组会在 K 个样本仍位于内存时直接生成；若要求之后重新计算，需设置 `save_all_samples: true`。

这次修改只改变后处理和物理单位输出，不改变训练 target，也不改变旧 checkpoint 的数值含义。若后续决定让 `0-5 mm` 完整映射到 `[0,1]`，则必须把训练标签统一改成 `depth_mm/5 mm`，同步调整阈值和 EDM 数据尺度，并从头训练；不能把现有 `/9 mm` checkpoint 直接按 `/5 mm` 解释。

### 17.5 配置与正式采样建议

```yaml
sample:
  uncertainty:
    num_samples: 16
    sample_seed: 20260712
    lower_quantile: 0.05
    upper_quantile: 0.95
    defect_threshold: 0.1
    consensus_probability_threshold: 0.5
    save_all_samples: false
```

- `K=16` 适合作为默认检查；正式报告可用 `K=32` 检查均值、标准差和分位数是否稳定。
- `K=2/4` 只适合 smoke test，不能给出平滑、可信的概率或 90% 分位区间。
- 正式不确定性采样建议保持 `s_churn=0`，使样本差异主要来自独立 initial noise，便于解释和复现。
- 每个数据样本使用互不重叠的 seed 区间；manifest 中记录 `sample_seed_start` 和 K。
- `defect_threshold` 应在验证集上按任务含义固定，不应针对测试标签逐样本调节。
- `consensus_probability_threshold=0.5` 表示至少半数采样支持缺陷。若更强调减少假阳性，可在验证集比较 `0.5/0.625/0.75`，同时检查召回率下降。

### 17.6 uncertainty 是否对应低覆盖区

这是需要数据验证的物理假设，而不是实现中的硬编码。当前代码不会把 `1-path_coverage` 直接当作 uncertainty，而是在评估阶段计算：

```text
Spearman(uncertainty, absolute_error)
Spearman(uncertainty, 1 - path_coverage)
低/高 coverage 四分位区域的平均 uncertainty
reliable/unreliable 区域的平均 uncertainty
```

只有当这些关系在独立验证集和测试集上稳定成立，才能解释为“高 uncertainty 对应低覆盖或物理欠约束区”。如果相关性很弱，uncertainty 仍可表示模型条件后验的多样性，但不能宣称具有覆盖区物理含义。

### 17.7 评估指标

`evaluate_edm.py` 保留后验均值的 MAE/RMSE/SSIM/IoU/Dice，同时增加：

- `interval_empirical_coverage`：标签落入分位区间的像素比例。
- `interval_calibration_error`：经验覆盖率与名义覆盖率之差的绝对值。
- `interval_mean_width`：分位区间平均宽度，需与覆盖率联合判断。
- `posterior_crps`：同时评价后验位置与离散程度，需保留原始 K 个样本。
- `uncertainty_error_spearman`：标准差和绝对误差的秩相关。
- `defect_probability_brier`：缺陷概率的 Brier score。
- `consensus_*`：共识预测对应的连续指标和阈值指标。
- `consensus_removed_fraction`：后验均值缺陷像素中被共识规则移除的比例。
- `mae_mm/rmse_mm`：根据 label metadata 反归一化后的毫米误差。
- `prediction_above_physical_limit_fraction`：后验均值超过 5 mm 物理上限的像素比例。

单模型 K 次采样主要描述 aleatoric/条件多解性，不完整覆盖模型参数不确定性。若后续需要 epistemic uncertainty，应在不改变本套后验统计接口的前提下，再比较多个独立 checkpoint 或深度集成。

## 18. `runs` 目录中的结果代表什么

### 18.1 当前目录的结论

当前 `simple/diffusion_EDM/runs` 中还没有正式的 `dataset_a_256_base48_edm` 或 `dataset_a_256_base64_edm` 完整训练结果。现有目录都是两样本、一轮训练、两步采样或后处理 smoke test，作用是验证代码链路，不能用于判断模型是否能定位缺陷，也不能用于论文指标。

| 当前目录                                        | 产生原因                                                                    | 是否应关注                        | 结论                                                                                                                    |
| ----------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `local_sfaf_fusion_smoke`                     | 旧 SFAF/物理位置结构的一轮两样本训练及单次采样                              | 仅作历史记录                      | `checkpoints/last.pt` 缺少当前物理编码参数，不能被当前模型 `strict=True` 加载；其中的噪声 prediction 不代表正式模型 |
| `local_sfaf_fusion_smoke/postprocessed_mm_v2` | 把旧噪声 prediction 按正确毫米单位重新绘制                                  | 只检查绘图效果                    | label 外观、物理坐标和色条是正确的，但 prediction 仍来自旧 smoke checkpoint                                             |
| `uncertainty_cli_smoke_train`                 | 当前结构的一轮、两个 optimization step 临时训练                             | 只检查 checkpoint 能否训练和加载  | checkpoint 与当前结构兼容，但训练量不足，不能检查精度                                                                   |
| `uncertainty_cli_smoke`                       | 使用上述临时 checkpoint 做 K=2、steps=2 的 uncertainty sample/evaluate 测试 | 只检查 uncertainty 文件和评估入口 | K=2 不能形成可信概率、分位区间或校准结论                                                                                |
| `postprocessing_cli_smoke`                    | 第一版毫米后处理测试                                                        | 不再关注                          | 其物理 std/interval 是逐统计量线性换算的旧实现，已由 v2 替代                                                            |
| `postprocessing_cli_smoke_v2`                 | 毫米后处理 v2 测试，生成时间早于 sample 子目录改动                          | 可作为文件内容样例                | 毫米 NPY 和 2x4 preview 内容正确，但仍是旧平铺布局；新运行会写入`sample0001/` 子目录                                  |
| `local_two_sample_smoke`                      | 为当前结构预留的干净 smoke run 目录                                         | 当前为空                          | 以后运行 README 中的新版 smoke 命令后才会出现结果                                                                       |

因此，目前最值得保留用于人工检查的是：

```text
postprocessing_cli_smoke_v2/manifest.csv
postprocessing_cli_smoke_v2/dataset_a_frequency_sample_0001_preview.png
postprocessing_cli_smoke_v2/*_prediction_mm.npy
postprocessing_cli_smoke_v2/*_uncertainty_mm.npy
```

这些文件只能证明后处理格式和物理单位正确。当前 prediction 仍是噪声，因为输入 checkpoint 只训练了一轮、两个 step。`local_sfaf_fusion_smoke`、`uncertainty_cli_smoke` 和 `postprocessing_cli_smoke` 中的数值不应拿来比较模型优劣。

### 18.2 正式训练后应出现的目录结构

base48 正式训练、采样和评估完成后，建议保持以下结构：

```text
runs/dataset_a_256_base48_edm/
├─ config_resolved.yaml
├─ data_split.json
├─ validation_sample_ids.txt
├─ loss_history.csv
├─ metrics.jsonl
├─ checkpoints/
│  ├─ best.pt
│  ├─ last.pt
│  └─ step_XXXXXXXX.pt
├─ samples/
│  ├─ manifest.csv
│  ├─ sample0001/
│  │  ├─ <id>_preview.png
│  │  ├─ <id>_prediction.npy
│  │  ├─ <id>_prediction_mm.npy
│  │  ├─ <id>_uncertainty.npy
│  │  ├─ <id>_uncertainty_mm.npy
│  │  ├─ <id>_defect_probability.npy
│  │  ├─ <id>_defect_entropy.npy
│  │  ├─ <id>_consensus_prediction_mm.npy
│  │  └─ <id>_posterior_summary.npz
│  ├─ sample0002/
│  │  └─ ...
│  └─ sampleNNNN/
│     └─ ...
└─ eval/
   ├─ metrics.csv
   ├─ summary.json
   └─ <id>_preview.png
```

训练、采样和评估应写入同一个正式 run 根目录下的不同子目录，不要把正式结果写进名称带 `smoke` 的目录。

### 18.3 每类文件的作用

| 文件                              | 内容                                                                           | 主要用途                                                          |
| --------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| `config_resolved.yaml`          | 本次运行实际使用的完整配置                                                     | 复现实验时首先检查；不能只看原始 YAML，因为命令行参数可能覆盖配置 |
| `data_split.json`               | 本次实际加载的完整 train/val sample ID、数量、seed 和划分模式                  | 审计数据集成员；确认训练中没有重新随机分配验证样本                |
| `validation_sample_ids.txt`     | 仅包含本次 validation 的 sample ID，每行一个                                   | 最快人工查看验证集组成                                             |
| `loss_history.csv`              | 每个训练 step 的总 loss、EDM loss、先验 loss、梯度和 sigma                     | 检查训练是否发散、梯度是否异常、各 loss 权重是否失衡              |
| `metrics.jsonl`                 | train/validation 过程日志                                                      | 查看随 epoch/step 的趋势；单个 step 下降不能证明模型有效          |
| `checkpoints/best.pt`           | validation loss 最优时的模型、EMA 和优化器状态                                 | 正式采样优先使用，并加`--use-ema`                               |
| `checkpoints/last.pt`           | 最近一个 epoch 的完整状态                                                      | 主要用于断点续训，不一定是精度最好的模型                          |
| `checkpoints/step_*.pt`         | 周期性快照                                                                     | 回退、比较过拟合开始前后的模型                                    |
| `samples/manifest.csv`          | sample 子目录、checkpoint、EMA/model、steps、K、seed、阈值、毫米尺度和输出路径 | 定位每个样本目录，并判断某张图由哪个模型和采样参数产生            |
| `*_prediction.npy`              | 归一化空间的 K 样本后验均值                                                    | 与现有训练 loss 和归一化指标保持兼容                              |
| `*_prediction_mm.npy`           | 每个样本映射到 0-5 mm 后重新求得的后验均值                                     | 实际厚度损失检查和工程使用时优先看                                |
| `*_uncertainty_mm.npy`          | 0-5 mm 物理后验的像素标准差                                                    | 检查边界、低覆盖区和可疑杂点是否不稳定                            |
| `*_defect_probability.npy`      | K 个样本超过缺陷阈值的比例                                                     | 判断某个缺陷是否得到多数采样支持                                  |
| `*_defect_entropy.npy`          | 缺陷存在性二元熵                                                               | 查找“有/无缺陷”判断最不一致的位置                               |
| `*_consensus_prediction_mm.npy` | 低共识像素清零后的毫米预测                                                     | 用于保守筛查稀疏幻觉；不替代默认 posterior mean                   |
| `*_posterior_summary.npz`       | mean/std/quantile/probability/entropy、阈值和 K                                | 离线重画与评估，不必逐个读取多个 NPY                              |
| `*_posterior_samples.npz`       | 可选的全部 K 个样本                                                            | 重新计算 CRPS、物理裁剪后统计和采样稳定性；占用磁盘较大           |
| `*_preview.png`                 | prediction、label、coarse、consensus、uncertainty 等 2x4 图                    | 最快发现全图噪声、错位、过度平滑或幻觉，但不能替代数值指标        |
| `eval/metrics.csv`              | 每个样本的完整指标                                                             | 定位困难样本、检查异常值和覆盖区关系                              |
| `eval/summary.json`             | 所有评估样本指标的均值与标准差                                                 | 正式比较模型和消融实验时优先使用                                  |

### 18.4 正式结果的关注优先级

1. 先检查 `config_resolved.yaml`，确认数据划分、模型宽度、checkpoint、EMA、采样 steps 和 K 正确。
2. 模型推理优先使用 `checkpoints/best.pt --use-ema`；`last.pt` 主要用于续训。
3. 首先阅读 `eval/summary.json` 判断整体性能，再打开 `eval/metrics.csv` 找最差样本，不能只挑一张好看的 preview。
4. 打开若干 `samples/sampleNNNN/*_preview.png`，同时核对 `prediction_mm`、label、uncertainty、defect probability 和 consensus，而不是只看 posterior mean。
5. 最后回看 `loss_history.csv` 和 `metrics.jsonl` 解释性能变化；训练 loss 低不等于反演结果正确。

### 18.5 重点指标及方向

| 指标                                               | 期望方向       | 说明                                                                      |
| -------------------------------------------------- | -------------- | ------------------------------------------------------------------------- |
| `mae_mm`、`rmse_mm`                            | 越低越好       | 最直接的毫米厚度损失误差                                                  |
| `ssim`、`pearson`                              | 越高越好       | 结构形状和连续场相关性；不能单独代表缺陷面积正确                          |
| `iou_01/dice_01` 等                              | 越高越好       | 当前阈值在归一化空间；分母为 9 mm 时，0.1/0.2/0.3 分别约为 0.9/1.8/2.7 mm |
| `volume_error`                                   | 越低越好       | 判断总损失体积是否严重高估；背景幻觉通常会使其很大                        |
| `prediction_above_physical_limit_fraction`       | 应接近 0       | 后验均值超过 5 mm 的比例                                                  |
| `posterior_sample_above_physical_limit_fraction` | 应接近 0       | 全部随机样本超过 5 mm 的比例，比只看均值更严格                            |
| `interval_empirical_coverage`                    | 接近名义覆盖率 | 默认 5%-95% 区间的名义覆盖率约为 0.90                                     |
| `interval_calibration_error`                     | 越低越好       | 经验覆盖率和名义覆盖率的差距                                              |
| `posterior_crps`                                 | 越低越好       | 同时评价后验位置和离散程度，需要保存原始 K 样本                           |
| `uncertainty_error_spearman`                     | 应稳定为正     | uncertainty 是否真的在误差大的位置升高；只看数值大并无意义                |
| `defect_probability_brier`                       | 越低越好       | 缺陷概率校准质量                                                          |
| `consensus_removed_fraction`                     | 无固定越大越好 | 需与 Dice/召回率联合判断；过大会把真实稀疏缺陷一并删除                    |

K=2、steps=2、单个样本得到的上述指标只用于检查代码可运行。正式 uncertainty 结论至少应使用 K=16，并在完整 validation 集上汇总；最终报告建议用 K=32 检查稳定性。

### 18.6 最小人工检查清单

```text
[ ] run 目录名称不含 smoke，配置与 checkpoint 来自同一次正式训练
[ ] 使用 best.pt 的 EMA 权重，而不是随意使用 last.pt
[ ] eval/summary.json 的毫米误差、结构指标和校准指标均已生成
[ ] prediction_above_physical_limit_fraction 接近 0
[ ] prediction_mm 背景接近 0，没有全图高频亮点
[ ] label 中的主要缺陷在 posterior mean 和 defect probability 中都有响应
[ ] 高 uncertainty 是否对应边界、低覆盖区或真实高误差位置经过统计验证
[ ] consensus 减少假阳性的同时没有明显删除真实缺陷
```
