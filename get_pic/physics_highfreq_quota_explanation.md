# physics_highfreq_quota 物理选频方法说明

本文档用于论文撰写，解释 `physics_highfreq_quota` 选频方法中各个指标的物理意义、超参数来源和参考文献依据。

实现位置：

```text
simple/f_domain/select_sensitive_frequencies.py
```

命令行入口：

```text
--metric physics_highfreq_quota
```

该方法不读取 label。label 只用于后验评价，不参与频点打分和频点选择。


## 1. 方法定位

`physics_highfreq_quota` 的目标不是直接重建缺陷图像，而是在频点预算有限时，从健康/损伤复频响 `H0, Hd` 中选择一组更适合 V1 ray-tube 粗图成像的频点。

输入：

```text
H0(tx, rx, f): healthy frequency response
Hd(tx, rx, f): damaged frequency response
completed_mask(tx, f)
PZT tx/rx geometry
```

输出：

```text
selected frequencies
```

核心思想：

```text
缺陷可见性
+ 空间聚焦性
+ 路径覆盖平衡
+ 高频分辨率
+ 健康响应稳定性
+ 低/中/高频配额
```

相比 `relative_l2`，该方法不只看 `|Hd-H0|` 的总能量；相比旧 `physics_tomography`，该方法避免 Fisher/logdet 指标过度偏向低/中频稳定响应。


## 2. 总体得分

每个频点 `f` 的分数为：

```text
score(f) =
  visibility(f)
  * focus_factor(f)
  * contrast_factor(f)
  * coverage_balance(f)
  * resolution_weight(f)
  * health_weight(f)
```

代码中对应：

```text
score = visibility
score *= 0.45 + 0.55 * focus_score
score *= 0.70 + 0.30 * path_contrast
score *= 0.65 + 0.35 * coverage_preference
score *= resolution_weight
score *= health_weight
```

这些乘性项的含义是：一个频点不能只在某一方面好。好的频点应同时满足有可观测扰动、扰动能在空间上形成局部聚焦、路径覆盖不过分偏置、频率足以提供较高分辨率，并且健康响应不能过弱。


## 3. Rytov 可见性 visibility

### 3.1 定义

对每条 tx-rx 路径和频点，先计算健康/损伤复频响比值：

```text
R_p(f) = Hd_p(f) / H0_p(f)
```

再使用 Rytov 形式的复对数扰动：

```text
g_p(f) = log(R_p(f))
```

其中：

```text
Re(g_p) = log(|Hd_p| / |H0_p|)
Im(g_p) = angle(Hd_p * conj(H0_p))
```

代码中的路径扰动强度为：

```text
signal_p(f) = |Re(g_p(f))| + alpha * |sin(Im(g_p(f)))|
```

默认：

```text
alpha = 0.35
```

之后对所有样本和路径做鲁棒统计：

```text
visibility(f) = 0.70 * median(signal_p(f)) + 0.30 * quantile_0.75(signal_p(f))
```

### 3.2 物理意义

`log(|Hd|/|H0|)` 表示损伤引起的相对幅值变化，对应散射损失、能量泄漏、传播衰减变化和局部刚度/厚度变化造成的响应差异。

`angle(Hd * conj(H0))` 表示损伤引起的相位变化，对应有效波速、传播常数、路径时延和模态耦合变化。

使用 `Hd/H0` 而不是 `Hd-H0` 的原因是：频响中包含传感器耦合、路径固定增益和传播距离导致的基线差异。比值形式能在一定程度上消除这些固定因素，使指标更接近损伤造成的相对扰动。

相位项使用 `sin(phase)` 是为了处理 wrapped phase，避免 `2*pi` 跳变造成虚假的大相位差。

### 3.3 超参数来源

`alpha=0.35`：相位对导波损伤敏感，但在当前模型中还没有做完整色散校准和模态分离，因此相位权重低于幅值项。它保留相位信息，但不让未校准相位主导选频。

`0.70 median + 0.30 q75`：中位数提供抗异常路径能力，75 分位保留“部分路径强烈受缺陷影响”的信息。若只用均值，少数异常 tx-rx 路径容易主导；若只用中位数，又会压低局部缺陷散射的贡献。


## 4. V1 无标签空间聚焦度 focus_score

### 4.1 定义

对每个频点，将路径扰动投影到低分辨率展开管壁网格：

```text
B_f(theta, z) = sum_p signal_p(f) * ray_kernel_p(theta, z)
```

这里 `ray_kernel_p` 与当前 V1 粗图中的 ray-tube 路径一致，但只使用低分辨率网格以降低计算量。

默认：

```text
highfreq_grid_size = 96
highfreq_sigma_ray_mm = 25
```

对 `B_f` 计算四个无标签聚焦指标：

```text
top5_energy / top20_energy
peak_to_median
Gini
entropy_score = 1 - normalized_entropy
```

组合：

```text
focus_score =
  0.35 * top5_over_top20
  + 0.25 * peak_to_median
  + 0.25 * Gini
  + 0.15 * entropy_score
```

进入总分时：

```text
focus_factor = 0.45 + 0.55 * focus_score
```

### 4.2 物理意义

对于损伤定位，路径扰动不能只是“总能量大”。如果某个频点上的所有路径都均匀变化，这更可能是整体耦合变化、边界效应或全局传播差异，而不是可定位缺陷。

好的频点应当满足：多条异常路径在管壁展开空间中交汇，形成相对局部的能量增强区域。`focus_score` 就是对这种“无标签可聚焦性”的近似度量。

各子指标含义：

```text
top5_energy / top20_energy
```

衡量前 5% 高响应区域是否集中在前 20% 区域内。值越高，能量越集中。

```text
peak_to_median
```

衡量峰值相对背景中位数是否突出。值越高，局部峰越明显。

```text
Gini
```

衡量空间分布的不均匀性。均匀场的 Gini 低，局部化场的 Gini 高。

```text
entropy_score
```

衡量空间分布的信息集中程度。越接近局部集中，熵越低，`entropy_score` 越高。

### 4.3 超参数来源

`grid_size=96`：这是计算成本与空间判断能力的折中。选频阶段只需要判断频点的空间聚焦倾向，不需要生成最终训练用 `256 x 256` 粗图。`96 x 96` 对 16x16 PZT 阵列和当前几何 ray-tube 假设已足以区分均匀场与局部聚焦场。这里的 helical order 不是由频域数据分离得到，而是管壁展开几何中的多路径假设。

### 4.4 helical order 的含义

当前代码中的：

```text
helical_orders = (-1, 0, 1)
```

不是从频域响应 `H(f)` 中识别出来的三个到达波包，也不是通过时域门控分离得到的三个传播阶次。

它的含义是：在圆柱管壁展开坐标中，tx 到 rx 可以通过不同周向绕行数连接。令 tx 和 rx 的周向角差为：

```text
Delta theta = wrap(theta_rx - theta_tx)
```

则第 `m` 个几何绕行假设为：

```text
Delta theta_m = Delta theta + 2*pi*m
```

当前只取：

```text
m = -1, 0, 1
```

也就是直接路径和左右各一圈的近邻绕行路径。每个 tx-rx-frequency 的标量扰动会被投影到这三类几何路径上。

因此，V1 和 `physics_highfreq_quota` 中的 helical order 是 **geometric path hypothesis**，不是 **measured arrival order**。

### 4.5 为什么频域下仍然使用多个 helical order

频域稳态响应 `H(tx,rx,f)` 是所有传播路径、模态、反射和耦合效应的复合结果。若没有时域门控、模态分解或相速度校准，单个频点上的 `H(f)` 不能可靠地区分能量来自哪一条绕行路径。

当前 V1 的策略是保守的：不强行判断哪一个 order 是真实主导路径，而是把每个 tx-rx 的损伤扰动投影到少量几何上可能的路径上。后续通过多 tx-rx、多频率的路径交汇来形成粗定位先验。

这种做法的代价是粗图更宽、更模糊，但优点是不需要时域到达时间，也不需要先验相速度曲线。

如果后续要真正区分 helical order，需要增加额外信息，例如：

```text
1. 时域信号和到达时间门控；
2. 已知色散曲线 c_p(f) 或波数 k(f)；
3. 基于 exp(+i k(f) L_order) 的频域相干补偿；
4. 对 healthy phase unwrap 后拟合路径长度 L_order；
5. 完整 Green function / Born kernel 的相干散射成像。
```

当前 `physics_highfreq_quota` 尚未做这些步骤，所以论文中应避免写成“从频域数据得到三个 helical order”，而应写成“在展开圆柱几何中引入三个低阶绕行路径假设”。

`sigma_ray_mm=25`：与 V1 粗图默认 ray-tube 宽度一致，代表导波路径附近的有效敏感区域。这个参数不应在选频阶段和成图阶段使用完全不同的物理尺度，否则选出的频点会和实际粗图生成目标不一致。

`0.35, 0.25, 0.25, 0.15`：最高权重给 `top5/top20`，因为当前任务最关心缺陷能量是否落入少量候选区域；`peak_to_median` 和 `Gini` 共同约束峰值突出和全局不均匀性；`entropy_score` 作为辅助，避免单一指标过于敏感。

`focus_factor = 0.45 + 0.55 * focus_score`：不把 focus 作为硬门控。原因是低分辨率 V1 聚焦图本身只是近似物理先验，不能因为 focus 暂时较低就完全丢弃某个有稳定损伤响应的频点。


## 5. 路径对比度 path_contrast

### 5.1 定义

```text
path_contrast =
  (quantile_0.90(signal_p) - quantile_0.50(signal_p))
  / (quantile_0.90(signal_p) + quantile_0.50(signal_p) + eps)
```

进入总分时：

```text
contrast_factor = 0.70 + 0.30 * path_contrast
```

### 5.2 物理意义

损伤散射通常不是让所有路径等幅变化，而是对穿过或接近缺陷区域的路径影响更强。因此路径响应应具有一定差异性。

如果 `path_contrast` 太低，说明所有路径几乎同样变化，可能来自整体幅值漂移、耦合变化或边界条件变化。若 `path_contrast` 较高，说明存在更明确的路径选择性，有利于定位。

### 5.3 超参数来源

使用 `q90` 和 `q50`，而不是最大值和均值，是为了避免少数异常路径或数值噪声主导。`0.70 + 0.30 * contrast` 表明路径对比度是重要辅助项，但不是唯一目标。


## 6. 路径覆盖平衡 coverage_balance

### 6.1 定义

先计算 participation ratio：

```text
P = (sum_p w_p)^2 / (N * sum_p w_p^2)
```

其中 `w_p = signal_p(f)`，`N` 是路径数。`P` 的范围为 `[0, 1]`。

再计算中间最优偏好：

```text
coverage_balance =
  exp(-((P - P0) / sigma_P)^2)
```

默认：

```text
P0 = 0.35
sigma_P = 0.25
```

进入总分时：

```text
coverage_factor = 0.65 + 0.35 * coverage_balance
```

### 6.2 物理意义

路径参与度太低，说明只有极少数路径响应强，可能是异常路径、局部耦合问题或数值噪声。路径参与度太高，说明几乎所有路径都强烈变化，可能是全局幅值漂移或整体边界变化，定位能力反而差。

缺陷定位理想状态是：一部分路径明显受影响，另一部分路径作为对照不明显受影响。也就是中等 participation ratio 最有利。

### 6.3 超参数来源

`P0=0.35` 表示期望约三分之一量级的有效路径参与。这个值不是精确物理常数，而是根据当前 16x16 tx-rx 布局、多几何绕行路径假设和局部缺陷成像需求设置的中间最优点。

`sigma_P=0.25` 让惩罚较宽松。它不会严厉排除 participation 稍偏高或稍偏低的频点，而是只压制极端情况。


## 7. 高频分辨率权重 resolution_weight

### 7.1 定义

```text
resolution_weight(f) =
  clip((f / f_ref)^gamma, w_min, w_max)
```

默认：

```text
f_ref = 50 kHz
gamma = 0.5
w_min = 0.70
w_max = 1.60
```

### 7.2 物理意义

导波波长与频率近似成反比：

```text
lambda ~ c_phase / f
```

在相速度变化不剧烈的频段内，频率越高，波长越短，理论空间分辨率越高。当前 V1 粗图后验评价也显示 `high_frequency_band_map` 是最有效的信息通道。

但是高频也有风险：传播衰减更强、PZT 响应可能变弱、相位更容易受色散和模态影响。因此不能无限偏向最高频，需要上下限裁剪。

### 7.3 超参数来源

`f_ref=50 kHz` 是当前 20-100 kHz 频段的中间参考频率，也接近低/中/高频过渡中心。低于该频率的点不会被完全排除，高于该频率的点会获得温和增益。

`gamma=0.5` 使用平方根增长，而不是线性增长，避免高频权重过强。

`w_min=0.70, w_max=1.60` 保留低频基本权重，同时限制高频最大增益。这样体现高频分辨率优势，但不让频率本身压过信号可见性和健康响应稳定性。


## 8. 健康响应稳定性 health_weight

### 8.1 定义

先计算健康响应平均幅值：

```text
A0(f) = mean_{tx,rx} |H0(tx,rx,f)|
```

低于低分位的频点直接排除：

```text
A0(f) < percentile_5(A0) -> excluded
```

其余频点使用：

```text
health_weight =
  clip(A0(f) / median(A0), 0.5, 1.5)
```

### 8.2 物理意义

如果健康响应在某个频点过弱，则该频点的 `Hd/H0` 对噪声、数值误差和耦合变化非常敏感。尤其高频段更容易出现响应弱和衰减强的问题，因此必须加入健康响应稳定性约束。

### 8.3 超参数来源

`5%` 低分位是温和的稳定性过滤，只排除最弱的一小段频点。

`0.5-1.5` 裁剪区间避免健康幅值成为主导因素。健康响应强只说明信噪比可能较好，不等价于缺陷定位能力强。


## 9. 频段配额

### 9.1 默认配额

当前 top15 频点使用：

```text
low:  20-40 kHz   3 个
mid:  40-65 kHz   4 个
high: 65-100 kHz  8 个
```

命令行参数：

```text
--highfreq-low-max-khz 40
--highfreq-mid-max-khz 65
--highfreq-low-quota 3
--highfreq-mid-quota 4
--highfreq-high-quota 8
```

### 9.2 物理意义

低频：

- 衰减相对较小；
- 对大尺度壁厚变化更稳定；
- 相位和幅值对噪声不那么敏感。

中频：

- 提供低频稳定性和高频分辨率之间的过渡；
- 避免频点集合只集中在一个窄频段。

高频：

- 波长更短；
- 空间分辨率更高；
- 当前后验评价显示高频通道对 V1 粗图最有帮助。

### 9.3 超参数来源

该配额来自两部分依据：

1. 物理原则：多频反演中低频提供稳定性，高频提供细节分辨率；
2. 当前 `output2` 开发集后验评价：`v1_label_guided` 和 `all_completed` 的优势主要来自高频通道。

论文中建议表述为：

```text
The band quota was selected on a development set and then fixed for all subsequent evaluations.
```

不要把它表述为对测试集逐样本调参。若后续有更多样本，建议保留独立测试集验证该配额。


## 10. 与 label 的关系

`physics_highfreq_quota` 不使用 label。它只使用：

```text
H0, Hd, completed_mask, tx/rx geometry
```

label 只用于：

```text
1. 后验评价不同选频策略；
2. 分析高频配额是否合理；
3. diffusion 训练监督目标。
```

不能用于：

```text
1. 单样本选频；
2. 生成正式粗图；
3. 对每个缺陷样本单独调权重。
```

如果写论文，应明确区分：

```text
physics_highfreq_quota: label-free frequency selection
v1_label_guided: oracle-like post-hoc upper bound
```


## 11. V2-lite 相干通道的物理意义

### 11.1 V1 当前缺少什么

当前 V1 和 `physics_highfreq_quota` 的核心是 ray-tube 标量反投影。它主要使用：

```text
|Hd-H0| / |H0|
log(|H0|) - log(|Hd|)
|angle(Hd * conj(H0))|
```

这些量能反映某条 tx-rx 路径在某个频点是否发生了异常，但 V1 在投影时基本没有利用“相位传播应该随路径长度变化”的约束。

换句话说，V1 主要回答：

```text
哪些路径变了？
这些路径在空间上大概穿过哪里？
```

它没有充分回答：

```text
如果缺陷真的在某个像素 p，不同 tx-rx-frequency 的复数扰动相位是否能在 p 处对齐？
```

V2-lite 的价值就在于补充这个问题。

### 11.2 V2-lite 引入的额外物理知识

V2-lite 使用现有 healthy 频响相位拟合一个 effective wave number：

```text
k_eff(f)
```

然后对候选像素 `p` 构造近似散射路径长度：

```text
L_scat(tx, p, rx, order) = L(tx, p, order_1) + L(p, rx, order_2)
```

或者在更简化的 ray-tube 版本中使用与 tx-rx 路径相关的近似长度。对健康-损伤复扰动：

```text
DeltaH(tx, rx, f) = Hd(tx, rx, f) - H0(tx, rx, f)
```

做相干补偿：

```text
B(p) = sum_{tx,rx,f} W(tx,rx,f)
       * DeltaH(tx,rx,f)
       * exp(+i * k_eff(f) * L_scat(tx,p,rx))
```

输出可作为额外粗图通道：

```text
born_coherent_abs(p) = |B(p)|
phase_consistency(p)
born_real_positive(p)
```

它引入的额外物理约束是：

```text
正确位置：多频、多路径相位经传播补偿后更容易相干增强；
错误位置：相位补偿不匹配，复数叠加更容易相互抵消。
```

因此，V2-lite 比 V1 多使用了 **复数相位的传播一致性**，而不仅仅是路径幅值异常或相位差绝对值。

### 11.3 为什么用 healthy phase unwrap 拟合 k_eff(f)

当前已有数据中包含 healthy 管道的复频响：

```text
H0(tx, rx, f)
```

对每条高信噪比 tx-rx 路径，healthy 相位近似满足：

```text
unwrap(angle(H0(tx,rx,f))) ~= -k_eff(f) * L(tx,rx,order) + phi0(tx,rx)
```

其中：

```text
L(tx,rx,order)
```

来自展开圆柱几何中的路径长度假设，`phi0` 表示 PZT 耦合、电路相位和局部安装相位偏置。

在多个 tx-rx 路径上做鲁棒拟合，可以得到一个经验的：

```text
k_eff(f)
```

它不是严格的模态色散曲线，也不能保证分离出单一 guided-wave 模态；但它能给出当前仿真/实验系统中“相位随频率和路径长度变化”的有效趋势。

这正适合 V2-lite：只作为附加相干通道，不作为唯一成像依据。

### 11.4 为什么这是性价比高的选择

V2-lite 的性价比高，原因是它只需要现有数据：

```text
healthy_H_complex.npz
damaged_H_complex.npz
tx/rx geometry metadata
```

不需要：

```text
1. 重新做时域仿真；
2. 重新做真实管道时域门控实验；
3. 完整模态分解；
4. 完整 Green function；
5. 高成本 3D/壳模型全波场数据库。
```

它的计算成本也低于完整 Born/Green 模型，因为只是在现有 tx-rx-frequency 复响应上增加一个相位补偿和复数叠加。

从信息增量看，V2-lite 正好补上 V1 最缺的一部分：

```text
V1: 几何路径覆盖 + 幅值/相位扰动强度
V2-lite: 多频复数相位传播一致性
x_matrix: 未压缩原始 tx-rx-frequency 观测
```

这三者一起输入 diffusion 时，模型可以同时看到：

```text
1. 粗空间先验；
2. 原始频域观测；
3. 物理相干性提示。
```

如果 V2-lite 相干核不够准，diffusion 仍可依靠 V1 和 `x_matrix` 修正它；如果相干核有效，它会给网络更明确的缺陷候选区域。

### 11.5 V2-lite 不能声称什么

论文中需要避免过度表述。V2-lite 不能声称：

```text
1. 已经准确分离 helical order；
2. 已经完成多模态 guided-wave Green function 建模；
3. k_eff(f) 是严格材料色散曲线；
4. born_coherent_abs 是最终缺陷反演结果。
```

更准确的表述是：

```text
V2-lite provides an auxiliary coherent scattering cue based on an empirically estimated effective wavenumber from the healthy frequency response. It is used as an optional conditioning channel rather than a standalone reconstruction.
```

中文可写为：

```text
V2-lite 基于健康管频响相位拟合得到的有效波数，对健康-损伤复扰动进行近似传播相位补偿，从而构造一个多频相干散射提示图。该通道不作为独立反演结果，而作为 diffusion 的可选物理条件输入。
```

### 11.6 与完整 V2/Born/Green 模型的区别

完整 Born/Green 模型需要：

```text
DeltaH(tx,rx,f) = integral G(rx,p,f) q(p) G(p,tx,f) dp
```

其中 `G` 应包含：

```text
1. 模态色散；
2. 多模态传播；
3. 衰减；
4. PZT 激励/接收耦合；
5. 管壁曲率和边界条件；
6. 缺陷散射机制。
```

这需要额外仿真或半解析导波模型，开发成本高。

V2-lite 则只保留其中最便宜、最可能有收益的一部分：

```text
exp(+i * k_eff(f) * L)
```

也就是相位传播补偿。它牺牲精确性，换取低成本和可快速消融验证。


## 12. output2 开发集结果

测试样本：

```text
1-10,21-31
```

共 21 个 damage response。

`physics_highfreq_quota` 选出的频点：

```text
40.0, 32.5, 20.0,
42.5, 50.0, 47.5, 52.5,
70.0, 72.5, 67.5, 75.0, 80.0, 82.5, 77.5, 95.0 kHz
```

`ray_relative_delta` 通道平均结果：

| method | Pearson | top5 hit | mass in label | centroid error | NRMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1_label_guided | 0.1657 | 0.2543 | 0.0564 | 479.2 mm | 2.146 |
| all_completed | 0.1026 | 0.1521 | 0.0514 | 472.3 mm | 2.428 |
| physics_highfreq_quota | 0.0975 | 0.1256 | 0.0511 | 471.6 mm | 2.376 |
| physics_tomography | 0.0834 | 0.1039 | 0.0501 | 463.1 mm | 2.430 |
| relative_l2 | 0.0785 | 0.0956 | 0.0497 | 454.7 mm | 2.460 |

`high_frequency_band_map` 通道平均结果：

| method | Pearson | top5 hit | mass in label | NRMSE |
| --- | ---: | ---: | ---: | ---: |
| v1_label_guided | 0.1772 | 0.2743 | 0.0575 | 2.123 |
| all_completed | 0.1742 | 0.2807 | 0.0572 | 2.113 |
| physics_highfreq_quota | 0.1669 | 0.2678 | 0.0567 | 2.094 |
| relative_l2 | 0.1587 | 0.2027 | 0.0561 | 2.117 |
| physics_tomography | 0.1487 | 0.2171 | 0.0554 | 2.033 |

结论：

```text
physics_highfreq_quota > physics_tomography > relative_l2
```

在最有效的高频通道上，`physics_highfreq_quota` 已接近 `all_completed` 和 `v1_label_guided`。


## 13. 论文中建议的表述

可以写成：

```text
We propose a label-free, physics-informed frequency selection strategy named physics_highfreq_quota. The method scores each candidate frequency using a Rytov-type healthy-damaged perturbation, a low-resolution ray-tube focusing measure, path participation balance, a bounded high-frequency resolution prior, and a healthy-response stability factor. A fixed low/mid/high frequency quota is then applied to preserve low-frequency robustness while emphasizing high-frequency spatial resolution.
```

中文表述：

```text
本文提出一种无标签物理约束选频方法 physics_highfreq_quota。该方法首先基于健康/损伤复频响比值构造 Rytov 型扰动，以表征缺陷引起的幅值和相位变化；随后利用与 V1 粗图一致的 ray-tube 几何在低分辨率网格上计算无标签空间聚焦度，以排除全局均匀响应或少数异常路径主导的频点；最后结合路径覆盖平衡、健康响应稳定性和有界高频分辨率权重，并通过低/中/高频配额选择最终频点集合。
```


## 14. 参考文献

1. Won-Kwang Park, *Multi-frequency subspace migration for imaging of perfectly conducting, arc-like cracks*, 2013.  
   https://arxiv.org/abs/1306.0265  
   作用：支持多频成像和频率加权能提升散射体/裂纹成像质量。

2. Carlos Borges, Manas Rachh, *Multifrequency inverse obstacle scattering with unknown impedance boundary conditions using recursive linearization*, 2021.  
   https://arxiv.org/abs/2104.13489  
   作用：支持低频到高频 continuation 思想，即低频稳定、高频提供更高分辨率。

3. Lars Ruthotto, Julianne Chung, Matthias Chung, *Optimal Experimental Design for Constrained Inverse Problems*, 2017.  
   https://arxiv.org/abs/1708.04740  
   作用：支持把频点选择看作逆问题中的最优实验设计或测量预算选择问题。

4. Sven Nordebo, Mats Gustafsson, Andrei Khrennikov, Borje Nilsson, Joachim Toft, *Fisher Information for Inverse Problems and Trace Class Operators*, 2012.  
   https://arxiv.org/abs/1203.5397  
   作用：支持使用 Fisher information 分析线性化逆问题中的测量信息量。

5. V. V. N. Sriram Malladi, Mohammad I. Albakri, Manu Krishnan, Serkan Gugercin, Pablo A. Tarazaga, *Estimating Experimental Dispersion Curves from Steady-State Frequency Response Measurements*, 2021.  
   https://arxiv.org/abs/2101.00155  
   作用：支持直接使用稳态频响 FRF 作为导波频域分析入口。

6. Marcus Haywood-Alexander, Nikolaos Dervilis, Keith Worden, Gordon Dobie, Timothy J. Rogers, *Informative Bayesian Tools for Damage Localisation by Decomposition of Lamb Wave Signals*, 2022.  
   https://arxiv.org/abs/2205.12161  
   作用：支持导波损伤定位中使用传播路径、反射/散射特征和物理先验。

7. M. Haywood-Alexander, N. Dervilis, K. Worden, R. S. Mills, P. Ladpli, T. J. Rogers, *A Bayesian Method for Material Identification of Composite Plates via Dispersion Curves*, 2022.  
   https://arxiv.org/abs/2209.03706  
   作用：说明导波频散曲线中频率与相速度/群速度/波数之间的关系是 guided-wave 方法的重要物理基础。

8. P. Huthwaite, F. Simonetti, *High-resolution guided wave tomography*, Wave Motion, 2013.  
   https://doi.org/10.1016/j.wavemoti.2013.04.004  
   作用：支持 guided-wave tomography 中利用多路径、多频信息进行壁厚/损伤成像的总体思想。

9. M. Huthwaite, *Evaluation of inversion approaches for guided wave thickness mapping*, Proceedings of the Royal Society A, 2014.  
   https://doi.org/10.1098/rspa.2014.0063  
   作用：说明 guided-wave 厚度反演中传播模型、频散和散射近似会显著影响成像结果，支持将 V2-lite 作为辅助通道而非最终反演。


## 15. 参数汇总表

| 参数 | 默认值 | 物理/算法意义 |
| --- | ---: | --- |
| `highfreq_phase_weight` | 0.35 | 相位扰动有用但未做完整色散校准，权重低于幅值项 |
| `highfreq_grid_size` | 96 | 低分辨率 V1 聚焦图网格，降低选频计算成本 |
| `highfreq_sigma_ray_mm` | 25 | 与 V1 ray-tube 粗图一致的路径敏感宽度 |
| `highfreq_f_ref_khz` | 50 | 20-100 kHz 频段的中间参考频率 |
| `highfreq_resolution_gamma` | 0.5 | 高频权重平方根增长，避免过度偏向最高频 |
| `highfreq_resolution_min` | 0.70 | 保留低频基本权重 |
| `highfreq_resolution_max` | 1.60 | 限制高频最大增益 |
| `highfreq_participation_target` | 0.35 | 中等路径参与度最利于定位 |
| `highfreq_participation_sigma` | 0.25 | 对 participation 的宽松惩罚尺度 |
| `min_healthy_abs_percentile` | 5 | 排除健康响应最弱的一小部分频点 |
| `low/mid/high quota` | 3/4/8 | 保留低中频稳定性，同时强化高频分辨率 |
