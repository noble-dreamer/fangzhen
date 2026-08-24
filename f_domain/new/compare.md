# 新旧管道频域仿真对比说明

本文对应 [README.md](README.md) 的“服务器执行顺序”，说明新增仿真为何存在、产物的物理意义，
以及最终送入 EDM 的数据。当前 README 默认最终选择 `F=12`；若以后修改 `--count`，下文的 `12`
应统一替换为实际的 `F_selected`。

## 1. 结论先行：新版并未增加每个训练样本的求解量

一次频域求解固定一个 TX 和一个频率，但会同时读出 16 个 RX。因此：

```text
COMSOL 求解次数/样本 = TX 数 × 频率数
H 的数据形状        = TX 数 × RX 数 × 频率数
```

旧版虽然输出 `H[16,16,15]`，实际是 `16×15=240` 次求解，不是 `16×16×15` 次。新版当前输出
`H[16,16,12]`，是 `16×12=192` 次求解，单个正式缺陷样本减少 48 次，即减少 20%。

新版增加的是整批数据共享的一次性仿真：导入 DC 标准曲线、建立可信 Shell 频散库、检查网格、
校正 Shell 偏差、映射标准模态身份，并用新缺陷重新选择频率。这些结果不会对每个训练样本重复计算。

| 对比项 | 旧版 | 新版 |
|---|---|---|
| 正式缺陷响应 | 旧缺陷，固定旧 top15 | image-texture-v4 小/中/大缺陷，新选 12 频点 |
| 每样本求解数 | `16×15=240` | `16×12=192` |
| 每样本复响应 | `H[16,16,15]` | `H[16,16,12]` |
| 选频依据 | 旧 pilot 的复响应 sensitivity | 新缺陷响应 + 校准频散覆盖/厚度敏感性 |
| 共享物理数据 | 无显式频散库 | DC 标准曲线、校准后的 `f(h,k,n,b_j)`、`b_j -> m`、速度/导数/可信度 |
| 标签 | 旧缺陷生成器 | `irregular_polygon_image_texture_v4_multiscale`，三档面积/尺度 |

有限管 COMSOL 执行路径没有改变：仍由同一个脚本逐样本建模并流式循环 `(TX,f)`，接收端、复响应
schema、checkpoint 和标签路径均保持不变。变化只发生在每个 damaged 样本建模前的壁损栅格：v4
用 OpenCV 读取 `Defect _models/1.jpg`，按旧 MATLAB 顺序完成旋转、多边形掩膜、灰度到深度映射和
边界高斯平滑。生成器改变后，旧 damaged/pilot/选频结果不能复用；非缺陷物理和频率轴完全一致时，
healthy 的物理模型本身不依赖缺陷纹理，但正式配对仍必须通过模型指纹审计。

## 2. 新增内容与 README 服务器命令一一对应

下表的数量按 README 当前参数计算。“求解点”是一次单胞特征频率求解；一个点可返回多个模态，
不能和有限管的一次 TX/频率求解按相同成本直接比较。

| README 步骤 | 是否 COMSOL | 当前规模 | 产物与用途 | 是否保留到 EDM 阶段 |
|---|---:|---:|---|---|
| `# 0 环境检查` | 否 | 0 | 确认 `mph`/COMSOL 6.4 环境 | 否 |
| `# 1 科学 smoke` | 是 | `3h×5k×3n=45` 点，每点 4 模态 | 验证模型树、schema、导数和 resume | 通过后删除 |
| `# 2 DC 导入` | 否 | 70 个 TXT、27 条标准模态/厚度 | 标准 `F(1..8,1..3)`、L/T 曲线和源 SHA | F 参考库与 metadata 进入 LUT 包 |
| `# 3 formal Shell` | 是 | `7h×41k×9n=2583` 点，每点 8 模态 | `n=0..8` 原始 Shell 分支；n=0 仅作锚点 | 作为共享校准来源，不进样本 batch |
| `# 4 有限管网格门禁` | 是 | `3h×3频率×3网格=27` 个 TX/频率工况 | 严格网格收敛报告 | 报告仅作门禁/追溯 |
| `# 5 Solid 校准` | 是 | `3h×3k×4n=36` 点，每点 4 模态 | Shell/Solid 频率比、分支匹配和置信度 | 校准来源，不进样本 batch |
| `# 6 发布校准库` | 否 | 0 | 合并步骤 3/4/5，重算可信导数 | 选频和解码 LUT 的 Shell 数值源 |
| `# 7 标准映射` | 否 | 8 次带 dummy 的全局指派 | 24 条 `b_j -> F(n,m)`、质量指标和两端 SHA | mapping NPZ/report 进入 LUT 包 |
| `# 8 image-texture-v4 pilot` | 是 | `(1 healthy+12 defect)×4 TX×33频率=1716` 工况 | 新缺陷的宽频复响应 | 只用于选频，不作正式训练样本 |
| `# 9 选择 12 频点` | 否 | 0 | ranking、summary、`selected_frequencies.txt` | 频率轴/追溯信息 |
| `# 10 正式 worker` | 是 | 每缺陷 `16×12=192` 工况 | `H_damaged[16,16,12]` 和标签 | 是，正式样本源数据 |
| 步骤 10 后的 healthy 命令 | 是 | 一次 `16×12=192` 工况 | 所有样本共用的 `H_healthy` | 是，生成差分特征所必需 |
| worker 后的 audit/tar | 否 | 0 | 完整性审计和最小传输清单 | 审计通过后按 manifest 取回 |

若有 200 个正式缺陷，旧版主体约为 `200×240` 个缺陷工况；新版主体约为 `200×192` 个缺陷工况，
外加上表的一次性共享成本。新增成本的目的不是增加样本维度，而是避免用未经验证的旧频率和物理先验。

## 3. 为什么这些仿真互相依赖

```text
#1 smoke（只验证代码）

#3 formal Shell -> #5 Solid 校准 --+
#4 有限管网格门禁 -----------------+-> #6 calibrated Shell --+
#2 DC 标准库 ------------------------------------------------------+-> #7 b_j -> m -> LUT/EDM

#6 calibrated Shell ──┐
#8 image-texture-v4 pilot ─┴─> #9 新选频 -> #10 healthy + defect workers
                                                │
                                                v
                                    audit -> get_pic -> EDM
```

- 步骤 5 必须读取步骤 3 的完全相同 `h/k/n/branch` 轴，否则 Solid 模态无法与 Shell 分支对应。
- 步骤 4 不直接修改频散曲线；它证明有限管接收响应不是粗网格误差主导，是发布器的质量门禁。
- 步骤 6 必须同时得到步骤 3、4、5。只有 Shell 曲线而没有网格与 Solid 校验时，只能标记
  `scientific_ready=false`。
- 步骤 7 只赋标准身份，不替换数值；它同时绑定步骤 2 与 6 的 SHA，任何一端变化都使旧映射失效。
- 步骤 8 在计算上可与步骤 2--7 并行，但步骤 9 必须同时拥有新缺陷响应和已校准频散库：前者说明
  “真实缺陷是否产生稳定散射”，后者说明“该频率是否有可观测、厚度敏感且可追踪的模态”。
- 步骤 10 必须等待步骤 9 固定有序频率轴；healthy 和所有 worker 的几何、网格、TX/RX、频率顺序
  必须完全一致，否则不能形成可信的 `H_damaged-H_healthy`。
- 缺陷形状/尺度分布变化时重跑步骤 8--10；材料、管径、厚度/频率/n 轴变化时重跑步骤 2、3、5--10。
  映射若因候选缺失而失败，步骤 3 以新根改为 16 模态，并重跑步骤 5--7；网格物理未变时步骤 4 可复用。

## 4. 各产物的物理意义及对任务的作用

| 产物 | 物理意义 | 对厚度反演的作用 |
|---|---|---|
| `H(tx,rx,f)` | 谐波轴向载荷下，接收贴片平均轴向位移的复振幅；包含幅值和相位，不是电压或时域波形 | 记录缺陷造成的传播、散射和相位变化，是每样本主要观测 |
| formal Shell 库 | 均匀剩余厚度 `h`、轴向波数 `k_z`、周向阶次 `n` 下的特征频率和模态分支 | 给出“某频率/模态如何随厚度变化”的物理坐标系 |
| DC 标准库 | 同管径、材料和厚度轴上的标准 `F(n,m)`、L/T 独立曲线 | 作为模态名称权威和 `k/c_g` 匹配参考，不覆盖 Shell 数值 |
| 标准映射 | calibrated `b_j` 与 DC `m` 的一对一身份及质量门禁 | 使解码 LUT 只使用 24 条唯一、可追溯的 F 分支 |
| `c_p=2πf/k_z` | 相位传播速度 | 对应传统速度图方法中的速度—频厚积关系 |
| `c_g=2π·df/dk_z` | 波包群速度 | 判断能量传播方向和分支稳定性 |
| `df/dh`、`dk/dh` | 厚度变化引起的频率变化，以及固定频率下的波数变化 | `|dk/dh|` 大表示相位对减薄更敏感，是选频/解码权重 |
| polarization/observability | 模态的径向、周向、轴向极化及轴向贴片可观测代理 | 屏蔽当前轴向激励/接收难以看到的模态 |
| 网格报告 | EPW 8 与 12 的有限管复响应/相位差 | 防止把离散误差误认为缺陷散射或厚度敏感性 |
| Solid 校准 | 完整壁厚实体单胞相对 Shell 的频率比、极化匹配和残差 | 修正/降权 Shell 模型偏差；它不是实际三维 PZT 管道认证 |
| image-texture-v4 pilot | 图像纹理小/中/大不规则减薄对宽频、多路径复响应的实际扰动 | 防止频率只对旧平滑生成器或某一种尺寸有效 |
| calibrated library | 经门禁和 Solid 修正后的共享频散表及置信度 | 当前显式参与选频；后续作为解码层的固定 LUT |

### 4.1 执行到步骤 6 会不会得到示例图那样的曲线

会得到可绘制类似曲线的数值，但步骤 6 当前只输出
`dispersion/axisymmetric_shell_dispersion_library.npz` 和 metadata JSON，不会自动生成 PNG。固定一个
厚度、周向阶次和 branch 后，以 `frequency_hz/1000` 为横轴、`phase_velocity_m_s/1000` 为纵轴，
并同时应用有效 mask 与 `frequency_in_range_mask`、跳过 `k_z=0`，就能得到“相速度—频率”分支；
也可改画 `group_velocity_m_s`。

它不会与示例图完全相同：当前正式带宽是 `15--110 kHz`，而示例覆盖约 `224--320 kHz` 或
`0--1000 kHz`；当前专用库扫描 `n=0..8`，示例中的 `n=21/35/43/46` 不在本次轴上。步骤 6 仍保存
`branch_id + n + polarization`；步骤 7 的 sidecar 才给 24 条 F 分支赋予 DC 标准名称，且不会改变
Shell 数值。若要复现示例的频段和阶次，必须另开输出根、
扩展频率/k 轴与 `circumferential-orders`，并重新做网格和 Solid 校准，不能外推现有库。

### 4.2 `F(n,b_j)` 与标准 `F(n,m)` 的关系

两者描述同一组非轴对称频散分支，`n` 都是明确给定的周向阶次；`F(n,b_j)` 是本项目的临时内部
记法，不是标准模态名称，区别在于第二个索引的物理资格：

| 标记 | 第二个索引的含义 | 当前是否可以直接得到 |
|---|---|---|
| `F(n,b_j)` | 本次数值库内部追踪分支 `j`，只保证在同一配置和同一 NPZ 中连续 | 可以 |
| `F(n,m)` | 由 DC 3.2 命名、再经全局门禁赋给 Shell 分支的标准阶次 `m` | 步骤 7 达到 24/24 后可以 |

不同文献对 m 的编号起点和分支归属可能不同，论文必须声明采用的命名规范与参考求解器；这里不把
m 简化成一个未经验证的“径向节点数”。

#### 4.2.1 当前 `b_j` 的数学来源（已实现）

指定厚度 h、轴向波数 `k_z` 和周向阶次 n 后，轴对称扩展单胞采用谐波形式：

```text
u(r,z,theta,t) = u_hat(r,z) exp(i*k_z*z + i*n*theta - i*omega*t)
```

离散后在每个网格点 `(h_i,k_l,n)` 求广义特征值问题：

```text
K(h_i,k_l,n) phi_q = omega_q^2 M(h_i) phi_q,    f_q = omega_q/(2*pi)
```

当前 COMSOL 配置返回距 `eigen_shift=60 kHz` 最近的 `M=8` 个正频率特征解，而不是从零频开始的
最低 8 阶。对第 q 个解，代码计算三分量位移能量和归一化极化：

```text
E_(q,a) = integral |u_(q,a)|^2 dS,    a in {radial,circumferential,axial}
p_(q,a) = E_(q,a) / sum_a E_(q,a)
```

当前 prescribed-order 实现取 `s_q=p_q/||p_q||_2` 作为 signature。对同一厚度中相邻的两个 k 点，
前一点分支 j 与后一点候选 q 的 MAC、频率差和极化差定义为：

```text
MAC_(j,q) = |s_j^H s_q|^2                         # signature 已归一化
d_f(j,q)  = |log(max(|f_j|,1)/max(|f_q|,1))|
d_p(j,q)  = 0.5 * ||p_j-p_q||_1
d_n(j,q)  = 1[n_j != n_q]
```

现有代码的匹配代价严格为：

```text
C_(j,q) = 1-MAC_(j,q) + 1.5*d_f(j,q) + 0.25*d_p(j,q) + 0.75*d_n(j,q)
```

由于 prescribed-order 扫描一次只处理一个固定 n，正常情况下 `d_n=0`。一个有效解与无效解配对时
代价设为 `10^6`。然后在 M 个候选的全排列集合 `S_M` 上求最小总成本指派：

```text
pi_l* = argmin_(pi in S_M) sum_(j=1..M) C_(j,pi(j))
```

实现使用 Hungarian 算法求 `pi_l*`，再按该排列重排当前 k 点的频率、极化、observability 和
signature。具体步骤为：

1. 对每个 h，在首个 `k_0` 将 M 个有限频率从低到高排序，并把这个顺序定义为局部分支 j。
2. 从 `k_1` 到 `k_(K-1)` 逐点执行上式，得到 k 方向的连续分支及匹配残差
   `r_k(h_i,k_l,j)=C_(j,pi_l*(j))`。
3. 取中间波数 `k_ref=k_(K//2)`，在相邻厚度 `h_(i-1)` 与 `h_i` 之间再次执行相同匹配，并将
   得到的排列应用于当前厚度的整条 k 轴，产生厚度残差 `r_h(h_i,j)`。
4. 定义追踪置信度：

```text
tracking_confidence = exp(-clip(nan_to_num(r_k,50)+nan_to_num(r_h,50),0,50))
```

5. 对排序后的第 j 条分支设置局部编号 j；多个 n 拼接时设置全局
   `branch_id=b_j=j+order_index*M`。

因此 `b_j` 的严格含义是“由本次离散网格、shift、M 和上述连续匹配共同定义的数组身份”。它不是
标准物理阶数，也不保证从最低 m 开始。改变 shift、`mode_count`、k/h 轴、模态特征或模型后，即使
数字相同，旧 `b_j` 与新 `b_j` 也不能直接等同。由于当前 signature 只有三分量极化，两个极化接近
但场形状不同的分支在交叉附近仍可能混淆，这正是它不能直接升级为标准 m 的原因。

#### 4.2.2 从 `b_j` 映射到标准 m（已实现，待服务器正式源数据）

`build_standard_mode_mapping.py import-dc` 已把七厚度水平 TXT 转为 schema-v1 标准库。高阶模态可在
15 kHz 以上出现物理截止；`valid_mask` 保留实际支持区，映射只插值、不向截止以下或 110 kHz 以上
外推。`map-shell` 只接受 `scientific_ready=true`、scope 正确、`h=[5,6,7,7.5,8,9,10]`、
`k-count=41`、`n=0..8` 和 8/16 候选的 calibrated schema-v3。

对固定 n、Shell 分支 `b_j` 和 DC 标准 m，在相同厚度及共同 k 支持上计算：

```text
d_f = |log(f_shell/f_DC)|
d_g = |cg_shell-cg_DC| / max(|cg_shell|+|cg_DC|, 200 m/s)
C   = median(d_f) + 0.25*median(d_g) + 0.5*(1-coverage)
Q   = coverage*consistency*exp(-C/0.12)*(1-exp(-max(margin,0)/0.05))
```

Shell 候选点还必须位于 `15--110 kHz`、`tracking_residual<=1.1` 且 `f/c_g` 有限。代码先比较同一
`b_j` 的三个点级代价 `d_f+0.25*d_g`，剔除最佳与次佳差 `<0.02` 的交叉模糊点。`coverage` 是保留
共同点数除以该 Shell 分支全部合格点数；`consistency` 是有覆盖的厚度中，该 m 获得最低逐厚度
中位点级代价的比例。

每个 n 构造 8（或 16）条 Shell 候选、3 条标准曲线及 dummy 行/列的方阵，以 `C` 做 Hungarian
一对一指派；dummy 代价为 `0.30`，因此允许 `unknown`，不会为了凑满三条而强制命名。被指派项同时要求：

```text
points>=20, thicknesses>=5, coverage>=0.60,
median(d_f)<=0.12, P95(d_f)<=0.25, C<=0.30,
margin>=0.05, consistency>=0.80, Q>=0.35
```

只有 24/24 通过才写 `standard_mode_mapping.npz`；它保存 `branch_id/n/family/standard_m`、核心质量
指标和 calibrated/DC 两端 SHA；report 保存完整门禁诊断。失败只写 report、非零退出，并把缺失/模糊项保留为 `unknown`。
合成乱序分支验证已精确恢复 24/24；删除 `F(8,3)` 候选后得到 23/24、无 NPZ，临时输出已删除。

步骤 5 的 Solid-to-Shell 匹配仍只负责数值修正，因为 Solid 解没有标准 m。步骤 7 只赋身份：

```text
D_(n,m)(h,k) := D_(n,b_j)(h,k),    m = Map(n,b_j)
```

DC、原始 Shell、Solid 校准和 calibrated Shell 数值始终分别保存，不能借重命名互相覆盖。

#### 4.2.3 是否只计算一次

是，但“一次”指完成覆盖全部目标 `h/k/n/m` 网格的一次离线仿真任务，不是只求一个 COMSOL 点：

- 当前 `F(n,b_j)` formal 库按一套物理配置计算一次，所有缺陷共享。
- 标准 `F(n,m)` 参考库和 `b_j -> m` 映射也只需建立一次；通过门禁后，pilot、正式 worker、EDM
  训练和推理均读取相同库，不为每个缺陷重复求频散。
- 第一次必须完成专用 `n=0..8` 扫描；若 8 候选不足则以新根补做 16 候选并重跑依赖校准。之后只要
  calibrated/DC 两端 SHA 不变就直接复用。
- 缺陷位置、面积、深度和 seed 改变不要求重算均匀管频散库；每个缺陷自己的 `H(tx,rx,f)` 仍需
  单独仿真。
- 管径、材料、壁厚范围、边界/结构物理、频率/k/n 轴改变时应重算；只改变换能器方向、尺寸或
  接收分量时，频散本征值可能可复用，但 observability 和选频资格必须重新计算与认证。

当前选频器可直接消费 calibrated `F(n,b_j)`；`plan.md` 的标准感知解码 LUT 则强制要求 24/24
映射。当前 DC 参考已齐，仍缺服务器步骤 3--7 的正式 formal、Solid、发布库和 mapping 结果。

### 4.3 步骤 6 根据什么把频散库称为“可信”

“可信”是当前 Shell-proxy 数据链内的受限资格，不是实验真值或真实三维 PZT 管道认证：

1. 步骤 3 对每个 `h/k_z/n` 求特征频率、三分量极化和模态 signature；Hungarian 匹配综合
   signature MAC、对数频率差、极化差和周向阶次，沿 k 轴及厚度轴追踪 branch。
2. 只有 tracking/thickness residual `<=1.1` 的相邻链接才接受速度和厚度导数；断裂链接保持无效，
   非有限值或 `|c_g|<1 m/s` 的点不能产生有效 `dk/dh`，拒绝值保持 `NaN`。
3. 步骤 4 必须全部通过有限管 EPW 8 对 EPW 12 的门禁：16 通道复响应相对 L2 `<=5%`、相位
   RMSE `<=0.05 rad`，且不存在 COMSOL problem、非有限值或零接收通道。它只证明数值网格收敛。
4. 步骤 5 用完整壁厚 Solid 单胞按频率和三分量极化匹配同一 Shell SHA 的分支；有效模态比例必须
   至少 `50%`，匹配结果提供 `solid/shell` 频率比和 calibration confidence。
5. 步骤 6 在 Solid 采样覆盖的 h/k 范围内按 branch、距离和置信度插值频率修正比，修正频率后
   重新计算速度与厚度导数；校准覆盖必须达到原有效导数点的 `10%`，并保存三项来源 SHA。

全部通过后才写入 `scientific_ready=true` 和
`readiness_scope=shell_proxy_simulation_only`。选频器同时检查这两个字段；它们不代表曲线已由实验、
真实焊缝、真实阻尼或完整换能器传递函数认证。

### 4.4 宽频 pilot 根据什么选出新的 12 个频率

步骤 8 对同一组 `20--100 kHz/2.5 kHz` 候选频率，计算一个 healthy 和 small/medium/large 各 4 个
缺陷的 `H`。步骤 9 不是逐缺陷选频，而是把全部 pilot 汇总成一份全局列表：

```text
ΔH = H_damaged - H_healthy
R_class(f) = 每个尺寸类别内 relative complex L2 的中位数
R_balanced(f) = 三个 R_class 的几何平均
Q_disp(f) = median_h max_branch(|dk/dh| × observability × tracking_confidence)
```

最终评分先对前五项做候选频率间的 `0--1` 排名，再加权：

| 项目 | 权重 | 含义 |
|---|---:|---|
| 三档平衡复响应变化 `R_balanced` | 40% | 缺陷相对 healthy 的复散射强度，避免大缺陷独占 |
| 样本稳定性 `mean/(mean+std)` | 15% | 不同位置/形状下是否稳定 |
| `abs(angle(Hd·conj(H0)))` | 10% | 缺陷引起的相位变化 |
| TX-RX path participation | 10% | 是否由较多传播路径共同支持，而非少数异常通道 |
| 频散质量 `Q_disp` | 20% | 厚度敏感、可观测且追踪/校准可信的分支强度 |
| 有效厚度覆盖率 | 5% | 7 个厚度点中有可靠分支覆盖的比例 |

候选还必须满足 healthy 幅值不低于其 5% 分位、频散厚度覆盖 `>=2/3`；频散点本身要求
`observability>=0.05`、residual `<=1.1`、有效有限 `dk/dh` 和 `n>0`。最后在 `<=40 kHz`、
`40--70 kHz`、`>70 kHz` 三段分别选 `3/4/5` 个，最小间隔 `2.5 kHz`。若某频段合格候选不足则
报错，不会回退旧 top15。输出的一份
`selected_frequencies.txt` 随后供 healthy 和所有正式样本共同使用，所以每个 `H[:,:,i]` 的 Hz 含义一致。

当前专用扫描使用 `n=1..8` 的 F 分支（`n=0` 仅作 L/T 锚点），测量是轴向激励/轴向接收，
主要支持可观测的 F/螺旋分支。
它没有证明纯 `T(0,1)` 可反演；启用 T 模态需要周向激励/接收和额外传播距离。

## 5. 最终进入 EDM 的数据：新旧差异

服务器取回的是可追溯源数据，训练前仍需用同一 healthy/damaged 复响应重新生成 `get_pic` 产物。

| 层级 | 旧版 EDM | 新版 EDM |
|---|---|---|
| damaged/healthy 源响应 | 各 `H[16,16,15]` | 各 `H[16,16,12]`，频率由新 pilot 重选 |
| 直接条件 `x_matrix` | `[7,15,16,16]` | `[7,12,16,16]`；7 通道定义不变 |
| 直接空间条件 `pic` | `[8,256,256]` | `[8,256,256]`，但必须由新 12 频率响应重新生成 |
| 物理坐标 | `frequency_hz[15]`、TX/RX 各 16 | `frequency_hz[12]`、TX/RX 各 16，必须与 F 轴同步排序 |
| 监督目标 | 旧缺陷的连续 `/9 mm` 壁损图 | image-texture-v4 连续 `/9 mm` 壁损图；源标签 512²，loader 调整到 256² |
| 显式共享频散条件 | 无 | calibrated Shell + DC + 24/24 mapping；当前选频已用 Shell，解码接入仍待实现 |

`x_matrix` 的 7 个通道仍为 `log_abs_delta`、`log_abs_reldelta`、`phase_cos`、`phase_sin`、
`healthy_log_abs`、`damaged_log_abs` 和 `valid_mask`。因此网络主体接收的信息类型不变，变化的是频率轴、
缺陷分布，以及新增的共享频散先验。

当前已实现的 EDM loader 直接读取：

```text
pic [B,8,256,256]
x_matrix [B,7,F_selected,16,16]
frequency_hz [B,F_selected]
tx_indices/rx_indices [B,16]
target [B,1,256,256]
```

现有正式 YAML 和 `XMatrixEncoder` 仍配置 `frequency_count=15`。使用当前 12 频点数据时必须创建
`frequency_count=12` 的新配置并从头训练；第一层输入通道数改变，旧 15 频点 checkpoint 不能作为
“已经适配新数据”的严格 checkpoint 直接加载。

校准频散库在现阶段通过步骤 7 改变最终频率集合，属于间接物理约束。`plan.md` 中的解码层
`LUT -> T_disp/M_disp` 尚未实现；实现后它才成为所有样本共享的直接模型条件，而不是每样本复制一份。

以下内容不会进入 EDM batch：MPH、COMSOL 全场、逐工况 CSV、progress、smoke、pilot 响应、网格 cases。
网格报告、Solid 校准文件和选频 summary 随传输包保留用于门禁与复现，但原始报告不是图像通道。
最终应以 `audit_irregular_dataset.py` 生成的 `transfer_manifest.txt` 为唯一取回清单。
