# 管道频散混合解码器实施计划

## 1. 目标与当前结论

目标是在 EDM 生成壁损厚度图的解码阶段显式使用管道频散关系，而不只在编码器中增加频率
特征。扩散状态继续保持单通道归一化壁损图，最终输出契约仍为 `[B,1,256,256]`。

当前 Dataset A 使用轴向 `z` 激励和轴向位移接收。现有两圈阵列只有一个轴向间距，复频响
也是多模态、绕行路径和反射的叠加，因此不能从每个 `TX-RX` 通道唯一恢复一张 T/F 相速度图。
阶段一只使用当前测量方式可观测的 F/轴向极化分支：

- 频散表内部使用 `k_z`，因为相位沿路径直接累积波数；诊断输出再计算 `c_p=2*pi*f/k_z`，
  与传统速度图表述保持对应。
- 纯 `T(0,1)` 以周向位移为主，在当前轴向接收下接近不可观测，而且通常对厚度不敏感；
  阶段一保留 T 模态接口，但通过 observability 和 `|dk/dh|` 门禁禁用。
- 阶段二增加周向激励/接收及额外传播距离后，再正式启用 T 模态，不能用当前数据宣称完成
  T 模态反演。

### 1.1 仿真前置链状态（2026-08-22）

代码链已完成：formal schema-v3 Shell 扫描、有限管严格网格门禁、最多 36 点轴对称实体校准、
Shell-proxy 校准库发布、irregular-v3 pilot 选频、16 TX worker 和 EDM 最小传输审计。服务器正式
结果仍待按 `README.md` 第 2 节依次运行；本机 smoke 均已在记录简要结果后删除。

发布器只在三项来源一致且门禁通过后写入 `scientific_ready=true` 和
`readiness_scope=shell_proxy_simulation_only`，不构成真实管道认证。最终候选频率只由新的
`irregular_polygon_texture_v3_multiscale` 三档缺陷 pilot 与该校准库重选；旧超高斯样本、旧 top15、
旧 healthy 和旧 341 样本不再参与。网络实现前必须先取回审计清单列出的最小数据包。

DC 3.2 标准管道曲线已人工导出到 `D:\lab_ultr\fz\dc_exports`。厚度轴为
`[5,6,7,7.5,8,9,10] mm`，固定内径 `300 mm`，外径为 `300+2h mm`，材料为
`E=70 GPa, nu=0.33, rho=2700 kg/m3`。七个目录各有 `F1..F8/L/T` 共 10 个 TXT；
每个厚度包含 24 条 `F(n,m)`、2 条 L 和 1 条 T。目标频带内所有曲线单段、波数严格递增，
`k=2*pi*f/cp` 最大相对误差小于 `9e-15`。所有 F/L 文件内容唯一；七份 `T(0,1)` 相同是
无频散扭转基模的正常物理结果，不得作为误导出拒绝。

## 2. 标准 `F(n,m)` 与 `b_j -> m` 前置阶段

该阶段只建立一次均匀管标准频散库和模态身份映射，不为每个缺陷重复计算。映射只赋予标准身份；
DC 数值、Shell 数值和 Solid 校准值必须分别保存，不能用重命名掩盖数值替换。

### 2.1 DC 标准参考库

新增单文件入口 `f_domain/new/build_standard_mode_mapping.py`，提供两个子命令：

- `import-dc` 读取七个厚度目录，只接受水平排列的 TXT；每个 m 必须是 11 列
  `f/cp/cE1/cE2/|cE|/skew/time/angle/wavelength/k/attenuation`。
- 要求目录、文件、`F(1..8,1..3)` 标签和单位完整，m 从 1 连续编号；F/L 内容哈希不得跨厚度
  重复，T 重复允许；目标频带不得出现 NaN 断段、波数回退或外推。
- 输出 `dc_standard_pipe_dispersion.npz` 和 metadata JSON，保存带 valid mask 的
  `[thickness,standard_mode,point]` 数组、全部源 TXT SHA-256、DC `3.2.0.0` 与可执行文件 SHA-256。
- 原始 TXT、生成 NPZ 和 JSON 都是运行产物，不提交 Git。

### 2.2 新 formal 与校准链

使用新输出根，不能恢复旧 `n=1..12` checkpoint：

```bash
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_shell_dispersion.py --formal \
  --thickness-mm 5 6 7 7.5 8 9 10 --k-count 41 \
  --circumferential-orders 0 1 2 3 4 5 6 7 8 \
  --mode-count 8 --eigen-shift-khz 60 --frequency-range-khz 15 110 \
  --cores 16 --output-root f_domain/new/outputs/axisymmetric_standard_modes_formal
```

`n=0` 仅校验 L/T 锚点，`n=1..8` 才参与 F 映射。有限管网格门禁保持现有
`h=5/7.5/10 mm, f=20/52.5/95 kHz, EPW=6/8/12`。Solid 校准固定为
`h=5/7.5/10`、`k-index=0/20/40`、`n=0/1/4/8`、`mode-count=4`，仍为 36 个点；随后由
现有发布器生成新的 calibrated schema-v3 library。

### 2.3 全局映射与门禁

`map-shell` 读取 calibrated schema-v3 和 DC 标准库。对固定 n 的 `(b_j,m)`，只在相同厚度、
`15--110 kHz`、`tracking_residual<=1.1` 和共同 k 支持内插值：

```text
d_f = |log(f_shell/f_DC)|
d_g = |cg_shell-cg_DC| / max(|cg_shell|+|cg_DC|, 200 m/s)
C   = median(d_f) + 0.25*median(d_g) + 0.5*(1-coverage)
Q   = coverage*consistency*exp(-C/0.12)*(1-exp(-margin/0.05))
```

同一 n 使用 Hungarian 一对一指派，额外 dummy 接收 `unknown` 分支。排除最佳与次佳点级代价差
小于 `0.02` 的交叉模糊点。正式映射同时要求：共同点不少于 20、覆盖至少 5 个厚度、
`coverage>=0.60`、median `d_f<=0.12`、P95 `d_f<=0.25`、`C<=0.30`、
`margin>=0.05`、厚度一致性 `>=0.80`、`Q>=0.35`。

只有 24 条 `F(1..8,1..3)` 均得到唯一有效映射时，才发布 `standard_mode_mapping.npz`；内容包括
`branch_id/n/family/standard_m/cost/margin/coverage/consistency/confidence/valid` 及两端 source SHA。
失败时只写诊断 JSON 并非零退出，不强制命名。若失败原因为候选缺失或换支，则用新根把
`mode-count` 提升到 16，并重跑 formal、Solid 校准、发布和映射。

### 2.4 进入 EDM 的数据

训练侧最终读取 calibrated Shell NPZ/metadata、DC 标准 NPZ/metadata 和 mapping NPZ/report。
DC 提供标准 `k(h,f,n,m)`；Shell 的 observability、导数有效性与映射置信度共同形成分支权重。
无需传输原始 TXT、MPH、progress、case CSV 或 smoke 输出。

## 3. 阶段一网络方案

采用“可微频散反查 + 置信度门控残差”的混合解码，而不是纯硬解码或仅辅助损失：

```text
U-Net decoder feature
        |-- learned thickness head ---------> x0_base
        `-- dispersion evidence head --------> k_pred(f, mode, z, theta)
                                                   |
fixed pipe dispersion LUT: k_z(h, f, mode) --------+
                                                   v
                                      soft thickness posterior P(h)
                                                   |
                         confidence/entropy gate --+--> x0_final
```

频散头在 `64x64` 空间尺度预测：

```text
k_pred: [B, F_selected, M=3, 64, 64]
```

其中每个真实频率最多选择三条质量最高的 F 分支。对离散剩余厚度候选
`h_q in [5.0,10.0] mm`、步长 `0.25 mm`，计算：

```text
cost(h_q) = sum_(f,m) w(f,m) * Huber(k_pred(f,m) - k_LUT(h_q,f,m))
P(h_q)    = softmax(-cost(h_q) / 0.1)
h_phys    = sum_q P(h_q) * h_q
d_phys    = clip((10 - h_phys) / 9, 0, 5/9)
x0_final  = x0_base + gate * (d_phys - x0_base)
```

`gate` 同时使用 LUT 有效覆盖、厚度后验熵、EDM 噪声等级和零初始化学习门。没有可靠频散
证据时必须严格退化为 `x0_base`，从而允许学习分支吸收阻尼、散射、模态混合和 Shell 偏差。

训练增加标签派生的波数辅助损失 `lambda_dispersion_k=0.05`。缺陷区与健康背景分别求均值后
等权组合，避免大量健康像素掩盖缺陷区梯度。该损失只监督可观测且导数有效的 F 分支。

## 4. 频散 LUT 契约

从 calibrated `axisymmetric_shell_dispersion_library.npz`、DC 标准库和标准映射 sidecar 联合构建
训练用紧凑 LUT，要求：

- 按样本真实 `frequency_hz` 排序，并以完全相同顺序重排 `x_matrix` 频率轴。
- 只插值，不允许在厚度、频率或波数轴上外推。
- 仅保留有限值、`tracking_residual <= 1.1`、`observability >= 0.05`、
  `derivative_valid_mask=true` 且 `dk/dh` 有限的点。
- 分支权重由 tracking confidence、observability 和归一化 `|dk/dh|` 联合确定。
- 保存 `frequency_hz`、`thickness_mm`、`k_z`、valid mask、权重、branch ID、周向阶次、
  标准 m、映射置信度、模态族、观测分量、schema 版本和三个源 NPZ SHA-256。
- 正式配置必须要求 24/24 标准 F 映射通过；`standard_m=-1` 或低置信分支不能进入解码 LUT。
- 当前 schema-v3 smoke 库只标记 `scientific_ready=false`，用于张量和集成 smoke；正式训练必须使用
  通过网格、有限长管和 Shell-to-solid 校准门禁的 LUT。

正式 `F_selected` 尚未固定，由新 pilot 的三段配额和逐厚度频散覆盖共同决定。构建器必须保留
逐频率/逐厚度 mask，不能用最近值静默填充，也不能回退到旧 top15。

## 5. 配置和兼容性

新增可选配置段：

```yaml
model:
  dispersion_decoder:
    enabled: true
    lut_path: <axisymmetric dispersion LUT>
    standard_reference_path: <DC standard dispersion library>
    mode_mapping_path: <b_j to standard m mapping>
    mode_families: [F]
    modes_per_frequency: 3
    decoder_resolution: 64
    thickness_min_mm: 5.0
    thickness_max_mm: 10.0
    thickness_step_mm: 0.25
    temperature: 0.1
    min_observability: 0.05
    residual_threshold: 1.1
    require_scientific_ready: false  # 仅 smoke 配置允许 false
loss:
  lambda_dispersion_k: 0.05
```

- 旧 YAML 不创建频散解码器，原模型参数名、输出和 strict checkpoint 加载保持不变。
- 新解码器必须使用新的 run 从头训练；旧 checkpoint 不得被描述为已经具备频散解码能力。
- LUT 张量注册为持久 buffer，随 raw model 和 EMA checkpoint 保存。
- checkpoint metadata 额外记录 LUT schema、SHA-256、模态族和观测分量。
- train/sample/evaluate 必须继续共用同一个 `build_model()`，避免推理时构造不同网络。
- 数据集、`pic [B,8,256,256]`、`x_matrix [B,7,F_selected,16,16]` 和标签 `/9 mm` 契约不变。

## 6. 增量实施与 Git 提交

遵守每次不超过 5 个文件、增删不超过 200 行，每个功能点验证后立即独立提交：

1. 实现 DC TXT 导入、schema 与真实七厚度质量门禁，产物继续由 `.gitignore` 排除。
2. 实现全局 `b_j -> m` 指派、诊断报告和 source SHA 门禁；先用临时合成乱序曲线验证。
3. 服务器完成新 formal、有限管、Solid 校准、发布和 24/24 标准映射。
4. 实现 LUT 构建器及 schema/质量门禁测试。
5. 实现独立 PyTorch 可微 LUT、软厚度反查和无证据恒等回退测试。
6. 在 U-Net 上采样端接入频散证据头和门控混合解码，不改变旧配置行为。
7. 接入 EDM 训练损失、DDP 指标归约、checkpoint 元数据和严格加载检查。
8. 增加两样本 smoke 并更新 `diffusion_EDM`、`f_domain/new` 文档及相关 skill。

不得提交 LUT/NPZ、训练 run、COMSOL 结果或工作区中已有的无关删除项。阶段一不修改 COMSOL
激励/接收和正式数据集；阶段二必须单独规划、验证和提交。

## 7. 验证与完成标准

- LUT 构建器拒绝错误 schema、重复/不匹配频率、外推、无可靠分支和误标可观测的 T 分支。
- DC 导入必须复现当前七厚度审计，映射必须覆盖全部 24 条标准 F；任何 unknown 都阻止正式 LUT。
- 合成单调频散表的软反查厚度误差不超过一个厚度档位，即 `0.25 mm`。
- 所有模态无效时，频散解码输出与原 `x0_base` 数值一致。
- CPU FP32 及可用 GPU FP16 的前向、反向和梯度均有限，输出仍为 `[B,1,256,256]`。
- 旧配置严格加载旧 state dict；新配置完成真实两样本训练、保存/重载和采样 smoke。
- 通过 `compileall`、聚焦测试、`git diff --check` 及每次提交的文件数/行数审计。
- 正式 calibrated LUT 上完成 held-out 消融：原 EDM、频散辅助损失、频散混合解码三组；
  比较毫米 MAE/RMSE、SSIM、Dice、体积误差、物理越界率和后验校准指标。
- smoke checkpoint 和 smoke 频散库只证明执行链，不作为模型精度或 T/F 模态识别结论。

## 8. 阶段二边界

阶段二才实施 T+F 联合数据：增加周向激励和周向位移接收，至少提供两个轴向传播距离用于波数
辨识，并重新验证换能器方向、healthy baseline、网格、吸收层和接收导出。完成新数据的模态可观测性
与厚度敏感性审计后，才允许把 `mode_families` 扩展为 `[F,T]`。
