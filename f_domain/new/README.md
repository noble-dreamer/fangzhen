# 新管道频散仿真任务

本目录隔离新的频散与 v2 数据任务。旧 `f_domain` 仿真代码和历史输出不得在这里被覆盖。
轴对称 Shell 频散、有限管网格门禁、少量实体校准、DC 标准库导入、`b_j -> m` 映射、
image-texture-v4 选频和正式数据审计代码均已实现；服务器仍需按第 2 节生成正式结果，不能用本机 smoke 代替。
新旧工作量、物理意义、依赖关系和 EDM 输入差异见 [compare.md](compare.md)。

## 1. 计划目录

```text
new/
|-- README.md
|-- compare.md                         # 新旧仿真、依赖与 EDM 输入对比
|-- shell_dispersion_common.py          # 周期 Shell 单胞公共构建逻辑
|-- axisymmetric_dispersion_common.py    # 指定周向阶次的 2D 轴对称 Shell 单胞
|-- solid_dispersion_common.py          # 已实现：少量 Shell-to-solid 校准用轴对称实体单胞
|-- shell_mode_features.py               # 周向阶次、极化和可观测性特征
|-- dispersion_tracking.py               # Hungarian 追踪、c_g、df/dh 和 dk/dh
|-- solve_shell_dispersion.py            # 已实现：h-k-n 流式扫描和 schema v3
|-- irregular_defect_common.py           # 不规则边界、纹理深度、标签和 COMSOL 适配
|-- generate_irregular_defect_examples.py # 无 COMSOL 的缺陷预览与 NPZ/JSON
|-- build_irregular_dataset_a_frequency.py # 可直接打开的 build-only MPH
|-- solve_irregular_dataset_a_frequency.py # 复用现有 H(tx,rx,f) 流式求解器
|-- audit_irregular_dataset.py          # 已实现：正式 worker 审计和最小 EDM 传输清单
|-- validate_shell_dispersion.py         # 已实现：有限管复响应网格收敛，不代表实管认证
|-- solve_solid_calibration.py           # 已实现：最多 36 点轴对称实体与 Shell 分支校准
|-- calibrate_dispersion_library.py      # 已实现：三项门禁后发布 Shell-proxy 校准库
|-- build_standard_mode_mapping.py       # 已实现：DC 导入及 calibrated b_j -> F(n,m) 映射
|-- select_dispersion_frequencies.py     # 已实现：仅用 image-texture-v4 pilot 和 schema-v3 频散库选频
|-- models/                              # 服务器生成的 MPH，不作为训练输入
`-- outputs/                             # NPZ、报告和 v2 数据集
```

### 不规则单缺陷入口

当前生成器按 MATLAB `Defect _models/Rand_thickness_defect_model_1m_1m.m` 的顺序读取
`Defect _models/1.jpg`：OpenCV 随 seed 随机旋转并保持原图尺寸、转灰度、缩放到局部网格，
再放入 20--50 顶点的随机多边形硬掩膜。掩膜外先设为健康厚度对应灰度，灰度反向映射为壁损，
最后只做与旧版一致的 `sigma=2` 高斯边界平滑；不再叠加程序化正弦纹理或 bowl 衰减。v4 在管道
展开坐标 `(arc=Rm*atan2(y,x), z)` 上按连续 seed 以 `1:1:1` 确定性轮换小型
`40--70 mm`、中型 `70--110 mm` 和大型 `110--170 mm` 等效半径；中型覆盖原 v2 的主要范围，
大小两端补足管道尺度下的数据覆盖。实际峰值仍为 `2.5--4.6 mm`，并截断在 Shell 的 `5 mm`
最大壁损范围内。预览仍固定使用 `0--5 mm` 色标，标签仍除以 `9 mm`，不通过放大显示对比度
制造更明显的缺陷。同一个 `depth_mm` 栅格同时生成：

- COMSOL 二维插值表和 Shell 厚度 `max(h_min,h0-min(defect_loss_max,int1(...)))`；
- EDM 的 `512x512` 连续深度、归一化深度和 mask 标签；
- JSON/manifest 的 seed、`size_class`、中心、尺度、峰值和 `active_area_mm2`，保证服务器可复现并按尺寸分层。

它仍是 Shell 外表面减薄模型，不表示裂纹、孔洞、倒扣或真实三维粗糙面。COMSOL 6.4 将
spreadsheet 插值值注册为内部函数 `int1`；在 Model Builder 中应按节点名
`irregular wall-loss depth` 查找，不要把 `int1` 误认为未命名物理量。
生成器标识为 `irregular_polygon_image_texture_v4_multiscale`。metadata 同时保存图片相对路径、
SHA-256 和旋转角；标签和 COMSOL 插值表使用相同的周向
周期采样，因此大型缺陷跨越 `0/360 deg` 接缝时不会被截断。不同生成器版本不得放入同一
数据根目录，也不得仅凭相同 sample ID 互相覆盖。服务器必须同步 `Defect _models/1.jpg`，并先用
`python -c "import cv2; print(cv2.__version__)"` 确认 COMSOL 环境可读取 OpenCV。

`512x512` 是监督标签网格，不等于可实现的物理分辨率。正式 image-v4 建议显式使用
`--grid-count 257`；更细的图片纹理只有在 COMSOL 网格、最高频率最短波长和 TX/RX 覆盖都能解析时
才会进入 `H(tx,rx,f)`。低于网格或波长尺度的细节只能作为标签细节，不能宣称已被超声观测到。

本机先生成示例和一个不求解的可视模型：

```powershell
# 生成 image-texture-v4 预览
conda run --no-capture-output -n comsol python -u `
  f_domain/new/generate_irregular_defect_examples.py --count 9 --seed0 20260820 `
  --output-dir f_domain/new/output_dataset/dataset_a_frequency_shell/examples_image_texture_v4

conda run --no-capture-output -n comsol python -u `
  f_domain/new/build_irregular_dataset_a_frequency.py `
  --sample-id 1 --seed 20260821 --grid-count 257 --cores 2
```

产物结构与旧频域任务对应，但全部隔离在 `new/output_dataset`：

```text
f_domain/new/output_dataset/
|-- dataset_a_frequency_shell/
|   |-- pipe_shell_frequency_irregular_sample_0001.mph
|   |-- defect_tables/
|   |-- examples/                       # 历史程序化 v3 示例
|   |-- examples_image_texture_v4/      # 当前图片纹理汇总 PNG、单图、NPZ 和 JSON
|   `-- metadata/
`-- streaming_dataset_a_frequency_shell/
    |-- defect_tables/
    |-- frequency_response/             # H_real/H_imag 和 completed_mask
    |-- labels/                          # depth_mm/depth_norm/mask/PNG/JSON
    |-- metadata/
    |-- progress/
    |-- csv/frequency_response/
    `-- manifest.csv
```

本机健康/缺陷单工况 smoke 命令如下。large 样本 0003 仅用于 smoke，不进入训练：

```powershell
conda run --no-capture-output -n comsol python -u `
  f_domain/new/solve_irregular_dataset_a_frequency.py `
  --samples 1 --start-id 3 --grid-count 65 `
  --tx 1 --frequencies 50000 `
  --include-healthy `
  --output-root f_domain/new/output_dataset/streaming_dataset_a_frequency_shell_v4_smoke `
  --skip-label-preview --heartbeat-s 10 --cores 2
```

image-v4 本机实测：健康/large 缺陷 50 kHz 单发单频求解约 53/60 秒，完整进程约 183 秒；两份输出
均为 `H.shape=(1,16,1)`，mask 完整且 16/16 接收通道有限非零。large 缺陷相对 healthy 的复响应
L2 差异为 `6.796%`，16 个通道全部变化，build/post-solve problems 均为空。该 smoke 使用
`grid-count=65`，只证明新纹理的建模、求解、导出、标签和 metadata 链路，不代表正式
`grid-count=257`、12 频点、16 发射的精度或服务器耗时。

## 2. 服务器执行顺序

服务器工作目录固定为仓库根目录 `simple/`，连接 COMSOL 的 conda 环境为 `comsol_lzx`；
因此脚本和默认输出均使用相对 `simple/` 的路径。本机 Git 工作目录同样为 `simple/`，但连接
COMSOL 的环境名为 `comsol`，只运行短时单工况和 smoke：

```powershell
conda run --no-capture-output -n comsol python -u `
  f_domain/new/solve_shell_dispersion.py `
  --thickness-mm 7.5 10 --k-count 2 --k-max-rad-m 25 `
  --circumferential-orders 10 --mode-count 4 `
  --output-root f_domain/new/outputs/axisymmetric_modal_quick_smoke --smoke
```

DC TXT 已位于本机 `D:\lab_ultr\fz\dc_exports`。先在本机导入标准参考；该步骤不调用 COMSOL，
但使用 `comsol` 环境中的 NumPy。生成的 NPZ/JSON 随后复制到服务器的同一相对目录，原始 TXT 不传输：

```powershell
conda run --no-capture-output -n comsol_lzx python -u `
  f_domain/new/build_standard_mode_mapping.py import-dc `
  --input-root D:\lab_ultr\fz\dc_exports `
  --output-root f_domain/new/outputs/standard_mode_mapping
```

当前实测导入 70 个 TXT，得到 `7 thickness x 27 modes x 1236 points`，24 条 F、2 条 L 和 1 条 T
全部通过；`k=2*pi*f/cp` 最大相对误差为 `8.93e-15`。以下入口以服务器的 `simple/` 目录为工作目录。正式结果尚未在本机
生成；每一步通过并检查输出后再运行下一步。第 1 步 smoke 通过并记录耗时/schema 摘要后，删除
它自己创建的 `axisymmetric_modal_science_smoke` 目录；失败或正式/断点续跑目录不得清理。

```bash
# 0. 环境检查
conda run --no-capture-output -n comsol_lzx python -c \
  "import mph; print(mph.discovery.backend())"

# 1. 当前科学 smoke
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_shell_dispersion.py \
  --thickness-mm 5 7.5 10 --k-count 5 --k-max-rad-m 100 \
  --circumferential-orders 8 10 12 --mode-count 4 \
  --output-root f_domain/new/outputs/axisymmetric_modal_science_smoke --smoke

# 3. 标准模态专用 formal 原始库；n=0 仅作 L/T 锚点，n=1..8 用于 F 映射
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_shell_dispersion.py --formal \
  --thickness-mm 5 6 7 7.5 8 9 10 --k-count 41 \
  --circumferential-orders 0 1 2 3 4 5 6 7 8 \
  --mode-count 8 --eigen-shift-khz 60 --frequency-range-khz 15 110 --cores 16 \
  --output-root f_domain/new/outputs/axisymmetric_standard_modes_formal

# 4. 有限长管 16 通道复响应网格门禁；通过后删除中间 case 输出，仅保留报告
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/validate_shell_dispersion.py \
  --thickness-mm 5 7.5 10 --frequencies-khz 20 52.5 95 \
  --elements-per-wavelength 6 8 12 --tx 1 \
  --max-complex-relative 0.05 --max-phase-rmse-rad 0.05 \
  --output-root f_domain/new/outputs/finite_pipe_mesh_validation \
  --delete-case-outputs --heartbeat-s 120 --cores 16

# 5. 最多 36 点实体校准；必须读取步骤 3 的原始 h/k/n/branch 轴
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_solid_calibration.py \
  --shell-library f_domain/new/outputs/axisymmetric_standard_modes_formal/dispersion/axisymmetric_shell_dispersion_library.npz \
  --thickness-mm 5 7.5 10 --k-indices 0 20 40 \
  --circumferential-orders 0 1 4 8 --mode-count 4 --max-solves 36 \
  --mesh-hmax-mm 1.25 --eigen-shift-khz 60 \
  --output-root f_domain/new/outputs/standard_modes_solid_calibration --cores 16

# 6. 合并 formal、严格 mesh 报告和实体修正，发布仅供 Shell-proxy 仿真的校准库
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/calibrate_dispersion_library.py \
  --shell-library f_domain/new/outputs/axisymmetric_standard_modes_formal/dispersion/axisymmetric_shell_dispersion_library.npz \
  --mesh-report f_domain/new/outputs/finite_pipe_mesh_validation/finite_pipe_mesh_validation.json \
  --solid-calibration f_domain/new/outputs/standard_modes_solid_calibration/calibration/solid_shell_calibration.npz \
  --output-root f_domain/new/outputs/axisymmetric_standard_modes_calibrated \
  --readiness-scope shell_proxy_simulation_only

# 7. 只从 calibrated Shell 与 DC 标准库建立全局 b_j -> m 映射；要求 24/24
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/build_standard_mode_mapping.py map-shell \
  --shell-library f_domain/new/outputs/axisymmetric_standard_modes_calibrated/dispersion/axisymmetric_shell_dispersion_library.npz \
  --dc-library f_domain/new/outputs/standard_mode_mapping/dc_standard_pipe_dispersion.npz \
  --output-root f_domain/new/outputs/standard_mode_mapping

# 8. 新缺陷 pilot：20--100 kHz/2.5 kHz、4 个等间隔 TX、每档尺寸 4 个样本
PILOT_FREQS=$(seq -s, 20000 2500 100000)
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_irregular_dataset_a_frequency.py \
  --include-healthy --samples 12 --start-id 1 --seed0 20260820 \
  --grid-count 257 \
  --tx 1,5,9,13 --frequencies "$PILOT_FREQS" \
  --output-root f_domain/new/outputs/irregular_frequency_pilot \
  --checkpoint-every-cases 33 --skip-label-preview --heartbeat-s 120

# 9. 用新缺陷响应与通过门禁的 schema-v3 频散库固定 12 个频点
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/select_dispersion_frequencies.py \
  --pilot-root f_domain/new/outputs/irregular_frequency_pilot \
  --dispersion-library f_domain/new/outputs/axisymmetric_standard_modes_calibrated/dispersion/axisymmetric_shell_dispersion_library.npz \
  --output-root f_domain/new/outputs/irregular_frequency_selection \
  --count 12 --band-quotas 3 4 5

# 10. 不规则缺陷 worker；正式运行必须显式读取新选出的频率
SELECTED_FREQS=$(tr -d '\r\n[:space:]' < f_domain/new/outputs/irregular_frequency_selection/selected_frequencies.txt)
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_irregular_dataset_a_frequency.py \
  --output-root f_domain/new/output_dataset/irregular_worker${WORKER_ID} \
  --start-id ${START_ID} --samples ${COUNT} \
  --grid-count 257 \
  --tx 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --frequencies "$SELECTED_FREQS" \
  --checkpoint-every-cases 12 --skip-label-preview --heartbeat-s 120
```

三个昂贵入口都按“求解 -> 提取 -> checkpoint -> `clearSolutionData()`”流式执行：步骤 3 的 Shell
模型复用于全部 `(h,k,n)`；步骤 5 每个厚度建一个 Solid 模型并复用其 `(k,n)`；步骤 8/10 每个物理
样本建一个有限管模型并复用全部 `(TX,f)`，一次提取 16 个 RX。改变缺陷或均匀厚度物理时必须重建
模型，不能只改参数冒充新样本。若步骤 7 因候选缺失/换支失败，步骤 3 使用全新根并改为
`--mode-count 16`，再以对应新根重跑步骤 5、6、7（Solid 仍为 4 模态/36 点）；步骤 4 的物理与网格
未变时可复用。任一源 NPZ SHA 改变都必须重建映射。

发布器会核对 formal Shell SHA、严格 `5%/0.05 rad` 网格报告、Solid 完整率和校准覆盖率，重算
校正后的 `frequency/c_g/df_dh/dk_dh`，并写入 `scientific_ready=true` 与
`readiness_scope=shell_proxy_simulation_only`。该标记只允许当前 Shell-proxy 仿真选频和 EDM 数据链
使用，不表示真实管道认证；选择器会拒绝缺少此 scope 的正式库。

选择器会核对所有 NPZ 的有序 frequency/TX/RX 轴和完整 mask，从 metadata 强制要求生成器为
`irregular_polygon_image_texture_v4_multiscale`，并要求 small/medium/large 三档均有 pilot。评分综合三档
平衡复响应、样本稳定性、相位变化、路径参与度、`|dk/dh|`、轴向可观测性和分支追踪置信度；
频散覆盖不足、healthy 幅值过低或来自旧超高斯生成器的候选不会进入正式频率集。

先单独生成一次同网格、同有序频率轴的 healthy，再启动不重叠 ID 的 defect workers：

```bash
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/solve_irregular_dataset_a_frequency.py \
  --only-healthy \
  --output-root f_domain/new/output_dataset/irregular_healthy_selected12 \
  --tx 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --frequencies "$SELECTED_FREQS" \
  --checkpoint-every-cases 12 --skip-label-preview --heartbeat-s 120
```

worker 目录不能共享 `manifest.csv`。合并时以每个 NPZ 的完整 `completed_mask`、有序 tx/frequency
轴、16 个有限非零接收通道、标签和 metadata 一一对应为准。正式 10/12 频点确定后，通过
`--frequencies` 显式传入固定列表；不得把不同频率数无 mask 混在同一训练 batch。

服务器先记录 smoke 的单次求解时间、峰值内存和网格倍率，再决定 worker 数。多 worker 必须使用
独立输出目录和不重叠 ID；不得依赖并发写入同一个 `manifest.csv`。

首批 worker 完成后必须先审计，再打包取回。`--roots` 可列出任意数量的独立 worker，示例要求
200 个样本；`--shared-files` 把频散、校准和选频来源一起纳入可追溯传输包：

```bash
conda run --no-capture-output -n comsol_lzx python -u \
  f_domain/new/audit_irregular_dataset.py \
  --roots f_domain/new/output_dataset/irregular_worker0 \
          f_domain/new/output_dataset/irregular_worker1 \
  --healthy-root f_domain/new/output_dataset/irregular_healthy_selected12 \
  --frequencies-file f_domain/new/outputs/irregular_frequency_selection/selected_frequencies.txt \
  --expected-samples 200 \
  --shared-files \
    f_domain/new/outputs/irregular_frequency_selection/frequency_ranking.csv \
    f_domain/new/outputs/irregular_frequency_selection/frequency_selection_summary.json \
    f_domain/new/outputs/axisymmetric_standard_modes_calibrated/dispersion/axisymmetric_shell_dispersion_library.npz \
    f_domain/new/outputs/axisymmetric_standard_modes_calibrated/metadata/axisymmetric_shell_dispersion.json \
    f_domain/new/outputs/standard_modes_solid_calibration/calibration/solid_shell_calibration.npz \
    f_domain/new/outputs/standard_modes_solid_calibration/metadata/solid_shell_calibration.json \
    f_domain/new/outputs/finite_pipe_mesh_validation/finite_pipe_mesh_validation.json \
    f_domain/new/outputs/standard_mode_mapping/dc_standard_pipe_dispersion.npz \
    f_domain/new/outputs/standard_mode_mapping/dc_standard_pipe_dispersion.json \
    f_domain/new/outputs/standard_mode_mapping/standard_mode_mapping.npz \
    f_domain/new/outputs/standard_mode_mapping/standard_mode_mapping_report.json \
  --output-root f_domain/new/outputs/irregular_dataset_audit

tar -czf f_domain/new/outputs/irregular_edm_transfer.tar.gz \
  -T f_domain/new/outputs/irregular_dataset_audit/transfer_manifest.txt
```

审计强制要求 `H.shape=(16,16,F)`、完整 mask、每工况 16 个有限非零接收通道、与 healthy 完全
相同的模型指纹和有序频率轴；同时检查 v4 generator、唯一图片 SHA、逐样本旋转角、三档尺寸、`512x512` 标签、`depth_norm=depth_mm/9`
和 `0--5 mm` 范围。EDM 取回包仅包含：

- healthy 和每个 damaged 的 `frequency_response/*_H_complex.npz`、`metadata/*.json`；
- 每个 damaged 的 `depth_mm.npy`、`depth_norm.npy`、`mask.npy`、label metadata JSON；
- 选频 ranking/summary/频率文件、通过门禁的频散库和校准元数据；
- 审计 CSV、summary 与 transfer manifest。

逐工况 CSV、progress、defect table、preview、MPH 和 smoke 输出不参与 EDM；不要上传或取回。

### 当前频散 smoke 验证

本机 `3 thickness x 5 k x 3 n = 45` 点实测约 100 秒，45/45 checkpoint 完整；15--110 kHz
内 105/105 个模态点具有有效 `c_g` 与 `dk/dh`。极化和最大误差为 `2.22e-16`，最大虚频率
为 `1.26e-9 Hz`，可观测性范围为 `[2.17e-31, 0.992]`。相同命令恢复耗时约 `0.016 s`，
只返回 `resumed=true`。这些结果验证执行链和 schema，不代表正式网格收敛或反演精度。
恢复需要完整 NPZ 和 metadata，且只能复用完全相同的物理参数；改变 `eigen-shift/cell-length/frequency-range`
等参数时必须使用新输出目录。模型侧使用派生量时需同时检查对应有效掩码与 `isfinite`。

`--formal` 仅解锁轴对称 prescribed-order 密网格，并在 NPZ/metadata 中固定写入
`scientific_ready=false`。本机 formal 执行链检查 `3 thickness x 5 k x 1 n = 15` 点约 70 秒，
15/15 完整、恢复约 `0.016 s`，full-ring formal 会被拒绝；检查目录在记录结果后已删除。

有限管网格脚本比较中等级与细等级的 16 通道复 L2 差异和相位 RMSE，并拒绝 COMSOL problems、
零接收通道和非有限响应。本机接口检查 `h=7.5 mm, 50 kHz, EPW=4/5/6` 约 92 秒；EPW 5→6
仍有 `8.60%` 复差异和 `0.0603 rad` 相位 RMSE，证明粗网格尚不能通过正式 `5%/0.05 rad`
门禁。该 smoke 使用放宽阈值，输出已删除，不代表网格收敛。

实体校准采用完整壁厚域的 2D 轴对称 SolidMechanics、指定周向阶次和轴向 Floquet 条件，按
频率与三分量极化匹配 Shell 分支。单点代码 smoke 的两条 `solid/shell` 频率比为 `0.9990/0.9940`，
有效率 `1.0`，checkpoint 恢复与 formal-library 门禁通过，输出已删除。正式产物保存 ratio、
match cost、branch ID 和 confidence，但仍写 `scientific_ready=false`。

校准发布器的纯 NumPy 合成 smoke 已验证成功发布、实际网格指标超限拒绝、放宽阈值拒绝和
Shell SHA 不匹配拒绝；生成目录在通过后已删除。服务器正式运行必须保留发布后的 NPZ/metadata，
不能保留或传输本机 smoke 产物。

## 3. 产物及其模型作用

| 仿真产物                                      | 提供的信息                                                                            | 对模型的优化                                               |
| --------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `axisymmetric_shell_dispersion_library.npz` | schema v3 的`k_z/c_p/c_g/df_dh/dk_dh`、指定周向阶次、分支、极化、可观测性及有效掩码 | 让频率 token 表示厚度敏感性，并屏蔽不可靠导数              |
| `dc_standard_pipe_dispersion.npz`           | DC 3.2 的标准`F/L/T` 名称及独立数值曲线                                             | 提供标准`m` 身份和独立 `k/c_g` 参考，不覆盖 Shell 数值 |
| `standard_mode_mapping.npz`                 | 两端 source SHA 绑定的 24 条`b_j -> F(n,m)`、代价、margin、覆盖和置信度             | 让 LUT 只消费身份唯一且通过门禁的 F 分支                   |
| `finite_pipe_mesh_validation.json`          | 有限长管接收复响应的网格收敛                                                          | 防止将数值离散误差当成缺陷敏感性                           |
| `solid_shell_calibration.npz`               | Shell-to-solid 修正、分支匹配残差和置信度                                             | 为 Shell-proxy 迁移提供门控，降低 Shell 模型偏差           |
| `selected_frequencies.txt`                  | 兼顾灵敏度、模态互补和可信度的 10/12 频点                                             | 将每样本 240 工况降到 160/192 工况                         |
| `dispersion_evidence.npz`                   | 路径厚度后验、熵、残差和空间反投影图                                                  | 通过 PicAdapter 给 EDM 显式、可视化的厚度证据              |

## 4. 网络融合数据流

```text
Shell periodic cell -> pipe dispersion library ----+
3D/finite validation -> correction + confidence ---+-> dispersion matcher
damaged H(tx,rx,f) -> x_matrix --------------------+       |-- physical tokens -> FiLM/cross-attention
                                                        `-- evidence maps -> PicAdapter
coarse maps -----------------------------------------------^                |
noisy thickness + sigma ----------------------------------------------------+-> EDM thickness posterior
```

该设计同时保留原始观测、粗图和学习残差。频散曲线负责解释“频率为何对某个厚度敏感”，
空间证据负责解释“哪些路径支持哪个位置”，校准置信度负责说明“这条物理先验是否可信”。
因此频散不是单一损失项，也不会强迫所有响应服从一条理想曲线。

### 4.1 样本数据与共享频散库

本轮每个缺陷样本使用选择器固定的同一套 `F=12` 有序频率轴。若三段配额或频散覆盖门禁无法
满足，应扩充新 image-v4 pilot/正式频散网格后重新选频，不能回退引用旧 top15。不同频率配置不能在
同一批次中无 mask 混用：

```text
H_damaged, H_healthy : complex[16, 16, F]
x = XMatrix(H_damaged, H_healthy) : float[7, F, 16, 16]
COMSOL cases/sample = 16 * F = 160 or 192
```

周期 Shell 库是主要物理先验；3D 和有限长管结果用于修正或评估它，得到全部样本共享的：

```text
D_shell(h, f, mode) = {k_z, c_g, dk/dh, order, polarization, observability}
D_corrected = Calibrate(D_shell, solid_3d)
D_star = AttachConfidence(D_corrected, finite_pipe)
D_star fields += {corrected_k_z, branch_residual, confidence}
```

`D_star` 一次生成，在训练和推理时按样本真实频率插值。它不是新的缺陷标签，也不要求每个缺陷
重复运行 3D；原始 COMSOL 位移场和 MPH 不装入训练 batch。

### 4.2 EDM 条件接口

当前条件形式可写为：

```text
y_hat = EDM_theta(y_sigma, sigma | E_x(x), A_pic(pic(x)))
```

加入频散后先构造两类条件：

```text
T_disp = DispersionEncoder(D_star, frequency_hz, TX, RX)
M_disp = BackProject(DispersionMatch(x, D_star, TX, RX))

M_disp channels = {path_thickness_posterior, entropy, residual, confidence}
```

最终模型接口为：

```text
y_hat = EDM_theta(
    y_sigma, sigma |
    x,
    concat(pic(x), M_disp),
    T_disp
)
```

在现有 EDM 内部，`T_disp` 与 XEncoder 的全局条件和 TX-RX token 融合后进入 FiLM 与
cross-attention；`M_disp` 与原粗图进入 PicAdapter。低置信度通过门控衰减，不删除原始 `x`，
使学习残差仍可吸收阻尼、散射、多模态混合及 Shell-to-3D 偏差。

### 4.3 各类仿真的进入边界

- 10/12 频点缺陷管响应直接生成每个样本的 `x`，是主要观测条件。
- Shell 频散、模态可观测权重直接形成 `T_disp`，并参与 `M_disp` 的路径厚度匹配。
- 3D 修正和有限管分支残差通过 `D_star` 与 `confidence` 间接进入；原始场不直接进入。
- 网格收敛只决定数据或频散分支能否通过质量门禁，不作为网络输入。
- `selected_frequencies.txt` 只定义频率轴，不作为额外特征。
- 推理阶段读取固定 `D_star` 后由实测 `x` 在线生成 token 和证据图，不再调用 COMSOL。

当前代码已有 XEncoder、FiLM/TX-RX token 融合和 PicAdapter 接口，但尚未实现
`DispersionEncoder`、`DispersionMatch` 与 `M_disp` 数据管线；以上是后续实现契约。

## 5. 数据源边界

本轮候选频率、正式响应、粗图和 EDM 训练集只使用
`irregular_polygon_image_texture_v4_multiscale`。旧程序化 v3/超高斯缺陷、旧 top15 排名、旧 damaged、
`output_dataset_new` 和 `get_pic/output_dataset_new` 不参与新频率评分，也不与本轮训练 batch 混合；
它们保持只读，仅可在未来另行定义的历史基线实验中使用。

新频率若不在旧频率轴上也不做复响应插值，而是按新 image-v4 几何直接重新求解。正式 healthy 必须和
damaged 使用完全相同的管道、材料、阻尼、最高频率网格、载荷、接收表达式、TX/RX 和有序频率轴。
审计脚本以这些字段的 SHA-256 模型指纹强制执行兼容性。

服务器取回后再由同一 healthy/damaged 复频响生成 `get_pic` 粗图和
`x_matrix [7,F,16,16]`；不得从旧 `x_matrix` 改写频率字段，也不得把 12/15 频点张量无 mask 混合。

## 6. 运行纪律

- `models/` 和 `outputs/` 使用新的任务名，不复用 `output_dataset_new`。
- 正式服务器运行前必须先完成 smoke、网格门禁和输出 schema 检查。
- smoke 通过并记录摘要后只删除本次命令创建且名称含 `smoke` 的目录；失败、formal、生产和 resume 根保留。
- 频率必须排序后与 `x_matrix` 同步 gather；不得平均掉频率、TX 或 RX 轴。
- 失败任务可以从最近一个 TX checkpoint 重跑，不覆盖已完成样本。
- 3D 只做校准；若超过 Shell 数据预算的 10%，立即停止扩展 3D 工况。
- smoke 输出不能用于判断反演精度或频散曲线质量。
