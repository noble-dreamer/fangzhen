# Simple shell simulation pipeline

这一套脚本用于批量数据集，和上一级目录里的 3D solid + PZT 实体模型解耦。

核心简化：

- 管道用 Shell 物理场，几何是圆柱中面。
- PZT 不再建实体，改为等效面载荷窗口。
- 缺陷不做几何切割，改为壳厚度局部减薄。
- 接收端为 16 个接收点的径向位移时域信号。
- 网格按最短波长控制，不按 PZT 厚度细化。

主要脚本：

- `simple_shell_common.py`: COMSOL Shell 模型构建公共模块，定义几何、材料、壳厚、吸收层、载荷、网格、study 和 metadata。
- `simple_defect_common.py`: 随机缺陷采样公共模块，生成多缺陷/凸瓣并转换成 shell 厚度减薄配置。
- `streaming_export_common.py`: 流式求解、接收点导出、特征提取、manifest 和 progress 日志公共模块。
- `build_dataset_a_healthy.py`: 构建 Dataset A 理想健康 MPH，不求解；用于模型树检查。
- `build_dataset_b_healthy.py`: 构建 Dataset B 带材料/位置/幅值扰动的健康 MPH，不求解；用于模型树检查。
- `generate_dataset_a_defects.py`: 构建 Dataset A 随机缺陷 MPH，不求解；用于模型树检查。
- `generate_dataset_b_defects.py`: 构建 Dataset B 随机缺陷 + 扰动 MPH，不求解；用于模型树检查。
- `solve_export_dataset_a_streaming.py`: 旧版 Dataset A 流式求解导出脚本，保留兼容；新任务优先用 validation/training 两个拆分脚本。
- `solve_export_dataset_a_validation_streaming.py`: Dataset A 理想验证集，单个规则圆形/椭圆外表面腐蚀缺陷。
- `solve_export_dataset_a_training_streaming.py`: Dataset A 训练集，随机多缺陷/不规则外表面腐蚀缺陷。
- `solve_export_dataset_b_streaming.py`: Dataset B 流式求解导出，保留自由端和真实实验扰动。
- `export_simple_waveforms.py`: 对已求解的 MPH 导出 16 路时域径向位移 CSV；不推荐大批量使用。
- `inspect_simple_model.py`: 检查接收点、参数扫描和网格设置。
- `check_simple_solution_response.py`: 对已求解 MPH 检查接收通道是否有非零响应。
- `debug_shell_model_tree.py`: 打印材料、壳厚、载荷、接收点等模型树关键属性。
- `debug_shell_load_fields.py`: 打印 Shell `FaceLoad` 节点字段，用于确认等效载荷写入位置。
- `debug_shell_feature_types.py`: 探测 COMSOL 6.4 Shell 支持的 feature 类型；已确认 Shell 不能直接创建 Solid 的 `Low-Reflecting Boundary`。
- `debug_shell_thickness_fields.py`: 打印 `ThicknessOffset` 字段，用于确认外表面腐蚀的 offset 表达式。
- `SIMPLE_SHELL_REVIEW_AND_SUGGESTIONS.md`: 审查结论和修改建议，已完成项用删除线标记。

## 目录结构

- `simple/`: 轻量 shell 仿真与导出脚本。
- `simple/output/dataset_a_shell/`: Dataset A 健康壳 MPH、metadata 和 build log。
- `simple/output/dataset_b_shell/`: Dataset B 健康壳 MPH、metadata 和 build log。
- `simple/output/generated_dataset_a_shell/`: Dataset A 随机缺陷 MPH、metadata 和 manifest。
- `simple/output/generated_dataset_b_shell/`: Dataset B 随机缺陷 MPH、metadata 和 manifest。
- `simple/output/streaming_dataset_a_validation_shell/`: 推荐的 A_validation 流式输出目录，运行后生成。
- `simple/output/streaming_dataset_a_training_shell/`: 推荐的 A_training 流式输出目录，运行后生成。
- `simple/output/streaming_dataset_b_shell/`: Dataset B 流式输出目录，运行后生成。

流式输出目录内部约定：

- `csv/waveforms/`: 每个 `sample + tx + frequency` 的 16 路接收波形 CSV。
- `csv/tomography_features/`: TOF、包络峰值、FFT 相位/幅值、螺旋阶次窗口等特征 CSV。
- `metadata/`: 样本级 JSON，记录缺陷、材料、吸收层、扰动、输出文件和 COMSOL self-check。
- `progress/`: 每个样本一个 JSONL 进度日志，可用于后台查看当前算到第几个工况和 ETA。
- `manifest.csv`: 本次任务输出索引。

Dataset A 的 simple 壳模型默认开启两端轴向渐变 Rayleigh 阻尼吸收层，用于抑制端部反射。COMSOL 6.4 的 Shell 物理场不能直接创建 Solid Mechanics 的 `Low-Reflecting Boundary` 节点，因此这里采用壳模型可运行的 absorbing-layer 代理方案。

缺陷默认按外表面腐蚀处理：壳厚度局部减薄，同时 damaged 模型在缺陷区设置 thickness offset，使内表面位置近似保持不变、外表面向内腐蚀。健康模型仍为中面无偏置。

Dataset B 中原来的 PZT 位置/胶层等误差，在这里改为：

- 发射端位置误差：`dz_mm`, `dtheta_deg`
- 接收端位置误差：`dz_mm`, `dtheta_deg`
- 发射/接收幅值比例误差：`AMPLITUDE_SCALE`
- 噪声、触发抖动、基线漂移：留到 CSV 后处理阶段

生成模型示例：

```powershell
conda run -n comsol python simple/build_dataset_a_healthy.py
conda run -n comsol python simple/generate_dataset_a_defects.py
```

计算完成后导出单工况示例：

```powershell
conda run -n comsol python simple/export_simple_waveforms.py --model simple/output/dataset_a_shell/pipe_shell_healthy.mph --output-root simple/output/dataset_a_shell --sample-id pipe_shell_healthy --tx 1 --frequencies 50000
```

## 推荐的大批量数据集流程：流式求解导出

如果用一个 MPH 保存全部瞬态参数化解，文件会非常大。服务器上的
`dataset_a_single_case_healthy.mph` 已经达到约 300 GB，说明“先保存大 MPH，
再重新读取导出”的方式不适合批量数据集。

新脚本采用下面的流程：

1. 在 COMSOL session 中建立一个工况模型。
2. 只设置一个 `tx` 和一个 `pzt_fc`，不建立 16 x 3 的参数化解。
3. 求解当前工况。
4. 直接从当前解对象中导出 16 路接收点径向位移和特征 CSV。
5. `client.remove(model)` 并默认 `client.clear()`，丢弃场解。
6. 进入下一个工况。

这种方式不会保存含瞬态场解的 `.mph` 文件，输出目录中也不会有
`models/*.mph`。保留的内容是：

- `csv/waveforms/*_waveforms.csv`: 16 路接收点时域位移。
- `csv/tomography_features/*_tomography_features.csv`: TOF、Hilbert 峰值、FFT 幅值/相位等。
- `csv/tomography_features/*_helical_order_projections.csv`: 不同螺旋阶次窗口峰值。
- `csv/tomography_features/*_receiver_summary.csv`: 接收通道摘要。
- `metadata/*.json`: 样本缺陷、材料、网格、求解参数、输出文件记录。
- `progress/*_progress.jsonl`: 每个样本的后台进度日志。
- `manifest.csv`: 本次生成的数据索引。

后台运行时，脚本会在控制台打印类似下面的进度：

```text
[progress] dataset_a_validation_single_defect case_start 1/48 tx=1 f=50000.0Hz elapsed=0s eta=unknown - building and solving COMSOL case without saving MPH
[progress] dataset_a_validation_single_defect case_done 1/48 tx=1 f=50000.0Hz elapsed=18m 24s eta=14h 25m 12s - wrote ...
```

如果通过任务调度或远程后台运行，可以直接查看 JSONL 进度文件：

```powershell
Get-Content simple\output\streaming_dataset_a_validation_shell\progress\dataset_a_validation_single_defect_progress.jsonl -Tail 5
```

其中 `case_index/case_count` 表示当前工况进度，`elapsed_s` 是已用秒数，`average_case_s` 是已完成工况平均耗时，`eta_s` 是估计剩余秒数。若某个工况失败，会记录 `case_failed` 和错误信息。

单工况健康验证：

```powershell
conda run -n comsol python simple/solve_export_dataset_a_streaming.py --only-healthy --tx 1 --frequencies 50000
```

Dataset A 先算健康基准，再算 1 个损伤样本，默认每个样本 16 个激励 x 3 个频率：

```powershell
conda run -n comsol python simple/solve_export_dataset_a_streaming.py --include-healthy --samples 1
```

Dataset A 理想验证集建议使用单规则缺陷：

```powershell
conda run -n comsol python simple/solve_export_dataset_a_validation_streaming.py --include-healthy
```

Dataset A 训练集建议使用随机多缺陷脚本：

```powershell
conda run -n comsol python simple/solve_export_dataset_a_training_streaming.py --include-healthy --samples 1
```

Dataset B 同样使用流式导出，但会加入材料、发射端、接收端和幅值扰动：

```powershell
conda run -n comsol python simple/solve_export_dataset_b_streaming.py --include-healthy --samples 1
```

如果已经有健康波形，可以在损伤样本导出时用于计算健康-损伤差分峰值：

```powershell
conda run -n comsol python simple/solve_export_dataset_a_streaming.py --samples 10 --healthy-waveform-root D:\lab_ultr\fz\simple\output\streaming_dataset_a_shell\csv\waveforms --healthy-waveform-sample-id dataset_a_shell_healthy
```

默认每个工况后会清理 COMSOL client 中的模型和结果缓存。`--keep-client-cache`
可以少做清理，速度可能略快，但内存和临时目录占用会更高，不建议大批量使用。

服务器输出路径在脚本顶部显式配置：

- `simple/solve_export_dataset_a_validation_streaming.py` 中的 `OUTPUT_ROOT`
- `simple/solve_export_dataset_a_training_streaming.py` 中的 `OUTPUT_ROOT`
- `simple/solve_export_dataset_b_streaming.py` 中的 `OUTPUT_ROOT`

默认不传 `--cores`，因此 `mph.start()` 不指定核心数，由 COMSOL 自行分配。
如果确实需要限制核心数，再手动加例如 `--cores 8`。

注意：流式脚本避免的是“把巨大瞬态解保存进 MPH”和“重新读取巨大 MPH”。
求解过程中 COMSOL 仍然会在内存和临时目录中放置当前工况的临时数据；因此服务器上
仍建议把 COMSOL 临时目录放到容量充足的本地 scratch，并在一批任务结束后检查
COMSOL recovery/temp 目录。

## COMSOL 模型树位置

如果在 COMSOL Model Builder 里查看：

- 壳厚度：`Component 1 > Shell Mechanics > shell thickness and defect wall loss`
- 线弹性材料：`Component 1 > Shell Mechanics > explicit aluminum shell elastic material`
- 等效激励：`Component 1 > Shell Mechanics > equivalent transducer face load`
- 当前激励编号：`Global Definitions > Parameters > tx`
- 当前激励频率：`Global Definitions > Parameters > pzt_fc`
- 激励脉冲函数：`Global Definitions > Functions > five-cycle Hanning sine`
- 接收点：`Results > Datasets > receiver PZT 17 point` 到 `receiver PZT 32 point`

注意：激励位置不是几何里的独立 PZT 面片。它们被写在 `equivalent transducer face load` 的 `F` 表达式里，通过以 `tx` 为开关的空间窗口函数激活对应发射点。这样做是为了避免 PZT 小尺寸强制局部加密网格。

健康模型的厚度显示为 `h0`。缺陷模型的厚度显示为 `max(h_min, h0 - (...))`，其中括号里是多个缺陷/凸瓣对应的厚度减薄窗口。

线弹性节点应显示：

- `E_mat = userdef`, `E = E_al`
- `nu_mat = userdef`, `nu = nu_al`
- `rho_mat = userdef`, `rho = rho_al`

如果仍显示 `rho_mat = from_mat`，说明打开的是旧模型，需要关闭该 MPH 后重新运行生成脚本。



 simple 模型计算完成后，COMSOL 里得到的是壳管道中面上的时域位移场。

  具体来说，每个参数工况都会有一组时间序列解：

  tx = 某个发射点
  pzt_fc = 某个频率
  t = 0 : dt_out : t_end

  当前默认是：

  tx = 1..16
  pzt_fc = 40000, 50000, 60000 Hz
  t = 0 到 0.8 ms
  dt_out = 0.5 us

  也就是一个模型里最多有：

  16 × 3 = 48 个时域工况

  每个工况里，COMSOL 存的是整个壳面上的位移自由度，比如：

- u：全局 X 方向位移
- v：全局 Y 方向位移
- w：全局 Z 方向位移
- Shell 相关转角/壳自由度

  这些是场数据，不是直接的 CSV 表格。你在 COMSOL 里看到的是“某个时刻整个管壁怎么振动”。

  simple 模型中数据代表什么：

- 管道：等效壳中面，不是 3D 实体厚壁管。
- 缺陷：局部厚度减薄导致的波速/散射变化。
- 激励：某个发射点位置的等效径向面载荷。
- 接收：16 个接收点处的径向位移。
- 信号单位：位移，单位是 m。

  导出脚本 simple/export_simple_waveforms.py 做的事情是：
  从 COMSOL 的场解中，在 16 个接收点取值，并把全局位移分量投影成径向位移：

  ur = cos(theta) * u + sin(theta) * v

  所以导出的 CSV 是16 路接收点径向位移随时间变化的波形。

  导出后一个 CSV 大概长这样：

  time_s,rx01_ur_m,rx02_ur_m,...,rx16_ur_m
  0,0.000000000000e+00,0.000000000000e+00,...,0.000000000000e+00
  5e-07,1.23e-12,8.91e-13,...,-2.10e-13
  1e-06,3.45e-12,1.02e-12,...,-4.33e-13
  ...
  8e-04,...

  文件命名例如：

  pipe_shell_healthy_tx01_f50000Hz_waveforms.csv

  含义是：

- tx01：第 1 个发射点激励
- f50000Hz：50 kHz 激励
- rx01..rx16：16 个接收通道，对应 PZT17..PZT32
- 每一行：一个时间点
- 每一列：一个接收通道的径向位移

  如果你导出完整默认扫描，会得到 48 个 CSV：

  16 个 tx × 3 个频率 = 48 个波形文件

  这些 CSV 才是后续层析反演最直接用的数据。后面可以再从这些波形里提取：

- TOF 到达时间
- Hilbert 包络峰值
- FFT 幅值/相位
- 健康-缺陷差分幅值
- 不同螺旋阶次的波包幅值

  简单总结：

  COMSOL 计算结果 = 整个壳管道的时域位移场
  export 导出结果 = 16 个接收点的径向位移时域波形 CSV

  对于你的 diffusion 数据集流程，建议保存和训练的主输入先用导出的 CSV，而不是直接用 MPH 里的全
  场数据。
