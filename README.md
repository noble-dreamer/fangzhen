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
- `defect_label_common.py`: 根据缺陷 metadata 重建展开 `theta-z` 厚度损失标签图，并提供相关系数、NRMSE、mask IoU 评价函数。
- `export_defect_labels.py`: 不调用 COMSOL，从已有 metadata 批量导出 `.npy` 标签、带坐标/色标的 `.png` 展开预览和 montage。
- `streaming_export_common.py`: 流式求解、接收点导出、特征提取、manifest 和 progress 日志公共模块。
- `build_dataset_a_healthy.py`: 构建 Dataset A 理想健康 MPH，不求解；用于模型树检查。
- `build_dataset_b_healthy.py`: 构建 Dataset B 带材料/位置/幅值扰动的健康 MPH，不求解；用于模型树检查。
- `generate_dataset_a_defects.py`: 构建 Dataset A 随机缺陷 MPH，不求解；用于模型树检查。
- `generate_dataset_b_defects.py`: 构建 Dataset B 随机缺陷 + 扰动 MPH，不求解；用于模型树检查。
- `solve_export_dataset_a_streaming.py`: Dataset A 通用流式求解导出脚本；可用 `--only-healthy --tx 1 --frequencies 50000` 做健康单工况求解、导出和波形检查。
- `solve_export_dataset_a_validation_streaming.py`: Dataset A 理想验证集流式脚本，单个规则圆形/椭圆外表面腐蚀缺陷。
- `solve_export_dataset_a_training_streaming.py`: Dataset A 训练集流式脚本，随机多缺陷/不规则外表面腐蚀缺陷。
- `solve_export_dataset_b_streaming.py`: Dataset B 流式求解导出脚本，保留自由端和真实实验扰动。
- `export_simple_waveforms.py`: 对已求解的 MPH 导出 16 路时域径向位移 CSV；不推荐大批量使用。
- `inspect_simple_model.py`: 检查接收点、参数扫描和网格设置。
- `check_simple_solution_response.py`: 对已求解 MPH 检查接收通道是否有非零响应。
- `debug_shell_model_tree.py`: 打印材料、壳厚、载荷、接收点等模型树关键属性。
- `debug_shell_load_fields.py`: 打印 Shell `FaceLoad` 节点字段，用于确认等效载荷写入位置。
- `debug_shell_feature_types.py`: 探测 COMSOL 6.4 Shell 支持的 feature 类型；已确认 Shell 不能直接创建 Solid 的 `Low-Reflecting Boundary`。
- `debug_shell_thickness_fields.py`: 打印 `ThicknessOffset` 字段，用于确认外表面腐蚀的 offset 表达式。
- `SIMPLE_SHELL_REVIEW_AND_SUGGESTIONS.md`: 审查结论和修改建议，已完成项用删除线标记。

## 流式脚本快速使用

优先使用流式脚本生成数据，不保存含瞬态场解的 `.mph`：

长时间求解建议使用 `conda run --no-capture-output -n comsol python -u ...`。`--no-capture-output` 避免 conda 捕获子进程输出，`python -u` 强制 Python 无缓冲输出。流式脚本当前使用普通 `print(..., flush=True)` 输出进度，不再依赖 `tqdm`。

- `solve_export_dataset_a_streaming.py`: Dataset A 通用脚本，可做健康单工况检查，也可生成旧版随机缺陷 A 数据。
- `solve_export_dataset_a_validation_streaming.py`: Dataset A 理想验证集，单个规则圆形/椭圆外表面腐蚀缺陷。
- `solve_export_dataset_a_training_streaming.py`: Dataset A 训练集，随机多缺陷/不规则外表面腐蚀缺陷。
- `solve_export_dataset_b_streaming.py`: Dataset B，带材料、位置、幅值扰动，端部保持自由边。

常用运行参数：

- `--tx`: 发射通道列表，默认 `1..16`，例如 `--tx 1` 或 `--tx 1,2,3`。
- `--frequencies`: 频率列表，默认 `40000,50000,60000`。
- `--include-healthy`: 先生成健康样本，再生成缺陷样本。
- `--only-healthy`: 只生成健康样本，适合单工况收敛/export 检查。
- `--samples`: 随机缺陷样本数，validation 脚本不用这个参数。
- `--output-root`: 输出目录；不传时默认写入 `simple/output/...`。
- `--healthy-waveform-root` 和 `--healthy-waveform-sample-id`: 使用已有健康波形计算健康-损伤差分特征。
- `--heartbeat-s`: COMSOL 阻塞求解期间的心跳间隔，默认 30 秒。
- `--dt-out-us`: 输出时间步和严格最大时间步，默认 `0.5`。
- `--linear-solver {cudss,pardiso,mumps}`: 线性求解器，默认 `pardiso`。
- `--cudss-precision {single,double}`: cuDSS 精度，默认 `double`；仅在 `--linear-solver cudss` 时生效。
- `--relative-tolerance`: 瞬态相对容差，默认 `1e-4`。
- `--rebuild-each-case`: 回退到每个 `tx + frequency` 工况重新建模/划网格；默认是每个样本复用一个模型/网格。
- `--keep-client-cache`: 样本结束后不调用 `client.clear()`；大批量通常不建议使用。
- `--cores`: 指定 COMSOL 核心数；不传时由 COMSOL 自行分配。

健康单工况检查，确认 COMSOL、求解、export 和 16 通道响应：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_streaming.py --only-healthy --tx 1 --frequencies 50000 --linear-solver pardiso --dt-out-us 0.5 --heartbeat-s 10
```

Dataset A 通用脚本，先算健康基准，再算 1 个随机缺陷样本：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_streaming.py --include-healthy --samples 1
```

Dataset A 理想验证集，单个规则缺陷：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_validation_streaming.py --include-healthy
```

Dataset A 训练集，随机多缺陷/不规则缺陷：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_training_streaming.py --include-healthy --samples 1
```

Dataset B，带实验扰动：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_b_streaming.py --include-healthy --samples 1
```

可选 GPU 加速测试配置。建议先用 `double` 精度验证通过，再尝试 `single`：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_training_streaming.py --include-healthy --samples 1 --dt-out-us 0.5 --linear-solver cudss --cudss-precision double
```

如果要输出到其他磁盘，不需要改源码：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_training_streaming.py --output-root <你的数据目录>\streaming_dataset_a_training_shell --samples 10
```

默认输出目录：

- `solve_export_dataset_a_validation_streaming.py`: `simple/output/streaming_dataset_a_validation_shell/`
- `solve_export_dataset_a_training_streaming.py`: `simple/output/streaming_dataset_a_training_shell/`
- `solve_export_dataset_a_streaming.py`: `simple/output/streaming_dataset_a_shell/`
- `solve_export_dataset_b_streaming.py`: `simple/output/streaming_dataset_b_shell/`

默认流式策略是每个样本只建立一次 COMSOL 模型、网格和 solver tree，然后逐工况修改 `tx` 与 `pzt_fc`，求解、导出 CSV/特征后调用 `clearSolutionData()` 清掉当前瞬态场解。这样能省掉重复建模和划网格时间，同时仍避免保存或累积 48 个工况的全场解。

## 目录结构

- `simple/`: 轻量 shell 仿真与导出脚本。
- `simple/output/dataset_a_shell/`: Dataset A 健康壳 MPH、metadata 和 build log。
- `simple/output/dataset_b_shell/`: Dataset B 健康壳 MPH、metadata 和 build log。
- `simple/output/generated_dataset_a_shell/`: Dataset A 随机缺陷 MPH、metadata 和 manifest。
- `simple/output/generated_dataset_b_shell/`: Dataset B 随机缺陷 MPH、metadata 和 manifest。
- `simple/output/streaming_dataset_a_validation_shell/`: 推荐的 A_validation 流式输出目录，运行后生成。
- `simple/output/streaming_dataset_a_training_shell/`: 推荐的 A_training 流式输出目录，运行后生成。
- `simple/output/streaming_dataset_b_shell/`: Dataset B 流式输出目录，运行后生成。

路径约定：默认输出路径都相对于 `simple/` 目录计算，即 `simple_shell_common.py` 中的 `ROOT / 'output'`。因此把整个项目移动到 Windows 服务器的其他目录后，默认输出仍会落在新位置的 `simple/output/` 下。需要把数据写到单独的数据盘时，再通过 `--output-root` 或 `--healthy-waveform-root` 传入显式路径。

流式输出目录内部约定：

- `csv/waveforms/`: 每个 `sample + tx + frequency` 的 16 路接收波形 CSV。
- `csv/tomography_features/`: TOF、包络峰值、FFT 相位/幅值、螺旋阶次窗口等特征 CSV。
- `metadata/`: 样本级 JSON，记录缺陷、材料、吸收层、扰动、输出文件和 COMSOL self-check。
- `labels/`: 展开的管外表面 `theta-z` 缺陷标签，包括深度 `.npy`、mask、归一化深度图、metadata 和预览 PNG。
- `progress/`: 每个样本一个 JSONL 进度日志，可用于后台查看当前算到第几个工况和 ETA。
- `manifest.csv`: 本次任务输出索引。

新构建的 MPH 还会在 `Results > Datasets` 里包含两个仅用于检查显示的点集：

- `transmitter PZT marker points`: 左侧 16 个等效发射载荷中心。
- `receiver PZT marker points`: 右侧 16 个接收点中心。

这两个点集只是后处理数据集，不参与几何、网格、物理场和求解。COMSOL 里可在 `3D Plot Group` 下添加 `More Plots > Point`，分别选择这两个数据集，用不同颜色/符号叠加到管表面位移图上检查位置。

Dataset A 的 simple 壳模型默认开启两端轴向渐变 Rayleigh 阻尼吸收层，用于抑制端部反射。COMSOL 6.4 的 Shell 物理场不能直接创建 Solid Mechanics 的 `Low-Reflecting Boundary` 节点，因此这里采用壳模型可运行的 absorbing-layer 代理方案。

缺陷默认按外表面腐蚀处理：壳厚度局部减薄，同时 damaged 模型在缺陷区设置 thickness offset，使内表面位置近似保持不变、外表面向内腐蚀。健康模型仍为中面无偏置。

`generate_defect/show_defect.png` 里的旧图是把管外表面展开成板后的腐蚀深度图。`simple` 当前缺陷与它在坐标意义上一致，都是外表面 `theta-z` 厚度损失；区别是旧 MATLAB 代码生成的是随机阶梯坑和二值边界平滑，`simple` 使用和 COMSOL 壳模型完全一致的超高斯厚度减薄场加可选凸瓣。因此训练标签应优先使用 `simple` 导出的标签，而不是仅追求旧图外观一致。

线性直接求解器默认设置为 `PARDISO`，这是当前 Dataset A 健康单工况已知更稳定的基线。该设置在 `simple_shell_common.py` 的 `SolverConfig.direct_linear_solver = 'pardiso'` 中统一控制，所有新建 simple 模型都会继承。NVIDIA `cuDSS` 仍可通过流式脚本参数启用，用于服务器 CUDA GPU 加速测试；若出现 `internal error cudss`、找不到一致初始值或最后时步不收敛，应先回退到 `--linear-solver pardiso --dt-out-us 0.5` 确认模型本身可解。

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

`build_dataset_a_healthy.py` 和流式 Dataset A 脚本都调用 `simple_shell_common.py` 的同一套建模函数，并使用同一类物理设置：壳几何、材料、两端 absorbing layer、外表面腐蚀约定、等效 PZT 面载荷、16 个接收点和默认求解器配置。区别在于：

- `build_dataset_a_healthy.py` 生成一个健康管 MPH，保留 16 x 3 参数扫描设置，但 `solve=False`，用于打开 COMSOL 检查模型树和参数配置。
- `solve_export_dataset_a*_streaming.py` 每次只建立一个 `tx + frequency` 工况，立即求解、导出 CSV/特征/metadata/labels，然后清理模型，不保存含解 MPH。

因此它们属于同一个 Dataset A simple shell 模型族，物理配置应保持一致；但不是同一个运行时 study 对象。正式数据生成以流式脚本输出为准，build 脚本用于模型检查。

## 推荐的大批量数据集流程：流式求解导出

如果用一个 MPH 保存全部瞬态参数化解，文件会非常大。服务器上的
`dataset_a_single_case_healthy.mph` 已经达到约 300 GB，说明“先保存大 MPH，
再重新读取导出”的方式不适合批量数据集。

新脚本采用下面的流程：

1. 在 COMSOL session 中为当前样本建立一个 reusable 模型，几何、网格和 solver tree 只初始化一次。
2. 建模时把本次要跑的全部 tx 写入等效载荷表达式，但不建立 COMSOL 参数化扫描。
3. 对每个工况只修改全局参数 `tx` 和 `pzt_fc`。
4. 求解当前工况。
5. 直接从当前解对象中导出 16 路接收点径向位移和特征 CSV。
6. 调用 COMSOL solution 的 `clearSolutionData()` 清除当前瞬态场解，保留几何、网格和 solver tree。
7. 进入下一个工况；样本结束后 `client.remove(model)` 并默认 `client.clear()`。

默认模式是“每个样本复用一个模型/网格”，可以省掉 48 个工况中重复建模、划网格和 solver tree 初始化的开销，同时仍然保持流式导出，不保存含解 MPH，也不会主动保留全部工况的瞬态场解。如果遇到 COMSOL 版本相关的重复求解问题，可加 `--rebuild-each-case` 回退到旧模式：每个 `tx + frequency` 重新建模、求解、导出并删除模型。

这种方式不会保存含瞬态场解的 `.mph` 文件，输出目录中也不会有
`models/*.mph`。保留的内容是：

- `csv/waveforms/*_waveforms.csv`: 16 路接收点时域位移。
- `csv/tomography_features/*_tomography_features.csv`: TOF、Hilbert 峰值、FFT 幅值/相位等。
- `csv/tomography_features/*_helical_order_projections.csv`: 不同螺旋阶次窗口峰值。
- `csv/tomography_features/*_receiver_summary.csv`: 接收通道摘要。
- `metadata/*.json`: 样本缺陷、材料、网格、求解参数、输出文件记录。
- `progress/*_progress.jsonl`: 每个样本的后台进度日志。
- `manifest.csv`: 本次生成的数据索引。
- `labels/<sample_or_model>_defect_depth_mm.npy`: 形状为 `(z_index, theta_index)` 的厚度损失标签，单位 mm；这是物理意义最直接的监督标签。
- `labels/<sample_or_model>_defect_depth_norm.npy`: 用 `h0 - h_min` 归一化到 `[0, 1]` 的深度标签，适合 diffusion 或神经网络训练。
- `labels/<sample_or_model>_defect_mask.npy`: 按 `mask_threshold_mm` 阈值生成的缺陷二值 mask，适合 IoU、面积和定位评价。
- `labels/<sample_or_model>_defect_label.png`: 管外表面展开预览图，仅用于人工检查；图上有样本名、横轴 `theta`、纵轴 `z` 和深度色标。
- `labels/<sample_or_model>_defect_label_metadata.json`: 标签说明文件，记录坐标范围、网格尺寸、归一化分母、mask 阈值、管道参数和各标签文件路径。

后台运行时，脚本会在控制台打印类似下面的进度：

```text
[progress] [------------------------]   0.0% dataset_a_validation_single_defect case_start 1/48 tx=1 f=50000.0Hz elapsed=0s eta=unknown - building and solving COMSOL case without saving MPH
[progress] [------------------------]   0.0% dataset_a_validation_single_defect comsol_solve_running 1/48 tx=1 f=50000.0Hz elapsed=5m 00s eta=unknown - COMSOL model.solve() is still running... case_elapsed=5m 00s
[progress] [#-----------------------]   2.1% dataset_a_validation_single_defect case_done 1/48 tx=1 f=50000.0Hz elapsed=18m 24s eta=14h 25m 12s - wrote ...
```

流式脚本使用普通 `print(..., flush=True)` 输出进度，避免 `tqdm` 在 Windows/conda/任务调度环境中被捕获或延迟。`comsol_solve_running` 是 Python 侧的心跳。COMSOL 的内部时间步百分比在 `mph` 的 `model.solve()` 阻塞调用期间不能稳定读取，所以终端进度显示的是样本级工况完成度、当前工况已用时和 ETA。默认每 30 秒输出一次，可以用 `--heartbeat-s 10` 调整。

如果通过任务调度或远程后台运行，可以直接查看 JSONL 进度文件：

```powershell
Get-Content simple\output\streaming_dataset_a_validation_shell\progress\dataset_a_validation_single_defect_progress.jsonl -Tail 5
```

其中 `case_index/case_count` 表示当前工况进度，`elapsed_s` 是已用秒数，`average_case_s` 是已完成工况平均耗时，`eta_s` 是估计剩余秒数。若某个工况失败，会记录 `case_failed` 和错误信息。

最小健康单工况收敛/输出检查：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_streaming.py --only-healthy --tx 1 --frequencies 50000 --linear-solver pardiso --dt-out-us 0.5 --heartbeat-s 10
```

该命令只跑健康管 `tx=1, f=50 kHz`，不保存 MPH。它会完成求解、从当前 COMSOL 解对象导出 16 路波形 CSV、提取特征 CSV、写 metadata，并在 metadata 里记录 `waveform_check`。完成后检查：

- `metadata.model.problems.case_problems[0].build_problems`
- `metadata.model.problems.case_problems[0].post_solve_problems`
- `metadata.model.problems.case_problems[0].waveform_check`

健康单工况应满足：`post_solve_problems` 没有严重报错，`finite=true`，`nonzero_channels=16/16`，并且导出的 CSV 时间点数应接近 `t_end / dt_out + 1`。

## 缺陷展开标签与评价

流式脚本在写出样本 metadata 时会自动生成 `labels/` 标签包。也可以在不启动 COMSOL 的情况下，对已有 metadata 单独补生成：

```powershell
python simple/export_defect_labels.py --metadata simple/output/generated_dataset_a_shell/metadata/dataset_a_shell_sample_0001.json
```

批量导出并生成类似 `generate_defect/show_defect.png` 的预览拼图：

```powershell
python simple/export_defect_labels.py --metadata simple/output/generated_dataset_a_shell/metadata --montage
```

标签坐标约定：

- 数组形状是 `(z_index, theta_index)`，作为图像看时横轴为周向角度 `theta`，纵轴为轴向 `z`。
- `theta` 坐标不包含 360 度端点，首尾按圆周周期相接；坐标范围、点数和步长写在 `defect_label_metadata.json`。
- `z` 坐标范围为 `0..L_pipe`，单位 mm；同样由 metadata 记录，不再额外生成单独的坐标轴 `.npy`。
- `depth_mm.npy` 是仿真里真正施加的厚度损失，即 `h0 - local_thickness`，并按 `h_min` 截断。
- 文件名前缀优先使用 metadata 中的 `model_path` 文件名，例如 `dataset_a_shell_sample_0001_defect_depth_mm.npy`；流式求解没有固定 MPH 时使用 `sample_id`。
- `output/label_check*` 这类目录只是本地验证临时产物，不是数据集必要内容；正式数据只需要每个数据集输出目录下的 `labels/`。

后续反演粗图只要重采样到同一 `(z, theta)` 网格，就可以用 `defect_label_common.compare_prediction()` 计算 Pearson 相关系数、归一化 RMSE 和 mask IoU。

如果已经有健康波形，可以在损伤样本导出时用于计算健康-损伤差分峰值：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_streaming.py --samples 10 --healthy-waveform-root simple\output\streaming_dataset_a_shell\csv\waveforms --healthy-waveform-sample-id dataset_a_shell_healthy
```

默认每个工况导出后会清除当前 solution data，但保留当前样本的几何、网格和 solver tree 供下一个工况复用。样本结束后会移除模型并清理 COMSOL client。`--keep-client-cache` 只是不在样本结束/回退模式每工况后调用 `client.clear()`，不建议大批量使用；它不会关闭默认的每工况 `clearSolutionData()`。

常用流式命令、加速参数和默认输出目录见前面的“流式脚本快速使用”。

默认不传 `--cores`，因此 `mph.start()` 不指定核心数，由 COMSOL 自行分配。
如果确实需要限制核心数，再手动加例如 `--cores 8`。

GPU/cuDSS 相关注意：

- simple 模型使用隐式瞬态 Shell 求解，GPU 加速来自直接线性求解器 `cuDSS`，不是 dG time-explicit 的 hardware acceleration。
- 当前稳定默认是 `linear_solver = pardiso`、`dt_out_us = 0.5`、`relative_tolerance = 1e-4`。
- cuDSS 建议测试顺序是先 `--linear-solver cudss --cudss-precision double --dt-out-us 0.5`，确认健康单工况通过后再测试 `--dt-out-us 1.0` 或 `--cudss-precision single`。
- 多 GPU 默认关闭；如果服务器明确有多张可用 GPU，可在 `SolverConfig.cudss_use_multiple_gpus` 中改为 `True`。
- 可用 `simple/debug_solver_tree.py <model.mph> --allowed` 检查新模型中 `Direct` 节点的 `linsolver` 是否为 `cudss`。
- 可用 `conda run -n comsol python simple/check_solver_cudss.py` 不保存 MPH 地检查内存模型是否已切到 cuDSS。
- 如果重新生成 `.mph` 时提示文件被锁定，通常是旧 MPH 正在被 COMSOL 或其他程序打开；关闭后再生成即可。

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
- 可视化标记点：`Results > Datasets > transmitter PZT marker points` 和 `receiver PZT marker points`

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

  对于你的 diffusion 数据集流程，建议保存和训练的主输入先用导出的 CSV 或由 CSV 提取的层析特征，监督标签使用 `labels/*_defect_depth_mm.npy` 或归一化后的 `labels/*_defect_depth_norm.npy`，不要直接依赖 MPH 里的全场数据。
