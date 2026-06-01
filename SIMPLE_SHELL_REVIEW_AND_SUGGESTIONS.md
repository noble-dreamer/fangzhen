# Simple shell 模型审查与修改建议

本文件基于 `simple/` 目录内 Python 脚本、`simple/README.md`、输出 build log，以及根目录 `dataset_create.md` 的审查编写。

## 总体结论

`simple/` 的总体方向是合理的：用壳中面代替 3D 厚壁实体，用等效面载荷代替 PZT 实体和压电耦合，用局部壳厚减薄代替几何切割缺陷，并采用“单 tx、单频率求解后立刻导出 CSV 并清理模型”的流式流程。这条路线适合先生成用于层析粗图和 diffusion 预训练的轻量仿真数据，能避开保存 300 GB 级含解 MPH 文件的问题。

2026-05-29 更新：当前 conda `comsol` 环境已确认调用 COMSOL 6.4，`mph.discovery.backend()` 指向 `D:/comsol6.4/COMSOL64/Multiphysics`。已用 COMSOL 6.4 重新构建 Dataset A 健康模型和一个 Dataset A 损伤模型，`model.problems()` 均为空。

2026-05-30 更新：simple 公共求解器配置曾统一改为 NVIDIA `cuDSS` 直接线性求解器，用于服务器 CUDA GPU 加速测试。已新增 `debug_solver_tree.py` 和 `check_solver_cudss.py` 用于检查 solver tree。

2026-05-30 更新：已新增展开管外表面缺陷标签导出。`defect_label_common.py` 会按 `simple_shell_common.py` 中同一个超高斯厚度损失公式重建 `theta-z` 深度图，`export_defect_labels.py` 可从 metadata 批量导出 `.npy` 标签、mask、带坐标轴/深度色标的 PNG 预览和 montage；流式脚本也会在样本完成后自动写入 `labels/`。

2026-05-30 更新：已移除流式脚本中的固定 `D:\...` 输出路径。现在默认输出全部从 `simple_shell_common.py` 的 `ROOT / 'output'` 派生，项目移动到 Windows 服务器其他目录后仍会写入新位置的 `simple/output/`；如需输出到独立数据盘，可用 `--output-root` 显式指定。

2026-05-31 更新：流式脚本已加入终端进度输出和求解中心跳。后续因 Windows/conda/任务调度下 `tqdm` 可能被捕获或延迟，已回退为普通 `print(..., flush=True)` 输出，并建议长时间求解使用 `conda run --no-capture-output -n comsol python -u ...`。由于 `mph` 调用 `model.solve()` 时是阻塞的，无法稳定取得 COMSOL 内部时间步百分比，因此心跳显示样本级工况完成度、当前工况已用时和 ETA，并写入 `progress/*.jsonl`。健康管 `tx=1, f=50 kHz` 单工况检查已统一使用 `solve_export_dataset_a_streaming.py --only-healthy --tx 1 --frequencies 50000`，该命令会求解并执行 export，原单独检查脚本已删除以避免重复维护。

2026-05-31 更新：已加入一组流式加速配置。流式脚本默认每个样本只建立一次模型、网格和 solver tree，然后循环修改 `tx` 与 `pzt_fc` 求解导出，每个工况导出后调用 `clearSolutionData()` 清除瞬态场解，避免保存 48 个工况的全场数据。保留 `--rebuild-each-case` 回退到旧的每工况重建模式。`dt_out_us`、`--linear-solver`、`--cudss-precision` 均可由命令行选择。

2026-05-31 更新：针对流式 `--only-healthy` 在 `cuDSS single + dt_out=1.0 us` 下出现“找不到一致的初始值 / internal error cudss / 最后一个时步不收敛”，而同一模型在 COMSOL GUI 中用 `PARDISO` 可正常求解的问题，已将公共默认求解配置回退为稳定基线：`linear_solver = pardiso`、`dt_out_us = 0.5`、`cudss_precision = double`。`cuDSS` 仍保留为显式加速测试参数，建议先测试 `--linear-solver cudss --cudss-precision double --dt-out-us 0.5`，再尝试 `single` 或 `dt_out=1.0 us`。

2026-06-01 更新：已修复流式导出阶段 `receiver 17` 点数据集/点插值不稳定的问题。原因是 MPh 的 `model.evaluate()` 不能直接对 CutPoint 派生数据集求解，而 COMSOL numerical `Interp/Eval` 如果没有显式选择 `open pipe cylindrical shell boundary`，会返回有时间步但空间点数为 0 的数据。当前导出逻辑已改为：先将 receiver 坐标投影到壳中面 `r = Rm`，尝试壳面点插值；若 COMSOL 仍不返回点数据，则读取命名壳面 selection 上的实际 `u/v` 场数据，并选取离目标 receiver 最近的真实壳面采样点导出径向位移。已用 `dt_out=20 us` 快速验证 Dataset A/B 健康单工况均能导出，`finite=true` 且 `nonzero_channels=16/16`。注意 `20 us` 仅用于快速验证导出链路，真实数据仍应使用 `0.5/0.7/1.0 us`。

2026-06-01 后续计划：建议将接收模型从“点接收/最近壳面点”升级为“小面积加权平均 receiver”。这需要修改 COMSOL 建模阶段：在壳面上用与 PZT 尺寸一致的接收窗口，对径向位移 `u_r = cos(theta_rx)*u + sin(theta_rx)*v` 做面积加权平均。该方案比单点插值和最近网格点更接近真实 PZT 接收，也更抗网格节点位置误差。由于这是较大的建模和导出接口改动，建议先将当前已验证可运行状态提交到 GitHub 存档，再单独分支实现。

但当前 `simple/` 更准确地说是“快速壳模型代理数据集”，还不能直接等同于真实 3D PZT 管道实验数据。后续需要重点补齐 Dataset A/B 的边界、健康-损伤配对、接收端增益、噪声后处理、标签图导出和螺旋阶次特征计算。

## 与数据集设计的匹配情况

### 已匹配的部分

- 管道几何参数匹配 Dataset A/B 的基本设定：长度 1 m，外半径 160 mm，内半径 150 mm，壁厚 10 mm。
- 发射环和接收环位置匹配：`z = 100 mm` 和 `z = 900 mm`，每环 16 个通道。
- 激励频率匹配当前计划：40、50、60 kHz。
- PZT 实体已简化为等效面载荷窗口，符合当前降低模型规模的目标。
- 接收数据导出为 16 路径向位移 `ur = cos(theta) * u + sin(theta) * v`，这对圆管壳中面的径向响应是合理的。
- 流式脚本按 `tx x frequency` 拆成单工况求解，求解后直接导出波形和特征，再 `client.remove(model)` / `client.clear()`，这个流程适合服务器批量运行。

### 当前不完全匹配的部分

- ~~`dataset_create.md` 中 Dataset A 原本是“理想基准”，建议单个规则圆形/椭圆形缺陷，并使用低反射边界或 PML。当前 `simple` 的 Dataset A 随机缺陷默认是 1-4 个不规则缺陷，并且没有实现低反射端部，壳端部等效为自由边。~~ 已新增 `solve_export_dataset_a_validation_streaming.py` 和 `solve_export_dataset_a_training_streaming.py`，Dataset A 默认开启壳端部 absorbing layer。
- Dataset B 要模拟真实实验，当前已加入材料、发射/接收位置和发射幅值扰动，但接收端幅值比例误差还没有实际作用到导出的接收波形中。
- Dataset B 的健康基准和损伤样本当前可能使用不同随机材料/PZT 位置/幅值扰动。如果后续要做健康-损伤差分或相位差，真实实验中同一样本的健康和损伤应共享同一组非缺陷扰动。
- 噪声、触发抖动、基线漂移目前只是写入 metadata 的 plan，还没有实际生成 noisy waveform。
- ~~diffusion 训练需要的缺陷标签图或厚度损失图目前没有由 `simple` 流式流程直接导出，只能从 metadata 中重建。~~ 已新增 `labels/` 输出，且标签由 metadata 按仿真厚度损失公式重建。

## 建模合理性判断

### 1. 壳模型

壳模型用于第一阶段数据集是可以接受的，尤其是在目标是生成层析粗图、学习缺陷位置/形状先验，而不是精确复现实验幅值时。当前管道 `Rm / h = 155 / 10 = 15.5`，属于可以尝试壳模型的范围。

需要注意的是，壳模型不能完整表达 3D 厚度方向 Lamb 波模态、真实 PZT 压电耦合和内壁腐蚀导致的中面偏移。因此后续必须用少量 3D 模型或实验健康/缺陷样本做相位、群速度和缺陷散射趋势验证。

### 2. 几何和边界

当前做法是在 3D 中建立圆柱支撑几何，再只选择圆柱侧壁作为 Shell 物理场边界，端盖边界不参与 Shell。这种方式在 COMSOL 中是常见的壳建模路线，build log 中 `problems = []` 也说明模型树没有明显错误。

主要风险在端部边界：COMSOL 6.4 的 Shell 物理场不能直接创建 Solid Mechanics 的 `Low-Reflecting Boundary` 节点。当前 Dataset A 使用两端轴向渐变 Rayleigh 阻尼吸收层作为壳模型下的低反射代理方案；Dataset B 继续保持自由端以匹配实验。仍建议后续用实际波形量化吸收层对端部反射的抑制效果。

### 3. 缺陷表达

当前缺陷是壳厚度的平滑减薄：

- 健康模型厚度为 `h0`。
- 缺陷模型厚度为 `max(h_min, h0 - loss_terms)`。
- 每个缺陷和凸瓣使用超高斯窗口。

这对快速生成连续厚度损失场是合理的，也适合生成 diffusion 标签。当前已按外表面腐蚀更新：缺陷处壳厚度减薄，同时设置 `ThicknessOffset` 为 `RelativeDistance`，`z_offset_rel` 为负的局部 `loss/thickness` 表达式，使内表面近似保持不变、外表面向内腐蚀。健康模型仍为中面无偏置。

### 4. 激励模型

等效径向面载荷窗口可以保留。它避免了 PZT 小尺寸导致的局部网格过密，适合批量数据集。

需要注意三点：

- `F0/pzt_A` 与平滑窗口相乘后，总积分力只是近似等于 `F0`，不是严格归一化。若只关心相位和相对特征，影响不大；若要比较幅值，需要做窗口积分归一化或统一归一化后处理。
- 载荷窗口宽度 6 mm，而当前 `hmax = 5.208 mm`，对波长来说够用，但对 PZT 窗口空间分辨率偏粗。建议做一次 `hmax` 收敛检查，例如 5.2 mm、3.5 mm、2.5 mm 对比相位和包络峰值。
- PZT 粘接层、极化方向和压电接收过程已被简化掉，因此导出位移不是实际电压信号。后续训练时应使用归一化、健康差分或频域相对变化，避免让模型学习不可靠的绝对幅值。

### 5. 接收模型

接收点径向位移投影方式正确，适合作为壳模型输出。当前流式导出已不再依赖保存的 receiver CutPoint 数据集；默认会把 receiver 坐标投影到壳中面，并在 `open pipe cylindrical shell boundary` 上求解/采样。若 COMSOL 点插值不可用，则退回到实际壳面采样点的最近邻导出，避免 `Dataset "... receiver 17 point" does not refer to a solution` 和空间点数为 0 的失败。

下一步更推荐的接收模型是“小面积加权平均 receiver”，而不是数学单点。真实 PZT 接收并不是一个无面积点，而是一个有限贴片；用接收贴片窗口对 `u_r` 做面积平均能减少网格点位置误差，并且更接近实验含义。

建议实现方式：

- 在 COMSOL 建模阶段创建一个作用在 `open pipe cylindrical shell boundary` 上的 Integration component coupling，例如 `intop_shell`。
- 对每个 receiver 17-32 生成接收窗口：

  ```text
  w_rx = exp(-((Rm*dtheta)/(pzt_w/2))^8 - ((z-z_rx)/(pzt_l/2))^8)
  ```

  其中 `dtheta` 使用与发射端窗口相同的角度差表达式，`z_rx` 和 `theta_rx` 包含 Dataset B 的 `dz_mm` / `dtheta_deg` 扰动，半径仍固定为 `Rm`。

- 每个通道导出以下全局表达式：

  ```text
  intop_shell(w_rx*((cos(theta_rx))*u+(sin(theta_rx))*v))/intop_shell(w_rx)
  ```

- 这样导出的 CSV 表示 16 个接收 PZT 贴片的加权平均径向位移，而不是最近网格点或单点插值。
- 如果后续要模拟接收端灵敏度误差，可在上述通道结果上再乘以 `rx_gain`，并把增益写入 metadata。

这个改动会影响 `simple_shell_common.py` 的建模函数和 `streaming_export_common.py` 的导出路径，属于较大修改。建议先提交当前最近壳面点导出版本作为稳定基线，再在新分支实现并用 Dataset A/B 健康单工况和一个损伤样本做对比验证。

但 Dataset B 中的接收端幅值误差目前没有实际施加到通道数据。`AMPLITUDE_SCALE` 对发射端面载荷有效，对 `rx17-rx32` 只写入 metadata，不会自动乘到导出的 CSV。后续如果要模拟接收灵敏度差异，应在导出或噪声后处理阶段对每个接收通道乘以对应增益，并记录随机种子。

### 6. 求解和流式导出

流式求解导出方案是当前最值得保留的部分。它避免将 16 x 3 个瞬态场解保存进一个 MPH 文件，也避免后续重新读取巨大 MPH 的导出成本。

流式脚本已增加后台进度日志：每个样本会写入 `progress/<sample_id>_progress.jsonl`，并在控制台输出 `case_index/case_count`、已用时、平均工况耗时和 ETA。若某个工况失败，会记录 `case_failed` 和错误信息。当前默认一个样本复用一个 COMSOL 模型和网格，逐工况求解导出后清除 solution data；这比每个工况重建模型更快，同时仍不保存含解 MPH。

求解器配置默认使用 PARDISO 和 `dt_out=0.5 us`，这是和当前 GUI 成功求解路径一致的稳定基线。cuDSS 需要服务器有兼容 NVIDIA GPU 和 CUDA driver；如果 COMSOL 进程检测不到 GPU，或 cuDSS factorization 在单精度下不稳定，会在求解阶段报错。此时 `--rebuild-each-case` 不能解决根因，因为它只改变建模/网格复用方式，不改变 Direct solver；应先用 `--linear-solver pardiso --dt-out-us 0.5` 复核健康单工况，再逐项测试 `cuDSS double`、`dt_out=1.0 us`、`cuDSS single` 对 TOF、FFT 相位和包络峰值的影响。

Windows 服务器运行时仍需注意：

- COMSOL 求解当前工况时仍会占用内存和临时目录。
- 建议将 COMSOL temp/recovery 目录设置到容量充足的本地 scratch，而不是系统盘。
- 默认不传 `--cores` 让 COMSOL 自行分配是合理的；批量并行时再限制核心数。
- 服务器为 Windows + COMSOL 6.4，根目录 `SERVER_COMSOL_ENV_SETUP.md` 里仍写有 COMSOL 6.2 和 Linux 命令示例，后续应更新为 Windows/6.4 版本说明。

## 需要优先修改或验证的点

### 高优先级

1. ~~将 Dataset A 拆成两个用途：~~

   - ~~`A_validation`: 单个规则圆形/椭圆形缺陷，用于验证螺旋路径相位延迟反演。~~
   - ~~`A_training`: 多缺陷/不规则缺陷，用于扩充 diffusion 训练分布。~~
   - 已新增 `simple/solve_export_dataset_a_validation_streaming.py` 和 `simple/solve_export_dataset_a_training_streaming.py`。
2. ~~为 Dataset A 的理想验证集处理端部反射：~~

   - ~~优先方案：加入低反射边界、吸收段或延长管道。~~
   - ~~简化方案：只在直达波和目标螺旋阶次到达窗口内提取特征，并明确避开端部反射。~~
   - 已在 Dataset A 中加入壳端部 absorbing layer；注意它不是 Solid Mechanics 的 Low-Reflecting Boundary 节点。
3. Dataset B 必须生成“同一随机扰动下的健康-损伤配对”：

   - 对每个损伤样本，先用同一材料、PZT 位置、发射/接收增益生成 paired healthy。
   - 再只改变缺陷厚度场生成 damaged。
   - 健康-损伤差分、相位差和频域比值只能在这对数据之间计算。
4. 实现接收端增益、噪声、触发抖动和基线漂移的后处理，并输出 noisy CSV：

   - 保留 clean waveform。
   - 另存 observed/noisy waveform。
   - metadata 中记录每个 tx/rx/frequency 的 SNR、通道增益、时间抖动和随机种子。
5. 将接收端从点接收升级为小面积加权平均 receiver：

   - 在建模阶段增加壳面 Integration component coupling，绑定 `open pipe cylindrical shell boundary`。
   - 复用 PZT 尺寸参数 `pzt_w` / `pzt_l` 定义 receiver 超高斯窗口。
   - 导出表达式使用 `intop_shell(w_rx*u_r)/intop_shell(w_rx)`，其中 `u_r = cos(theta_rx)*u + sin(theta_rx)*v`。
   - Dataset B 的接收位置扰动只改变 `theta_rx` 和 `z_rx`，接收窗口仍在 `r=Rm` 的壳面上。
   - 与当前最近壳面点导出对比：健康单工况应保持 `finite=true`、`nonzero_channels=16/16`；波形相位和包络峰值不应出现不合理突变。
   - 该项会改动 COMSOL 模型树和导出表达式，建议在当前稳定版本提交 GitHub 后再做。
6. ~~为 diffusion 增加标签图导出：~~

   - ~~建议导出 `theta-z` 网格上的厚度损失图，例如 `defect_depth_mm[theta_index, z_index]`。~~ 已导出 `defect_depth_mm[z_index, theta_index]`，作为图像使用时横轴为 `theta`，纵轴为 `z`。
   - ~~同时导出二值 mask、归一化深度图和 metadata。~~ 已输出 `defect_mask.npy`、`defect_depth_norm.npy`、带坐标/色标预览图和 `defect_label_metadata.json`；坐标轴由 metadata 记录，不再额外输出不必要的坐标 `.npy`。
   - ~~粗图生成阶段和 diffusion 训练阶段都应使用同一坐标约定。~~ 已在 README 中固定展开坐标约定，并提供 Pearson 相关系数、NRMSE、mask IoU 辅助函数。
7. 检查螺旋阶次特征计算：

   - 当前 `helical_order_projections` 使用固定群速度和预测到达时间窗，这只是粗特征。
   - 后续建议改成路径窗内的互相关延迟、包络峰值、带通后 FFT 相位或健康-损伤复数比值。
   - 需要特别检查正负螺旋阶次的角度展开逻辑，避免把 `+1/-1` 阶路径多加一圈。

### 中优先级

1. 做 mesh/time convergence 表：

   - `hmax = 5.2, 3.5, 2.5 mm`
   - `dt_out = 1.0, 0.5, 0.25 us`
   - 对比直达波 TOF、FFT 相位、包络峰值和缺陷差分特征。
2. 用单个已知缺陷做 sanity check：

   - 健康模型所有接收通道非零。
   - 对称 tx/rx 对在健康模型中满足近似圆周对称性。
   - 缺陷位于某条路径附近时，该路径特征变化应强于远离路径的通道。
3. 清理和补充 Markdown：

   - `dataset_create.md` 中部分公式是复制后的乱码，应重写为标准 Markdown/LaTeX。
   - `simple/README.md` 建议补充本文档中的关键限制：Dataset A 端部反射、Dataset B paired healthy、接收端增益尚未施加、噪声尚未实际生成。
   - 生成的 build log 是一次构建快照，后续配置变化后需要重新生成，避免文档和实际 MPH 不一致。
4. 校准群速度：

   - 现在默认 `group_velocity = 2522 m/s`。
   - 建议从健康壳模型中根据 tx-rx 到达时间反推不同频率的有效群速度，或用色散曲线/实验健康信号校准。

### 低优先级

1. 如果后续关心幅值，可把等效载荷窗口做积分归一化。
2. 可增加每个样本的 preview 图：缺陷厚度图、16x16 通道特征矩阵、粗层析图。
3. 可增加批处理断点续算：已有 CSV/metadata 时跳过已完成 case。

### 2026-05-31 已处理的脚本维护项

1. ~~流式后台运行时终端没有清晰进度显示。~~ 已在 `streaming_export_common.py` 中使用普通 `print(..., flush=True)` 显示样本级工况进度，并保留 `progress/*.jsonl` 作为后台日志；`model.solve()` 阻塞期间用 `--heartbeat-s` 周期刷新当前工况已用时。长时间任务建议使用 `conda run --no-capture-output ...`。
2. ~~健康单工况检查脚本与 `solve_export_dataset_a_streaming.py --only-healthy --tx 1 --frequencies 50000` 功能重复。~~ 已删除独立 `check_single_healthy_case.py`，统一使用流式脚本；该命令会求解并执行 export，metadata 中的 `waveform_check` 用于确认 CSV 导出和 16 通道有效性。
3. ~~README 中缺少流式脚本清单、单工况 export 检查说明，以及 build 脚本和 streaming 脚本模型关系说明。~~ 已更新 `simple/README.md`：列出四个流式运行脚本，说明 `tqdm`/心跳/JSONL 进度，给出健康单工况检查命令，并明确 `build_dataset_a_healthy.py` 与 Dataset A streaming 脚本共享 `simple_shell_common.py` 建模配置但用途不同。
4. ~~COMSOL 中检查模型时不容易直接看到 16 个等效发射点和 16 个接收点位置。~~ 已在新构建模型的 `Results > Datasets` 中加入 `transmitter PZT marker points` 和 `receiver PZT marker points`，它们只是可视化点集，不参与几何、网格、物理场或求解。
5. ~~每个流式工况都重建几何、网格和 solver tree，造成额外初始化开销。~~ 已改为默认每个样本复用一个模型/网格，逐工况修改 `tx` 和 `pzt_fc`，导出后清除 solution data；如需旧行为可加 `--rebuild-each-case`。
6. ~~默认输出时间步和线性求解器精度偏保守，批量生成代理数据较慢。~~ 已在流式脚本中加入 `--dt-out-us`、`--relative-tolerance`、`--linear-solver`、`--cudss-precision` 和 `--pardiso-use-cluster` 参数，可显式测试 `cuDSS`、`single` 和 `dt_out=1.0 us` 加速组合。
7. ~~流式 `--only-healthy` 使用默认加速组合时，`--rebuild-each-case` 仍会出现 cuDSS 内部错误和最后时步不收敛。~~ 原因是流式默认曾使用 `cuDSS single + dt_out=1.0 us`，而 GUI 成功路径使用 `PARDISO`；`--rebuild-each-case` 只影响是否复用模型/网格，不影响 Direct solver。已将默认改回 `PARDISO + dt_out=0.5 us`，并在启动时打印求解器配置；选择 `cuDSS single` 时会输出稳定性警告。

## 建议的 Windows 服务器验证顺序

先只跑单工况，确认 COMSOL 6.4、MPh、JPype 和 conda 环境都正常：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_streaming.py --only-healthy --tx 1 --frequencies 50000 --linear-solver pardiso --dt-out-us 0.5 --heartbeat-s 10
```

确认输出 CSV 后，检查一组波形是否非零、时间长度是否为 0-0.8 ms、接收通道是否为 16 路。然后再跑一个健康 + 一个损伤样本：

```powershell
conda run --no-capture-output -n comsol python -u simple/solve_export_dataset_a_streaming.py --include-healthy --samples 1 --tx 1 --frequencies 50000
```

最后再扩大到 16 个发射、3 个频率。不要一开始就批量提交大量样本。

## 当前是否可以继续用 simple 路线

可以继续使用，但建议把它定位为第一阶段代理数据集。短期内最重要的不是恢复 3D PZT，而是把 Dataset A 的理想验证、Dataset B 的健康-损伤配对、噪声/增益后处理和缺陷标签图导出补齐。只要这些环节完成，`simple` 目录的流式壳模型就可以作为 diffusion 前两阶段数据生成的主流程。
