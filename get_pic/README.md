# get_pic V1 粗图生成流程

`get_pic` 下的所有 Python 脚本都应在 `get_pic` conda 环境中运行。

```powershell
conda run -n get_pic python simple/get_pic/generate_coarse_maps.py
```

默认输入根目录是：

```text
simple/f_domain/output/streaming_dataset_a_frequency_shell/
```

默认期望的输入文件包括：

```text
frequency_response/dataset_a_frequency_healthy_H_complex.npz
frequency_response/dataset_a_frequency_sample_0001_H_complex.npz
metadata/dataset_a_frequency_healthy.json
metadata/dataset_a_frequency_sample_0001.json
```

如果服务器上传后的缺陷样本使用标准命名，可以这样批量生成：

```powershell
conda run -n get_pic python simple/get_pic/generate_coarse_maps.py --sample-ids 1-20
```

输出目录：

```text
simple/get_pic/output/coarse_maps/
simple/get_pic/output/x_matrix/
simple/get_pic/output/manifest.csv
```

可选的预览和评价命令：

```powershell
conda run -n get_pic python simple/get_pic/preview_coarse_maps.py --coarse simple/get_pic/output/coarse_maps/dataset_a_frequency_sample_0001_coarse_maps.npz
conda run -n get_pic python simple/get_pic/evaluate_coarse_maps.py --coarse simple/get_pic/output/coarse_maps/dataset_a_frequency_sample_0001_coarse_maps.npz
```

V1 粗图生成只使用健康/损伤频域响应和 metadata。生成粗图时不会读取 label；label 只用于后续评价和调参。

## 预览 PNG 怎么看

`preview_coarse_maps.py` 会把下面文件中的 `pic` 各通道都画出来：

```text
simple/get_pic/output/coarse_maps/<sample>_coarse_maps.npz
```

所有子图都使用和缺陷标签一致的管外表面展开坐标：

```text
横轴 = theta，0..360 deg，首尾周期相接
纵轴 = z，0..1000 mm
```

因此源缺陷标签：

```text
simple/f_domain/output/streaming_dataset_a_frequency_shell/labels/<sample>_defect_depth_norm.npy
```

或对应的预览图：

```text
simple/f_domain/output/streaming_dataset_a_frequency_shell/labels/<sample>_defect_label.png
```

可以按同样方向直接对照：横向是周向角度，纵向是管道轴向位置。

预览图中的各个通道含义如下：

| 通道 | 含义 | 观察方式 |
| --- | --- | --- |
| `ray_log_amp_loss` | 对 `log(abs(H0)) - log(abs(Hd))` 做路径反投影。 | 反映损伤后幅值相对健康样本变弱的路径。腐蚀减薄区域可能在多条异常路径交汇处变亮。 |
| `ray_relative_delta` | 对 `abs(Hd - H0) / abs(H0)` 做路径反投影。 | V1 中最直接的异常图。亮区表示很多 `tx-rx-frequency` 组合在该区域附近出现变化。 |
| `ray_phase_change` | 对 `abs(angle(Hd * conj(H0)))` 做路径反投影。 | 反映相位变化，和波速变化/路径延迟有关；但相位未精确校准时可能比较噪。 |
| `ray_delta_abs` | 对 `abs(Hd - H0)` 做路径反投影。 | 绝对复差分。容易受高幅值通道主导，应结合 `ray_relative_delta` 看。 |
| `low_frequency_band_map` | 只用低频段的 `ray_relative_delta` 生成。 | 通常更平滑、更发散，用来看低频信息是否仍指向缺陷附近。 |
| `mid_frequency_band_map` | 只用中频段的 `ray_relative_delta` 生成。 | 通常优先检查，因为它更接近当前主要激励频段。 |
| `high_frequency_band_map` | 只用高频段的 `ray_relative_delta` 生成。 | 可能更锐利，但也更容易受噪声、未完成频点和相位不稳影响。 |
| `path_coverage` | 归一化后的路径/核覆盖度。 | 不是缺陷图。它表示传感器路径对各区域的照射程度。覆盖度低的位置不可靠。 |
| `valid_case_count` | 每个像素附近有效 `tx-rx-frequency` 工况数量。 | 不是缺陷图。用于判断该区域是否由足够多的有效数据支撑。 |
| `reliability_mask` | 由 `path_coverage >= reliability_threshold` 得到的二值可靠区域。 | 不是缺陷图。值为 0 的区域表示粗图通道不应被强信任。 |

默认预览画的是归一化后的 `pic`。这便于看图，但每个通道都有自己的归一化色标，所以不同子图之间不能直接比较“谁更亮”。如果要看未归一化值，可以加 `--raw`：

```powershell
conda run -n get_pic python simple/get_pic/preview_coarse_maps.py --raw --coarse simple/get_pic/output/coarse_maps/dataset_a_frequency_sample_0001_coarse_maps.npz
```

## 如何判断是否与源缺陷符合

先打开频域流程生成的 label 预览图：

```text
simple/f_domain/output/streaming_dataset_a_frequency_shell/labels/<sample>_defect_label.png
```

然后和 `get_pic` 的 preview PNG 对照。

一个合理的 V1 粗图通常应满足：

- `ray_relative_delta`、`ray_log_amp_loss` 或 `mid_frequency_band_map` 的亮区应与真实缺陷区域重叠，或者至少有多条亮带穿过真实缺陷区域。
- 最亮区域不一定和 label 形状完全一致。V1 是 ray-tube 反投影，常见结果是条带、拖尾或拉长的亮斑，而不是精确腐蚀坑边界。
- 如果多个独立通道都在相同的 `(theta, z)` 附近变亮，并且该位置靠近 label 中的缺陷，这是比较好的信号。
- `path_coverage` 在缺陷附近应为非零且不能太低。如果缺陷本身处在低覆盖区，那么粗图与 label 不匹配的意义要降低。
- 如果只有 `ray_delta_abs` 很亮，但 `ray_relative_delta` 和 `ray_phase_change` 不明显，可能只是绝对幅值通道主导，不一定是真缺陷定位。
- 如果所有异常图都只在发射环或接收环附近很亮，也就是 `z=100 mm` 或 `z=900 mm` 附近，通常说明端点屏蔽、归一化或频点选择还需要调整。

可以用评价脚本做定量比较：

```powershell
conda run -n get_pic python simple/get_pic/evaluate_coarse_maps.py --coarse simple/get_pic/output/coarse_maps/dataset_a_frequency_sample_0001_coarse_maps.npz
```

该命令会写出：

```text
simple/get_pic/output/reports/<sample>_coarse_maps_metrics.json
```

主要指标含义：

- `pearson`：越高越好，表示粗图是否在 label 高的位置也变高。
- `nrmse`：越低越好，表示粗图和 label 的归一化误差。
- `mask_iou`：越高越好，表示阈值化后缺陷区域的重叠程度。
- `top5_hit_rate`：粗图响应最高的 5% 像素中，有多少比例落在真实缺陷 mask 内。这个指标比 IoU 更适合 V1 这种条带状粗图。
- `prediction_mass_in_label`：粗图总响应能量中有多少比例落在真实缺陷 mask 内。值越高，说明异常能量越集中到真实缺陷附近。
- `centroid_error_mm`：越低越好，表示粗图异常中心和真实缺陷中心的距离。

注意：V1 粗图不是最终缺陷深度图。它的目标是给 diffusion 提供一个物理上合理的空间先验，而不是精确复原 label 的边界和深度。

## 频点选择和对比

原始 `relative_l2` 频点选择只衡量：

```text
||Hd(f)-H0(f)|| / ||H0(f)||
```

这能找出“频响变化大”的频点，但不保证这些频点经过 V1 反投影后更接近真实缺陷位置。

更适合作为论文主方法的是无 label 的物理选频指标 `physics_tomography`：

```powershell
conda run -n get_pic python simple/f_domain/select_sensitive_frequencies.py --sample-ids 1-12,14-23 --metric physics_tomography --top-n 15 --output-root simple/f_domain/output/frequency_selection_physics_tomography_tuned --prefix physics_tomography_tuned --frequency-min-khz 20 --frequency-max-khz 100
```

该指标不读取缺陷 label，只使用健康/损伤复频响。它综合考虑稳健相对扰动、相位扰动、路径参与度、路径对比度、tx/rx 覆盖均衡性。物理解释见：

```text
simple/get_pic/PHYSICS_FREQUENCY_SELECTION.md
```

`v1_label_guided` 依赖缺陷 label，因此更适合作为仿真阶段的后验验证和调参工具，不建议作为论文主选频依据：

```powershell
conda run -n get_pic python simple/f_domain/select_sensitive_frequencies.py --sample-ids 1-12,14-23 --metric v1_label_guided --top-n 15 --output-root simple/f_domain/output/frequency_selection_v1_label_guided --prefix v1_label_guided --v1-grid-size 128 --v1-sigma-ray-mm 25 --frequency-min-khz 20 --frequency-max-khz 100
```

该指标的思想是：对每个频点，检查异常路径强度是否更集中在穿过真实 label 的路径上。它综合了路径异常与 label 路径重叠的相关性、top label 路径和非 label 路径的对比度，以及异常能量落在 label 路径上的比例。

对比某个样本的新选频和全部频点：

```powershell
conda run -n get_pic python simple/get_pic/compare_frequency_selection.py --sample-id 14 --selected-frequencies simple/f_domain/output/frequency_selection_v1_label_guided/v1_label_guided_top15_frequencies.txt --output-root simple/get_pic/output/frequency_selection_compare_sample14 --preview
```

如果要对比旧 `relative_l2` 选频：

```powershell
conda run -n get_pic python simple/get_pic/compare_frequency_selection.py --sample-id 14 --selected-frequencies simple/f_domain/output/frequency_selection/frequency_sensitivity_top15_frequencies.txt --output-root simple/get_pic/output/frequency_selection_compare_sample14_old_relative_l2 --preview
```

汇总评价报告：

```powershell
conda run -n get_pic python simple/get_pic/summarize_reports.py --report simple/get_pic/output/frequency_selection_compare_sample14/selected/reports/dataset_a_frequency_sample_0014_coarse_maps_metrics.json simple/get_pic/output/frequency_selection_compare_sample14/all_completed/reports/dataset_a_frequency_sample_0014_coarse_maps_metrics.json
```

如果要检查 `physics_tomography` 与全频点的差异：

```powershell
conda run -n get_pic python simple/get_pic/compare_frequency_selection.py --sample-id 14 --selected-frequencies simple/f_domain/output/frequency_selection_physics_tomography_tuned/physics_tomography_tuned_top15_frequencies.txt --output-root simple/get_pic/output/frequency_selection_compare_sample14_physics_tomography_tuned --preview
```
