# physics_highfreq_quota 频域选频鲁棒性汇总

## 频点集合

- output2 40 样本选出的 top15/kHz: 40, 32.5, 20, 42.5, 50, 47.5, 52.5, 70, 72.5, 67.5, 75, 80, 82.5, 77.5, 95
- 旧 output 99 样本选出的 top15/kHz: 37.5, 32.5, 35, 47.5, 50, 55, 45, 75, 92.5, 90, 80, 87.5, 85, 72.5, 95
- 两者交集/kHz: 32.5, 47.5, 50, 72.5, 75, 80, 95

## output2 当前 z 向模型，40 样本评价

ray_relative_delta 通道:

| method | pearson | IoU | top5_hit | mass_in_label | centroid_mm |
| --- | --- | --- | --- | --- | --- |
| all_frequencies | 0.1097 | 0.0552 | 0.1750 | 0.0576 | 470.59 |
| old99_physics_top15 | 0.1259 | 0.0552 | 0.2080 | 0.0588 | 465.63 |
| physics_highfreq_quota_40 | 0.1052 | 0.0552 | 0.1475 | 0.0573 | 470.58 |
| relative_l2_40 | 0.0849 | 0.0552 | 0.1082 | 0.0557 | 436.32 |
| v1_label_guided_40 | 0.1731 | 0.0552 | 0.2717 | 0.0629 | 479.68 |

high_frequency_band_map 通道:

| method | pearson | IoU | top5_hit | mass_in_label | centroid_mm |
| --- | --- | --- | --- | --- | --- |
| all_frequencies | 0.1814 | 0.0552 | 0.2889 | 0.0637 | 480.81 |
| old99_physics_top15 | 0.1900 | 0.0552 | 0.2731 | 0.0646 | 481.13 |
| physics_highfreq_quota_40 | 0.1746 | 0.0552 | 0.2803 | 0.0632 | 479.13 |
| relative_l2_40 | 0.1662 | 0.0552 | 0.2185 | 0.0625 | 473.29 |
| v1_label_guided_40 | 0.1844 | 0.0552 | 0.2870 | 0.0640 | 480.05 |

## 旧 output x/y 模型，99 样本同分布评价

ray_relative_delta 通道:

| method | pearson | IoU | top5_hit | mass_in_label | centroid_mm |
| --- | --- | --- | --- | --- | --- |
| old99_all_frequencies | 0.1896 | 0.0557 | 0.3320 | 0.0646 | 456.02 |
| old99_physics_top15 | 0.1944 | 0.0557 | 0.3241 | 0.0651 | 456.70 |
| old99_relative_l2_top15 | 0.1775 | 0.0557 | 0.3343 | 0.0634 | 450.14 |
| old99_v1_label_guided_top15 | 0.2074 | 0.0557 | 0.3106 | 0.0667 | 461.53 |
| output2_physics40_top15 | 0.1938 | 0.0557 | 0.3436 | 0.0649 | 451.11 |

high_frequency_band_map 通道:

| method | pearson | IoU | top5_hit | mass_in_label | centroid_mm |
| --- | --- | --- | --- | --- | --- |
| old99_all_frequencies | 0.2055 | 0.0557 | 0.2879 | 0.0668 | 457.89 |
| old99_physics_top15 | 0.2039 | 0.0557 | 0.2814 | 0.0667 | 452.73 |
| old99_relative_l2_top15 | 0.2008 | 0.0557 | 0.2773 | 0.0663 | 456.19 |
| old99_v1_label_guided_top15 | 0.2069 | 0.0557 | 0.2834 | 0.0670 | 460.31 |
| output2_physics40_top15 | 0.2022 | 0.0557 | 0.2991 | 0.0663 | 455.71 |

## 结论

- `physics_highfreq_quota_40` 是当前 output2 40 样本上的主结果；`relative_l2_40`、`v1_label_guided_40` 和 `all_frequencies` 是对照。
- 旧 output 99 样本与当前 output2 的频点集合并不完全一致，说明旧 x/y 载荷和接收方向数据不能与当前 z 向数据等权混合评分。
- 判断 `physics_highfreq_quota` 方法是否鲁棒，应看旧 output 99 内部的同分布比较：`old99_physics_top15` 是否仍优于 `old99_relative_l2_top15`，并接近 `old99_v1_label_guided_top15`。
- 旧 output 可以作为方法鲁棒性验证集；当前 z 向模型的最终频点仍应以 output2 40 样本为主。
