# Dataset-A EDM 固定划分

`dataset_a_1200_train_ids.txt` 与 `dataset_a_1200_val_ids.txt` 是 base48/base64 正式配置使用的显式数据划分。

- 数据范围：`dataset_a_frequency_sample_0001` 至 `dataset_a_frequency_sample_1200`。
- 划分方式：对 1200 个 ID 使用 `torch.Generator().manual_seed(20260708)` 生成一次随机排列。
- 训练集：1080 个样本。
- 验证集：120 个样本。
- 两个清单互不重叠，按 sample ID 升序保存，便于人工检查。

训练开始时，`train_edm.py` 会把这次实际加载的完整成员再次写入：

```text
runs/<run_name>/data_split.json
runs/<run_name>/validation_sample_ids.txt
```

请不要在同一次实验中修改这两个清单。若需要新的实验划分，应创建一对新的清单和新的 run 目录。
