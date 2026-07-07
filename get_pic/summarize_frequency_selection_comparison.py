"""Summarize frequency-selection robustness experiments.

This script reads existing evaluate_coarse_maps.py JSON reports and writes
compact CSV/Markdown summaries for the current output2-vs-old-output check.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev


ROOT = Path(__file__).resolve().parent
SIMPLE_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "output2" / "frequency_selection_physics_highfreq_quota_40_vs_old99"

METRIC_NAMES = (
    "pearson",
    "nrmse",
    "mask_iou",
    "top5_hit_rate",
    "prediction_mass_in_label",
    "centroid_error_mm",
)

FOCUS_CHANNELS = (
    "ray_relative_delta",
    "high_frequency_band_map",
    "ray_phase_change",
    "ray_log_amp_loss",
)

METHODS = [
    {
        "dataset": "output2_current_z",
        "method": "physics_highfreq_quota_40",
        "reports": ROOT / "output2" / "freqsel_new40_on_output2_40" / "reports",
        "note": "physics_highfreq_quota selected from current output2 40 samples",
    },
    {
        "dataset": "output2_current_z",
        "method": "relative_l2_40",
        "reports": ROOT / "output2" / "freqsel_relative_l2_40_on_output2_40" / "reports",
        "note": "relative_l2 selected from current output2 40 samples",
    },
    {
        "dataset": "output2_current_z",
        "method": "v1_label_guided_40",
        "reports": ROOT / "output2" / "freqsel_v1_label_guided_40_on_output2_40" / "reports",
        "note": "label-guided reference selected from current output2 40 samples",
    },
    {
        "dataset": "output2_current_z",
        "method": "all_frequencies",
        "reports": ROOT / "output2" / "freqsel_all_frequencies_on_output2_40" / "reports",
        "note": "all valid frequency points on current output2 40 samples",
    },
    {
        "dataset": "output2_current_z",
        "method": "old99_physics_top15",
        "reports": ROOT / "output2" / "freqsel_old99_on_output2_40" / "reports",
        "note": "physics_highfreq_quota selected from old output 99 samples, tested on current output2 40 samples",
    },
    {
        "dataset": "output_old_xy",
        "method": "output2_physics40_top15",
        "reports": ROOT / "output" / "freqsel_output2_physics40_on_old99" / "reports",
        "note": "current output2 physics_highfreq_quota top15 transferred to old output 99 samples",
    },
    {
        "dataset": "output_old_xy",
        "method": "old99_physics_top15",
        "reports": ROOT / "output" / "freqsel_old99_physics_on_old99" / "reports",
        "note": "physics_highfreq_quota selected and tested on old output 99 samples",
    },
    {
        "dataset": "output_old_xy",
        "method": "old99_relative_l2_top15",
        "reports": ROOT / "output" / "freqsel_old99_relative_l2_on_old99" / "reports",
        "note": "relative_l2 selected and tested on old output 99 samples",
    },
    {
        "dataset": "output_old_xy",
        "method": "old99_v1_label_guided_top15",
        "reports": ROOT / "output" / "freqsel_old99_v1_label_guided_on_old99" / "reports",
        "note": "label-guided reference selected and tested on old output 99 samples",
    },
    {
        "dataset": "output_old_xy",
        "method": "old99_all_frequencies",
        "reports": ROOT / "output" / "freqsel_old99_all_frequencies_on_old99" / "reports",
        "note": "all valid frequency points on old output 99 samples",
    },
]

FREQUENCY_FILES = [
    {
        "name": "physics_highfreq_quota_40",
        "path": SIMPLE_ROOT
        / "f_domain"
        / "output2"
        / "frequency_selection_physics_highfreq_quota_40samples"
        / "physics_highfreq_quota_top15_frequencies.txt",
    },
    {
        "name": "relative_l2_40",
        "path": SIMPLE_ROOT
        / "f_domain"
        / "output2"
        / "frequency_selection_relative_l2_40samples"
        / "relative_l2_top15_frequencies.txt",
    },
    {
        "name": "v1_label_guided_40",
        "path": SIMPLE_ROOT
        / "f_domain"
        / "output2"
        / "frequency_selection_v1_label_guided_40samples"
        / "v1_label_guided_top15_frequencies.txt",
    },
    {
        "name": "old99_physics_top15",
        "path": SIMPLE_ROOT
        / "f_domain"
        / "output2"
        / "frequency_selection_physics_highfreq_quota_old99"
        / "physics_highfreq_quota_old99_top15_frequencies.txt",
    },
    {
        "name": "old99_relative_l2_top15",
        "path": SIMPLE_ROOT
        / "f_domain"
        / "output2"
        / "frequency_selection_relative_l2_old99"
        / "relative_l2_old99_top15_frequencies.txt",
    },
    {
        "name": "old99_v1_label_guided_top15",
        "path": SIMPLE_ROOT
        / "f_domain"
        / "output2"
        / "frequency_selection_v1_label_guided_old99"
        / "v1_label_guided_old99_top15_frequencies.txt",
    },
]


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def stats(values: list[float]) -> dict[str, float | int]:
    values = finite_values(values)
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
    }


def load_reports(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    reports = []
    for item in sorted(path.glob("*_metrics.json")):
        reports.append(json.loads(item.read_text(encoding="utf-8")))
    if not reports:
        raise RuntimeError(f"No metrics reports found in {path}")
    return reports


def aggregate_method(method: dict) -> list[dict[str, object]]:
    reports = load_reports(method["reports"])
    channels = sorted({channel for report in reports for channel in report["metrics"]})
    rows: list[dict[str, object]] = []
    for channel in channels:
        for metric in METRIC_NAMES:
            values = [
                float(report["metrics"][channel][metric])
                for report in reports
                if channel in report["metrics"] and metric in report["metrics"][channel]
            ]
            item = stats(values)
            rows.append(
                {
                    "dataset": method["dataset"],
                    "method": method["method"],
                    "channel": channel,
                    "metric": metric,
                    "n": item["n"],
                    "mean": item["mean"],
                    "std": item["std"],
                    "median": item["median"],
                    "note": method["note"],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_paper_wide_table(rows: list[dict[str, object]], path: Path) -> None:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["method"]), str(row["channel"]))
        item = grouped.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "sample_count": row["n"],
                "selection_method": row["method"],
                "coarse_map_channel": row["channel"],
                "note": row["note"],
            },
        )
        metric = str(row["metric"])
        item[f"{metric}_mean"] = row["mean"]
        item[f"{metric}_std"] = row["std"]
        item[f"{metric}_median"] = row["median"]

    method_order = {
        "physics_highfreq_quota_40": 10,
        "relative_l2_40": 20,
        "v1_label_guided_40": 30,
        "all_frequencies": 40,
        "old99_physics_top15": 50,
        "old99_relative_l2_top15": 60,
        "old99_v1_label_guided_top15": 70,
        "old99_all_frequencies": 80,
        "output2_physics40_top15": 90,
    }
    channel_order = {
        "ray_relative_delta": 10,
        "high_frequency_band_map": 20,
        "ray_phase_change": 30,
        "ray_log_amp_loss": 40,
        "ray_delta_abs": 50,
        "low_frequency_band_map": 60,
        "mid_frequency_band_map": 70,
        "path_coverage": 80,
        "valid_case_count": 90,
        "reliability_mask": 100,
    }
    table = sorted(
        grouped.values(),
        key=lambda item: (
            str(item["dataset"]),
            method_order.get(str(item["selection_method"]), 999),
            channel_order.get(str(item["coarse_map_channel"]), 999),
        ),
    )
    fieldnames = [
        "dataset",
        "sample_count",
        "selection_method",
        "coarse_map_channel",
        "pearson_mean",
        "pearson_std",
        "nrmse_mean",
        "nrmse_std",
        "mask_iou_mean",
        "mask_iou_std",
        "top5_hit_rate_mean",
        "top5_hit_rate_std",
        "prediction_mass_in_label_mean",
        "prediction_mass_in_label_std",
        "centroid_error_mm_mean",
        "centroid_error_mm_std",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in table:
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def write_key_metrics_table(
    rows: list[dict[str, object]],
    frequency_meta: dict[str, object],
    path: Path,
) -> None:
    metric_lookup = {
        (str(row["dataset"]), str(row["method"]), str(row["channel"]), str(row["metric"])): row
        for row in rows
    }
    methods = []
    seen = set()
    for row in rows:
        key = (str(row["dataset"]), str(row["method"]))
        if key in seen:
            continue
        seen.add(key)
        methods.append(key)

    frequency_key = {
        "physics_highfreq_quota_40": "physics_highfreq_quota_40",
        "relative_l2_40": "relative_l2_40",
        "v1_label_guided_40": "v1_label_guided_40",
        "old99_physics_top15": "old99_physics_top15",
        "old99_relative_l2_top15": "old99_relative_l2_top15",
        "old99_v1_label_guided_top15": "old99_v1_label_guided_top15",
        "output2_physics40_top15": "physics_highfreq_quota_40",
    }
    ordered = frequency_meta["ordered_khz"]

    method_order = {
        "physics_highfreq_quota_40": 10,
        "relative_l2_40": 20,
        "v1_label_guided_40": 30,
        "all_frequencies": 40,
        "old99_physics_top15": 50,
        "old99_relative_l2_top15": 60,
        "old99_v1_label_guided_top15": 70,
        "old99_all_frequencies": 80,
        "output2_physics40_top15": 90,
    }

    def value(dataset: str, method: str, channel: str, metric: str) -> object:
        row = metric_lookup.get((dataset, method, channel, metric))
        return "" if row is None else row["mean"]

    table: list[dict[str, object]] = []
    for dataset, method in sorted(methods, key=lambda item: (item[0], method_order.get(item[1], 999))):
        f_key = frequency_key.get(method)
        if f_key is None:
            frequency_text = "all_valid_frequencies" if "all_frequencies" in method else ""
        else:
            frequency_text = ",".join(f"{freq:g}" for freq in ordered[f_key])
        table.append(
            {
                "dataset": dataset,
                "selection_method": method,
                "selected_frequencies_khz": frequency_text,
                "ray_relative_delta_pearson": value(dataset, method, "ray_relative_delta", "pearson"),
                "ray_relative_delta_top5_hit_rate": value(dataset, method, "ray_relative_delta", "top5_hit_rate"),
                "ray_relative_delta_prediction_mass_in_label": value(
                    dataset,
                    method,
                    "ray_relative_delta",
                    "prediction_mass_in_label",
                ),
                "ray_relative_delta_centroid_error_mm": value(
                    dataset,
                    method,
                    "ray_relative_delta",
                    "centroid_error_mm",
                ),
                "high_frequency_band_map_pearson": value(dataset, method, "high_frequency_band_map", "pearson"),
                "high_frequency_band_map_top5_hit_rate": value(
                    dataset,
                    method,
                    "high_frequency_band_map",
                    "top5_hit_rate",
                ),
                "high_frequency_band_map_prediction_mass_in_label": value(
                    dataset,
                    method,
                    "high_frequency_band_map",
                    "prediction_mass_in_label",
                ),
                "high_frequency_band_map_centroid_error_mm": value(
                    dataset,
                    method,
                    "high_frequency_band_map",
                    "centroid_error_mm",
                ),
            }
        )
    write_csv(path, table)


def read_frequency_file(path: Path) -> list[float]:
    values = []
    text = path.read_text(encoding="utf-8").replace("\n", ",")
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token) / 1000.0)
    return values


def frequency_summary() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    sets: dict[str, set[float]] = {}
    ordered: dict[str, list[float]] = {}
    for item in FREQUENCY_FILES:
        values = read_frequency_file(item["path"])
        ordered[item["name"]] = values
        sets[item["name"]] = set(values)
        rows.append(
            {
                "name": item["name"],
                "count": len(values),
                "frequencies_khz": ",".join(f"{value:g}" for value in values),
            }
        )

    reference = sets["physics_highfreq_quota_40"]
    comparisons = {}
    for name, values in sets.items():
        if name == "physics_highfreq_quota_40":
            continue
        intersection = sorted(reference & values)
        union = sorted(reference | values)
        comparisons[name] = {
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union) if union else 0.0,
            "intersection_khz": intersection,
            "only_reference_khz": sorted(reference - values),
            "only_other_khz": sorted(values - reference),
        }
    return rows, {"ordered_khz": ordered, "comparisons_to_physics_highfreq_quota_40": comparisons}


def select_rows(rows: list[dict[str, object]], *, dataset: str, channel: str, metric: str) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["dataset"] == dataset and row["channel"] == channel and row["metric"] == metric
    ]


def fmt(value: object, digits: int = 4) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(lines)


def write_report(all_rows: list[dict[str, object]], frequency_meta: dict[str, object]) -> None:
    physics = frequency_meta["ordered_khz"]["physics_highfreq_quota_40"]
    old = frequency_meta["ordered_khz"]["old99_physics_top15"]
    intersection = frequency_meta["comparisons_to_physics_highfreq_quota_40"]["old99_physics_top15"][
        "intersection_khz"
    ]

    def compact_methods(dataset: str, channel: str) -> list[dict[str, str]]:
        methods = []
        for method in sorted({row["method"] for row in all_rows if row["dataset"] == dataset}):
            metric_lookup = {
                row["metric"]: row
                for row in all_rows
                if row["dataset"] == dataset and row["method"] == method and row["channel"] == channel
            }
            if not metric_lookup:
                continue
            methods.append(
                {
                    "method": method,
                    "pearson": fmt(metric_lookup["pearson"]["mean"]),
                    "IoU": fmt(metric_lookup["mask_iou"]["mean"]),
                    "top5_hit": fmt(metric_lookup["top5_hit_rate"]["mean"]),
                    "mass_in_label": fmt(metric_lookup["prediction_mass_in_label"]["mean"]),
                    "centroid_mm": fmt(metric_lookup["centroid_error_mm"]["mean"], 2),
                }
            )
        return methods

    lines = [
        "# physics_highfreq_quota 频域选频鲁棒性汇总",
        "",
        "## 频点集合",
        "",
        f"- output2 40 样本选出的 top15/kHz: {', '.join(f'{value:g}' for value in physics)}",
        f"- 旧 output 99 样本选出的 top15/kHz: {', '.join(f'{value:g}' for value in old)}",
        f"- 两者交集/kHz: {', '.join(f'{value:g}' for value in intersection)}",
        "",
        "## output2 当前 z 向模型，40 样本评价",
        "",
        "ray_relative_delta 通道:",
        "",
        markdown_table(compact_methods("output2_current_z", "ray_relative_delta")),
        "",
        "high_frequency_band_map 通道:",
        "",
        markdown_table(compact_methods("output2_current_z", "high_frequency_band_map")),
        "",
        "## 旧 output x/y 模型，99 样本同分布评价",
        "",
        "ray_relative_delta 通道:",
        "",
        markdown_table(compact_methods("output_old_xy", "ray_relative_delta")),
        "",
        "high_frequency_band_map 通道:",
        "",
        markdown_table(compact_methods("output_old_xy", "high_frequency_band_map")),
        "",
        "## 结论",
        "",
        "- `physics_highfreq_quota_40` 是当前 output2 40 样本上的主结果；`relative_l2_40`、`v1_label_guided_40` 和 `all_frequencies` 是对照。",
        "- 旧 output 99 样本与当前 output2 的频点集合并不完全一致，说明旧 x/y 载荷和接收方向数据不能与当前 z 向数据等权混合评分。",
        "- 判断 `physics_highfreq_quota` 方法是否鲁棒，应看旧 output 99 内部的同分布比较：`old99_physics_top15` 是否仍优于 `old99_relative_l2_top15`，并接近 `old99_v1_label_guided_top15`。",
        "- 旧 output 可以作为方法鲁棒性验证集；当前 z 向模型的最终频点仍应以 output2 40 样本为主。",
    ]
    (OUTPUT_DIR / "frequency_selection_robustness_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for method in METHODS:
        all_rows.extend(aggregate_method(method))

    write_csv(OUTPUT_DIR / "all_channel_metrics_summary.csv", all_rows)
    write_paper_wide_table(all_rows, OUTPUT_DIR / "paper_frequency_selection_metrics_wide.csv")

    focused = [
        row
        for row in all_rows
        if row["channel"] in FOCUS_CHANNELS
        and row["metric"] in {"pearson", "mask_iou", "top5_hit_rate", "prediction_mass_in_label", "centroid_error_mm"}
    ]
    write_csv(OUTPUT_DIR / "focused_metrics_summary.csv", focused)

    frequency_rows, frequency_meta = frequency_summary()
    write_csv(OUTPUT_DIR / "frequency_sets_summary.csv", frequency_rows)
    (OUTPUT_DIR / "frequency_sets_summary.json").write_text(
        json.dumps(frequency_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_key_metrics_table(
        all_rows,
        frequency_meta,
        OUTPUT_DIR / "paper_frequency_selection_key_metrics.csv",
    )
    write_report(all_rows, frequency_meta)

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
