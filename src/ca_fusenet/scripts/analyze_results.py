from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ca_fusenet.utils.class_mapping import ITWPolimiClassMapping


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid")

LOGGER = logging.getLogger("analyze_results")

MACRO_CATEGORY_COLORS: dict[str, str] = {
    "sitting": "#1f77b4",
    "standing": "#ff7f0e",
    "walking": "#2ca02c",
    "other": "#7f7f7f",
}

EXPERIMENT_COLORS: list[str] = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
]

GLOBAL_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy", "acc", "test_acc"),
    "balanced_accuracy": ("balanced_accuracy", "test_balanced_accuracy"),
    "macro_f1": ("macro_f1", "test_f1", "test_macro_f1"),
    "weighted_f1": ("weighted_f1", "test_weighted_f1"),
    "matthews_corrcoef": ("matthews_corrcoef", "test_matthews_corrcoef"),
}


@dataclass
class ExperimentData:
    path: Path
    name: str
    metrics: dict[str, Any] | None
    per_class: pd.DataFrame | None
    confusion_raw: np.ndarray | None


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "experiment"


def _save_figure(fig: plt.Figure, path: Path, generated_files: list[Path]) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(path)


def _get_metric_value(metrics: dict[str, Any] | None, keys: tuple[str, ...]) -> float:
    if metrics is None:
        return float("nan")
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float("nan")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        LOGGER.warning("Missing metrics file: %s", path)
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load metrics file %s (%s)", path, exc)
        return None
    if not isinstance(data, dict):
        LOGGER.warning("Metrics file is not a JSON object: %s", path)
        return None
    return data


def _read_per_class_csv(path: Path, class_ids: list[int], class_name_lookup: dict[int, str]) -> pd.DataFrame | None:
    if not path.exists():
        LOGGER.warning("Missing per-class CSV: %s", path)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load per-class CSV %s (%s)", path, exc)
        return None

    required_columns = {"class_index", "precision", "recall", "f1", "support"}
    if not required_columns.issubset(df.columns):
        LOGGER.warning(
            "Per-class CSV missing columns %s in %s",
            sorted(required_columns - set(df.columns)),
            path,
        )
        return None

    parsed = df.copy()
    parsed["class_index"] = pd.to_numeric(parsed["class_index"], errors="coerce").astype("Int64")
    parsed = parsed[parsed["class_index"].notna()].copy()
    parsed["class_index"] = parsed["class_index"].astype(int)
    parsed = parsed[parsed["class_index"].isin(class_ids)].copy()

    for col in ("precision", "recall", "f1", "support"):
        parsed[col] = pd.to_numeric(parsed[col], errors="coerce")
    parsed["precision"] = parsed["precision"].fillna(0.0).astype(float)
    parsed["recall"] = parsed["recall"].fillna(0.0).astype(float)
    parsed["f1"] = parsed["f1"].fillna(0.0).astype(float)
    parsed["support"] = parsed["support"].fillna(0.0).astype(float)

    parsed["class_name"] = parsed["class_index"].map(class_name_lookup).fillna(parsed["class_index"].astype(str))
    parsed = parsed.sort_values("class_index").reset_index(drop=True)
    return parsed


def _read_confusion_raw(path: Path) -> np.ndarray | None:
    if not path.exists():
        LOGGER.warning("Missing raw confusion matrix file: %s", path)
        return None
    try:
        matrix = np.load(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load raw confusion matrix %s (%s)", path, exc)
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        LOGGER.warning("Raw confusion matrix is not square in %s", path)
        return None
    return matrix


def _load_experiment(
    exp_dir: Path,
    exp_name: str,
    class_ids: list[int],
    class_name_lookup: dict[int, str],
) -> ExperimentData:
    metrics_path = exp_dir / "metrics_eval.json"
    per_class_path = exp_dir / "per_class_metrics.csv"
    confusion_raw_path = exp_dir / "confusion_matrix_raw.npy"

    metrics = _read_json(metrics_path)
    per_class = _read_per_class_csv(per_class_path, class_ids, class_name_lookup)
    confusion_raw = _read_confusion_raw(confusion_raw_path)
    return ExperimentData(
        path=exp_dir,
        name=exp_name,
        metrics=metrics,
        per_class=per_class,
        confusion_raw=confusion_raw,
    )


def _category_of_class(class_idx: int, category_lookup: dict[int, str]) -> str:
    return category_lookup.get(int(class_idx), "other")


def _plot_per_class_f1(
    exp: ExperimentData,
    output_dir: Path,
    category_lookup: dict[int, str],
    generated_files: list[Path],
) -> None:
    if exp.per_class is None:
        return

    data = exp.per_class.sort_values("f1", ascending=False).reset_index(drop=True)
    colors = [MACRO_CATEGORY_COLORS[_category_of_class(idx, category_lookup)] for idx in data["class_index"]]
    macro_f1 = _get_metric_value(exp.metrics, GLOBAL_METRIC_KEYS["macro_f1"])
    if np.isnan(macro_f1):
        macro_f1 = float(data["f1"].mean())

    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(data))
    ax.barh(y, data["f1"].values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(data["class_name"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("F1-score")
    ax.set_ylabel("Class")
    ax.set_title(f"{exp.name} - Per-Class F1")
    ax.axvline(macro_f1, linestyle="--", color="black", linewidth=1.2, label=f"Macro-F1={macro_f1:.3f}")

    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=label.title())
        for label, color in MACRO_CATEGORY_COLORS.items()
    ]
    handles.append(plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="Macro-F1"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    out_path = output_dir / f"{_slugify(exp.name)}_per_class_f1.png"
    _save_figure(fig, out_path, generated_files)


def _plot_precision_recall(
    exp: ExperimentData,
    output_dir: Path,
    generated_files: list[Path],
) -> None:
    if exp.per_class is None:
        return

    data = exp.per_class.sort_values("f1", ascending=False).reset_index(drop=True)
    y = np.arange(len(data))
    bar_height = 0.38

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(y - bar_height / 2, data["precision"].values, height=bar_height, label="Precision", color="#4c72b0")
    ax.barh(y + bar_height / 2, data["recall"].values, height=bar_height, label="Recall", color="#dd8452")
    ax.set_yticks(y)
    ax.set_yticklabels(data["class_name"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_ylabel("Class")
    ax.set_title(f"{exp.name} - Per-Class Precision vs Recall")
    ax.legend(loc="lower right")

    out_path = output_dir / f"{_slugify(exp.name)}_precision_recall.png"
    _save_figure(fig, out_path, generated_files)


def _plot_support_distribution(
    exp: ExperimentData,
    output_dir: Path,
    category_lookup: dict[int, str],
    generated_files: list[Path],
) -> None:
    if exp.per_class is None:
        return

    data = exp.per_class.sort_values("support", ascending=False).reset_index(drop=True)
    colors = [MACRO_CATEGORY_COLORS[_category_of_class(idx, category_lookup)] for idx in data["class_index"]]
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(y, data["support"].values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(data["class_name"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Support (samples)")
    ax.set_ylabel("Class")
    ax.set_title(f"{exp.name} - Class Support Distribution")

    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=label.title())
        for label, color in MACRO_CATEGORY_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    out_path = output_dir / f"{_slugify(exp.name)}_support_distribution.png"
    _save_figure(fig, out_path, generated_files)


def _plot_confusion_topk(
    exp: ExperimentData,
    output_dir: Path,
    top_k: int,
    class_name_lookup: dict[int, str],
    generated_files: list[Path],
) -> None:
    if exp.confusion_raw is None:
        return

    matrix = exp.confusion_raw
    if matrix.shape[0] == 0:
        LOGGER.warning("Empty confusion matrix for %s", exp.name)
        return

    support = matrix.sum(axis=1)
    k = min(max(1, int(top_k)), matrix.shape[0])
    top_indices = np.argsort(-support)[:k]
    reduced = matrix[np.ix_(top_indices, top_indices)].astype(float)
    row_sums = reduced.sum(axis=1, keepdims=True)
    reduced_norm = np.divide(reduced, row_sums, out=np.zeros_like(reduced), where=row_sums != 0)
    labels = [class_name_lookup.get(int(idx), str(int(idx))) for idx in top_indices]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        reduced_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{exp.name} - Reduced Confusion Matrix (Top {k} by Support)")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    out_path = output_dir / f"{_slugify(exp.name)}_confusion_matrix_top{k}.png"
    _save_figure(fig, out_path, generated_files)


def _plot_macro_category_metrics(
    exp: ExperimentData,
    output_dir: Path,
    macro_groups: dict[str, list[int]],
    generated_files: list[Path],
) -> None:
    if exp.per_class is None:
        return

    categories = ["sitting", "standing", "walking", "other"]
    df = exp.per_class.set_index("class_index")
    means_f1: list[float] = []
    means_precision: list[float] = []
    means_recall: list[float] = []

    for category in categories:
        class_ids = [idx for idx in macro_groups.get(category, []) if idx in df.index]
        if not class_ids:
            means_f1.append(0.0)
            means_precision.append(0.0)
            means_recall.append(0.0)
            continue
        subset = df.loc[class_ids]
        means_f1.append(float(subset["f1"].mean()))
        means_precision.append(float(subset["precision"].mean()))
        means_recall.append(float(subset["recall"].mean()))

    x = np.arange(len(categories))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width, means_f1, width=width, label="F1", color="#4c72b0")
    ax.bar(x, means_precision, width=width, label="Precision", color="#dd8452")
    ax.bar(x + width, means_recall, width=width, label="Recall", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels([cat.title() for cat in categories])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average score")
    ax.set_title(f"{exp.name} - Macro-Category Aggregated Metrics")
    ax.legend(loc="upper right")

    out_path = output_dir / f"{_slugify(exp.name)}_macro_category_metrics.png"
    _save_figure(fig, out_path, generated_files)


def _build_global_comparison_dataframe(experiments: list[ExperimentData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        row = {"experiment": exp.name}
        for target_key, aliases in GLOBAL_METRIC_KEYS.items():
            row[target_key] = _get_metric_value(exp.metrics, aliases)
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "experiment",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "matthews_corrcoef",
        ],
    )


def _plot_global_comparison_table(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    generated_files: list[Path],
) -> None:
    csv_path = output_dir / "comparison_global_metrics.csv"
    comparison_df.to_csv(csv_path, index=False)
    generated_files.append(csv_path)

    display_df = comparison_df.copy()
    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")

    fig_height = max(2.5, 0.6 * len(display_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("Global Metrics Comparison", pad=12)

    out_path = output_dir / "comparison_global_metrics.png"
    _save_figure(fig, out_path, generated_files)


def _plot_comparison_per_class_f1(
    experiments: list[ExperimentData],
    output_dir: Path,
    class_ids: list[int],
    class_name_lookup: dict[int, str],
    generated_files: list[Path],
) -> None:
    available = [exp for exp in experiments if exp.per_class is not None]
    if len(available) < 2:
        LOGGER.warning("Skipping comparative per-class F1: need at least 2 experiments with per_class_metrics.csv")
        return

    anchor = None
    for exp in reversed(experiments):
        if exp.per_class is not None:
            anchor = exp
            break
    if anchor is None:
        return

    anchor_lookup = anchor.per_class.set_index("class_index")["f1"].to_dict()
    sorted_class_ids = sorted(class_ids, key=lambda idx: anchor_lookup.get(idx, -1.0), reverse=True)
    class_labels = [class_name_lookup.get(idx, str(idx)) for idx in sorted_class_ids]

    y = np.arange(len(sorted_class_ids))
    n_experiments = len(available)
    total_height = 0.82
    bar_height = total_height / n_experiments
    start = -total_height / 2 + bar_height / 2

    fig, ax = plt.subplots(figsize=(14, 12))
    for i, exp in enumerate(available):
        series = exp.per_class.set_index("class_index")["f1"] if exp.per_class is not None else pd.Series(dtype=float)
        values = [float(series.get(idx, 0.0)) for idx in sorted_class_ids]
        ax.barh(
            y + start + i * bar_height,
            values,
            height=bar_height,
            color=EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)],
            label=exp.name,
            edgecolor="black",
            linewidth=0.2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("F1-score")
    ax.set_ylabel("Class")
    ax.set_title("Comparative Per-Class F1")
    ax.legend(loc="lower right")

    out_path = output_dir / "comparison_per_class_f1.png"
    _save_figure(fig, out_path, generated_files)


def _plot_comparison_macro_category_f1(
    experiments: list[ExperimentData],
    output_dir: Path,
    macro_groups: dict[str, list[int]],
    generated_files: list[Path],
) -> None:
    available = [exp for exp in experiments if exp.per_class is not None]
    if len(available) < 2:
        LOGGER.warning("Skipping comparative macro-category F1: need at least 2 experiments with per_class_metrics.csv")
        return

    categories = ["sitting", "standing", "walking", "other"]
    x = np.arange(len(categories))
    width = 0.82 / len(available)
    start = -0.41 + width / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, exp in enumerate(available):
        series = exp.per_class.set_index("class_index")["f1"] if exp.per_class is not None else pd.Series(dtype=float)
        means: list[float] = []
        for category in categories:
            indices = macro_groups.get(category, [])
            values = [float(series.get(idx, np.nan)) for idx in indices]
            values = [v for v in values if not np.isnan(v)]
            means.append(float(np.mean(values)) if values else 0.0)
        ax.bar(
            x + start + i * width,
            means,
            width=width,
            color=EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)],
            label=exp.name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([cat.title() for cat in categories])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average F1-score")
    ax.set_title("Comparative Macro-Category F1")
    ax.legend(loc="upper right")

    out_path = output_dir / "comparison_macro_category_f1.png"
    _save_figure(fig, out_path, generated_files)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved evaluation outputs and generate static plots.",
    )
    parser.add_argument(
        "--experiment_dirs",
        nargs="+",
        required=True,
        help="One or more experiment output directories.",
    )
    parser.add_argument(
        "--experiment_names",
        nargs="+",
        default=None,
        help="Optional display names for experiments, same order as --experiment_dirs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_output",
        help="Directory for generated analysis outputs.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=12,
        help="Number of classes to show in reduced confusion matrix.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    exp_dirs = [Path(path).expanduser().resolve() for path in args.experiment_dirs]
    if args.experiment_names is not None and len(args.experiment_names) != len(exp_dirs):
        raise ValueError("--experiment_names must match the number of --experiment_dirs")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k = max(1, int(args.top_k))

    mapping = ITWPolimiClassMapping()
    class_name_lookup = {int(idx): mapping.get_class_name(int(idx)) for idx in mapping.get_mapping().keys()}
    macro_groups = mapping.get_macro_categories()
    category_lookup = {
        int(idx): category
        for category, indices in macro_groups.items()
        for idx in indices
    }
    class_ids = sorted(class_name_lookup.keys())

    experiments: list[ExperimentData] = []
    for i, exp_dir in enumerate(exp_dirs):
        exp_name = args.experiment_names[i] if args.experiment_names else exp_dir.name
        if not exp_dir.exists():
            LOGGER.warning("Experiment directory does not exist: %s", exp_dir)
            experiments.append(
                ExperimentData(
                    path=exp_dir,
                    name=exp_name,
                    metrics=None,
                    per_class=None,
                    confusion_raw=None,
                )
            )
            continue
        experiments.append(_load_experiment(exp_dir, exp_name, class_ids, class_name_lookup))

    generated_files: list[Path] = []

    for exp in experiments:
        _plot_per_class_f1(exp, output_dir, category_lookup, generated_files)
        _plot_precision_recall(exp, output_dir, generated_files)
        _plot_support_distribution(exp, output_dir, category_lookup, generated_files)
        _plot_confusion_topk(exp, output_dir, top_k, class_name_lookup, generated_files)
        _plot_macro_category_metrics(exp, output_dir, macro_groups, generated_files)

    if len(experiments) > 1:
        comparison_df = _build_global_comparison_dataframe(experiments)
        _plot_global_comparison_table(comparison_df, output_dir, generated_files)
        _plot_comparison_per_class_f1(experiments, output_dir, class_ids, class_name_lookup, generated_files)
        _plot_comparison_macro_category_f1(experiments, output_dir, macro_groups, generated_files)

    print("Generated files:")
    if not generated_files:
        print("- none")
    else:
        for path in generated_files:
            print(f"- {path}")


if __name__ == "__main__":
    main()
