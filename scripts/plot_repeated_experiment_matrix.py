#!/usr/bin/env python3
"""Plot mean and distribution metrics from repeated matrix experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from common import configure_matplotlib, project_path


SEVERITY_ORDER = ("mild", "medium", "severe", "all", "control", "combined")
MODEL_ORDER = ("small", "medium", "large")
GROUP_LABELS = {
    "all": "All dysarthric",
    "control": "Controls",
    "combined": "Dysarthric + controls",
}


def group_label(name: str) -> str:
    return GROUP_LABELS.get(name, name.title())


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-runs", default="results/repeated_experiment_matrix/all_runs.csv")
    parser.add_argument("--aggregate", default="results/repeated_experiment_matrix/aggregate.csv")
    parser.add_argument("--output-dir", default="results/repeated_experiment_matrix/plots")
    args = parser.parse_args()

    all_rows = [row for row in load_rows(project_path(args.all_runs)) if row["status"] == "complete"]
    aggregate_rows = load_rows(project_path(args.aggregate))
    if not all_rows or not aggregate_rows:
        raise SystemExit("No completed repeated experiments are available to plot")

    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    severities = [name for name in SEVERITY_ORDER if any(row["severity"] == name for row in aggregate_rows)]
    models = [name for name in MODEL_ORDER if any(row["model_size"] == name for row in aggregate_rows)]
    aggregate = {(row["severity"], row["model_size"]): row for row in aggregate_rows}
    test_groups = {row.get("test_group", row["severity"]) for row in aggregate_rows}
    test_note = f" — tested on {group_label(next(iter(test_groups)))}" if len(test_groups) == 1 else ""
    runs: dict[tuple[str, str], list[float]] = {}
    for severity in severities:
        for model in models:
            runs[(severity, model)] = [
                float(row["accuracy"]) * 100 for row in all_rows
                if row["severity"] == severity and row["model_size"] == model
            ]
    observed_max = max(value for samples in runs.values() for value in samples)
    axis_max = min(100, max(10, math.ceil((observed_max + 5) / 10) * 10))

    x = np.arange(len(severities))
    width = 0.8 / max(1, len(models))
    fig, axis = plt.subplots(figsize=(10, 6))
    for index, model in enumerate(models):
        means = [
            float(aggregate[(severity, model)]["mean_accuracy"]) * 100
            if (severity, model) in aggregate else np.nan
            for severity in severities
        ]
        stds = [
            float(aggregate[(severity, model)]["std_accuracy"]) * 100
            if (severity, model) in aggregate else 0.0
            for severity in severities
        ]
        positions = x - 0.4 + width / 2 + index * width
        bars = axis.bar(positions, means, width, yerr=stds, capsize=4, label=model.title())
        axis.bar_label(bars, fmt="%.1f", padding=5, fontsize=8)
    axis.set_xticks(x, [group_label(severity) for severity in severities])
    axis.set_ylabel("Mean test accuracy (%)")
    axis.set_title("Mean HTK accuracy across repeated splits (±1 SD)" + test_note)
    axis.set_ylim(0, axis_max)
    axis.legend(title="Model")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mean_test_accuracy.png", dpi=180)
    plt.close(fig)

    matrix = np.full((len(severities), len(models)), np.nan)
    for row_index, severity in enumerate(severities):
        for column_index, model in enumerate(models):
            if (severity, model) in aggregate:
                matrix[row_index, column_index] = float(
                    aggregate[(severity, model)]["mean_accuracy"]
                ) * 100
    fig, axis = plt.subplots(figsize=(8, 5.5))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=axis_max, aspect="auto")
    fig.colorbar(image, ax=axis, label="Mean test accuracy (%)")
    axis.set_xticks(range(len(models)), [model.title() for model in models])
    axis.set_yticks(range(len(severities)), [group_label(severity) for severity in severities])
    axis.set_xlabel("Model size")
    axis.set_ylabel("Speaker group")
    axis.set_title("Mean test accuracy across repeated splits" + test_note)
    for row_index in range(len(severities)):
        for column_index in range(len(models)):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                axis.text(column_index, row_index, f"{value:.1f}%", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output / "mean_accuracy_heatmap.png", dpi=180)
    plt.close(fig)

    plot_rows = math.ceil(len(severities) / 2)
    fig, axes = plt.subplots(plot_rows, 2, figsize=(11, 4 * plot_rows), sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    for axis, severity in zip(axes_flat, severities):
        values = [runs[(severity, model)] for model in models]
        axis.boxplot(values, tick_labels=[model.title() for model in models], showmeans=True)
        for model_index, samples in enumerate(values, start=1):
            axis.scatter([model_index] * len(samples), samples, alpha=0.55, s=18)
        axis.set_title(group_label(severity))
        axis.set_ylim(0, axis_max)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylabel("Test accuracy (%)")
    for axis in axes_flat[len(severities):]:
        axis.set_visible(False)
    fig.suptitle("Accuracy distribution across repeated splits" + test_note)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "accuracy_distribution.png", dpi=180)
    plt.close(fig)

    print(f"Wrote repeated-experiment plots to {output}")


if __name__ == "__main__":
    main()
