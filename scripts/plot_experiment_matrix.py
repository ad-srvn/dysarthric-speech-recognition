#!/usr/bin/env python3
"""Create aggregate plots for the speaker-group/model-size experiment matrix."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/experiment_matrix/summary.csv")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary_path = project_path(args.summary)
    if not summary_path.is_file():
        raise SystemExit(f"Matrix summary not found: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "complete"]
    if not rows:
        raise SystemExit("No completed matrix runs are available to plot")

    output = project_path(args.output_dir) if args.output_dir else summary_path.parent / "plots"
    output.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    severities = [name for name in SEVERITY_ORDER if any(row["severity"] == name for row in rows)]
    models = [name for name in MODEL_ORDER if any(row["model_size"] == name for row in rows)]
    lookup = {(row["severity"], row["model_size"]): row for row in rows}
    max_components = max(int(row["selected_gaussian_components_per_word"]) for row in rows)
    test_groups = {row.get("test_group", row["severity"]) for row in rows}
    test_note = f" — tested on {group_label(next(iter(test_groups)))}" if len(test_groups) == 1 else ""

    width = 0.8 / max(1, len(models))
    x = np.arange(len(severities))
    fig, axis = plt.subplots(figsize=(10, 6))
    for index, model in enumerate(models):
        values = [float(lookup[(severity, model)]["accuracy"]) * 100 if (severity, model) in lookup else np.nan for severity in severities]
        positions = x - 0.4 + width / 2 + index * width
        bars = axis.bar(positions, values, width, label=model.title())
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_xticks(x, [group_label(name) for name in severities])
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("HTK recognition accuracy by training group and model size" + test_note)
    axis.set_ylim(0, 105)
    axis.legend(title="Model")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "test_accuracy_by_model.png", dpi=180)
    plt.close(fig)

    matrix = np.full((len(severities), len(models)), np.nan)
    for row_index, severity in enumerate(severities):
        for column_index, model in enumerate(models):
            if (severity, model) in lookup:
                matrix[row_index, column_index] = float(lookup[(severity, model)]["accuracy"]) * 100
    fig, axis = plt.subplots(figsize=(8, 5.5))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    fig.colorbar(image, ax=axis, label="Test accuracy (%)")
    axis.set_xticks(range(len(models)), [name.title() for name in models])
    axis.set_yticks(range(len(severities)), [group_label(name) for name in severities])
    axis.set_xlabel("Model size")
    axis.set_ylabel("Speaker group")
    axis.set_title("Test accuracy heatmap" + test_note)
    for row_index in range(len(severities)):
        for column_index in range(len(models)):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                axis.text(column_index, row_index, f"{value:.1f}%", ha="center", va="center", color="white" if value > 55 else "black")
    fig.tight_layout()
    fig.savefig(output / "test_accuracy_heatmap.png", dpi=180)
    plt.close(fig)

    plot_rows = math.ceil(len(severities) / 2)
    fig, axes = plt.subplots(plot_rows, 2, figsize=(11, 4 * plot_rows), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    for axis, severity in zip(axes_flat, severities):
        severity_rows = [lookup[(severity, model)] for model in models if (severity, model) in lookup]
        components = [int(row["selected_gaussian_components_per_word"]) for row in severity_rows]
        accuracy = [float(row["accuracy"]) * 100 for row in severity_rows]
        axis.plot(components, accuracy, marker="o", linewidth=2)
        for row, component, score in zip(severity_rows, components, accuracy):
            offset = (-42, 7) if row["model_size"] == "small" else (6, 7)
            axis.annotate(
                f"{row['model_size'].title()}\n{score:.1f}%", (component, score),
                xytext=offset, textcoords="offset points", fontsize=8,
            )
        axis.set_title(group_label(severity))
        axis.set_xlim(0, max_components * 1.1)
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.25)
        axis.set_ylabel("Test accuracy (%)")
        axis.set_xlabel("Selected Gaussian components per word")
    for axis in axes_flat[len(severities):]:
        axis.set_visible(False)
    fig.suptitle("Model complexity versus test accuracy" + test_note)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "complexity_vs_accuracy.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    severity_positions = np.arange(len(severities))
    for index, model in enumerate(models):
        values = [int(lookup[(severity, model)]["selected_mixtures"]) if (severity, model) in lookup else 0 for severity in severities]
        positions = severity_positions - 0.4 + width / 2 + index * width
        bars = axis.bar(positions, values, width, label=model.title())
        axis.bar_label(bars, padding=3, fontsize=8)
    axis.set_xticks(severity_positions)
    axis.set_xticklabels([group_label(name) for name in severities])
    axis.set_xlim(-0.6, len(severities) - 0.4)
    axis.set_ylabel("Validation-selected mixtures per emitting state")
    axis.set_title("Selected mixture count")
    axis.legend(title="Model")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "selected_mixtures.png", dpi=180)
    plt.close(fig)

    print(f"Wrote aggregate plots to {output}")


if __name__ == "__main__":
    main()
