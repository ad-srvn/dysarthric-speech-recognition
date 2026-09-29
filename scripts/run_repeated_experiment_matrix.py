#!/usr/bin/env python3
"""Repeat the speaker-group/model matrix with different split seeds and average it."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from common import load_config, preserve_path, project_path, unique_output_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def completed_repeat_matches(
    repeat_root: Path,
    expected: dict,
    expected_count: int,
) -> bool:
    try:
        saved = json.loads((repeat_root / "repeat_config.json").read_text(encoding="utf-8"))
        rows = read_csv(repeat_root / "summary.csv")
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return saved == expected and len(rows) == expected_count and all(
        row.get("status") == "complete" for row in rows
    )


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            grouped[(str(row["severity"]), str(row["model_size"]))].append(row)

    output: list[dict[str, object]] = []
    for (severity, model_size), items in sorted(grouped.items()):
        accuracies = [float(row["accuracy"]) for row in items]
        macro_accuracies = [float(row["macro_word_accuracy"]) for row in items]
        mixtures = [float(row["selected_mixtures"]) for row in items]
        output.append({
            "severity": severity,
            "model_size": model_size,
            "test_group": items[0].get("test_group", severity),
            "successful_repeats": len(items),
            "mean_accuracy": statistics.fmean(accuracies),
            "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "mean_macro_word_accuracy": statistics.fmean(macro_accuracies),
            "std_macro_word_accuracy": statistics.stdev(macro_accuracies) if len(macro_accuracies) > 1 else 0.0,
            "mean_selected_mixtures": statistics.fmean(mixtures),
            "hmm_states": items[0]["hmm_states"],
            "emitting_states": items[0]["emitting_states"],
            "maximum_mixtures": items[0]["maximum_mixtures"],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--matrix-config", default="config/experiment_matrix.json")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--output-root", default="results/repeated_experiment_matrix")
    parser.add_argument("--groups", "--severities", dest="severities", nargs="+")
    parser.add_argument("--test-group")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2 for an averaged experiment")
    if args.seed_step < 1:
        raise SystemExit("--seed-step must be positive")

    base = load_config(args.config)
    matrix_definition = json.loads(project_path(args.matrix_config).read_text(encoding="utf-8"))
    expected_severities = args.severities or matrix_definition["severities"]
    expected_models = args.models or list(matrix_definition["models"])
    base_seed = int(base["random_seed"])
    requested_output = project_path(args.output_root)
    output_root = requested_output if args.skip_completed else unique_output_path(requested_output)
    if output_root != requested_output:
        print(f"Keeping existing results; new repeated output: {output_root}", flush=True)
    if args.skip_completed:
        for previous in (output_root / "all_runs.csv", output_root / "aggregate.csv", output_root / "plots"):
            preserved = preserve_path(previous)
            if preserved:
                print(f"Preserved previous output as {preserved}", flush=True)
    repetitions_root = output_root / "repetitions"
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="dysarthric-htk-repeat-") as temporary:
        temporary_root = Path(temporary)
        for repeat_index in range(1, args.repeats + 1):
            seed = base_seed + (repeat_index - 1) * args.seed_step
            repeat_name = f"repeat_{repeat_index:02d}"
            repeat_root = repetitions_root / repeat_name
            summary_path = repeat_root / "summary.csv"
            repeat_config = dict(base)
            repeat_config["random_seed"] = seed
            repeat_fingerprint = {
                "experiment": repeat_config,
                "matrix": matrix_definition,
                "severities": expected_severities,
                "models": expected_models,
                "test_group": args.test_group,
            }
            expected_count = len(expected_severities) * len(expected_models)

            if args.skip_completed and completed_repeat_matches(
                repeat_root, repeat_fingerprint, expected_count
            ):
                repeat_rows = read_csv(summary_path)
                print(f"Skipping completed {repeat_name} (seed {seed})", flush=True)
                for row in repeat_rows:
                    all_rows.append({"repeat": repeat_index, "random_seed": seed, **row})
                continue

            config_path = temporary_root / f"{repeat_name}.json"
            config_path.write_text(json.dumps(repeat_config, indent=2) + "\n", encoding="utf-8")
            preserved = preserve_path(repeat_root)
            if preserved:
                print(f"Preserved incomplete or incompatible repeat as {preserved}", flush=True)

            command = [
                sys.executable, str(project_path("scripts/run_experiment_matrix.py")),
                "--config", str(config_path), "--matrix-config", str(project_path(args.matrix_config)),
                "--output-root", str(repeat_root),
            ]
            if args.severities:
                command.extend(("--groups", *args.severities))
            if args.models:
                command.extend(("--models", *args.models))
            if args.test_group:
                command.extend(("--test-group", args.test_group))
            print(f"\n########## {repeat_name.upper()} / SEED {seed} ##########", flush=True)
            subprocess.run(command, check=True)

            (repeat_root / "repeat_config.json").write_text(
                json.dumps(repeat_fingerprint, indent=2) + "\n", encoding="utf-8"
            )

            repeat_rows = read_csv(summary_path)
            for row in repeat_rows:
                all_rows.append({"repeat": repeat_index, "random_seed": seed, **row})
            write_csv(output_root / "all_runs.csv", all_rows)
            write_csv(output_root / "aggregate.csv", aggregate(all_rows))

    aggregate_rows = aggregate(all_rows)
    write_csv(output_root / "all_runs.csv", all_rows)
    write_csv(output_root / "aggregate.csv", aggregate_rows)
    aggregate_lookup = {(row["severity"], row["model_size"]): row for row in aggregate_rows}
    incomplete = []
    for severity in expected_severities:
        for model in expected_models:
            row = aggregate_lookup.get((severity, model))
            count = int(row["successful_repeats"]) if row else 0
            if count != args.repeats:
                incomplete.append(f"{severity}/{model} ({count}/{args.repeats})")
    if incomplete:
        raise SystemExit("Some averages contain fewer than the requested repeats: " + ", ".join(incomplete))

    subprocess.run([
        sys.executable, str(project_path("scripts/plot_repeated_experiment_matrix.py")),
        "--all-runs", str(output_root / "all_runs.csv"),
        "--aggregate", str(output_root / "aggregate.csv"),
        "--output-dir", str(output_root / "plots"),
    ], check=True)
    print(f"\nCompleted {args.repeats} repetitions. Averaged results: {output_root}")


if __name__ == "__main__":
    main()
