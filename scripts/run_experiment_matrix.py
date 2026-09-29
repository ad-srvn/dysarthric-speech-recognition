#!/usr/bin/env python3
"""Train and evaluate every configured speaker-group/model-size combination."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from common import (
    load_config,
    portable_path,
    preserve_path,
    project_path,
    setup_logging,
    unique_output_path,
)
from run_pipeline import archive_existing_results


def run_pipeline(config_path: Path, *arguments: str) -> None:
    command = [sys.executable, str(project_path("scripts/run_pipeline.py")), "--config", str(config_path), *arguments]
    print("$ " + " ".join(f"'{part}'" if " " in part else part for part in command), flush=True)
    subprocess.run(command, check=True)


def completed_run_matches(destination: Path, expected_config: dict) -> bool:
    try:
        saved_config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        int((destination / "selected_mixture.txt").read_text(encoding="ascii").strip())
        json.loads((destination / "test/experiment_summary.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return saved_config == expected_config


def clear_model_outputs() -> None:
    logs = project_path("results/logs")
    for pattern in ("hcompv.log", "herest_*.log", "hhed_*.log", "hvite_*.log", "hresults_*.log"):
        for path in logs.glob(pattern):
            path.unlink()
    for path in project_path("results").glob("mixture_*"):
        if path.is_dir():
            shutil.rmtree(path)


def archive_run(destination: Path, config: dict, model_name: str) -> dict[str, object]:
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {destination}")
    destination.mkdir(parents=True)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    selected_path = project_path("results/selected_mixture.txt")
    selected_mixture = int(selected_path.read_text(encoding="ascii").strip())
    (destination / "selected_mixture.txt").write_text(f"{selected_mixture}\n", encoding="ascii")

    test_source = project_path(f"results/mixture_{selected_mixture}/test")
    shutil.copytree(test_source, destination / "test")
    selected_model = project_path(f"models/final_mix_{selected_mixture}")
    if selected_model.is_dir():
        shutil.copytree(selected_model, destination / "selected_model")
    validation_root = destination / "validation"
    for mixture in range(1, int(config["target_mixtures"]) + 1):
        source = project_path(f"results/mixture_{mixture}/validation")
        if source.is_dir():
            shutil.copytree(source, validation_root / f"mixture_{mixture}")
    logs = project_path("results/logs")
    if logs.is_dir():
        shutil.copytree(logs, destination / "logs")

    return archive_summary(destination, config, model_name)


def archive_summary(destination: Path, config: dict, model_name: str) -> dict[str, object]:
    summary = json.loads((destination / "test/experiment_summary.json").read_text(encoding="utf-8"))
    selected = int((destination / "selected_mixture.txt").read_text(encoding="ascii").strip())
    return {
        "severity": config["severity_filter"], "model_size": model_name,
        "test_group": config.get("test_group", config["severity_filter"]),
        "hmm_states": config["num_hmm_states"], "emitting_states": config["num_emitting_states"],
        "maximum_mixtures": config["target_mixtures"], "selected_mixtures": selected,
        "selected_gaussian_components_per_word": int(config["num_emitting_states"]) * selected,
        "vocabulary_size": summary["vocabulary_size"], "speakers": summary["speakers"],
        "training_utterances": summary["training_utterances"],
        "validation_utterances": summary["validation_utterances"],
        "testing_utterances": summary["testing_utterances"], "accuracy": summary["accuracy"],
        "macro_word_accuracy": summary["macro_word_accuracy"], "status": "complete",
        "result_directory": portable_path(destination),
    }


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json", help="Base experiment configuration")
    parser.add_argument("--matrix-config", default="config/experiment_matrix.json")
    parser.add_argument("--output-root", default="results/experiment_matrix")
    parser.add_argument("--groups", "--severities", dest="severities", nargs="+", help="Optional speaker groups")
    parser.add_argument("--test-group", help="Use a different speaker group for the held-out test set")
    parser.add_argument("--models", nargs="+", help="Optional subset, e.g. small large")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    base = load_config(args.config)
    matrix = json.loads(project_path(args.matrix_config).read_text(encoding="utf-8"))
    severities = args.severities or matrix["severities"]
    model_names = args.models or list(matrix["models"])
    available_groups = matrix.get("available_groups", matrix["severities"])
    unknown_severities = sorted(set(severities) - set(available_groups))
    unknown_models = sorted(set(model_names) - set(matrix["models"]))
    unknown_test_group = args.test_group and args.test_group not in available_groups
    if unknown_severities or unknown_models or unknown_test_group:
        raise SystemExit(
            f"Unknown matrix selections; groups={unknown_severities}, "
            f"test_group={args.test_group if unknown_test_group else None}, models={unknown_models}"
        )

    requested_output = project_path(args.output_root)
    output_root = requested_output if args.skip_completed else unique_output_path(requested_output)
    if output_root != requested_output:
        logging.info("Keeping existing results; new matrix output: %s", output_root)
    if args.skip_completed:
        for previous in (output_root / "summary.csv", output_root / "plots"):
            preserved = preserve_path(previous)
            if preserved:
                logging.info("Preserved previous output as %s", preserved)
    runs_root = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    archive_existing_results()
    summaries: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="dysarthric-htk-matrix-") as temporary:
        temporary_root = Path(temporary)
        for severity in severities:
            severity_base = dict(base)
            severity_base["severity_filter"] = severity
            if args.test_group:
                severity_base["test_group"] = args.test_group
            else:
                severity_base.pop("test_group", None)
            runs: list[tuple[str, Path, dict]] = []
            for model_name in model_names:
                destination = runs_root / severity / model_name
                config = dict(severity_base)
                config.update(matrix["models"][model_name])
                if args.skip_completed and completed_run_matches(destination, config):
                    summaries.append(archive_summary(destination, config, model_name))
                    logging.info("Reused completed run %s/%s", severity, model_name)
                else:
                    runs.append((model_name, destination, config))

            if not runs:
                continue

            preparation_config = runs[0][2]
            preparation_path = temporary_root / f"{severity}_prepare.json"
            preparation_path.write_text(json.dumps(preparation_config, indent=2) + "\n", encoding="utf-8")
            print(f"\n##### PREPARE GROUP: {severity.upper()} #####", flush=True)
            run_pipeline(preparation_path, "--stop-after", "features")

            for model_name, destination, config in runs:
                config_path = temporary_root / f"{severity}_{model_name}.json"
                config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
                print(f"\n##### TRAIN {severity.upper()} / {model_name.upper()} #####", flush=True)
                preserved = preserve_path(destination)
                if preserved:
                    logging.info("Preserved incompatible run as %s", preserved)
                clear_model_outputs()
                try:
                    run_pipeline(config_path, "--start-at", "initialize")
                    summaries.append(archive_run(destination, config, model_name))
                except subprocess.CalledProcessError as exc:
                    logging.error("Run failed for %s/%s with exit code %d", severity, model_name, exc.returncode)
                    summaries.append({
                        "severity": severity, "model_size": model_name,
                        "test_group": config.get("test_group", severity),
                        "hmm_states": config["num_hmm_states"],
                        "emitting_states": config["num_emitting_states"],
                        "maximum_mixtures": config["target_mixtures"],
                        "selected_mixtures": "", "selected_gaussian_components_per_word": "",
                        "vocabulary_size": "", "speakers": "", "training_utterances": "",
                        "validation_utterances": "", "testing_utterances": "",
                        "accuracy": "", "macro_word_accuracy": "", "status": "failed",
                        "result_directory": portable_path(destination),
                    })
                write_summary(summaries, output_root / "summary.csv")

    write_summary(summaries, output_root / "summary.csv")
    subprocess.run([
        sys.executable, str(project_path("scripts/plot_experiment_matrix.py")),
        "--summary", str(output_root / "summary.csv"),
        "--output-dir", str(output_root / "plots"),
    ], check=True)
    completed = sum(row["status"] == "complete" for row in summaries)
    print(f"\nMatrix complete: {completed}/{len(summaries)} runs succeeded. Results: {output_root}")
    if completed != len(summaries):
        raise SystemExit("One or more matrix runs failed; see summary.csv and the run logs")


if __name__ == "__main__":
    main()
