#!/usr/bin/env python3
"""Run the complete TORGO-to-HTK experiment, stopping on the first failure."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

from common import HTK_TOOLS, load_config, project_path, require_tools, run_command, setup_logging, timestamped_path

STAGES = ("discover", "vocabulary", "split", "htk-files", "features", "initialize", "train", "decode", "evaluate")
RESULT_FILES = (
    "recognized.mlf", "hresults.txt", "predictions.csv", "per_speaker_accuracy.csv",
    "per_word_accuracy.csv", "confusion_matrix.csv", "confusion_matrix.png",
    "experiment_summary.txt", "experiment_summary.json", "selected_mixture.txt",
)


def python(script: str, *arguments: object) -> list[object]:
    return [sys.executable, project_path(f"scripts/{script}"), *arguments]


def archive_existing_results() -> None:
    results = project_path("results")
    files = [results / name for name in RESULT_FILES if (results / name).is_file()]
    mixture_dirs = [path for path in results.glob("mixture_*") if path.is_dir()]
    if not files and not mixture_dirs:
        return

    selected_file = results / "selected_mixture.txt"
    selected = selected_file.read_text(encoding="ascii").strip() if selected_file.is_file() else ""
    destination = timestamped_path(results / "history" / "run")
    destination.mkdir(parents=True)
    for path in files:
        path.rename(destination / path.name)
    for path in mixture_dirs:
        path.rename(destination / path.name)

    logs = results / "logs"
    if logs.is_dir():
        shutil.copytree(logs, destination / "logs")
        for path in logs.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    selected_model = project_path(f"models/final_mix_{selected}") if selected else None
    if selected_model and selected_model.is_dir():
        shutil.copytree(selected_model, destination / "selected_model")
    print(f"Archived previous standalone results to {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--start-at", choices=STAGES[:7], default="discover")
    parser.add_argument("--stop-after", choices=STAGES)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    config_path = project_path(args.config)
    config = load_config(config_path)
    require_tools(HTK_TOOLS)
    start_index = STAGES.index(args.start_at)
    if args.stop_after and STAGES.index(args.stop_after) < start_index:
        raise SystemExit("--stop-after cannot occur before --start-at")
    if args.start_at == "discover" and args.stop_after is None:
        archive_existing_results()

    commands: list[tuple[str, list[object]]] = [
        ("discover", python("prepare_dataset.py", "--config", config_path)),
        ("vocabulary", python("select_vocabulary.py", "--config", config_path)),
        ("split", python("make_splits.py", "--config", config_path)),
        ("htk-files", python("generate_htk_files.py", "--compile-grammar")),
        ("features", python("extract_features.py")),
        ("initialize", python("initialize_models.py", "--config", config_path)),
        ("train", python("train.py", "--config", config_path)),
    ]
    for stage, command in commands:
        if STAGES.index(stage) < start_index:
            continue
        print(f"\n=== {stage} ===", flush=True)
        run_command(command)
        if args.stop_after == stage:
            return

    target = int(config["target_mixtures"])
    if target > 3:
        print(f"Note: requested {target} mixtures; comparison outputs include every mixture from 1 through {target}.")
    validation_scores: dict[int, float] = {}
    for mixture in range(1, target + 1):
        output = project_path(f"results/mixture_{mixture}/validation")
        print(f"\n=== decode validation mixture {mixture} ===", flush=True)
        run_command(python("decode.py", "--mixture", mixture, "--split", "validation", "--output-dir", output))
        if args.stop_after == "decode" and mixture == target:
            return
        print(f"\n=== evaluate validation mixture {mixture} ===", flush=True)
        run_command(python("evaluate.py", "--config", config_path, "--mixture", mixture, "--split", "validation", "--output-dir", output))
        validation_scores[mixture] = float(json.loads((output / "experiment_summary.json").read_text())["accuracy"])

    best_mixture = max(validation_scores, key=lambda mixture: (validation_scores[mixture], -mixture))
    final = project_path(f"results/mixture_{best_mixture}/test")
    print(f"\n=== decode test with validation-selected mixture {best_mixture} ===", flush=True)
    run_command(python("decode.py", "--mixture", best_mixture, "--split", "test", "--output-dir", final))
    print(f"\n=== evaluate test with validation-selected mixture {best_mixture} ===", flush=True)
    run_command(python("evaluate.py", "--config", config_path, "--mixture", best_mixture, "--split", "test", "--output-dir", final))

    for name in RESULT_FILES[:-1]:
        shutil.copy2(final / name, project_path("results") / name)
    (project_path("results") / "selected_mixture.txt").write_text(
        f"{best_mixture}\n", encoding="ascii"
    )
    print(
        f"\nComplete. Validation selected {best_mixture} mixture(s); "
        f"held-out test results are in {project_path('results')}"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Pipeline stopped because a command failed with exit code {exc.returncode}: "
            f"{exc.cmd}"
        ) from None
