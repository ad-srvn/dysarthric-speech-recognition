#!/usr/bin/env python3
"""Evaluate one-word MLF output and write metrics, confusion matrix, and summary."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from common import configure_matplotlib, load_config, project_path, read_csv, require_file, setup_logging, write_csv


def parse_mlf(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line == "#!MLF!#":
            continue
        if line.startswith('"') and line.endswith('"'):
            current = Path(line.strip('"')).stem
            continue
        if line == ".":
            current = None
            continue
        if current is None:
            continue
        parts = line.split()
        token = parts[2] if len(parts) >= 3 and parts[0].lstrip("-").isdigit() else parts[0]
        if token not in {"!NULL", "sil", "sp"}:
            labels[current] = token.upper()
    return labels


def accuracy_rows(predictions: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in predictions:
        grouped[str(row[key])].append(bool(row["correct"]))
    return [
        {key: value, "correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
        for value, values in sorted(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--mixture", type=int, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    output = project_path(args.output_dir or f"results/mixture_{args.mixture}/{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    references = parse_mlf(require_file(f"labels/{args.split}.mlf"))
    recognized = parse_mlf(require_file(output / "recognized.mlf", "recognized MLF"))
    metadata = {row["utterance_id"]: row for row in read_csv("metadata/metadata.csv") if row["split"] == args.split}
    predictions: list[dict[str, object]] = []
    for utterance_id, reference in sorted(references.items()):
        row = metadata.get(utterance_id)
        if row is None:
            raise SystemExit(f"Reference utterance {utterance_id} is absent from {args.split} metadata")
        prediction = recognized.get(utterance_id, "<MISSING>")
        predictions.append({
            "speaker": row["speaker_id"], "file": utterance_id,
            "reference": reference, "prediction": prediction,
            "correct": reference == prediction,
        })
    if not predictions:
        raise SystemExit(f"No {args.split} utterances could be parsed from labels/{args.split}.mlf")
    write_csv(output / "predictions.csv", predictions, ("speaker", "file", "reference", "prediction", "correct"))
    speaker_metrics = accuracy_rows(predictions, "speaker")
    word_metrics = accuracy_rows(predictions, "reference")
    write_csv(output / "per_speaker_accuracy.csv", speaker_metrics, ("speaker", "correct", "total", "accuracy"))
    write_csv(output / "per_word_accuracy.csv", word_metrics, ("reference", "correct", "total", "accuracy"))

    vocabulary = [line.strip() for line in require_file("lists/words.list").read_text().splitlines() if line.strip()]
    labels = vocabulary + (["<MISSING>"] if any(row["prediction"] == "<MISSING>" for row in predictions) else [])
    counts = Counter((str(row["reference"]), str(row["prediction"])) for row in predictions)
    matrix_rows = []
    for reference in vocabulary:
        matrix_rows.append({"reference": reference, **{prediction: counts[(reference, prediction)] for prediction in labels}})
    write_csv(output / "confusion_matrix.csv", matrix_rows, ("reference", *labels))

    configure_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = [[counts[(reference, prediction)] for prediction in labels] for reference in vocabulary]
    size = max(7.0, min(16.0, 0.55 * len(labels) + 4))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(vocabulary)), labels=vocabulary)
    axis.set_xlabel("Predicted word")
    axis.set_ylabel("Reference word")
    axis.set_title(f"TORGO isolated-word confusion matrix ({args.mixture} mixtures)")
    threshold = max((max(row) for row in matrix), default=0) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center", color="white" if value > threshold else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    correct = sum(bool(row["correct"]) for row in predictions)
    overall = correct / len(predictions)
    macro = sum(float(row["accuracy"]) for row in word_metrics) / len(word_metrics)
    all_metadata = read_csv("metadata/metadata.csv")
    group = str(config.get("severity_filter", "all"))
    test_group = str(config.get("test_group", group))
    speech_types = {
        "all": "all dysarthric isolated monosyllabic words",
        "control": "control isolated monosyllabic words",
        "combined": "dysarthric and control isolated monosyllabic words",
    }
    if test_group != group:
        speech_type = f"trained on {group}; evaluated on {test_group} isolated monosyllabic words"
    else:
        speech_type = speech_types.get(group, f"{group} dysarthric isolated monosyllabic words")
    training_speakers = {row["speaker_id"] for row in all_metadata if row["split"] == "train"}
    testing_speakers = {row["speaker_id"] for row in all_metadata if row["split"] == "test"}
    summary = {
        "dataset": "TORGO",
        "speech_type": speech_type,
        "speaker_group": group,
        "training_group": group,
        "test_group": test_group,
        "severity_filter": group,
        "evaluation_split": args.split,
        "experiment_type": config["experiment_type"],
        "vocabulary_size": len(vocabulary),
        "speakers": len({row["speaker_id"] for row in all_metadata}),
        "training_speakers": len(training_speakers),
        "testing_speakers": len(testing_speakers),
        "training_utterances": sum(row["split"] == "train" for row in all_metadata),
        "validation_utterances": sum(row["split"] == "validation" for row in all_metadata),
        "testing_utterances": sum(row["split"] == "test" for row in all_metadata),
        "evaluated_utterances": len(predictions),
        "feature_kind": "MFCC_0_D_A",
        "feature_dimension": 39,
        "window_ms": 25,
        "frame_shift_ms": 10,
        "hmm_states": int(config["num_hmm_states"]),
        "emitting_states": int(config["num_emitting_states"]),
        "gaussian_mixtures_per_emitting_state": args.mixture,
        "training_algorithm": "Baum-Welch / HERest",
        "decoding_algorithm": "Viterbi / HVite",
        "correct": correct,
        "accuracy": overall,
        "macro_word_accuracy": macro,
    }
    (output / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text = (
        f"Dataset: TORGO\nSpeech type: {speech_type}\n"
        f"Training group: {group}\nTest group: {test_group}\nEvaluation split: {args.split}\n"
        f"Experiment: {config['experiment_type']}\nVocabulary size: {len(vocabulary)}\n"
        f"Speakers: {summary['speakers']}\nTraining utterances: {summary['training_utterances']}\n"
        f"Validation utterances: {summary['validation_utterances']}\n"
        f"Testing utterances: {summary['testing_utterances']}\nEvaluated utterances: {len(predictions)}\n\n"
        f"Feature:\nMFCC_0_D_A\n39 dimensions\n"
        f"25 ms window\n10 ms frame shift\n\nModel:\n{config['num_hmm_states']}-state left-to-right GMM-HMM\n"
        f"{config['num_emitting_states']} emitting states\n"
        f"{args.mixture} Gaussian mixtures per emitting state\n\n"
        f"Training:\nBaum-Welch / HERest\n\nDecoding:\nViterbi / HVite\n\n"
        f"Accuracy: {overall:.2%}\nMacro word accuracy: {macro:.2%}\n"
    )
    (output / "experiment_summary.txt").write_text(text, encoding="utf-8")
    logging.info("Mixture %d %s accuracy: %.2f%% (macro %.2f%%)", args.mixture, args.split, overall * 100, macro * 100)


if __name__ == "__main__":
    main()
