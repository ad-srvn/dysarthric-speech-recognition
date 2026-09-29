#!/usr/bin/env python3
"""Create deterministic train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
from collections import defaultdict

from common import load_config, read_csv, setup_logging, write_csv


def seeded_rng(seed: int, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def speaker_dependent(rows: list[dict[str, str]], config: dict) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["speaker_id"], row["word"])].append(row)
    output: list[dict[str, str]] = []
    skipped = 0
    minimum = int(config.get("minimum_examples_per_speaker_word", 2))
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda row: row["wav_path"])
        if len(items) < minimum or len(items) < 3:
            skipped += len(items)
            continue
        seeded_rng(int(config["random_seed"]), "/".join(key)).shuffle(items)
        validation_count = max(1, round(len(items) * float(config["validation_ratio"])))
        test_count = max(1, round(len(items) * float(config["test_ratio"])))
        while validation_count + test_count > len(items) - 1:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        train_count = len(items) - validation_count - test_count
        for index, row in enumerate(items):
            updated = dict(row)
            if index < train_count:
                updated["split"] = "train"
            elif index < train_count + validation_count:
                updated["split"] = "validation"
            else:
                updated["split"] = "test"
            output.append(updated)
    logging.info("Skipped %d items from speaker/word cells with too few repetitions", skipped)
    return output


def speaker_independent(rows: list[dict[str, str]], config: dict) -> list[dict[str, str]]:
    speakers = sorted({row["speaker_id"] for row in rows})
    requested_test = [str(item).upper() for item in config.get("held_out_speakers", [])]
    requested_validation = [str(item).upper() for item in config.get("validation_speakers", [])]
    requested = set(requested_test + requested_validation)
    if requested:
        unknown = sorted(requested - set(speakers))
        if unknown:
            raise SystemExit(f"Configured held-out speakers not present: {', '.join(unknown)}")
        if set(requested_test) & set(requested_validation):
            raise SystemExit("A speaker cannot be in both validation_speakers and held_out_speakers")
        test_speakers = set(requested_test)
        validation_speakers = set(requested_validation)
    else:
        if len(speakers) < 3:
            raise SystemExit("Speaker-independent train/validation/test splitting requires at least 3 speakers")
        shuffled = speakers[:]
        seeded_rng(int(config["random_seed"]), "held-out-speakers").shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * float(config["test_ratio"])))
        validation_count = max(1, round(len(shuffled) * float(config["validation_ratio"])))
        while test_count + validation_count > len(shuffled) - 1:
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
        test_speakers = set(shuffled[:test_count])
        validation_speakers = set(shuffled[test_count:test_count + validation_count])
    output = []
    for row in rows:
        updated = dict(row)
        if row["speaker_id"] in test_speakers:
            updated["split"] = "test"
        elif row["speaker_id"] in validation_speakers:
            updated["split"] = "validation"
        else:
            updated["split"] = "train"
        output.append(updated)
    logging.info("Validation speakers: %s", ", ".join(sorted(validation_speakers)))
    logging.info("Test speakers: %s", ", ".join(sorted(test_speakers)))
    return output


def cross_group_split(rows: list[dict[str, str]], config: dict) -> list[dict[str, str]]:
    training = [row for row in rows if row.get("dataset_role") == "training"]
    testing = [row for row in rows if row.get("dataset_role") == "testing"]
    if not training or not testing:
        raise SystemExit("Cross-group evaluation requires both training and testing recordings")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in training:
        groups[(row["speaker_id"], row["word"])].append(row)

    output: list[dict[str, str]] = []
    skipped = 0
    minimum = max(2, int(config.get("minimum_examples_per_speaker_word", 2)))
    validation_ratio = float(config["validation_ratio"]) / (
        float(config["train_ratio"]) + float(config["validation_ratio"])
    )
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda row: row["wav_path"])
        if len(items) < minimum:
            skipped += len(items)
            continue
        seeded_rng(int(config["random_seed"]), "/".join(key)).shuffle(items)
        validation_count = min(len(items) - 1, max(1, round(len(items) * validation_ratio)))
        for index, row in enumerate(items):
            output.append({**row, "split": "validation" if index < validation_count else "train"})
    output.extend({**row, "split": "test"} for row in testing)
    logging.info("Skipped %d training items from cells with too few repetitions", skipped)
    return output


def validate(rows: list[dict[str, str]]) -> None:
    required_splits = {"train", "validation", "test"}
    present_splits = {row["split"] for row in rows}
    if not rows or not required_splits <= present_splits:
        raise SystemExit("Split must produce train, validation, and test recordings; adjust vocabulary/split settings.")
    identities: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        identities[row["wav_path"]].add(row["split"])
    leakage = [path for path, splits in identities.items() if len(splits) > 1]
    if leakage:
        raise RuntimeError(f"Recording leakage detected for {len(leakage)} files")
    train_words = {row["word"] for row in rows if row["split"] == "train"}
    evaluation_words = {row["word"] for row in rows if row["split"] in {"validation", "test"}}
    missing = sorted(evaluation_words - train_words)
    if missing:
        raise SystemExit("Validation/test words absent from training: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    rows = read_csv("metadata/selected.csv")
    if config.get("test_group"):
        output = cross_group_split(rows, config)
    else:
        output = speaker_dependent(rows, config) if config["experiment_type"] == "speaker-dependent" else speaker_independent(rows, config)
    validate(output)
    output.sort(key=lambda row: (row["split"], row["speaker_id"], row["word"], row["wav_path"]))
    counters: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in output:
        key = (row["split"], row["speaker_id"], row["word"])
        counters[key] += 1
        row["utterance_id"] = f"{row['speaker_id']}_{row['word']}_{counters[key]:04d}"
    write_csv("metadata/metadata.csv", output, list(rows[0].keys()) + ["split", "utterance_id"])
    counts = {split: sum(row["split"] == split for row in output) for split in ("train", "validation", "test")}
    logging.info("Wrote metadata/metadata.csv: %d train, %d validation, %d test", counts["train"], counts["validation"], counts["test"])


if __name__ == "__main__":
    main()
