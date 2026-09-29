#!/usr/bin/env python3
"""Select a data-driven monosyllabic vocabulary using CMUdict."""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict

import pronouncing

from common import load_config, project_path, read_csv, setup_logging, write_csv

GROUPS = ("mild", "medium", "severe", "all", "control", "combined")


def matches_group(row: dict[str, str], group: str, speakers: set[str]) -> bool:
    if row["isolated_word"] != "true":
        return False
    if group == "control":
        return row["dysarthric"] == "false"
    if group == "combined":
        return True
    if row["dysarthric"] != "true":
        return False
    return group == "all" or row["speaker_id"] in speakers


def is_monosyllabic(word: str) -> tuple[bool, str]:
    pronunciations = pronouncing.phones_for_word(word.lower())
    one_syllable = [phones for phones in pronunciations if pronouncing.syllable_count(phones) == 1]
    return bool(one_syllable), one_syllable[0] if one_syllable else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--vocabulary-size", type=int)
    parser.add_argument("--minimum-examples", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    size = args.vocabulary_size or int(config["vocabulary_size"])
    minimum = args.minimum_examples or int(config.get("minimum_word_examples", 6))
    group = str(config.get("severity_filter", "all")).lower()
    test_group = str(config.get("test_group", "")).lower() or None
    if group not in GROUPS:
        raise SystemExit(f"Unknown speaker group {group!r}; choose one of: {', '.join(GROUPS)}")
    if test_group and test_group not in GROUPS:
        raise SystemExit(f"Unknown test group {test_group!r}; choose one of: {', '.join(GROUPS)}")
    group_speakers = {
        str(speaker).upper()
        for speaker in config.get("severity_speakers", {}).get(group, [])
    }
    if group in {"mild", "medium", "severe"} and not group_speakers:
        raise SystemExit(f"No speakers configured for group={group!r}")
    minimum_duration = float(config.get("minimum_audio_seconds", 0.25))
    maximum_duration = float(config.get("maximum_audio_seconds", 15.0))
    discovered = read_csv("metadata/discovered.csv")
    candidates = [
        row for row in discovered
        if matches_group(row, group, group_speakers)
    ]
    test_speakers = {
        str(speaker).upper()
        for speaker in config.get("severity_speakers", {}).get(test_group or "", [])
    }
    test_candidates = [
        row for row in discovered
        if test_group and matches_group(row, test_group, test_speakers)
    ]
    if test_group and not test_candidates:
        raise SystemExit(
            f"No isolated-word recordings matched test_group={test_group!r}. "
            "Install the corresponding TORGO speaker folders under data/."
        )
    training_ids = {row["speaker_id"] for row in candidates}
    testing_ids = {row["speaker_id"] for row in test_candidates}
    overlap = sorted(training_ids & testing_ids)
    if overlap:
        raise SystemExit(
            "Training and test groups overlap for speakers: " + ", ".join(overlap)
        )

    rows, test_rows = [], []
    rejected_duration = 0
    for source, destination in ((candidates, rows), (test_candidates, test_rows)):
        for row in source:
            try:
                duration = float(row.get("duration_seconds", ""))
            except ValueError:
                duration = 0.0
            if minimum_duration <= duration <= maximum_duration:
                destination.append(row)
            else:
                rejected_duration += 1
    logging.info(
        "Rejected %d recordings outside the %.2f-%.2f second isolated-word range",
        rejected_duration, minimum_duration, maximum_duration,
    )
    if not rows:
        raise SystemExit(
            f"No isolated-word recordings matched group={group!r}. "
            "Make sure the corresponding TORGO speaker folders are installed under data/."
        )
    totals = Counter(row["word"] for row in rows)
    test_totals = Counter(row["word"] for row in test_rows)
    by_speaker: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_speaker[row["word"]][row["speaker_id"]] += 1

    report: list[dict[str, object]] = []
    eligible: list[tuple[str, int, int]] = []
    cell_minimum = int(config.get("minimum_examples_per_speaker_word", 2))
    for word in sorted(totals):
        mono, pronunciation = is_monosyllabic(word)
        speaker_counts = by_speaker[word]
        repeated_speakers = sum(count >= cell_minimum for count in speaker_counts.values())
        usable_examples = sum(count for count in speaker_counts.values() if count >= cell_minimum)
        enough_data = totals[word] >= minimum
        if config["experiment_type"] == "speaker-dependent":
            enough_data = usable_examples >= minimum and repeated_speakers > 0
        if test_group:
            enough_data = enough_data and test_totals[word] >= int(
                config.get("minimum_test_word_examples", 1)
            )
        report.append({
            "word": word,
            "total_examples": totals[word],
            "speaker_count": len(speaker_counts),
            "usable_examples": usable_examples,
            "speakers_with_repetitions": repeated_speakers,
            "cmu_pronunciation": pronunciation,
            "monosyllabic": str(mono).lower(),
            "eligible": str(mono and enough_data).lower(),
        })
        if mono and enough_data:
            eligible.append((word, usable_examples if config["experiment_type"] == "speaker-dependent" else totals[word], repeated_speakers if config["experiment_type"] == "speaker-dependent" else len(speaker_counts)))
    eligible.sort(key=lambda item: (-item[2], -item[1], item[0]))
    vocabulary = sorted(item[0] for item in eligible[:size])
    if not vocabulary:
        raise SystemExit(
            f"No eligible monosyllabic words: found {len(rows)} isolated dysarthric recordings, "
            f"but none met minimum_word_examples={minimum}. Lower it in the config if appropriate."
        )
    project_path("metadata/vocabulary.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    vocabulary_set = set(vocabulary)
    selected = []
    for row in rows:
        if row["word"] in vocabulary_set:
            selected.append({**row, "dataset_role": "training" if test_group else "standard"})
    if test_group:
        for row in test_rows:
            if row["word"] in vocabulary_set:
                selected.append({**row, "dataset_role": "testing"})
    write_csv("metadata/selected.csv", selected, selected[0].keys())
    write_csv(
        "metadata/vocabulary_counts.csv", report,
        ("word", "total_examples", "speaker_count", "usable_examples", "speakers_with_repetitions", "cmu_pronunciation", "monosyllabic", "eligible"),
    )
    detail = []
    for word in sorted(totals):
        for speaker, count in sorted(by_speaker[word].items()):
            detail.append({"word": word, "speaker_id": speaker, "count": count, "selected": str(word in vocabulary).lower()})
    write_csv("metadata/vocabulary_by_speaker.csv", detail, ("word", "speaker_id", "count", "selected"))
    logging.info(
        "Selected %d words for training group %s%s and %d recordings from speakers %s: %s",
        len(vocabulary), group, f" with test group {test_group}" if test_group else "", len(selected),
        ", ".join(sorted({row["speaker_id"] for row in selected})), ", ".join(vocabulary),
    )


if __name__ == "__main__":
    main()
