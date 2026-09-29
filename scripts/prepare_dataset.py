#!/usr/bin/env python3
"""Discover TORGO recordings and pair them with their prompts."""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path

from common import ensure_dirs, load_config, project_path, setup_logging, write_csv

SPEAKER_RE = re.compile(r"^(?P<gender>[FM])(?P<control>C)?(?P<number>\d+)$", re.I)
FIELDS = (
    "speaker_id", "speaker_group", "wav_path", "transcription", "normalized",
    "dysarthric", "severity", "isolated_word", "word", "duration_seconds",
    "microphone", "session", "utterance_key",
)


def normalize_transcription(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[A-Z]+", ascii_text.upper()))


def speaker_for(path: Path, root: Path) -> tuple[str, bool, str] | None:
    for part in path.relative_to(root).parts:
        match = SPEAKER_RE.fullmatch(part)
        if match:
            speaker = part.upper()
            control = bool(match.group("control"))
            return speaker, not control, "control" if control else "dysarthric"
    return None


def session_for(path: Path, speaker: str) -> Path:
    parts = list(path.parents)
    speaker_dir = next((p for p in parts if p.name.upper() == speaker), None)
    if speaker_dir is None:
        return path.parent
    relative = path.relative_to(speaker_dir)
    return speaker_dir / relative.parts[0] if len(relative.parts) > 1 else speaker_dir


def microphone_name(path: Path) -> str:
    for parent in path.parents:
        lower = parent.name.lower()
        if lower.startswith("wav_"):
            return lower[4:]
        if lower in {"audio", "wav"}:
            stem = path.stem.lower()
            return stem.split("_")[0] if "_" in stem else lower
    return path.parent.name.lower()


def utterance_number(path: Path) -> str:
    match = re.search(r"(\d+)$", path.stem)
    return match.group(1) if match else path.stem


def wav_duration_seconds(path: Path) -> float | None:
    """Read PCM WAV duration without decoding the complete recording."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (EOFError, OSError, wave.Error):
        return None


def prompt_index(session: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    candidates = [p for p in session.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".prompt"}]
    for prompt in candidates:
        if "prompt" not in "/".join(part.lower() for part in prompt.parts):
            continue
        result.setdefault(prompt.stem.lower(), prompt)
        result.setdefault(utterance_number(prompt).lower(), prompt)
    return result


def mic_rank(name: str, preferences: list[str]) -> int:
    compact = re.sub(r"[^a-z]", "", name.lower())
    for index, preference in enumerate(preferences):
        if re.sub(r"[^a-z]", "", preference.lower()) in compact:
            return index
    return len(preferences)


def discover(
    dataset_root: Path,
    preferences: list[str],
    severity_by_speaker: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    severity_by_speaker = severity_by_speaker or {}
    wavs = sorted(p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not wavs:
        raise SystemExit(
            f"No WAV files found below {dataset_root}. Download/extract TORGO there first; "
            "expected paths resemble F01/Session1/wav_headMic/0001.wav."
        )
    indices: dict[Path, dict[str, Path]] = {}
    grouped: dict[str, list[tuple[int, Path, dict[str, object]]]] = defaultdict(list)
    skipped_speaker = skipped_prompt = 0
    for wav in wavs:
        speaker_info = speaker_for(wav, dataset_root)
        if speaker_info is None:
            skipped_speaker += 1
            continue
        speaker, dysarthric, group = speaker_info
        session = session_for(wav, speaker)
        if session not in indices:
            indices[session] = prompt_index(session)
        number = utterance_number(wav).lower()
        prompt = indices[session].get(wav.stem.lower()) or indices[session].get(number)
        if prompt is None:
            skipped_prompt += 1
            continue
        transcription = " ".join(prompt.read_text(encoding="utf-8", errors="replace").split())
        normalized = normalize_transcription(transcription)
        tokens = normalized.split()
        isolated = len(tokens) == 1 and tokens[0].isalpha()
        microphone = microphone_name(wav)
        relative_session = session.relative_to(dataset_root).as_posix()
        key = f"{relative_session}/{number}"
        row: dict[str, object] = {
            "speaker_id": speaker,
            "speaker_group": group,
            "wav_path": str(wav.resolve()),
            "transcription": transcription,
            "normalized": normalized,
            "dysarthric": str(dysarthric).lower(),
            "severity": severity_by_speaker.get(speaker, "control" if not dysarthric else "unknown"),
            "isolated_word": str(isolated).lower(),
            "word": tokens[0] if isolated else "",
            "duration_seconds": (
                f"{duration:.6f}" if (duration := wav_duration_seconds(wav)) is not None else ""
            ),
            "microphone": microphone,
            "session": relative_session,
            "utterance_key": key,
        }
        grouped[key].append((mic_rank(microphone, preferences), wav, row))
    rows = [min(candidates, key=lambda item: (item[0], str(item[1])))[2] for candidates in grouped.values()]
    rows.sort(key=lambda row: (str(row["speaker_id"]), str(row["session"]), str(row["wav_path"])))
    logging.info("Discovered %d canonical utterances from %d WAV files", len(rows), len(wavs))
    logging.info("Skipped %d WAVs without recognized speaker IDs and %d without prompts", skipped_speaker, skipped_prompt)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    ensure_dirs()
    config = load_config(args.config)
    root = project_path(config["dataset_root"])
    severity_by_speaker = {
        str(speaker).upper(): str(severity).lower()
        for severity, speakers in config.get("severity_speakers", {}).items()
        for speaker in speakers
    }
    rows = discover(root, config.get("preferred_microphones", []), severity_by_speaker)
    write_csv("metadata/discovered.csv", rows, FIELDS)
    dysarthric_words = sum(r["dysarthric"] == "true" and r["isolated_word"] == "true" for r in rows)
    logging.info("Wrote metadata/discovered.csv (%d dysarthric isolated-word items)", dysarthric_words)


if __name__ == "__main__":
    main()
