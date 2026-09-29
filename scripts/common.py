"""Shared helpers for the TORGO/HTK pipeline."""

from __future__ import annotations

import csv
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
HTK_TOOLS = ("HCopy", "HCompV", "HERest", "HHEd", "HParse", "HVite", "HResults")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def configure_matplotlib() -> None:
    cache = Path(tempfile.gettempdir()) / "dysarthric-speech-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def portable_path(value: str | Path) -> str:
    resolved = project_path(value)
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def timestamped_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}_{stamp}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{stamp}_{counter}")
        counter += 1
    return candidate


def unique_output_path(path: Path) -> Path:
    return timestamped_path(path) if path.exists() else path


def preserve_path(path: Path) -> Path | None:
    if not path.exists():
        return None
    destination = timestamped_path(path)
    path.rename(destination)
    return destination


def load_config(path: str | Path) -> dict:
    config_path = project_path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration not found: {config_path}") from exc
    required = {
        "dataset_root", "experiment_type", "vocabulary_size",
        "train_ratio", "validation_ratio", "test_ratio", "random_seed", "num_hmm_states", "num_emitting_states",
        "feature_dimension", "target_mixtures", "baum_welch_iterations",
        "mixture_training_iterations",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise SystemExit(f"Configuration is missing: {', '.join(missing)}")
    if config["feature_dimension"] != 39:
        raise SystemExit("feature_dimension must be 39 for MFCC_0_D_A features.")
    total_states = int(config["num_hmm_states"])
    emitting_states = int(config["num_emitting_states"])
    if total_states < 3 or emitting_states < 1 or total_states != emitting_states + 2:
        raise SystemExit(
            "num_hmm_states must equal num_emitting_states + 2 "
            "(one non-emitting entry and one non-emitting exit state)."
        )
    if config["experiment_type"] not in {"speaker-dependent", "speaker-independent"}:
        raise SystemExit("experiment_type must be speaker-dependent or speaker-independent")
    split_total = sum(float(config[key]) for key in ("train_ratio", "validation_ratio", "test_ratio"))
    if abs(split_total - 1.0) > 1e-9:
        raise SystemExit("train_ratio + validation_ratio + test_ratio must equal 1.0")
    return config


def ensure_dirs() -> None:
    for relative in (
        "features/mfc/train", "features/mfc/validation", "features/mfc/test",
        "metadata", "lists", "labels", "dictionary",
        "grammar", "models", "results/logs",
    ):
        project_path(relative).mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    resolved = project_path(path)
    if not resolved.is_file():
        raise SystemExit(f"Required CSV not found: {resolved}")
    with resolved.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quote_command(command: Sequence[object]) -> str:
    return shlex.join(str(part) for part in command)


def run_command(
    command: Sequence[object],
    log_path: str | Path | None = None,
    *,
    include_command_in_log: bool = True,
) -> None:
    rendered = quote_command(command)
    print(f"$ {rendered}", flush=True)
    if log_path is None:
        subprocess.run([str(part) for part in command], check=True)
        return
    resolved = project_path(log_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as log:
        if include_command_in_log:
            log.write(f"$ {rendered}\n\n")
            log.flush()
        subprocess.run(
            [str(part) for part in command], check=True,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )


def find_executable(name: str) -> str | None:
    htk_bin = os.environ.get("HTK_BIN")
    if htk_bin:
        candidate = Path(htk_bin).expanduser() / name
        return str(candidate.resolve()) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    local_candidate = ROOT / "tools" / "htk" / "bin" / name
    if local_candidate.is_file() and os.access(local_candidate, os.X_OK):
        return str(local_candidate.resolve())
    return shutil.which(name)


def require_tools(names: Sequence[str] = HTK_TOOLS) -> dict[str, str]:
    found = {name: find_executable(name) for name in names}
    missing = [name for name, path in found.items() if path is None]
    if missing:
        raise SystemExit(
            "Missing required HTK 3.4.x binaries: " + ", ".join(missing) + "\n"
            f"Checked repository-local directory: {ROOT / 'tools' / 'htk' / 'bin'}\n"
            "Install HTK there, add its bin directory to PATH, or set for example:\n"
            "  export HTK_BIN=/usr/local/bin\n"
            "No alternative recognizer will be substituted."
        )
    return {name: path for name, path in found.items() if path is not None}


def require_file(path: str | Path, description: str = "file") -> Path:
    resolved = project_path(path)
    if not resolved.is_file():
        raise SystemExit(f"Missing {description}: {resolved}")
    return resolved


def htk_path(path: Path) -> str:
    return path.resolve().as_posix()
