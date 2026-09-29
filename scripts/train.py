#!/usr/bin/env python3
"""Train configurable-state and configurable-mixture whole-word HMMs."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

from common import load_config, project_path, require_file, require_tools, run_command, setup_logging


def herest(tool: str, source: Path, destination: Path, iteration: int) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    run_command([
        tool, "-T", "1", "-I", require_file("labels/train.mlf"),
        "-t", "250.0", "150.0", "1000.0",
        "-S", require_file("lists/train.scp"),
        "-H", require_file(source / "macros"),
        "-H", require_file(source / "hmmdefs"),
        "-M", destination, require_file("lists/words.list"),
    ], f"results/logs/herest_{iteration:02d}.log")
    require_file(destination / "macros", "HERest macros")
    require_file(destination / "hmmdefs", "HERest HMM definitions")


def snapshot(source: Path, mixture: int) -> None:
    target = project_path(f"models/final_mix_{mixture}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def validate_mixtures(model_dir: Path, expected: int, emitting_states: int) -> None:
    """Ensure HHEd actually expanded every emitting state to the requested count."""
    if expected == 1:
        return
    definitions = require_file(model_dir / "hmmdefs", "trained HMM definitions").read_text(encoding="ascii")
    observed = [int(value) for value in re.findall(r"<NUMMIXES>\s+(\d+)", definitions, flags=re.I)]
    word_count = sum(1 for line in require_file("lists/words.list").read_text().splitlines() if line.strip())
    expected_states = word_count * emitting_states
    if len(observed) != expected_states or any(value != expected for value in observed):
        raise SystemExit(
            f"Model validation failed: expected {expected_states} emitting states with {expected} mixtures; "
            f"found {len(observed)} explicit mixture declarations with values {sorted(set(observed))}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    tools = require_tools(("HERest", "HHEd"))
    current = project_path("models/hmm0")
    require_file(current / "macros", "initialized macros")
    require_file(current / "hmmdefs", "initialized HMM definitions")
    serial = 0
    for _ in range(int(config["baum_welch_iterations"])):
        serial += 1
        destination = project_path(f"models/hmm{serial}")
        herest(tools["HERest"], current, destination, serial)
        current = destination
    snapshot(current, 1)
    logging.info("Saved 1-mixture experiment at models/final_mix_1")

    target_mixtures = int(config["target_mixtures"])
    emitting_states = int(config["num_emitting_states"])
    last_emitting_state = int(config["num_hmm_states"]) - 1
    if target_mixtures < 1:
        raise SystemExit("target_mixtures must be at least 1")
    for mixtures in range(2, target_mixtures + 1):
        hed = project_path(f"models/mix_{mixtures}.hed")
        hed.write_text(f"MU {mixtures} {{*.state[2-{last_emitting_state}].mix}}\n", encoding="ascii")
        serial += 1
        split_dir = project_path(f"models/hmm{serial}")
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        run_command([
            tools["HHEd"], "-T", "1", "-H", require_file(current / "macros"),
            "-H", require_file(current / "hmmdefs"), "-M", split_dir,
            hed, require_file("lists/words.list"),
        ], f"results/logs/hhed_mix_{mixtures}.log")
        if not (split_dir / "macros").exists():
            shutil.copy2(current / "macros", split_dir / "macros")
        require_file(split_dir / "hmmdefs", "HHEd HMM definitions")
        current = split_dir
        for _ in range(int(config["mixture_training_iterations"])):
            serial += 1
            destination = project_path(f"models/hmm{serial}")
            herest(tools["HERest"], current, destination, serial)
            current = destination
        validate_mixtures(current, mixtures, emitting_states)
        snapshot(current, mixtures)
        logging.info("Saved %d-mixture experiment at models/final_mix_%d", mixtures, mixtures)


if __name__ == "__main__":
    main()
