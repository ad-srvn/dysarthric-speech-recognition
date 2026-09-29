#!/usr/bin/env python3
"""Create a configurable left-to-right prototype and initialize HMMs with HCompV."""

from __future__ import annotations

import argparse
import logging
import shutil

from common import load_config, project_path, require_file, require_tools, run_command, setup_logging


def vector(values: list[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def create_prototype(dimension: int, total_states: int = 5) -> str:
    if total_states < 3:
        raise ValueError("An HMM needs entry, emitting, and exit states")
    zeros = vector([0.0] * dimension)
    ones = vector([1.0] * dimension)
    lines = [
        f"~o <VECSIZE> {dimension} <MFCC_0_D_A>",
        '~h "proto"',
        "<BEGINHMM>",
        f"<NUMSTATES> {total_states}",
    ]
    for state in range(2, total_states):
        lines.extend((f"<STATE> {state}", f"<MEAN> {dimension}", zeros, f"<VARIANCE> {dimension}", ones))
    transitions: list[str] = []
    for row in range(total_states):
        values = [0.0] * total_states
        if row == 0:
            values[1] = 1.0
        elif row < total_states - 1:
            values[row] = 0.6
            values[row + 1] = 0.4
        transitions.append(" ".join(f"{value:.1f}" for value in values))
    lines.extend((f"<TRANSP> {total_states}", *transitions, "<ENDHMM>"))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    hcompv = require_tools(("HCompV",))["HCompV"]
    models = project_path("models")
    for old_model in models.iterdir():
        if old_model.name == ".gitkeep":
            continue
        if old_model.is_dir():
            shutil.rmtree(old_model)
        else:
            old_model.unlink()
    hmm0 = project_path("models/hmm0")
    hmm0.mkdir(parents=True)
    prototype = project_path("models/proto")
    prototype.write_text(
        create_prototype(int(config["feature_dimension"]), int(config["num_hmm_states"])),
        encoding="ascii",
    )
    run_command([
        hcompv, "-T", "1", "-f", "0.01", "-m",
        "-S", require_file("lists/train.scp"),
        "-M", hmm0, prototype,
    ], "results/logs/hcompv.log")
    initialized = require_file(hmm0 / "proto", "HCompV prototype").read_text(encoding="ascii")
    marker = initialized.find('~h "proto"')
    if marker < 0:
        raise SystemExit("HCompV output does not contain the expected ~h \"proto\" header")
    global_options = initialized[:marker].strip() + "\n"
    hmm_body = initialized[marker:]
    vfloors = require_file(hmm0 / "vFloors", "HCompV variance floors").read_text(encoding="ascii")
    (hmm0 / "macros").write_text(global_options + vfloors, encoding="ascii")
    words = [line.strip() for line in require_file("lists/words.list").read_text().splitlines() if line.strip()]
    definitions = []
    for word in words:
        definitions.append(hmm_body.replace('~h "proto"', f'~h "{word}"', 1).rstrip())
    (hmm0 / "hmmdefs").write_text("\n".join(definitions) + "\n", encoding="ascii")
    logging.info("Initialized %d word HMMs in models/hmm0", len(words))


if __name__ == "__main__":
    main()
