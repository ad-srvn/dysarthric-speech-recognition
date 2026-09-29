#!/usr/bin/env python3
"""Decode isolated words with HVite and score them with HResults."""

from __future__ import annotations

import argparse

from common import project_path, require_file, require_tools, run_command, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture", type=int, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    tools = require_tools(("HVite", "HResults"))
    model = project_path(f"models/final_mix_{args.mixture}")
    output = project_path(args.output_dir or f"results/mixture_{args.mixture}/{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    recognized = output / "recognized.mlf"
    run_command([
        tools["HVite"], "-T", "1",
        "-H", require_file(model / "macros", "model macros"),
        "-H", require_file(model / "hmmdefs", "model HMM definitions"),
        "-S", require_file(f"lists/{args.split}.scp"), "-i", recognized,
        "-w", require_file("grammar/wdnet", "compiled word network"),
        require_file("dictionary/dict", "HTK dictionary"), require_file("lists/words.list"),
    ], f"results/logs/hvite_{args.split}_mix_{args.mixture}.log")
    require_file(recognized, "HVite recognition MLF")
    run_command([
        tools["HResults"], "-T", "1", "-I", require_file(f"labels/{args.split}.mlf"),
        require_file("lists/words.list"), recognized,
    ], output / "hresults.txt", include_command_in_log=False)


if __name__ == "__main__":
    main()
