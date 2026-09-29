#!/usr/bin/env python3
"""Generate SCP, MLF, word-list, dictionary, and isolated-word grammar files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common import htk_path, project_path, read_csv, require_tools, run_command, setup_logging


def quoted(path: Path) -> str:
    return '"' + htk_path(path).replace('"', '\\"') + '"'


def write_mlf(path: Path, rows: list[dict[str, str]]) -> None:
    lines = ["#!MLF!#"]
    for row in rows:
        lines.extend((f'"*/{row["utterance_id"]}.lab"', row["word"], "."))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-grammar", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    rows = read_csv("metadata/metadata.csv")
    vocabulary = [line.strip() for line in project_path("metadata/vocabulary.txt").read_text().splitlines() if line.strip()]
    present_words = {row["word"] for row in rows}
    vocabulary = [word for word in vocabulary if word in present_words]
    if not vocabulary:
        raise SystemExit("No selected vocabulary remains in metadata/metadata.csv")

    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            raise SystemExit(f"No {split} rows in metadata/metadata.csv")
        hcopy_lines, feature_lines = [], []
        for row in split_rows:
            wav = Path(row["wav_path"])
            if not wav.is_file():
                raise SystemExit(f"TORGO WAV is missing: {wav}")
            mfc = project_path("features/mfc") / split / f"{row['utterance_id']}.mfc"
            hcopy_lines.append(f"{quoted(wav)} {quoted(mfc)}")
            feature_lines.append(quoted(mfc))
        project_path(f"lists/{split}_hcopy.scp").write_text("\n".join(hcopy_lines) + "\n", encoding="utf-8")
        project_path(f"lists/{split}.scp").write_text("\n".join(feature_lines) + "\n", encoding="utf-8")
        write_mlf(project_path(f"labels/{split}.mlf"), split_rows)

    project_path("lists/words.list").write_text("\n".join(vocabulary) + "\n", encoding="ascii")
    project_path("dictionary/dict").write_text("".join(f"{word} {word}\n" for word in vocabulary), encoding="ascii")
    alternatives = " | ".join(vocabulary)
    project_path("grammar/grammar").write_text(f"$WORD = {alternatives};\n\n( $WORD )\n", encoding="ascii")
    logging.info("Generated HTK files for %d words and %d utterances", len(vocabulary), len(rows))

    if args.compile_grammar:
        hparse = require_tools(("HParse",))["HParse"]
        run_command(
            [hparse, project_path("grammar/grammar"), project_path("grammar/wdnet")],
            "results/logs/hparse.log",
        )


if __name__ == "__main__":
    main()
