#!/usr/bin/env python3
"""Remove generated experiment artifacts, never raw TORGO data."""

from __future__ import annotations

import shutil

from common import project_path

DIRECTORIES = (
    "features/mfc/train", "features/mfc/validation", "features/mfc/test", "models",
)
FILES = (
    "metadata/discovered.csv", "metadata/selected.csv", "metadata/metadata.csv",
    "metadata/vocabulary.txt", "metadata/vocabulary_counts.csv", "metadata/vocabulary_by_speaker.csv",
    "lists/train.scp", "lists/validation.scp", "lists/test.scp",
    "lists/train_hcopy.scp", "lists/validation_hcopy.scp", "lists/test_hcopy.scp", "lists/words.list",
    "labels/train.mlf", "labels/validation.mlf", "labels/test.mlf",
    "dictionary/dict", "grammar/grammar", "grammar/wdnet",
)


def main() -> None:
    for relative in DIRECTORIES:
        path = project_path(relative)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        (path / ".gitkeep").touch()
    for relative in FILES:
        path = project_path(relative)
        if path.exists():
            path.unlink()
    results = project_path("results")
    for item in results.iterdir():
        if item.name == "logs":
            for log in item.iterdir():
                if log.name != ".gitkeep":
                    log.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print("Removed generated features, models, metadata, and results. data was not touched.")


if __name__ == "__main__":
    main()
