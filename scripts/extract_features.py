#!/usr/bin/env python3
"""Extract 39-dimensional MFCC_0_D_A features with HTK HCopy."""

from __future__ import annotations

import argparse

from common import project_path, require_file, require_tools, run_command, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    hcopy = require_tools(("HCopy",))["HCopy"]
    config = require_file("config/mfcc.cfg", "MFCC configuration")
    for split in ("train", "validation", "test"):
        for old_feature in (project_path("features/mfc") / split).glob("*.mfc"):
            old_feature.unlink()
        scp = require_file(f"lists/{split}_hcopy.scp", f"{split} HCopy SCP")
        run_command([hcopy, "-C", config, "-S", scp], f"results/logs/hcopy_{split}.log")
        expected = [line for line in project_path(f"lists/{split}.scp").read_text().splitlines() if line.strip()]
        missing = [line.strip('"') for line in expected if not project_path(line.strip('"')).is_file()]
        if missing:
            raise SystemExit(f"HCopy completed but {len(missing)} feature files are missing; first: {missing[0]}")


if __name__ == "__main__":
    main()
