#!/usr/bin/env python3
"""Report the exact HTK executables selected by the pipeline."""

from __future__ import annotations

import subprocess

from common import HTK_TOOLS, require_tools


def main() -> None:
    tools = require_tools(HTK_TOOLS)
    print("HTK tool resolution:")
    for name in HTK_TOOLS:
        path = tools[name]
        subprocess.run([path, "-V"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  {name}: {path} (version check passed)")


if __name__ == "__main__":
    main()
