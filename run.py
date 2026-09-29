#!/usr/bin/env python3
"""Choose an experiment and start the appropriate runner."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEVERITY_ALIASES = {
    "moderate": "medium",
    "controls": "control",
    "everyone": "combined",
}
SEVERITY_MENU = {
    "1": ["mild"],
    "2": ["medium"],
    "3": ["severe"],
    "4": ["all"],
    "5": ["control"],
    "6": ["combined"],
    "7": ["mild", "medium", "severe", "all", "control", "combined"],
}


def choose_severities() -> list[str]:
    print(
        "\nChoose a speaker group:\n"
        "  1. Mild\n"
        "  2. Moderate\n"
        "  3. Severe\n"
        "  4. All dysarthric speakers combined\n"
        "  5. Control speakers\n"
        "  6. Dysarthric and control speakers combined\n"
        "  7. Run every group separately"
    )
    while True:
        choice = input("Selection [1]: ").strip() or "1"
        if choice in SEVERITY_MENU:
            return SEVERITY_MENU[choice]
        print("Enter a number from 1 to 7.")


def choose_runs() -> int:
    while True:
        value = input("Number of runs [1]: ").strip() or "1"
        try:
            runs = int(value)
        except ValueError:
            runs = 0
        if runs >= 1:
            return runs
        print("Enter a positive whole number.")


def build_command(
    severities: list[str],
    models: list[str],
    runs: int,
    resume: bool,
    output_root: Path | None = None,
    test_group: str | None = None,
) -> list[str]:
    if runs == 1:
        runner = "scripts/run_experiment_matrix.py"
        command = [sys.executable, str(ROOT / runner)]
    else:
        runner = "scripts/run_repeated_experiment_matrix.py"
        command = [sys.executable, str(ROOT / runner), "--repeats", str(runs)]
    command.extend(("--groups", *severities, "--models", *models))
    if test_group:
        command.extend(("--test-group", test_group))
    if output_root is not None:
        command.extend(("--output-root", str(output_root)))
    if resume:
        command.append("--skip-completed")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare HTK models for selected TORGO speaker groups."
    )
    parser.add_argument(
        "--group", "--groups", "--severity", "--severities", dest="severity", nargs="+",
        choices=(
            "mild", "moderate", "medium", "severe", "all",
            "control", "controls", "combined", "everyone",
        ),
        help=(
            "Speaker groups to run. 'all' means all dysarthric speakers; "
            "'combined' includes dysarthric and control speakers."
        ),
    )
    parser.add_argument(
        "--runs", "--repeats", type=int,
        help="Total number of runs. Values above 1 are averaged.",
    )
    parser.add_argument(
        "--test-group",
        choices=("mild", "moderate", "medium", "severe", "all", "control", "controls"),
        help="Evaluate on a different held-out speaker group.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=("small", "medium", "large"),
        default=["small", "medium", "large"],
    )
    parser.add_argument("--resume", action="store_true", help="Reuse matching completed runs")
    parser.add_argument("--output-root", help="Custom result directory")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args = parser.parse_args()

    interactive = args.severity is None and sys.stdin.isatty()
    if args.severity is None:
        if not interactive:
            parser.error("choose --group, or run this command in a terminal for the menu")
        severities = choose_severities()
    else:
        severities = [SEVERITY_ALIASES.get(value, value) for value in args.severity]
        severities = list(dict.fromkeys(severities))

    runs = choose_runs() if interactive and args.runs is None else (
        args.runs if args.runs is not None else 1
    )
    if runs < 1:
        parser.error("--runs must be at least 1")

    test_group = SEVERITY_ALIASES.get(args.test_group, args.test_group) if args.test_group else None
    test_label = f"__to-{test_group}" if test_group else ""
    selection = f"{'-'.join(severities)}{test_label}__{'-'.join(args.models)}__{runs}-runs"
    output_root = Path(args.output_root).expanduser() if args.output_root else (
        ROOT / "results" / "experiments" / selection
    )
    command = build_command(
        severities, args.models, runs, args.resume, output_root, test_group
    )
    print("\nStarting:")
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
