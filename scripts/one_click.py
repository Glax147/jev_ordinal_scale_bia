#!/usr/bin/env python3
"""Verify released outputs and reproduce all aggregate tables and figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*parts: str) -> None:
    command = [sys.executable, *parts]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("-m", "pytest", "-q")
    run("scripts/verify_artifacts.py")
    run("scripts/analyze.py")
    run("scripts/make_figures.py")
    print("\nReproduction complete. See outputs/metrics and outputs/figures.")


if __name__ == "__main__":
    main()
