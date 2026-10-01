#!/usr/bin/env python3
"""Backward-compatible alias for the full artifact verifier."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    raise SystemExit(subprocess.call([sys.executable, str(ROOT / "scripts" / "verify_artifacts.py")], cwd=ROOT))


if __name__ == "__main__":
    main()
