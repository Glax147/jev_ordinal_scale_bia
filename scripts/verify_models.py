#!/usr/bin/env python3
"""Verify downloaded KEV adapter/base trees before local inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.model_integrity import load_and_verify_downloaded_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="kev-0.8b,kev-4b")
    parser.add_argument("--model-root", type=Path, default=ROOT / "artifacts" / "models")
    args = parser.parse_args()
    report = {}
    for model in [value.strip() for value in args.models.split(",") if value.strip()]:
        report[model] = load_and_verify_downloaded_model(args.model_root, model)
        print(f"OK {model}: adapter and base trees match their SHA-256 inventories")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
