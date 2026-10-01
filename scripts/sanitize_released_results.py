#!/usr/bin/env python3
"""Maintainer tool: remove relay-specific metadata from released predictions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.io import iter_jsonl, write_jsonl


REMOVED_FIELDS = {"endpoint", "provider_request_id"}
RELAY_MARKERS = ("tu-zi", "tuzi", "knox", "openlux", "openlunx")


def sanitized(rows):
    for original in rows:
        row = {key: value for key, value in original.items() if key not in REMOVED_FIELDS}
        schema = str(row.get("schema_version", ""))
        if any(marker in schema.casefold() for marker in RELAY_MARKERS):
            row["schema_version"] = "jev-answer-v1.0"
        yield row


def main() -> None:
    released = ROOT / "results" / "released"
    for path in sorted(released.glob("*.jsonl.gz")):
        temporary = path.with_name(path.name + ".writing.gz")
        rows = write_jsonl(temporary, sanitized(iter_jsonl(path)))
        os.replace(temporary, path)
        print(f"Sanitized/canonicalized {path.name}: {rows:,} rows")


if __name__ == "__main__":
    main()
