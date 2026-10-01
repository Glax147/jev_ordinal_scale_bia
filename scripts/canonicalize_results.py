#!/usr/bin/env python3
"""Deduplicate append-only runner output, preferring the last successful row."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_bias.io import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--order-index", type=Path, help="Optional JSONL(.gz) index that fixes output order")
    args = parser.parse_args()
    latest, source_rows = {}, 0
    for row in iter_jsonl(args.input):
        source_rows += 1
        sample_id = row.get("sample_id") or row.get("row_uid")
        if not sample_id:
            continue
        previous = latest.get(sample_id)
        if row.get("status") == "ok" or previous is None or previous.get("status") != "ok":
            latest[sample_id] = row
    if args.order_index:
        order = [row["sample_id"] for row in iter_jsonl(args.order_index)]
        missing = [sample_id for sample_id in order if sample_id not in latest]
        if missing:
            raise SystemExit(f"Missing {len(missing)} indexed rows; first={missing[:5]}")
        rows = [latest[sample_id] for sample_id in order]
    else:
        rows = [latest[key] for key in sorted(latest)]
    write_jsonl(args.output, rows)
    print(json.dumps({"source_rows": source_rows, "unique_rows": len(rows),
                      "duplicates_removed": source_rows - len(rows),
                      "statuses": dict(Counter(row.get("status", "missing") for row in rows))}, indent=2))


if __name__ == "__main__":
    main()

