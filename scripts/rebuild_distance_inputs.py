#!/usr/bin/env python3
"""Rebuild only the four-source candidate-distance prompts from upstream data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.io import iter_jsonl, sha256_file, write_jsonl
from scripts.rebuild_inputs import find_batch3

SOURCES = (
    "word_concreteness",
    "wmt20_translation_quality",
    "measuring_hate_speech",
    "stanford_politeness",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", type=Path,
                        default=ROOT / "supplement" / "startlux_distance_release_20261009")
    parser.add_argument("--datasets", type=Path, default=ROOT / "artifacts" / "datasets")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "frozen_inputs" / "distance7.jsonl")
    parser.add_argument("--base-input", type=Path,
                        help="Use an already reconstructed input containing the 20,000 indexed source rows")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="Required for Stanford Politeness's unresolved historical revision")
    args = parser.parse_args()
    manifest = json.loads((args.supplement / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = manifest["input_indexes"]["distance7"]["source_input_sha256"]
    if args.download and args.base_input:
        parser.error("--download and --base-input cannot be combined")
    if args.download:
        command = [sys.executable, str(ROOT / "scripts" / "download_datasets.py"),
                   "--sources", ",".join(SOURCES), "--output", str(args.datasets)]
        if args.allow_unpinned:
            command.append("--allow-unpinned")
        subprocess.run(command, cwd=ROOT, check=True)

    index = {row["sample_id"]: row for row in iter_jsonl(
        ROOT / "data" / "indexes" / "main40.paper-v1.jsonl.gz"
    ) if row["source"] in SOURCES}
    if len(index) != 20_000:
        raise RuntimeError(f"Expected 20,000 four-source rows; found {len(index)}")
    if args.base_input:
        from scripts.rebuild_inputs import verify
        rows_by_id = {}
        for row in iter_jsonl(args.base_input):
            sample_id = row["sample_id"]
            if sample_id in index:
                if sample_id in rows_by_id:
                    raise RuntimeError(f"Duplicate base input item: {sample_id}")
                rows_by_id[sample_id] = verify(row, index[sample_id])
    else:
        from scripts.source_builders import batch3
        rows_by_id = find_batch3(index, args.datasets, set(), batch3)
    if rows_by_id.keys() != index.keys():
        raise RuntimeError("The upstream snapshots did not reproduce all 20,000 indexed rows")
    base_path = args.output.with_name("distance7_base20k.jsonl")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(base_path, [rows_by_id[sample_id] for sample_id, meta in sorted(
        index.items(), key=lambda item: item[1]["ordinal"]
    )])

    generator = args.supplement / manifest["generator"]["file"]
    if sha256_file(generator) != manifest["generator"]["sha256"]:
        raise RuntimeError("The candidate-distance generator does not match the release manifest")
    report = args.output.with_suffix(".report.json")
    subprocess.run([sys.executable, str(generator), "--input", str(base_path),
                    "--output", str(args.output), "--report", str(report)],
                   cwd=ROOT, check=True)
    actual = sha256_file(args.output)
    if actual != expected:
        raise RuntimeError(f"Generated prompts differ from the release: {actual} != {expected}")
    print(f"Rebuilt {args.output} with verified SHA-256 {actual}")


if __name__ == "__main__":
    main()
