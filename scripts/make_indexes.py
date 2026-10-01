#!/usr/bin/env python3
"""Maintainer tool: derive compact frozen indexes from the original input JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.integrity import scan_gzip_jsonl
from jev_bias.io import canonical_json_sha256, iter_jsonl, write_jsonl


def context_sha256(row: dict) -> str:
    return hashlib.sha256(row["context"].encode("utf-8")).hexdigest()


def main_record(row: dict, ordinal: int) -> dict:
    record = {
        "ordinal": ordinal, "sample_id": row["sample_id"], "source": row["source"],
        "original_id": row.get("original_id"), "split": row.get("split"),
        "sampling_stratum": row.get("sampling_stratum"), "sampling_seed": row.get("sampling_seed"),
        "source_repo": row.get("source_repo"), "source_revision": row.get("source_revision"),
        "task_type": row.get("task_type"), "option_count": len(row["choices"]),
        "gold_position": row.get("gold_position"), "row_sha256": canonical_json_sha256(row),
        "context_sha256": context_sha256(row),
    }
    record["row_without_context"] = {k: v for k, v in row.items() if k != "context"}
    return record


def position_record(row: dict, ordinal: int) -> dict:
    record = {
        "ordinal": ordinal, "sample_id": row["sample_id"], "base_sample_id": row["base_sample_id"],
        "source": row["source"], "order_variant": row["order_variant"],
        "option_count": len(row["choices"]), "gold_position": row["gold_position"],
        "original_gold_position": row["original_gold_position"],
        "permutation_new_to_canonical": row["choice_permutation_new_to_original"],
        "row_sha256": canonical_json_sha256(row),
        "context_sha256": context_sha256(row),
    }
    record["row_without_context"] = {k: v for k, v in row.items() if k != "context"}
    return record


def kcurve_record(row: dict, ordinal: int) -> dict:
    payload = {
        "ordinal": ordinal, "sample_id": row["sample_id"], "base_sample_id": row["base_sample_id"],
        "source": row["source"], "task_type": row["task_type"], "option_count": row["option_count"],
        "gold_position": row["gold_position"], "binning_method": row.get("binning_method"),
        "candidate_set_id": row.get("candidate_set_id"), "row_sha256": canonical_json_sha256(row),
        "context_sha256": context_sha256(row),
    }
    if row["source"] == "dbpedia14":
        payload.update({
            "gold_canonical_level": row["gold_source_class_index"],
            "permutation_new_to_canonical": row["choice_permutation_new_to_source_class"],
            "candidate_set_canonical": row["candidate_source_class_indices"],
        })
    else:
        payload.update({
            "gold_canonical_level": int(row["gold_canonical_level"]) - 1,
            "permutation_new_to_canonical": row["choice_permutation_new_to_canonical"],
            "bin_edges": row.get("bin_edges"),
        })
    payload["row_without_context"] = {k: v for k, v in row.items() if k != "context"}
    return payload


def build(paths: list[Path], output: Path, converter) -> dict:
    ordinal = 0

    def records():
        nonlocal ordinal
        for path in paths:
            for row in iter_jsonl(path):
                yield converter(row, ordinal)
                ordinal += 1

    row_count = write_jsonl(output, records())
    try:
        portable_output = output.relative_to(ROOT).as_posix()
    except ValueError:
        portable_output = output.name
    scan = scan_gzip_jsonl(output)
    return {
        "file": portable_output,
        "rows": row_count,
        "sha256": scan["archive_sha256"],
        "archive_bytes": scan["archive_bytes"],
        "content_sha256": scan["content_sha256"],
        "content_bytes": scan["bytes"],
        "ordered_ids_sha256": scan["ordered_ids_sha256"],
        "selection_sha256": scan["selection_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT / "artifacts" / "frozen_inputs",
                        help="Directory made by rebuild_inputs.py, or the private historical input directory")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "indexes")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    combined = args.source_root / "main40.jsonl"
    main_paths = [combined] if combined.is_file() else [
        args.source_root / "ordinal_bias_10x5000.jsonl",
        args.source_root / "ordinal_bias_batch2_10x5000.jsonl",
        args.source_root / "ordinal_bias_batch3_20x5000.jsonl",
    ]
    position_path = (args.source_root / "position9.jsonl" if (args.source_root / "position9.jsonl").is_file()
                     else args.source_root / "core9_two_random_orders_90000.jsonl")
    kcurve_path = (args.source_root / "kcurve4.jsonl" if (args.source_root / "kcurve4.jsonl").is_file()
                   else args.source_root / "controlled_k_curve_127400.jsonl")
    manifest = {
        "schema_version": "paper-v1-frozen-index-v2",
        "main40": build(main_paths, args.output / "main40.paper-v1.jsonl.gz", main_record),
        "position9": build([position_path],
                           args.output / "position9.paper-v1.jsonl.gz", position_record),
        "kcurve4": build([kcurve_path],
                         args.output / "kcurve4.paper-v1.jsonl.gz", kcurve_record),
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
