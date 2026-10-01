#!/usr/bin/env python3
"""Verify code-only release indexes and data reconstructed at runtime."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(str(text).casefold().split()).encode("utf-8")).hexdigest()


def raw_text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_index(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    eval_index_path = BUNDLE_ROOT / "data/indexes/main40.paper-v1.jsonl.gz"
    train_index_path = ROOT / "configs/fixed_train_index_32000.jsonl.gz"
    eval_index = read_index(eval_index_path)
    train_index = read_index(train_index_path)
    require(len(eval_index) == 200000, f"evaluation index has {len(eval_index)} rows, expected 200000")
    require(len(train_index) == 32000, f"training index has {len(train_index)} rows, expected 32000")
    require(len({r["sample_id"] for r in eval_index}) == 200000, "evaluation index contains duplicate sample IDs")
    require(len({r["record_id"] for r in train_index}) == 32000, "training index contains duplicate record IDs")
    require(set(Counter(r["source"] for r in eval_index).values()) == {5000},
            "evaluation index is not balanced at 5,000 rows per dataset")

    report = {
        "status": "ok",
        "packaging": "optional-posttraining-over-bundled-main40-panel",
        "fixed_indexes": {
            "evaluation": {"rows": len(eval_index), "sha256": sha256(eval_index_path)},
            "training": {"rows": len(train_index), "sha256": sha256(train_index_path)},
        },
    }
    if not args.code_only:
        expected_raw = json.loads((ROOT / "manifests/raw_source_manifest.json").read_text(encoding="utf-8"))
        runtime_raw = json.loads((ROOT / "data/raw/raw_source_manifest.json").read_text(encoding="utf-8"))
        require(runtime_raw == expected_raw, "runtime raw-source manifest differs from the bundled expected manifest")
        for entry in expected_raw["files"]:
            raw_path = ROOT / entry["path"]
            require(raw_path.is_file(), f"missing raw source file: {entry['path']}")
            require(raw_path.stat().st_size == int(entry["bytes"]), f"raw source byte-size mismatch: {entry['path']}")
            require(sha256(raw_path) == entry["sha256"], f"raw source SHA-256 mismatch: {entry['path']}")

        eval_path = BUNDLE_ROOT / "artifacts/frozen_inputs/main40.jsonl"
        expected = {r["sample_id"]: r for r in eval_index}
        eval_counts = Counter()
        seen = set()
        with eval_path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                require(sample_id in expected, f"unexpected evaluation sample: {sample_id}")
                require(sample_id not in seen, f"duplicate evaluation sample: {sample_id}")
                item = expected[sample_id]
                require(raw_text_hash(row["context"]) == item["context_sha256"],
                        f"raw context SHA-256 mismatch: {sample_id}")
                require(int(row["gold_position"]) == int(item["gold_position"]),
                        f"gold-position mismatch: {sample_id}")
                seen.add(sample_id)
                eval_counts[row["source"]] += 1
        require(seen == set(expected), "evaluation file does not contain the exact indexed sample-ID set")
        require(set(eval_counts.values()) == {5000}, "evaluation file is not balanced at 5,000 rows per dataset")

        expected_train = {r["record_id"]: r for r in train_index}
        train_seen = set()
        path = ROOT / "data/prepared/kev_bias_train.jsonl"
        expected_prepared = json.loads((ROOT / "manifests/prepared_data_manifest.json").read_text(encoding="utf-8"))
        runtime_prepared = json.loads((ROOT / "data/prepared/prepared_data_manifest.json").read_text(encoding="utf-8"))
        require(runtime_prepared == expected_prepared,
                "runtime prepared-data manifest differs from the bundled expected manifest")
        expected_train_file = expected_prepared["files"]["train"]
        require(path.stat().st_size == int(expected_train_file["bytes"]), "prepared training JSONL byte-size mismatch")
        require(sha256(path) == expected_train_file["sha256"], "prepared training JSONL SHA-256 mismatch")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = row["_meta"]["id"]
                require(key in expected_train, f"unexpected training record: {key}")
                require(key not in train_seen, f"duplicate training record: {key}")
                item = expected_train[key]
                require(normalized_text_hash(row["state"]) == item["context_sha256"],
                        f"normalized training-text SHA-256 mismatch: {key}")
                require(int(row["_meta"]["gold_canonical_level"]) == int(item["gold_canonical_level"]),
                        f"training label mismatch: {key}")
                train_seen.add(key)
        require(train_seen == set(expected_train), "reconstructed training file does not match the frozen index")
        report["runtime_data"] = {
            "evaluation_rows": len(seen),
            "training_rows": len(train_seen),
            "evaluation_counts": dict(sorted(eval_counts.items())),
            "evaluation_sha256": sha256(eval_path),
        }
    report_path = ROOT / "artifacts" / "runtime_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
