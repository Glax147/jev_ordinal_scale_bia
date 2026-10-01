#!/usr/bin/env python3
"""Build or verify the optional post-training release manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[2]
ROOT = BUNDLE_ROOT / "posttraining"
SCRIPT_ROOT = BUNDLE_ROOT / "scripts" / "posttraining"
REQUIREMENTS_PATH = BUNDLE_ROOT / "requirements" / "posttraining.txt"
EXCLUDED_TOP_LEVEL = {".cache", "artifacts", "code", "data", "logs", "models", "runs"}
EXCLUDED_RESULT_DIRS = {"generated"}
GENERATED_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path.name == "MANIFEST.json" or any(part in GENERATED_NAMES for part in relative.parts):
        return False
    if relative.parts[0] in EXCLUDED_TOP_LEVEL:
        return False
    if len(relative.parts) > 1 and relative.parts[0] == "results" and relative.parts[1] in EXCLUDED_RESULT_DIRS:
        return False
    return path.is_file()


def scan_result(path: Path) -> dict:
    ids: list[str] = []
    statuses: Counter[str] = Counter()
    canonical = hashlib.sha256()
    forbidden = (b'"endpoint"', b'"provider_request_id"', b'tu-zi', b'knox', b'openlux')
    with gzip.open(path, "rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            if any(token in raw.lower() for token in forbidden):
                raise ValueError(f"provider/relay metadata found in {path.name}")
            row = json.loads(raw)
            ids.append(str(row["sample_id"]))
            statuses[str(row.get("status", "unknown"))] += 1
            canonical.update(
                (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            )
    return {
        "rows": len(ids),
        "unique_ids": len(set(ids)),
        "statuses": dict(sorted(statuses.items())),
        "sample_ids_ordered_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "canonical_records_sha256": canonical.hexdigest(),
    }


def build() -> dict:
    files = []
    for path in sorted(p for p in ROOT.rglob("*") if included(p)):
        files.append({
            "file": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    for path in sorted(p for p in SCRIPT_ROOT.rglob("*") if p.is_file() and not any(part in GENERATED_NAMES for part in p.parts)):
        files.append({
            "file": path.relative_to(BUNDLE_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    files.append({
        "file": REQUIREMENTS_PATH.relative_to(BUNDLE_ROOT).as_posix(),
        "bytes": REQUIREMENTS_PATH.stat().st_size,
        "sha256": sha256_file(REQUIREMENTS_PATH),
    })
    files.sort(key=lambda item: item["file"])
    tree = hashlib.sha256()
    for item in files:
        tree.update(f"{item['sha256']}  {item['bytes']}  {item['file']}\n".encode())

    config = json.loads((ROOT / "configs/posttraining.json").read_text(encoding="utf-8"))
    released_checkpoints = config.get("released_checkpoints", [])
    if len(released_checkpoints) != 2:
        raise ValueError("expected two released BA-LoRA checkpoints")
    for checkpoint in released_checkpoints:
        if not checkpoint.get("repository") or len(str(checkpoint.get("revision", ""))) != 40:
            raise ValueError(f"invalid released checkpoint pin: {checkpoint}")

    train_index = ROOT / "configs/fixed_train_index_32000.jsonl.gz"
    with gzip.open(train_index, "rb") as handle:
        train_payload = handle.read()
    if len(train_payload.splitlines()) != 32_000:
        raise ValueError("training index must contain exactly 32,000 rows")

    input_manifest_path = BUNDLE_ROOT / "data/frozen_inputs/MANIFEST.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    main_input_spec = input_manifest["experiments"]["main40"]
    main_index = BUNDLE_ROOT / "data/indexes/main40.paper-v1.jsonl.gz"
    with gzip.open(main_index, "rt", encoding="utf-8") as handle:
        main_ids = [str(json.loads(line)["sample_id"]) for line in handle if line.strip()]
    if len(main_ids) != 200_000 or len(set(main_ids)) != 200_000:
        raise ValueError("shared main40 index is incomplete or contains duplicate sample IDs")
    main_ordered_ids_sha256 = hashlib.sha256(("\n".join(main_ids) + "\n").encode()).hexdigest()
    if int(main_input_spec["rows"]) != 200_000 or main_input_spec["ordered_ids_sha256"] != main_ordered_ids_sha256:
        raise ValueError("shared main40 input manifest and compact index disagree")
    released = []
    for path in sorted((ROOT / "results/released").glob("*.jsonl.gz")):
        scan = scan_result(path)
        if scan["rows"] != 200_000 or scan["unique_ids"] != 200_000 or scan["statuses"] != {"ok": 200_000}:
            raise ValueError(f"incomplete post-training result: {path.name}: {scan}")
        released.append({"file": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), **scan})
    if len(released) != 2 or len({r["sample_ids_ordered_sha256"] for r in released}) != 1:
        raise ValueError("expected two complete released post-training results over one ordered input")
    if any(r["sample_ids_ordered_sha256"] != main_ordered_ids_sha256 for r in released):
        raise ValueError("released post-training result order/IDs do not match the shared main40 index")

    return {
        "schema_version": "jev-bias-posttraining-release-v1",
        "hash_algorithm": "sha256",
        "release_tree_sha256": tree.hexdigest(),
        "files": files,
        "shared_assets": {
            "main40_input": {
                "materialized_file": "../artifacts/frozen_inputs/main40.jsonl",
                "source_manifest": "../data/frozen_inputs/MANIFEST.json",
                "source_manifest_sha256": sha256_file(input_manifest_path),
                "bytes": int(main_input_spec["bytes"]),
                "sha256": str(main_input_spec["content_sha256"]),
                "sample_ids_ordered_sha256": str(main_input_spec["ordered_ids_sha256"]),
            },
            "main40_index": {
                "file": "../data/indexes/main40.paper-v1.jsonl.gz",
                "bytes": main_index.stat().st_size,
                "sha256": sha256_file(main_index),
                "sample_ids_ordered_sha256": main_ordered_ids_sha256,
            },
        },
        "training_index": {
            "file": "configs/fixed_train_index_32000.jsonl.gz",
            "rows": 32_000,
            "archive_sha256": sha256_file(train_index),
            "decompressed_sha256": hashlib.sha256(train_payload).hexdigest(),
        },
        "released_results": released,
        "released_checkpoints": released_checkpoints,
        "checkpoint_replay": "checkpoints-released-full-main40-replay-not-yet-independently-verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    observed = build()
    path = ROOT / "MANIFEST.json"
    if args.verify:
        expected = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("post-training manifest mismatch")
        print("OK post-training manifest")
    else:
        path.write_text(json.dumps(observed, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote {path.name}: {observed['release_tree_sha256']}")


if __name__ == "__main__":
    main()
