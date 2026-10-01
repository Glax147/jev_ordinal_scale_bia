#!/usr/bin/env python3
"""Rebuild paper-v1 inputs from upstream sources and compact frozen indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.io import (
    canonical_json_sha256,
    iter_jsonl,
    ordered_selection_digest,
    selection_digest,
    sha256_file,
    write_jsonl,
)
def verify(row: dict, meta: dict) -> dict:
    observed = canonical_json_sha256(row)
    if observed != meta["row_sha256"]:
        raise RuntimeError(f"Reconstructed row mismatch for {row['sample_id']}: {observed} != {meta['row_sha256']}")
    return row


def find_from_loaders(index: dict[str, dict], loaders, cache: Path) -> dict[str, dict]:
    found = {}
    for name, loader in loaders:
        wanted = {sample_id for sample_id, meta in index.items() if meta["source"] == name}
        if not wanted: continue
        print(f"Reconstructing {name}: {len(wanted)} rows", flush=True)
        for row in loader():
            if row and row["sample_id"] in wanted:
                found[row["sample_id"]] = verify(row, index[row["sample_id"]])
                if wanted <= found.keys(): break
    return found


def maybe_truncate(context: str, row: dict) -> str:
    if not row.get("context_truncated"):
        return context.strip()
    marker = "\n...[middle truncated to satisfy provider request-size limit]...\n"
    keys = [chr(ord("A") + i) for i in range(len(row["choices"]))]
    def size(text):
        payload = {"state": text, "model": "jev-1.13", "questions": {"decision": {
            "type": "choice", "instructions": row["question"], "criteria": dict(zip(keys, row["choices"]))}}}
        return len(json.dumps(payload, ensure_ascii=True).encode("utf-8"))
    low, high = 0, len(context)
    while low < high:
        keep = (low + high + 1) // 2; left = (keep + 1) // 2; right = keep // 2
        candidate = context[:left] + marker + (context[-right:] if right else "")
        if size(candidate) <= 24000: low = keep
        else: high = keep - 1
    left = (low + 1) // 2; right = low // 2
    return context[:left] + marker + (context[-right:] if right else "")


def find_batch3(index: dict[str, dict], artifacts: Path, earlier_sources: set[str], builder) -> dict[str, dict]:
    by_source_hash = defaultdict(dict)
    for sample_id, meta in index.items():
        if meta["source"] not in earlier_sources:
            by_source_hash[meta["source"]][meta["context_sha256"]] = (sample_id, meta)
    found = {}
    for source, wanted in sorted(by_source_hash.items()):
        print(f"Reconstructing {source}: {len(wanted)} contexts", flush=True)
        for context in builder.iter_contexts(source, artifacts):
            variants = [context.strip()]
            for sample_id, meta in list(wanted.items()):
                if meta[1]["row_without_context"].get("context_truncated"):
                    variants.append(maybe_truncate(context, meta[1]["row_without_context"]))
            matched = None
            for variant in variants:
                digest = hashlib.sha256(variant.encode("utf-8")).hexdigest()
                if digest in wanted:
                    matched = (digest, variant); break
            if matched:
                digest, exact_context = matched
                sample_id, meta = wanted.pop(digest)
                row = dict(meta["row_without_context"]); row["context"] = exact_context
                found[sample_id] = verify(row, meta)
                if not wanted: break
        if wanted:
            raise RuntimeError(f"{source}: {len(wanted)} paper-v1 contexts were not recovered; first hashes={list(wanted)[:5]}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, default=ROOT / "artifacts" / "datasets")
    parser.add_argument("--indexes", type=Path, default=ROOT / "data" / "indexes")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "frozen_inputs")
    parser.add_argument("--download", action="store_true",
                        help="Download all configured upstream snapshots before rebuilding")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="Allow the two sources whose historical upstream commits were not recorded")
    args = parser.parse_args()
    # Keep --help usable in the lightweight analysis environment. Dataset
    # dependencies are imported only when reconstruction actually starts.
    from scripts.source_builders import batch1, batch2, batch3

    if args.download:
        command = [sys.executable, str(ROOT / "scripts/download_datasets.py"),
                   "--sources", "all", "--output", str(args.datasets)]
        if args.allow_unpinned:
            command.append("--allow-unpinned")
        subprocess.run(command, cwd=ROOT, check=True)
    index = {row["sample_id"]: row for row in iter_jsonl(args.indexes / "main40.paper-v1.jsonl.gz")}
    batch1.RAW = args.datasets / "_builder_cache" / "batch1"
    batch2.RAW = args.datasets / "_builder_cache" / "batch2"
    batch2.configure_base()
    batch1.RAW = args.datasets / "_builder_cache" / "batch1"
    found = {}
    found.update(find_from_loaders(index, batch1.LOADERS, batch1.RAW))
    found.update(find_from_loaders(index, batch2.LOADERS, batch2.RAW))
    earlier_sources = {name for name, _ in batch1.LOADERS} | {name for name, _ in batch2.LOADERS}
    found.update(find_batch3(index, args.datasets, earlier_sources, batch3))
    missing = sorted(set(index) - set(found))
    if missing: raise RuntimeError(f"Missing {len(missing)} main rows; first={missing[:5]}")
    args.output.mkdir(parents=True, exist_ok=True)
    main_rows = [found[sample_id] for sample_id, meta in sorted(index.items(), key=lambda item: item[1]["ordinal"])]
    outputs = []

    def commit_panel(name: str, rows: list[dict]) -> None:
        target = args.output / f"{name}.jsonl"
        temporary = target.with_suffix(".jsonl.part")
        count = write_jsonl(temporary, rows)
        if count != len(rows):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to write every {name} row")
        os.replace(temporary, target)
        ids = [str(row["sample_id"]) for row in rows]
        outputs.append({
            "experiment": name,
            "file": target.name,
            "rows": count,
            "sha256": sha256_file(target),
            "selection_sha256": selection_digest(ids),
            "ordered_ids_sha256": ordered_selection_digest(ids),
            "identity": "every row matched the committed paper-v1 canonical row SHA-256",
        })

    commit_panel("main40", main_rows)
    context_by_base = {row["sample_id"]: row["context"] for row in main_rows}
    for name in ("position9", "kcurve4"):
        rows = []
        for meta in iter_jsonl(args.indexes / f"{name}.paper-v1.jsonl.gz"):
            context = context_by_base.get(meta["base_sample_id"])
            if context is None: raise RuntimeError(f"Missing base context {meta['base_sample_id']}")
            row = dict(meta["row_without_context"]); row["context"] = context
            rows.append(verify(row, meta))
        commit_panel(name, rows)
    (args.output / "MANIFEST.json").write_text(
        json.dumps({
            "schema_version": "jev-public-rebuilt-inputs-v1",
            "source_indexes": "data/indexes/*.paper-v1.jsonl.gz",
            "hash_algorithm": "sha256",
            "files": outputs,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"Rebuilt 200,000 + 90,000 + 127,400 prompts in {args.output}")


if __name__ == "__main__":
    main()
