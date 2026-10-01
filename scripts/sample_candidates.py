#!/usr/bin/env python3
"""Create a new versioned, order-independent hash-stratified sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_bias.io import (canonical_json_sha256, iter_jsonl, ordered_selection_digest,
                         selection_digest, sha256_file, write_jsonl)
from jev_bias.sampling import hash_stratified_sample_with_stats, sampling_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Canonical unsampled candidate JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    candidates = list(iter_jsonl(args.input))
    rows, pool_stats = hash_stratified_sample_with_stats(candidates, args.n, args.seed)
    write_jsonl(args.output, rows)
    content_hashes = sorted(canonical_json_sha256(row) for row in candidates)
    candidate_pool_content = hashlib.sha256("".join(f"{value}\n" for value in content_hashes).encode()).hexdigest()
    manifest = {
        "schema_version": "jev-hash-sample-manifest-v2",
        "method": "sha256-stratified-lowest-rank",
        "rank_formula": "SHA256(seed \\x1f source \\x1f stratum \\x1f sample_id)",
        "deduplication": "deterministic-within-source-question-context-choices",
        "seed": args.seed,
        "candidate_rows": len(candidates),
        "eligible_rows_after_dedup": pool_stats["eligible_rows_after_dedup"],
        "eligible_by_stratum_after_dedup": pool_stats["eligible_by_stratum_after_dedup"],
        "allocation_by_source": pool_stats["allocation_by_source"],
        "allocation_by_source_and_stratum": pool_stats["allocation_by_stratum"],
        "candidate_pool_selection_sha256": selection_digest(str(row["sample_id"]) for row in candidates),
        "candidate_pool_content_sha256": candidate_pool_content,
        "available_by_source_and_stratum": dict(sorted(Counter(sampling_cell(row) for row in candidates).items())),
        "rows": len(rows),
        "selected_by_source_and_stratum": dict(sorted(Counter(sampling_cell(row) for row in rows).items())),
        "selection_sha256": selection_digest(str(row["sample_id"]) for row in rows),
        "ordered_ids_sha256": ordered_selection_digest(str(row["sample_id"]) for row in rows),
        "output_sha256": sha256_file(args.output),
        "sampler_code_sha256": sha256_file(ROOT / "src" / "jev_bias" / "sampling.py"),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
