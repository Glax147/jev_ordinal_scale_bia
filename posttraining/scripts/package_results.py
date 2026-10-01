#!/usr/bin/env python3
"""Package a completed BA-LoRA evaluation without recording machine paths."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_deterministic_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["kev-0.8b", "kev-4b"])
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    payload = args.answers.read_bytes()
    if not payload.endswith(b"\n"):
        payload += b"\n"
    ids: list[str] = []
    statuses: Counter[str] = Counter()
    canonical = hashlib.sha256()
    for line in payload.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids.append(str(row["sample_id"]))
        statuses[str(row.get("status", "unknown"))] += 1
        canonical.update(
            (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        )
    if len(ids) != 200_000 or len(set(ids)) != 200_000:
        raise ValueError(f"expected 200,000 unique rows; got rows={len(ids)}, unique={len(set(ids))}")
    if statuses != {"ok": 200_000}:
        raise ValueError(f"released evaluation is not complete: {dict(statuses)}")

    stem = f"{args.model}__ba-lora-seed42__main40"
    answer_out = args.output_dir / f"{stem}.jsonl.gz"
    write_deterministic_gzip(answer_out, payload)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    summary["checkpoint"] = f"{args.model}-ba-lora-seed42"
    summary["output"] = f"results/released/{answer_out.name}"
    summary["release_provenance"] = {
        "source_answers_bytes": len(payload),
        "source_answers_sha256": sha256_bytes(payload),
        "canonical_records_sha256": canonical.hexdigest(),
        "row_count": len(ids),
        "unique_sample_ids": len(set(ids)),
        "sample_ids_ordered_sha256": sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")),
    }
    summary_out = args.output_dir / f"{stem}__score_summary.json"
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "answers": answer_out.name,
        "answers_gzip_sha256": sha256_bytes(answer_out.read_bytes()),
        "summary": summary_out.name,
        **summary["release_provenance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
