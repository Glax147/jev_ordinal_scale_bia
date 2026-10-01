from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterable

from .io import selection_digest, sha256_file
from .results import stable_decision_record, stable_prediction_record


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def scan_jsonl_binary(handle: BinaryIO, *, predictions: bool = False) -> dict:
    """Stream-scan JSONL without normalizing its bytes.

    ``content_sha256`` covers the exact decompressed byte stream.  The ordered
    and unordered ID digests provide a second, format-independent identity for
    the benchmark selection.  For prediction files, ``prediction_digest``
    excludes latency, timestamps and token-usage fields that naturally vary on
    a rerun.
    """
    content = hashlib.sha256()
    ordered_ids = hashlib.sha256()
    ordered_prediction_digest = hashlib.sha256()
    prediction_rows: list[tuple[str, str]] = []
    decision_rows: list[tuple[str, str]] = []
    sample_ids: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    statuses: Counter[str] = Counter()
    rows = 0
    byte_count = 0

    for line_number, raw in enumerate(handle, 1):
        content.update(raw)
        byte_count += len(raw)
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid UTF-8 JSONL at line {line_number}: {exc}") from exc
        rows += 1
        sample_id = row.get("sample_id") or row.get("row_uid")
        if sample_id is None:
            raise ValueError(f"Missing sample_id at line {line_number}")
        sample_id = str(sample_id)
        sample_ids.append(sample_id)
        ordered_ids.update(sample_id.encode("utf-8"))
        ordered_ids.update(b"\n")
        if sample_id in seen and len(duplicates) < 10:
            duplicates.append(sample_id)
        seen.add(sample_id)
        if predictions:
            statuses[str(row.get("status", "missing"))] += 1
            stable = stable_prediction_record(row)
            stable_bytes = canonical_bytes(stable)
            ordered_prediction_digest.update(stable_bytes)
            ordered_prediction_digest.update(b"\n")
            prediction_rows.append((sample_id, hashlib.sha256(stable_bytes).hexdigest()))
            decision_bytes = canonical_bytes(stable_decision_record(row))
            decision_rows.append((sample_id, hashlib.sha256(decision_bytes).hexdigest()))

    result = {
        "rows": rows,
        "bytes": byte_count,
        "content_sha256": content.hexdigest(),
        "ordered_ids_sha256": ordered_ids.hexdigest(),
        "selection_sha256": selection_digest(sample_ids),
        "unique_ids": len(seen),
        "duplicate_ids": duplicates,
    }
    if predictions:
        prediction_digest = hashlib.sha256()
        for sample_id, row_digest in sorted(prediction_rows):
            prediction_digest.update(sample_id.encode("utf-8"))
            prediction_digest.update(b"\0")
            prediction_digest.update(row_digest.encode("ascii"))
            prediction_digest.update(b"\n")
        result["statuses"] = dict(sorted(statuses.items()))
        result["prediction_digest_sha256"] = prediction_digest.hexdigest()
        result["ordered_prediction_digest_sha256"] = ordered_prediction_digest.hexdigest()
        decision_digest = hashlib.sha256()
        for sample_id, row_digest in sorted(decision_rows):
            decision_digest.update(sample_id.encode("utf-8"))
            decision_digest.update(b"\0")
            decision_digest.update(row_digest.encode("ascii"))
            decision_digest.update(b"\n")
        result["decision_digest_sha256"] = decision_digest.hexdigest()
    return result


def scan_gzip_jsonl(path: str | Path, *, predictions: bool = False) -> dict:
    path = Path(path)
    with gzip.open(path, "rb") as handle:
        result = scan_jsonl_binary(handle, predictions=predictions)
    result["archive_bytes"] = path.stat().st_size
    result["archive_sha256"] = sha256_file(path)
    return result


def tree_manifest(root: str | Path, *, exclude_names: Iterable[str] = (),
                  exclude_dirs: Iterable[str] = (".git", ".cache", "__pycache__")) -> dict:
    """Hash every regular file and then hash the sorted portable manifest."""
    root = Path(root)
    excluded = set(exclude_names)
    excluded_dirs = set(exclude_dirs)
    files = []
    for path in sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name not in excluded and not (set(p.relative_to(root).parts) & excluded_dirs)
    ):
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    digest = hashlib.sha256()
    for item in files:
        digest.update(f"{item['sha256']}  {item['bytes']}  {item['path']}\n".encode("utf-8"))
    return {"files": files, "tree_sha256": digest.hexdigest(), "file_count": len(files)}
