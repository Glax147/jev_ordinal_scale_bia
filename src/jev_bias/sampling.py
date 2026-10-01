from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Iterable


def stable_hash(seed: int | str, source: str, stratum: str, sample_id: str) -> str:
    """Stable rank key: independent of Python version and input row order."""
    payload = f"{seed}\x1f{source}\x1f{stratum}\x1f{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sampling_cell(row: dict) -> str:
    """Portable composite key so equal stratum names from two sources never merge."""
    return json.dumps(
        [str(row["source"]), str(row["sampling_stratum"])],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def capped_balanced_allocation(available: dict[str, int], total: int) -> dict[str, int]:
    """Round-robin equal allocation with deterministic deficit redistribution."""
    available = {str(k): int(v) for k, v in available.items() if int(v) > 0}
    if sum(available.values()) < total:
        raise ValueError(f"Need {total} rows but only {sum(available.values())} are available")
    allocation = Counter()
    active = sorted(available)
    cursor = 0
    for _ in range(total):
        while allocation[active[cursor % len(active)]] >= available[active[cursor % len(active)]]:
            active.pop(cursor % len(active))
            cursor %= len(active)
        key = active[cursor % len(active)]
        allocation[key] += 1
        cursor += 1
    return dict(allocation)


def _prepare_buckets(rows: Iterable[dict], seed: int) -> tuple[dict[str, list[tuple[str, dict]]], int]:
    buckets: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    seen_ids: set[str] = set()
    representatives: dict[str, tuple[str, str, dict]] = {}
    labels_by_signature: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        # Deduplicate only within a source and include the candidate set.  The
        # signature deliberately excludes the answer so contradictory labels
        # can be detected rather than silently collapsed.
        signature_payload = {
            "source": str(row["source"]),
            "question": " ".join(str(row.get("question", "")).split()).casefold(),
            "context": " ".join(str(row.get("context", "")).split()).casefold(),
            "choices": [" ".join(str(x).split()).casefold() for x in row.get("choices", [])],
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        label = row.get("gold_label_text")
        if label is None:
            label = row.get("gold_position")
        if label is None:
            label = row.get("label")
        labels_by_signature[signature].add(json.dumps(label, sort_keys=True, ensure_ascii=False))
        stratum = str(row["sampling_stratum"])
        rank = stable_hash(seed, str(row["source"]), stratum, sample_id)
        candidate = (rank, sample_id, row)
        previous = representatives.get(signature)
        if previous is None or candidate[:2] < previous[:2]:
            representatives[signature] = candidate
    conflicts = sorted(signature for signature, labels in labels_by_signature.items() if len(labels) > 1)
    if conflicts:
        raise ValueError(f"Conflicting labels for {len(conflicts)} duplicate content groups; first={conflicts[0]}")
    for rank, _, row in representatives.values():
        buckets[sampling_cell(row)].append((rank, row))
    return buckets, len(seen_ids)


def hash_stratified_sample_with_stats(rows: Iterable[dict], total: int, seed: int) -> tuple[list[dict], dict]:
    """Return the deterministic sample plus the actual post-deduplication pool counts."""
    buckets, candidate_rows = _prepare_buckets(rows, seed)
    cells_by_source: dict[str, dict[str, int]] = defaultdict(dict)
    for cell, values in buckets.items():
        source = str(values[0][1]["source"])
        cells_by_source[source][cell] = len(values)
    source_allocation = capped_balanced_allocation(
        {source: sum(cells.values()) for source, cells in cells_by_source.items()}, total
    )
    allocation: dict[str, int] = {}
    for source, count in source_allocation.items():
        allocation.update(capped_balanced_allocation(cells_by_source[source], count))
    selected: list[tuple[str, dict]] = []
    for stratum, count in allocation.items():
        selected.extend(sorted(buckets[stratum], key=lambda item: (item[0], item[1]["sample_id"]))[:count])
    output = [row for _, row in sorted(selected, key=lambda item: (item[0], item[1]["sample_id"]))]
    stats = {
        "candidate_rows": candidate_rows,
        "eligible_rows_after_dedup": sum(len(values) for values in buckets.values()),
        "eligible_by_stratum_after_dedup": dict(sorted((key, len(values)) for key, values in buckets.items())),
        "allocation_by_source": dict(sorted(source_allocation.items())),
        "allocation_by_stratum": dict(sorted(allocation.items())),
    }
    return output, stats


def hash_stratified_sample(rows: Iterable[dict], total: int, seed: int) -> list[dict]:
    """Select the lowest SHA-256 ranks in each stratum, then hash-order the output.

    Required fields: ``sample_id``, ``source`` and ``sampling_stratum``.
    Duplicate sample IDs are rejected. Identical question/context/candidate
    groups are deduplicated before selection.
    """
    return hash_stratified_sample_with_stats(rows, total, seed)[0]


def select_frozen(rows: Iterable[dict], frozen_index: dict[str, dict]) -> list[dict]:
    """Recover paper-v1 rows by ID and verify their canonical row hashes."""
    found: dict[str, dict] = {}
    from .io import canonical_json_sha256

    for row in rows:
        sample_id = str(row["sample_id"])
        expected = frozen_index.get(sample_id)
        if expected is None:
            continue
        observed = canonical_json_sha256(row)
        if observed != expected["row_sha256"]:
            raise RuntimeError(f"Upstream row changed for {sample_id}: {observed} != {expected['row_sha256']}")
        found[sample_id] = row
    missing = sorted(set(frozen_index) - set(found))
    if missing:
        raise RuntimeError(f"Could not reconstruct {len(missing)} frozen rows; first: {missing[:5]}")
    return [found[sample_id] for sample_id, _ in sorted(frozen_index.items(), key=lambda item: item[1]["ordinal"])]
