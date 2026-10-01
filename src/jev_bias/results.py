from __future__ import annotations

import re
from pathlib import Path

from .io import iter_jsonl


KEY_ORDER = {chr(ord("A") + i): i for i in range(26)}
LEVEL_RE = re.compile(r"(?:level|toxicity level)\s*0*(\d+)", re.I)


def probability_vector(value, k: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value[:k]]
    if isinstance(value, dict):
        return [float(v) for _, v in sorted(value.items(), key=lambda item: KEY_ORDER.get(item[0].upper(), 999))][:k]
    return None


def normalize_result(row: dict) -> dict:
    pred = row.get("predicted_position", row.get("prediction_position"))
    gold = row.get("gold_position", row.get("expected_position"))
    k = row.get("option_count", row.get("choice_count"))
    if k is None and isinstance(row.get("choices"), list):
        k = len(row["choices"])
    if k is None and isinstance(row.get("probabilities"), (list, dict)):
        k = len(row["probabilities"])
    probs = probability_vector(row.get("probabilities"), int(k)) if k is not None else None
    return {
        "sample_id": row.get("sample_id"), "source": row.get("source"),
        "status": row.get("status"), "pred_position": None if pred is None else int(pred),
        "gold_position": None if gold is None else int(gold), "k": None if k is None else int(k),
        "pred_text": row.get("predicted_label_text", row.get("prediction_text")),
        "gold_text": row.get("gold_label_text", row.get("expected_text")),
        "probabilities": probs, "raw": row,
    }


def stable_prediction_record(row: dict) -> dict:
    """Normalize historical and current runner schemas to comparable semantics."""
    normalized = normalize_result(row)
    probabilities = normalized["probabilities"]
    # Released KEV files stored four-decimal probabilities whereas current
    # PyTorch runners emit full floats.  Four decimals are the shared recorded
    # precision and avoid treating harmless accelerator noise as a new result.
    stable_probabilities = None if probabilities is None else [round(value, 4) for value in probabilities]
    return {
        "sample_id": normalized["sample_id"],
        "source": normalized["source"],
        "status": normalized["status"],
        "requested_model": row.get("requested_model") or row.get("model"),
        "returned_model": row.get("returned_model"),
        "predicted_position": normalized["pred_position"],
        "gold_position": normalized["gold_position"],
        "option_count": normalized["k"],
        "predicted_text": normalized["pred_text"],
        "gold_text": normalized["gold_text"],
        "probabilities": stable_probabilities,
        "correct": row.get("correct"),
        "error_type": row.get("error_type"),
        "error": row.get("error"),
    }


def stable_decision_record(row: dict) -> dict:
    """Stable argmax-level identity, independent of probability precision."""
    stable = stable_prediction_record(row)
    return {key: value for key, value in stable.items() if key != "probabilities"}


def load_latest(path: str | Path) -> dict[str, dict]:
    latest = {}
    for row in iter_jsonl(path):
        normalized = normalize_result(row)
        if normalized["sample_id"]:
            latest[normalized["sample_id"]] = normalized
    return latest


def canonical_level_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = LEVEL_RE.search(text)
    return int(match.group(1)) - 1 if match else None
