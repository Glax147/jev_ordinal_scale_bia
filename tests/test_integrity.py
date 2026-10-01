import io
import json

from jev_bias.integrity import canonical_bytes, scan_jsonl_binary
from jev_bias.results import stable_prediction_record


def test_historical_and_current_prediction_schemas_normalize_identically():
    historical = {
        "sample_id": "x", "source": "d", "status": "ok", "model": "kev-0.8b",
        "prediction_position": 1, "expected_position": 0, "choice_count": 2,
        "prediction_text": "B", "expected_text": "A", "probabilities": {"A": 0.4, "B": 0.6},
        "correct": False,
    }
    current = {
        "schema_version": "kev-answer-public-v1", "sample_id": "x", "source": "d", "status": "ok",
        "model": "kev-0.8b", "predicted_position": 1, "gold_position": 0, "option_count": 2,
        "predicted_label_text": "B", "gold_label_text": "A", "probabilities": [0.4, 0.6],
        "correct": False,
    }
    assert stable_prediction_record(historical) == stable_prediction_record(current)


def test_prediction_digest_is_order_independent_but_order_digest_is_not():
    rows = [
        {"sample_id": "a", "source": "d", "status": "ok", "model": "m", "choice_count": 2,
         "prediction_position": 0, "expected_position": 0, "probabilities": [0.8, 0.2], "correct": True},
        {"sample_id": "b", "source": "d", "status": "ok", "model": "m", "choice_count": 2,
         "prediction_position": 1, "expected_position": 0, "probabilities": [0.1, 0.9], "correct": False},
    ]
    payload_a = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    payload_b = b"".join(canonical_bytes(row) + b"\n" for row in reversed(rows))
    a = scan_jsonl_binary(io.BytesIO(payload_a), predictions=True)
    b = scan_jsonl_binary(io.BytesIO(payload_b), predictions=True)
    assert a["prediction_digest_sha256"] == b["prediction_digest_sha256"]
    assert a["ordered_prediction_digest_sha256"] != b["ordered_prediction_digest_sha256"]


def test_stable_digest_uses_shared_four_decimal_probability_precision():
    historical = {
        "sample_id": "x", "source": "d", "status": "ok", "model": "m", "choice_count": 2,
        "prediction_position": 0, "expected_position": 0,
        "probabilities": {"A": 0.6667, "B": 0.3333}, "correct": True,
    }
    current = {
        "sample_id": "x", "source": "d", "status": "ok", "model": "m", "option_count": 2,
        "predicted_position": 0, "gold_position": 0,
        "probabilities": [0.66669999, 0.33330001], "correct": True,
    }
    assert stable_prediction_record(historical) == stable_prediction_record(current)
    a = scan_jsonl_binary(io.BytesIO(canonical_bytes(historical) + b"\n"), predictions=True)
    b = scan_jsonl_binary(io.BytesIO(canonical_bytes(current) + b"\n"), predictions=True)
    assert a["prediction_digest_sha256"] == b["prediction_digest_sha256"]
    assert a["decision_digest_sha256"] == b["decision_digest_sha256"]
