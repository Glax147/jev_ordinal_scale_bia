from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def entropy(values: Sequence[float]) -> float:
    return -sum(float(p) * math.log(float(p)) for p in values if p > 0)


def distribution(labels: Iterable[int], k: int) -> np.ndarray:
    counts = Counter(int(x) for x in labels)
    n = sum(counts.values())
    if not n:
        return np.zeros(k, dtype=float)
    return np.array([counts[i] / n for i in range(k)], dtype=float)


def effective_support(values: Sequence[float]) -> float:
    return math.exp(entropy(values))


def qwk_safe(gold: list[int], pred: list[int]) -> float | None:
    """Quadratic weighted Cohen's kappa without a scikit-learn dependency."""
    if len(gold) != len(pred) or not gold:
        return None
    k = max(max(gold), max(pred)) + 1
    observed = np.zeros((k, k), dtype=float)
    for a, b in zip(gold, pred):
        observed[int(a), int(b)] += 1.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(gold)
    if k == 1:
        return None
    indices = np.arange(k, dtype=float)
    weights = ((indices[:, None] - indices[None, :]) / (k - 1)) ** 2
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return None
    return float(1.0 - (weights * observed).sum() / denominator)


def summarize(gold: list[int], pred: list[int], probabilities: list[list[float]] | None, k: int,
              ordinal: bool) -> dict:
    if not gold or len(gold) != len(pred):
        raise ValueError("gold and prediction arrays must have the same nonzero length")
    q = distribution(gold, k)
    p = distribution(pred, k)
    accuracy = sum(a == b for a, b in zip(gold, pred)) / len(gold)
    u_arg = effective_support(p) / k
    gold_coverage = effective_support(q) / k
    ratio = effective_support(p) / effective_support(q)
    tvd = 0.5 * float(np.abs(p - q).sum())
    payload = {
        "n": len(gold), "k": k, "accuracy": accuracy,
        "u_arg": u_arg, "gold_coverage": gold_coverage,
        "gold_relative_utilization": ratio,
        "support_deviation": abs(ratio - 1.0),
        "tvd": tvd,
        "gold_counts": [int((q[i] * len(gold)) + 0.5) for i in range(k)],
        "prediction_counts": [int((p[i] * len(gold)) + 0.5) for i in range(k)],
    }
    if ordinal:
        payload["nmae"] = sum(abs(a - b) for a, b in zip(gold, pred)) / (len(gold) * (k - 1))
        payload["qwk"] = qwk_safe(gold, pred)
    if probabilities:
        matrix = np.asarray(probabilities, dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        mean_probs = matrix.mean(axis=0)
        payload["u_soft"] = effective_support(mean_probs) / k
        payload["normalized_entropy"] = float(np.mean([
            entropy(row) / math.log(k) if k > 1 else 1.0 for row in matrix
        ]))
        top2 = np.partition(matrix, -2, axis=1)[:, -2:]
        payload["margin"] = float(np.mean(top2[:, 1] - top2[:, 0]))
    return payload
