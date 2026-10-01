#!/usr/bin/env python3
"""Compare released KEV baselines with the released BA-LoRA runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from itertools import zip_longest
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[2]
ROOT = BUNDLE_ROOT / "posttraining"
PAIRS = {
    "kev-0.8b": (
        BUNDLE_ROOT / "results/released/kev-0.8b__main-40datasets__200000.jsonl.gz",
        ROOT / "results/released/kev-0.8b__ba-lora-seed42__main40.jsonl.gz",
    ),
    "kev-4b": (
        BUNDLE_ROOT / "results/released/kev-4b__main-40datasets__200000.jsonl.gz",
        ROOT / "results/released/kev-4b__ba-lora-seed42__main40.jsonl.gz",
    ),
}


def entropy(values: list[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0)


def accumulator(k: int, task_type: str) -> dict:
    return {
        "k": k, "task_type": task_type, "n": 0, "gold": Counter(),
        "pred_before": Counter(), "pred_after": Counter(),
        "joint_before": Counter(), "joint_after": Counter(),
        "correct_before": 0, "correct_after": 0,
        "abs_before": 0, "abs_after": 0,
        "prob_before": [0.0] * k, "prob_after": [0.0] * k,
        "margin_before": 0.0, "margin_after": 0.0,
    }


def add(acc: dict, gold: int, before: int, after: int, pb: list[float], pa: list[float]) -> None:
    acc["n"] += 1
    acc["gold"][gold] += 1
    for phase, pred, probs in (("before", before, pb), ("after", after, pa)):
        acc[f"pred_{phase}"][pred] += 1
        acc[f"joint_{phase}"][(gold, pred)] += 1
        acc[f"correct_{phase}"] += gold == pred
        acc[f"abs_{phase}"] += abs(gold - pred)
        for index, value in enumerate(probs):
            acc[f"prob_{phase}"][index] += value
        ranked = sorted(probs, reverse=True)
        acc[f"margin_{phase}"] += ranked[0] - ranked[1]


def qwk(acc: dict, phase: str) -> float:
    n, k = acc["n"], acc["k"]
    observed = expected = 0.0
    for i in range(k):
        for j in range(k):
            weight = ((i - j) / (k - 1)) ** 2
            observed += weight * acc[f"joint_{phase}"][(i, j)] / n
            expected += weight * (acc["gold"][i] / n) * (acc[f"pred_{phase}"][j] / n)
    return 1.0 - observed / expected if expected else 1.0


def metrics(acc: dict, phase: str) -> dict:
    n, k = acc["n"], acc["k"]
    gold = [acc["gold"][i] / n for i in range(k)]
    pred = [acc[f"pred_{phase}"][i] / n for i in range(k)]
    probabilities = [value / n for value in acc[f"prob_{phase}"]]
    probability_total = sum(probabilities)
    probabilities = [value / probability_total for value in probabilities]
    effective_gold = math.exp(entropy(gold))
    effective_pred = math.exp(entropy(pred))
    ratio = effective_pred / effective_gold
    ordinal = acc["task_type"] == "ordinal_choice"
    return {
        "accuracy": acc[f"correct_{phase}"] / n,
        "uarg": effective_pred / k,
        "r": ratio,
        "dr": abs(ratio - 1.0),
        "tvd": 0.5 * sum(abs(a - b) for a, b in zip(pred, gold)),
        "nmae": acc[f"abs_{phase}"] / (n * (k - 1)) if ordinal else None,
        "qwk": qwk(acc, phase) if ordinal else None,
        "usoft": math.exp(entropy(probabilities)) / k,
        "margin": acc[f"margin_{phase}"] / n,
    }


def normalized(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("probability vector has non-positive mass")
    return [value / total for value in values]


def load_gold_index() -> dict[str, tuple[str, int, int]]:
    index = {}
    path = BUNDLE_ROOT / "data/indexes/main40.paper-v1.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index[str(row["sample_id"])] = (
                str(row["source"]), int(row["gold_position"]), int(row["option_count"])
            )
    if len(index) != 200_000:
        raise RuntimeError(f"expected 200,000 indexed main40 IDs; got {len(index)}")
    return index


def analyze_pair(
    model: str,
    before_path: Path,
    after_path: Path,
    gold_index: dict[str, tuple[str, int, int]],
) -> tuple[list[dict], dict]:
    grouped: dict[str, dict] = {}
    input_pairs = comparable = 0
    excluded = []
    with gzip.open(before_path, "rt", encoding="utf-8") as before_handle, gzip.open(
        after_path, "rt", encoding="utf-8"
    ) as after_handle:
        for before_line, after_line in zip_longest(before_handle, after_handle):
            if before_line is None or after_line is None:
                raise RuntimeError(f"row-count mismatch for {model}")
            before = json.loads(before_line)
            after = json.loads(after_line)
            if before["sample_id"] != after["sample_id"]:
                raise RuntimeError(f"sample order mismatch for {model}: {before['sample_id']} != {after['sample_id']}")
            sample_id = str(before["sample_id"])
            if sample_id not in gold_index:
                raise RuntimeError(f"released ID is absent from main40 index: {sample_id}")
            source, gold, k = gold_index[sample_id]
            if source != str(before["source"]) or source != str(after["source"]):
                raise RuntimeError(f"source mismatch for {model}: {sample_id}")
            if k != int(after["option_count"]) or gold != int(after["gold_position"]):
                raise RuntimeError(f"post-training gold/schema mismatch for {model}: {sample_id}")
            input_pairs += 1
            if before.get("status") != "ok" or after.get("status") != "ok":
                excluded.append({
                    "sample_id": sample_id,
                    "source": source,
                    "before_status": str(before.get("status")),
                    "after_status": str(after.get("status")),
                })
                continue
            if k != int(before["choice_count"]) or gold != int(before["expected_position"]):
                raise RuntimeError(f"baseline gold/schema mismatch for {model}: {sample_id}")
            pb = normalized([float(before["probabilities"].get(chr(65 + i), 0.0)) for i in range(k)])
            pa = normalized([float(value) for value in after["probabilities"]])
            acc = grouped.setdefault(source, accumulator(k, str(after["task_type"])))
            add(acc, gold, int(before["prediction_position"]), int(after["predicted_position"]), pb, pa)
            comparable += 1
    if input_pairs != 200_000:
        raise RuntimeError(f"expected 200,000 input pairs for {model}; got {input_pairs}")
    if comparable + len(excluded) != input_pairs:
        raise RuntimeError(f"complete-case accounting mismatch for {model}")

    output = []
    metric_names = ("accuracy", "uarg", "r", "dr", "tvd", "nmae", "qwk", "usoft", "margin")
    for source, acc in sorted(grouped.items()):
        before = metrics(acc, "before")
        after = metrics(acc, "after")
        record = {"model": model, "source": source, "task_type": acc["task_type"], "n": acc["n"], "k": acc["k"]}
        for name in metric_names:
            record[f"{name}_before"] = before[name]
            record[f"{name}_after"] = after[name]
            record[f"{name}_delta"] = None if before[name] is None else after[name] - before[name]
        output.append(record)
    return output, {
        "model": model,
        "input_pairs": input_pairs,
        "comparable_pairs": comparable,
        "excluded_count": len(excluded),
        "excluded": excluded,
        "dataset_n": {record["source"]: record["n"] for record in output},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/generated/analysis/ba_lora_changes_all40.csv")
    args = parser.parse_args()
    gold_index = load_gold_index()
    records, exclusions = [], []
    for model, paths in PAIRS.items():
        model_records, model_exclusions = analyze_pair(model, *paths, gold_index)
        records.extend(model_records)
        exclusions.append(model_exclusions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    exclusions_path = args.output.with_name(args.output.stem + "__exclusions.json")
    exclusions_path.write_text(
        json.dumps({"policy": "paired complete cases; every exclusion is enumerated", "models": exclusions},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": args.output.relative_to(ROOT).as_posix() if args.output.is_relative_to(ROOT) else args.output.name,
        "rows": len(records),
        "datasets_by_model": dict(Counter(record["model"] for record in records)),
        "comparable_items_by_model": {
            model: sum(record["n"] for record in records if record["model"] == model) for model in PAIRS
        },
        "excluded_items_by_model": {item["model"]: item["excluded_count"] for item in exclusions},
        "exclusions": exclusions_path.relative_to(ROOT).as_posix() if exclusions_path.is_relative_to(ROOT) else exclusions_path.name,
    }, indent=2))


if __name__ == "__main__":
    main()
