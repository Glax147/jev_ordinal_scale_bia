#!/usr/bin/env python3
"""Recompute the released main, position, and candidate-count metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.io import iter_jsonl
from jev_bias.metrics import summarize
from jev_bias.results import load_latest, probability_vector

MODELS = ["jev-1.13", "kev-0.8b", "kev-4b", "kev-9b"]
NOMINAL = {"snips7", "yahoo_answers10", "scotus13", "dbpedia14"}


def result_file(results: Path, model: str, experiment: str) -> Path:
    tokens = {"main": "main-40datasets", "position": "position-ablation-9datasets",
              "kcurve": "choice-count-ablation-4datasets-k2to14"}
    matches = sorted(results.glob(f"{model}__{tokens[experiment]}__*.jsonl.gz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {model}/{experiment} file in {results}; found {matches}")
    return matches[0]


def load_index(path: Path) -> dict[str, dict]:
    return {row["sample_id"]: row for row in iter_jsonl(path)}


def remap_probability(probs: list[float] | None, permutation: list[int], canonical_k: int) -> list[float] | None:
    if probs is None:
        return None
    out = [0.0] * canonical_k
    for position, canonical in enumerate(permutation):
        if position < len(probs):
            out[int(canonical)] = float(probs[position])
    return out


def metrics_for(rows: list[dict], ordinal: bool, k: int) -> dict:
    return summarize([r["gold"] for r in rows], [r["pred"] for r in rows],
                     [r["probs"] for r in rows] if rows and all(r["probs"] is not None for r in rows) else None,
                     k, ordinal)


def main_metrics(results: Path, indexes: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    index = load_index(indexes / "main40.paper-v1.jsonl.gz")
    details, normalized = [], {}
    for model in MODELS:
        latest = load_latest(result_file(results, model, "main"))
        groups = defaultdict(list)
        for sample_id, meta in index.items():
            row = latest.get(sample_id)
            if not row or row["status"] != "ok":
                continue
            k = int(meta["option_count"])
            groups[meta["source"]].append({"gold": int(meta["gold_position"]), "pred": row["pred_position"],
                                            "probs": probability_vector(row["raw"].get("probabilities"), k)})
        normalized[model] = groups
        for source, rows in sorted(groups.items()):
            k = index[next(sid for sid, meta in index.items() if meta["source"] == source)]["option_count"]
            task = "nominal" if source in NOMINAL else "ordinal"
            m = metrics_for(rows, task == "ordinal", int(k))
            details.append({"model": model, "source": source, "task": task, **m})
    detail_df = pd.DataFrame(details)
    metric_columns = ["accuracy", "u_arg", "gold_relative_utilization", "support_deviation", "tvd", "nmae", "qwk",
                      "u_soft", "normalized_entropy", "margin"]
    macros = []
    for (model, task), group in detail_df.groupby(["model", "task"], sort=False):
        record = {"model": model, "task": task, "datasets": len(group)}
        for metric in metric_columns:
            if metric in group:
                record[metric] = group[metric].dropna().mean() if group[metric].notna().any() else None
        macros.append(record)
    return detail_df, pd.DataFrame(macros), normalized


def position_metrics(results: Path, indexes: Path, main_detail: pd.DataFrame, main_groups: dict) -> pd.DataFrame:
    index = load_index(indexes / "position9.paper-v1.jsonl.gz")
    records = []
    for model in MODELS:
        latest = load_latest(result_file(results, model, "position"))
        groups = defaultdict(list)
        paired = defaultdict(dict)
        for sample_id, meta in index.items():
            row = latest.get(sample_id)
            if not row or row["status"] != "ok":
                continue
            perm = [int(x) for x in meta["permutation_new_to_canonical"]]
            k = int(meta["option_count"])
            canonical_pred = perm[row["pred_position"]]
            canonical_probs = remap_probability(probability_vector(row["raw"].get("probabilities"), k), perm, k)
            item = {"gold": int(meta["original_gold_position"]), "pred": canonical_pred, "probs": canonical_probs}
            key = (meta["source"], meta["order_variant"])
            groups[key].append(item)
            paired[(meta["source"], meta["base_sample_id"])][meta["order_variant"]] = item
        for source in sorted({key[0] for key in groups}):
            task = "nominal" if source in NOMINAL else "ordinal"
            variants = []
            for variant in sorted(v for s, v in groups if s == source):
                rows = groups[(source, variant)]
                variants.append(metrics_for(rows, task == "ordinal", len(rows[0]["probs"])))
            avg = {key: float(np.mean([m[key] for m in variants if m.get(key) is not None]))
                   for key in variants[0] if key not in {"gold_counts", "prediction_counts"} and
                   any(m.get(key) is not None for m in variants)}
            pairs = [value for (s, _), value in paired.items() if s == source and len(value) == 2]
            agreement, shifts = [], []
            for value in pairs:
                a, b = [value[k] for k in sorted(value)]
                agreement.append(a["pred"] == b["pred"])
                if a["probs"] is not None and b["probs"] is not None:
                    shifts.append(0.5 * sum(abs(x - y) for x, y in zip(a["probs"], b["probs"])))
            original = main_detail[(main_detail.model == model) & (main_detail.source == source)].iloc[0]
            records.append({
                "model": model, "source": source, "task": task,
                "original_accuracy": original.accuracy,
                "randomized_accuracy": avg["accuracy"],
                "original_r": original.gold_relative_utilization,
                "randomized_r": avg["gold_relative_utilization"],
                "original_d_r": original.support_deviation,
                "randomized_d_r": avg["support_deviation"],
                "original_tvd": original.tvd, "randomized_tvd": avg["tvd"],
                "pairwise_agreement": float(np.mean(agreement)),
                "probability_shift_tvd": float(np.mean(shifts)) if shifts else None,
            })
    return pd.DataFrame(records)


def kcurve_metrics(results: Path, indexes: Path) -> pd.DataFrame:
    index = load_index(indexes / "kcurve4.paper-v1.jsonl.gz")
    records = []
    for model in MODELS:
        latest = load_latest(result_file(results, model, "kcurve"))
        groups = defaultdict(list)
        for sample_id, meta in index.items():
            row = latest.get(sample_id)
            if not row or row["status"] != "ok":
                continue
            perm = [int(x) for x in meta["permutation_new_to_canonical"]]
            canonical_k = 14 if meta["source"] == "dbpedia14" else int(meta["option_count"])
            observed_k = int(meta["option_count"])
            groups[(meta["source"], meta.get("binning_method") or "candidate_subset", int(meta["option_count"]))].append({
                "gold": int(meta["gold_canonical_level"]), "pred": perm[row["pred_position"]],
                "probs": remap_probability(
                    probability_vector(row["raw"].get("probabilities"), observed_k), perm, canonical_k
                ),
            })
        for (source, binning, k), rows in sorted(groups.items()):
            if source == "dbpedia14":
                accuracy = sum(r["gold"] == r["pred"] for r in rows) / len(rows)
                probs = [r["probs"] for r in rows if r["probs"] is not None]
                margins = []
                for item in probs:
                    values = sorted(item, reverse=True)
                    margins.append(values[0] - values[1])
                records.append({"model": model, "source": source, "binning": binning, "k": k,
                                "n": len(rows), "accuracy": accuracy,
                                "margin": float(np.mean(margins)) if margins else None})
            else:
                records.append({"model": model, "source": source, "binning": binning, "k": k,
                                **metrics_for(rows, True, k)})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "released")
    parser.add_argument("--indexes", type=Path, default=ROOT / "data" / "indexes")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "metrics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main_detail, main_macro, main_groups = main_metrics(args.results, args.indexes)
    position = position_metrics(args.results, args.indexes, main_detail, main_groups)
    kcurve = kcurve_metrics(args.results, args.indexes)
    for name, frame in [("main_by_dataset", main_detail), ("main_macro", main_macro),
                        ("position_by_dataset", position), ("kcurve_by_condition", kcurve)]:
        frame.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8")
        (args.output / f"{name}.json").write_text(
            json.dumps(frame.replace({np.nan: None}).to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(main_macro.to_string(index=False))
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
