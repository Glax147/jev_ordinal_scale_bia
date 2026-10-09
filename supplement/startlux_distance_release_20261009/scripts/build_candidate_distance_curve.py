"""Build a seven-point candidate-distance curve for four ordinal datasets.

This extends ``build_candidate_distance_pairs.py`` while preserving its frozen
source rows, label balancing, displayed gold positions, and option-order
counterbalancing.  Each distance condition partitions the complete ordinal
scale into disjoint candidate sets.  Consequently every label in a displayed
set serves equally often as gold, so the candidate set alone cannot reveal the
answer.  No LLM or manual rewriting is used.
"""

from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
import random
import statistics
from pathlib import Path


SOURCES = (
    "word_concreteness",
    "wmt20_translation_quality",
    "measuring_hate_speech",
    "stanford_politeness",
)
VERSION = "ordinal_candidate_distance_curve_balanced_v1.0"
SEED = 20261007
LEVELS = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rng_for(seed: int, key: str) -> random.Random:
    value = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(value[:16], "big"))


def load_rows(path: Path) -> tuple[dict[str, list[dict]], dict[str, int]]:
    selected = {source: [] for source in SOURCES}
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            if source not in selected:
                continue
            sample_id = row["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id at line {number}: {sample_id}")
            seen_ids.add(sample_id)
            choices = row["choices"]
            gold = row["gold_position"]
            if not isinstance(gold, int) or not 0 <= gold < len(choices):
                raise ValueError(f"Invalid gold position: {sample_id}")
            if choices != [f"Level {i:02d}" for i in range(1, len(choices) + 1)]:
                raise ValueError(f"Nonstandard ordinal labels: {sample_id}")
            if choices[gold] != row["gold_label_text"]:
                raise ValueError(f"Incorrect gold text: {sample_id}")
            if not row.get("context") or not row.get("question"):
                raise ValueError(f"Missing text: {sample_id}")
            score = row.get("source_score")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"Missing continuous source score: {sample_id}")
            selected[source].append(row)

    duplicate_removed = {}
    for source, rows in selected.items():
        counts = collections.Counter(row["context"] for row in rows)
        selected[source] = [row for row in rows if counts[row["context"]] == 1]
        duplicate_removed[source] = len(rows) - len(selected[source])
        if not selected[source]:
            raise ValueError(f"No rows for {source}")
    return selected, duplicate_removed


def balance_original_labels(rows_by_source: dict[str, list[dict]], seed: int) -> dict[str, int]:
    removed = {}
    for source, rows in rows_by_source.items():
        groups: dict[int, list[dict]] = collections.defaultdict(list)
        scale_size = len(rows[0]["choices"])
        choice_count = 3 if scale_size % 3 == 0 else 4
        for row in rows:
            if len(row["choices"]) != scale_size:
                raise ValueError(f"Mixed scale sizes in {source}")
            groups[row["gold_position"]].append(row)
        if set(groups) != set(range(scale_size)):
            raise ValueError(f"Missing original gold level in {source}")
        ordering_count = math.factorial(choice_count)
        keep_per_label = (min(map(len, groups.values())) // ordering_count) * ordering_count
        if keep_per_label == 0:
            raise ValueError(f"Insufficient rows for {source}")
        kept = []
        for label in range(scale_size):
            group = sorted(groups[label], key=lambda row: row["sample_id"])
            rng_for(seed, f"balance|{source}|{label}").shuffle(group)
            kept.extend(group[:keep_per_label])
        removed[source] = len(rows) - len(kept)
        rows_by_source[source] = kept
    return removed


def partitions(items: tuple[int, ...], group_size: int):
    """Yield each unordered equal-size partition exactly once."""
    if not items:
        yield ()
        return
    first = items[0]
    for rest in itertools.combinations(items[1:], group_size - 1):
        group = (first,) + rest
        remaining = tuple(item for item in items if item not in group)
        for tail in partitions(remaining, group_size):
            yield (group,) + tail


def group_gap(levels: tuple[int, ...], scale_size: int) -> Fraction:
    distances = [
        abs(a - b)
        for index, a in enumerate(levels)
        for b in levels[index + 1 :]
    ]
    return Fraction(sum(distances), len(distances) * (scale_size - 1))


def select_distance_partitions(scale_size: int, choice_count: int) -> list[dict]:
    """Choose seven distinct, approximately equally spaced partition means.

    For every attainable mean, retain the partition with the smallest
    within-condition distance dispersion.  Endpoints are therefore the same
    contiguous and evenly interleaved constructions used by the paired study.
    """
    best_by_mean: dict[Fraction, tuple] = {}
    for partition in partitions(tuple(range(scale_size)), choice_count):
        gaps = tuple(group_gap(group, scale_size) for group in partition)
        mean_gap = sum(gaps, Fraction()) / len(gaps)
        dispersion = statistics.pstdev(float(gap) for gap in gaps)
        candidate = (dispersion, partition, gaps)
        if mean_gap not in best_by_mean or candidate < best_by_mean[mean_gap]:
            best_by_mean[mean_gap] = candidate

    available = sorted(best_by_mean)
    if len(available) < LEVELS:
        raise ValueError(
            f"Only {len(available)} balanced partition distances for "
            f"K0={scale_size}, K={choice_count}"
        )
    low, high = available[0], available[-1]
    selected_means: list[Fraction] = []
    for index in range(LEVELS):
        target = low + (high - low) * Fraction(index, LEVELS - 1)
        chosen = min(
            (value for value in available if value not in selected_means),
            key=lambda value: (abs(value - target), value),
        )
        selected_means.append(chosen)
    selected_means.sort()

    result = []
    for index, mean_gap in enumerate(selected_means, 1):
        dispersion, partition, gaps = best_by_mean[mean_gap]
        result.append(
            {
                "distance_level": index,
                "condition": f"distance_curve_d{index:02d}",
                "mean_gap": mean_gap,
                "relative_distance": Fraction(mean_gap - low, high - low),
                "dispersion": dispersion,
                "partition": partition,
                "group_gaps": gaps,
            }
        )
    return result


def make_orderings(rows: list[dict], choice_count: int, seed: int, source: str) -> dict:
    by_gold: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_gold[row["gold_position"]].append(row)
    codes = [
        (gold_slot, permutation)
        for gold_slot in range(choice_count)
        for permutation in itertools.permutations(range(choice_count - 1))
    ]
    assigned = {}
    for label, group in by_gold.items():
        group = sorted(group, key=lambda row: row["sample_id"])
        rng_for(seed, f"position|{source}|{label}").shuffle(group)
        for index, row in enumerate(group):
            assigned[row["sample_id"]] = codes[index % len(codes)]
    return assigned


def build_row(original: dict, levels: tuple[int, ...], ordering: tuple,
              condition_spec: dict, seed: int) -> dict:
    gold = original["gold_position"]
    gold_slot, distractor_permutation = ordering
    distractors = [level for level in levels if level != gold]
    chosen: list[int | None] = [None] * len(levels)
    chosen[gold_slot] = gold
    for slot, permutation_index in zip(
        (slot for slot in range(len(levels)) if slot != gold_slot),
        distractor_permutation,
    ):
        chosen[slot] = distractors[permutation_index]
    if any(level is None for level in chosen):
        raise AssertionError("Unfilled option slot")
    ranks = [int(level) for level in chosen]
    actual_gap = group_gap(tuple(sorted(ranks)), len(original["choices"]))
    level = condition_spec["distance_level"]
    return {
        "sample_id": f"{original['sample_id']}::distance_curve_d{level:02d}",
        "pair_id": original["sample_id"],
        "source": original["source"],
        "original_id": original.get("original_id"),
        "split": original.get("split"),
        "condition": condition_spec["condition"],
        "distance_level": level,
        "distance_levels_total": LEVELS,
        "relative_distance": round(float(condition_spec["relative_distance"]), 8),
        "partition_mean_pairwise_rank_distance_normalized": round(
            float(condition_spec["mean_gap"]), 8
        ),
        "mean_pairwise_rank_distance_normalized": round(float(actual_gap), 8),
        "task_type": "ordinal_choice",
        "decision_format": "multiple_choice",
        "context": original["context"],
        "question": original["question"],
        "choices": [original["choices"][rank] for rank in ranks],
        "gold_position": gold_slot,
        "gold_label_text": original["gold_label_text"],
        "option_count": len(levels),
        "original_option_count": len(original["choices"]),
        "original_gold_rank": gold,
        "choice_to_original_rank": ranks,
        "gold_ordinal_rank_among_shown": sum(rank < gold for rank in ranks),
        "source_score": original["source_score"],
        "source_repo": original.get("source_repo"),
        "source_revision": original.get("source_revision"),
        "original_context_truncated": original.get("context_truncated", False),
        "sampling_seed": seed,
        "construction_version": VERSION,
    }


def build_curve(rows_by_source: dict[str, list[dict]], seed: int):
    output = []
    specifications = {}
    for source in SOURCES:
        rows = rows_by_source[source]
        scale_size = len(rows[0]["choices"])
        choice_count = 3 if scale_size % 3 == 0 else 4
        conditions = select_distance_partitions(scale_size, choice_count)
        specifications[source] = conditions
        orderings = make_orderings(rows, choice_count, seed, source)
        for original in rows:
            gold = original["gold_position"]
            for condition in conditions:
                candidate_set = next(
                    group for group in condition["partition"] if gold in group
                )
                output.append(
                    build_row(
                        original,
                        candidate_set,
                        orderings[original["sample_id"]],
                        condition,
                        seed,
                    )
                )
    rng_for(seed, "curve_output_order").shuffle(output)
    return output, specifications


def balance_report(rows: list[dict]) -> dict:
    unordered: dict[tuple[str, ...], collections.Counter] = collections.defaultdict(collections.Counter)
    ordered: dict[tuple[str, ...], collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        unordered[tuple(sorted(row["choices"]))][row["gold_label_text"]] += 1
        ordered[tuple(row["choices"])][row["gold_label_text"]] += 1

    def validate(groups: dict) -> dict:
        largest_difference = 0
        for choices, gold_counts in groups.items():
            if set(gold_counts) != set(choices):
                raise AssertionError(f"Candidate-set gold leakage: {choices}, {gold_counts}")
            difference = max(gold_counts.values()) - min(gold_counts.values())
            largest_difference = max(largest_difference, difference)
        if largest_difference:
            raise AssertionError(f"Unequal gold counts within a candidate set: {largest_difference}")
        return {
            "candidate_sets": len(groups),
            "max_gold_count_difference_within_set": largest_difference,
        }

    return {"unordered": validate(unordered), "ordered": validate(ordered)}


def validate_and_report(rows: list[dict], specifications: dict) -> dict:
    expected_ids = {row["sample_id"] for row in rows}
    if len(expected_ids) != len(rows):
        raise AssertionError("Duplicate output sample_id")
    pair_counts = collections.Counter(row["pair_id"] for row in rows)
    if set(pair_counts.values()) != {LEVELS}:
        raise AssertionError("Not every source item has seven distance versions")

    rows_by_pair: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        rows_by_pair[row["pair_id"]].append(row)
        if row["choices"][row["gold_position"]] != row["gold_label_text"]:
            raise AssertionError(f"Displayed gold mismatch: {row['sample_id']}")
        if row["choice_to_original_rank"][row["gold_position"]] != row["original_gold_rank"]:
            raise AssertionError(f"Gold-rank mapping mismatch: {row['sample_id']}")
    held_fixed = (
        "source",
        "original_id",
        "split",
        "context",
        "question",
        "gold_label_text",
        "gold_position",
        "option_count",
        "original_option_count",
        "original_gold_rank",
        "source_score",
    )
    for pair_id, versions in rows_by_pair.items():
        reference = versions[0]
        for version in versions[1:]:
            changed = [field for field in held_fixed if version.get(field) != reference.get(field)]
            if changed:
                raise AssertionError(f"Pair controls changed for {pair_id}: {changed}")

    source_report = {}
    for source in SOURCES:
        source_rows = [row for row in rows if row["source"] == source]
        condition_report = []
        previous_mean = -1.0
        for spec in specifications[source]:
            condition_rows = [
                row for row in source_rows if row["distance_level"] == spec["distance_level"]
            ]
            distances = [row["mean_pairwise_rank_distance_normalized"] for row in condition_rows]
            mean_distance = statistics.mean(distances)
            if mean_distance <= previous_mean:
                raise AssertionError(f"Non-increasing curve for {source}")
            previous_mean = mean_distance
            gold_counts = collections.Counter(row["gold_label_text"] for row in condition_rows)
            position_counts = collections.Counter(row["gold_position"] for row in condition_rows)
            ordinal_rank_counts = collections.Counter(
                row["gold_ordinal_rank_among_shown"] for row in condition_rows
            )
            if len(set(gold_counts.values())) != 1:
                raise AssertionError(f"Unbalanced gold labels: {source}, {spec['condition']}")
            if len(set(position_counts.values())) != 1:
                raise AssertionError(f"Unbalanced gold positions: {source}, {spec['condition']}")
            if len(set(ordinal_rank_counts.values())) != 1:
                raise AssertionError(f"Unbalanced ordinal ranks: {source}, {spec['condition']}")
            condition_report.append(
                {
                    "distance_level": spec["distance_level"],
                    "condition": spec["condition"],
                    "question_rows": len(condition_rows),
                    "relative_distance": round(float(spec["relative_distance"]), 8),
                    "mean_pairwise_rank_distance_normalized": round(mean_distance, 8),
                    "min_item_distance": min(distances),
                    "max_item_distance": max(distances),
                    "item_distance_sd": round(statistics.pstdev(distances), 8),
                    "partition_zero_based_ranks": [list(group) for group in spec["partition"]],
                    "gold_label_counts": dict(sorted(gold_counts.items())),
                    "gold_position_counts": {str(key): value for key, value in sorted(position_counts.items())},
                    "gold_ordinal_rank_counts": {
                        str(key): value for key, value in sorted(ordinal_rank_counts.items())
                    },
                    "candidate_set_balance": balance_report(condition_rows),
                }
            )
        source_report[source] = {
            "unique_source_items": len(source_rows) // LEVELS,
            "question_rows": len(source_rows),
            "original_scale_levels": source_rows[0]["original_option_count"],
            "displayed_choices_per_question": source_rows[0]["option_count"],
            "conditions": condition_report,
        }
    return source_report


def write_jsonl(rows: list[dict], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(output)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project / "data" / "ordinal_bias_batch3_20x5000.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "data" / "candidate_distance_curve_4datasets_7levels.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project / "data" / "candidate_distance_curve_4datasets_7levels_report.json",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows_by_source, duplicate_removed = load_rows(args.input)
    original_counts = {
        source: len(rows_by_source[source]) + duplicate_removed[source] for source in SOURCES
    }
    balance_removed = balance_original_labels(rows_by_source, args.seed)
    rows, specifications = build_curve(rows_by_source, args.seed)
    source_report = validate_and_report(rows, specifications)
    output_sha256 = write_jsonl(rows, args.output)
    report = {
        "construction_version": VERSION,
        "seed": args.seed,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": output_sha256,
        "distance_conditions_per_source_item": LEVELS,
        "unique_source_items": len(rows) // LEVELS,
        "produced_question_rows": len(rows),
        "intervention": (
            "Seven monotonically increasing candidate-distance conditions. Each condition "
            "partitions the original ordinal scale into disjoint candidate sets."
        ),
        "held_fixed_across_distance_conditions": [
            "context",
            "question",
            "gold_label_text",
            "gold_position",
            "option_count",
        ],
        "balanced_within_every_source_and_distance": [
            "gold labels",
            "gold display positions",
            "gold ordinal ranks among shown choices",
            "gold labels within each ordered and unordered candidate set",
        ],
        "selection_rule": (
            "Enumerate all equal-size partitions; for seven equally spaced target means, "
            "select the nearest distinct attainable mean and then the partition with the "
            "smallest within-condition distance dispersion."
        ),
        "source_preprocessing": {
            source: {
                "original_rows": original_counts[source],
                "duplicate_context_rows_removed": duplicate_removed[source],
                "rows_removed_to_balance_gold_labels": balance_removed[source],
            }
            for source in SOURCES
        },
        "sources": source_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "unique_source_items": len(rows) // LEVELS,
                "questions": len(rows),
                "output": str(args.output),
                "report": str(args.report),
                "sha256": output_sha256,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
