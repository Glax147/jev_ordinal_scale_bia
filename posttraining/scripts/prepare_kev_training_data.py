#!/usr/bin/env python3
"""Create leakage-free KEV Choice training JSONL from the eight pinned sources.

The output keeps only datasets that showed severe choice-space compression in
the frozen 40-dataset experiment.  Each record receives a deterministic random
mapping from semantic levels to neutral option keys.  KEV's own training-time
augmentation can then reshuffle option order and apply permutation KL without
letting a fixed key reveal a score level across records.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "prepared"
SEED = 20260928
TARGET_PER_DATASET = 4000

REVISIONS = {
    "ibm_argument_quality": "590726b",
    "helpsteer2_correctness": "990b2711a36180dd19d9c94b8627844866f8982a",
    "civil_comments_toxicity": "f2970eb3a55777454c94069077cc8d9b5866312d",
    "measuring_hate_speech": "5468f6e118396646b02a2f691e771f6b6d9502ea",
    "wine_quality": "e6b10f4db3091a6fed8c5b294c0cc885e7f6e99d",
    "word_concreteness": "10f460a4d535800d89c579db7cd2664dbaf6b1d6",
    "wmt20_translation_quality": "0783ed2bd75f44835df4ea664f9ccb85812c8563",
    "realtoxicity_continuation": "f21629712ffd6a3d13a54fd2807ccd521c55ef74",
}

CHOICES = {
    "ibm_argument_quality": [
        "Q1 - Lowest quality", "Q2 - Low quality", "Q3 - Medium quality",
        "Q4 - High quality", "Q5 - Highest quality",
    ],
    "helpsteer2_correctness": [
        "0 - Incorrect", "1 - Mostly incorrect", "2 - Partly correct",
        "3 - Mostly correct", "4 - Fully correct",
    ],
    "civil_comments_toxicity": [f"Toxicity level {i:02d}" for i in range(1, 8)],
    "measuring_hate_speech": [f"Level {i:02d}" for i in range(1, 9)],
    "wine_quality": [
        "80–81 points", "82–83 points", "84–85 points", "86–87 points",
        "88–89 points", "90–91 points", "92–93 points", "94–95 points",
        "96–97 points", "98–99 points", "100 points",
    ],
    "word_concreteness": [f"Level {i:02d}" for i in range(1, 10)],
    "wmt20_translation_quality": [f"Level {i:02d}" for i in range(1, 13)],
    "realtoxicity_continuation": [f"Level {i:02d}" for i in range(1, 15)],
}

QUESTIONS = {
    "ibm_argument_quality": "Which crowd-derived argument-quality quintile was assigned to this argument?",
    "helpsteer2_correctness": "What correctness rating was assigned to this assistant response?",
    "civil_comments_toxicity": "Which toxicity level was assigned to this comment, from level 01 (lowest) to level 07 (highest)?",
    "measuring_hate_speech": "Which hate-speech severity band was assigned to this text, from Level 01 (lowest) to Level 08 (highest)?",
    "wine_quality": "Which Wine Enthusiast point band was assigned to this wine review?",
    "word_concreteness": "Which concreteness band was assigned to this word or expression, from Level 01 (most abstract) to Level 09 (most concrete)?",
    "wmt20_translation_quality": "Which human translation-quality band was assigned to this pair, from Level 01 (lowest) to Level 12 (highest)?",
    "realtoxicity_continuation": "Which toxicity band was assigned to this continuation, from Level 01 (lowest) to Level 14 (highest)?",
}

CONCRETENESS_THRESHOLDS = [1.81, 2.07, 2.33, 2.68, 3.07, 3.52, 4.04, 4.59]
HATE_SPEECH_THRESHOLDS = [-3.46, -2.35, -1.45, -0.72, -0.09, 0.56, 1.32]
WMT_THRESHOLDS = [
    70.25, 78.33333587646484, 83.33333587646484, 85.33333587646484,
    87.0, 88.66666412353516, 90.0, 91.0, 92.33333587646484,
    93.66666412353516, 96.66666412353516,
]
REALTOX_THRESHOLDS = [
    0.04303850314285715, 0.065443045, 0.07906512285714286,
    0.10924429142857144, 0.14550563285714282, 0.20702842714285716,
    0.27817565, 0.36329269999999997, 0.4597955371428572,
    0.5671150228571428, 0.7143426, 0.8368374714285715,
    0.8998802285714286,
]

LANGUAGE_NAMES = {
    "en": "English", "de": "German", "zh": "Chinese", "et": "Estonian",
    "ne": "Nepali", "ro": "Romanian", "si": "Sinhala",
}

SENSITIVE_PATTERNS = [
    r"\bporn(?:ography|ographic)?\b", r"\bsexual(?:ly)?\b", r"\bsex\b",
    r"\bintercourse\b", r"\borgasm\w*\b", r"\bmasturbat\w*\b",
    r"\bprostitut\w*\b", r"\brape(?:d|s)?\b|\brapist\b",
    r"\bnude\b|\bnaked\b", r"\bgenitals?\b|\bpenis\b|\bvagina\b",
    r"\bgambl\w*\b", r"\bcasino\b", r"\broulette\b", r"\bblackjack\b",
    r"\bpoker\b", r"\bslot machines?\b", r"\bwager\w*\b", r"\bjackpot\b",
    r"\bbookmaker\b|\bsportsbook\b", r"\blottery\b|\blotto\b",
    r"\bcocaine\b", r"\bheroin\b", r"\bmeth(?:amphetamine)?\b",
    r"\bfentanyl\b", r"\bopium\b", r"\bcrack cocaine\b",
]
SENSITIVE = [re.compile(pattern, re.IGNORECASE) for pattern in SENSITIVE_PATTERNS]


def normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(str(text).casefold().split()).encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass
class Candidate:
    dataset: str
    original_id: str
    context: str
    gold: int
    source_score: float | int | None
    source_split: str = "train"
    extra: dict = field(default_factory=dict)


class Reservoir:
    def __init__(self, capacity_per_label: int, seed: int):
        self.capacity = capacity_per_label
        self.rng = random.Random(seed)
        self.seen: Counter[int] = Counter()
        self.rows: dict[int, list[Candidate]] = defaultdict(list)

    def add(self, row: Candidate) -> None:
        label = row.gold
        self.seen[label] += 1
        bucket = self.rows[label]
        if len(bucket) < self.capacity:
            bucket.append(row)
            return
        j = self.rng.randrange(self.seen[label])
        if j < self.capacity:
            bucket[j] = row


def balanced_take(reservoir: Reservoir, target: int, seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    buckets = {label: list(rows) for label, rows in reservoir.rows.items()}
    for rows in buckets.values():
        rng.shuffle(rows)
    selected: list[Candidate] = []
    labels = sorted(buckets)
    # Round-robin produces equal strata until a stratum is exhausted, then
    # redistributes its unused quota without replacement.
    while len(selected) < target:
        progressed = False
        for label in labels:
            if buckets[label] and len(selected) < target:
                selected.append(buckets[label].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def parquet_rows(paths: Iterable[Path], columns: list[str] | None = None) -> Iterator[dict]:
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=4096, columns=columns):
            yield from batch.to_pylist()


def ibm_thresholds() -> list[float]:
    values = []
    for name in ("train.csv", "dev.csv", "test.csv"):
        with (RAW / "ibm_argument_quality" / name).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    values.append(float(row["WA"]))
                except (KeyError, TypeError, ValueError):
                    pass
    values.sort()
    return [values[math.ceil(len(values) * q / 5) - 1] for q in range(1, 5)]


def iter_ibm() -> Iterator[Candidate]:
    thresholds = ibm_thresholds()
    with (RAW / "ibm_argument_quality" / "train.csv").open(encoding="utf-8-sig", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            try:
                score = float(row["WA"])
            except (KeyError, TypeError, ValueError):
                continue
            context = f"Topic: {row.get('topic', '')}\nArgument: {row.get('argument', '')}"
            yield Candidate("ibm_argument_quality", f"train:{i}", context,
                            bisect.bisect_right(thresholds, score), score,
                            extra={"quintile_thresholds": thresholds})


def iter_helpsteer2() -> Iterator[Candidate]:
    path = RAW / "helpsteer2_correctness" / "train.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            row = json.loads(line)
            score = int(row["correctness"])
            context = f"User prompt:\n{row['prompt']}\n\nAssistant response:\n{row['response']}"
            yield Candidate("helpsteer2_correctness", f"train:{i}", context, score, score)


def iter_civil_comments() -> Iterator[Candidate]:
    paths = sorted((RAW / "civil_comments_toxicity" / "data").glob("train-*.parquet"))
    for i, row in enumerate(parquet_rows(paths, ["text", "toxicity"])):
        score = float(row["toxicity"])
        gold = min(6, max(0, int(score * 7)))
        yield Candidate("civil_comments_toxicity", f"train:{i}", row["text"], gold, score)


def iter_measuring_hate_speech() -> Iterator[Candidate]:
    path = RAW / "measuring_hate_speech" / "data" / "train-00000-of-00001.parquet"
    for i, row in enumerate(parquet_rows([path], ["text", "hate_speech_score"])):
        score = row.get("hate_speech_score")
        if score is None:
            continue
        score = float(score)
        yield Candidate("measuring_hate_speech", str(i), row["text"],
                        bisect.bisect_right(HATE_SPEECH_THRESHOLDS, score), score)


def iter_wine_quality() -> Iterator[Candidate]:
    path = RAW / "wine_quality" / "data" / "train-00000-of-00001.parquet"
    fields = ["title", "country", "province", "variety", "description", "points"]
    for i, row in enumerate(parquet_rows([path], fields)):
        try:
            score = float(row["points"])
        except (KeyError, TypeError, ValueError):
            continue
        if score < 80 or score > 100:
            continue
        gold = min(10, max(0, int((score - 80) // 2)))
        context = (
            f"Wine: {row.get('title') or ''}\n"
            f"Origin: {row.get('country') or ''}, {row.get('province') or ''}\n"
            f"Variety: {row.get('variety') or ''}\n"
            f"Review: {row.get('description') or ''}"
        )
        yield Candidate("wine_quality", f"train:{i}", context, gold, score)


def iter_word_concreteness() -> Iterator[Candidate]:
    path = RAW / "word_concreteness" / "concreteness_ratings.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            word = row.get("Word") or row.get("word") or row.get("Expression") or row.get("expression")
            score_text = (row.get("Conc.M") or row.get("concreteness") or row.get("Concreteness")
                          or row.get("mean") or row.get("Mean"))
            if not word or score_text is None:
                continue
            try:
                score = float(score_text)
            except (TypeError, ValueError):
                continue
            context = f"Word or expression: {word}"
            yield Candidate("word_concreteness", str(i), context,
                            bisect.bisect_right(CONCRETENESS_THRESHOLDS, score), score)


def iter_wmt() -> Iterator[Candidate]:
    root = RAW / "wmt20_translation_quality"
    for config_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        config = config_dir.name
        parts = config.split("-")
        if len(parts) != 2:
            continue
        source_lang, target_lang = parts
        paths = sorted(config_dir.glob("train-*.parquet"))
        for row in parquet_rows(paths, ["segid", "translation", "mean"]):
            score = float(row["mean"])
            trans = row["translation"]
            context = (
                f"Source ({LANGUAGE_NAMES.get(source_lang, source_lang)}): {trans[source_lang]}\n"
                f"Translation ({LANGUAGE_NAMES.get(target_lang, target_lang)}): {trans[target_lang]}"
            )
            yield Candidate("wmt20_translation_quality", f"{config}:{row['segid']}", context,
                            bisect.bisect_right(WMT_THRESHOLDS, score), score,
                            extra={"language_pair": config})


def iter_realtoxicity() -> Iterator[Candidate]:
    path = RAW / "realtoxicity_continuation" / "prompts.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            score = row.get("continuation", {}).get("toxicity")
            if score is None:
                continue
            score = float(score)
            context = f"Prompt: {row['prompt']['text']}\nContinuation: {row['continuation']['text']}"
            original_id = f"{row['filename']}:{row['begin']}"
            yield Candidate("realtoxicity_continuation", original_id, context,
                            bisect.bisect_right(REALTOX_THRESHOLDS, score), score)


ITERATORS = {
    "ibm_argument_quality": iter_ibm,
    "helpsteer2_correctness": iter_helpsteer2,
    "civil_comments_toxicity": iter_civil_comments,
    "measuring_hate_speech": iter_measuring_hate_speech,
    "wine_quality": iter_wine_quality,
    "word_concreteness": iter_word_concreteness,
    "wmt20_translation_quality": iter_wmt,
    "realtoxicity_continuation": iter_realtoxicity,
}


def is_sensitive(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE)


def make_kev_record(row: Candidate) -> dict:
    choices = CHOICES[row.dataset]
    order = list(range(len(choices)))
    random.Random(stable_seed(SEED, row.dataset, row.original_id, "initial_order")).shuffle(order)
    keys = [chr(ord("a") + i) for i in range(len(choices))]
    criteria = {keys[i]: choices[canonical] for i, canonical in enumerate(order)}
    correct_key = keys[order.index(row.gold)]
    return {
        "state": row.context,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": QUESTIONS[row.dataset],
                "criteria": criteria,
                "label": correct_key,
                "src": row.dataset,
            }
        },
        "_meta": {
            "source": row.dataset,
            "id": f"{row.dataset}::{row.original_id}",
            "group_id": f"{row.dataset}::{row.original_id}",
            "original_id": row.original_id,
            "split": row.source_split,
            "revision": REVISIONS[row.dataset],
            "gold_canonical_level": row.gold,
            "source_score": row.source_score,
            **row.extra,
        },
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-per-dataset", type=int, default=TARGET_PER_DATASET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--filter-sensitive", type=int, choices=(0, 1), default=1)
    parser.add_argument("--blacklist", type=Path, default=ROOT / "data" / "evaluation_blacklist.json")
    parser.add_argument(
        "--fixed-index",
        type=Path,
        default=ROOT / "configs" / "fixed_train_index_32000.jsonl.gz",
        help="Frozen source-ID index. When present, no sampling is performed.",
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation fraction must be between 0 and 0.5")

    if args.fixed_index.is_file():
        wanted: dict[tuple[str, str], dict] = {}
        index_rows = 0
        with gzip.open(args.fixed_index, "rt", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                key = (item["source"], str(item["original_id"]))
                if key in wanted:
                    raise RuntimeError(f"duplicate record in frozen training index: {key}")
                wanted[key] = item
                index_rows += 1
        if index_rows != 32_000:
            raise RuntimeError(f"frozen training index has {index_rows} rows, expected 32000")
        found: dict[tuple[str, str], Candidate] = {}
        for dataset, iterator in ITERATORS.items():
            for row in iterator():
                key = (dataset, str(row.original_id))
                item = wanted.get(key)
                if item is None:
                    continue
                if normalized_hash(row.context) != item["context_sha256"]:
                    raise RuntimeError(f"source text changed for {key}")
                if row.gold != int(item["gold_canonical_level"]):
                    raise RuntimeError(f"source label changed for {key}")
                if key in found:
                    raise RuntimeError(f"source parser produced duplicate frozen record: {key}")
                found[key] = row
        missing = sorted(set(wanted) - set(found))
        if missing:
            raise RuntimeError(f"fixed training index has {len(missing)} unresolved rows; first={missing[:5]}")

        train_rows: list[dict] = []
        for item in sorted(wanted.values(), key=lambda x: int(x["ordinal"])):
            record = make_kev_record(found[(item["source"], str(item["original_id"]))])
            # The frozen 40-dataset panel is the only evaluation set.  The
            # former internal validation slice is folded back into training;
            # it is never used for checkpoint or hyperparameter selection.
            train_rows.append(record)
        OUT.mkdir(parents=True, exist_ok=True)
        train_path = OUT / "kev_bias_train.jsonl"
        write_jsonl(train_path, train_rows)
        manifest = {
            "schema_version": "kev_bias_posttrain_data_v2.0-fixed-index",
            "selection": "no sampling; all 32,000 frozen source IDs are used for training",
            "fixed_index": args.fixed_index.relative_to(ROOT).as_posix(),
            "seed": SEED,
            "totals": {"selected": len(train_rows), "train": len(train_rows), "validation": 0},
            "files": {
                "train": {"path": train_path.relative_to(ROOT).as_posix(), "sha256": file_sha256(train_path), "bytes": train_path.stat().st_size},
            },
            "source_revisions": REVISIONS,
        }
        expected = json.loads((ROOT / "manifests/prepared_data_manifest.json").read_text(encoding="utf-8"))
        if manifest != expected:
            raise RuntimeError("reconstructed training JSONL does not match the bundled bytes/SHA-256 manifest")
        manifest_path = OUT / "prepared_data_manifest.json"
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        temporary_manifest.replace(manifest_path)
        print(json.dumps({"status": "ok", **manifest["totals"]}, indent=2))
        return

    blacklist_raw = json.loads(args.blacklist.read_text(encoding="utf-8"))["datasets"]
    black_ids = {d: set(v["original_ids"]) for d, v in blacklist_raw.items()}
    black_contexts = {d: set(v["context_sha256"]) for d, v in blacklist_raw.items()}

    selected_by_dataset: dict[str, list[Candidate]] = {}
    stats = {}
    for dataset, iterator in ITERATORS.items():
        reservoir = Reservoir(args.target_per_dataset, stable_seed(args.seed, dataset, "reservoir"))
        seen_contexts: set[str] = set()
        counts = Counter()
        eligible_labels = Counter()
        for row in iterator():
            counts["source_rows"] += 1
            context_hash = normalized_hash(row.context)
            if row.original_id in black_ids.get(dataset, set()) or context_hash in black_contexts.get(dataset, set()):
                counts["excluded_evaluation"] += 1
                continue
            if context_hash in seen_contexts:
                counts["excluded_duplicate_context"] += 1
                continue
            seen_contexts.add(context_hash)
            if len(row.context) > args.max_context_chars:
                counts["excluded_too_long"] += 1
                continue
            if args.filter_sensitive and is_sensitive(row.context):
                counts["excluded_sensitive"] += 1
                continue
            counts["eligible"] += 1
            eligible_labels[row.gold] += 1
            reservoir.add(row)
        selected = balanced_take(
            reservoir,
            args.target_per_dataset,
            stable_seed(args.seed, dataset, "balanced_take"),
        )
        selected_by_dataset[dataset] = selected
        stats[dataset] = {
            **dict(counts),
            "eligible_label_counts": {str(k): v for k, v in sorted(eligible_labels.items())},
            "selected": len(selected),
            "selected_label_counts": dict(sorted(Counter(r.gold for r in selected).items())),
        }
        print(dataset, json.dumps(stats[dataset], ensure_ascii=False), flush=True)

    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    split_stats = {}
    for dataset, rows in selected_by_dataset.items():
        by_label: dict[int, list[Candidate]] = defaultdict(list)
        for row in rows:
            by_label[row.gold].append(row)
        ds_train, ds_validation = [], []
        for label, bucket in sorted(by_label.items()):
            random.Random(stable_seed(args.seed, dataset, label, "split")).shuffle(bucket)
            n_val = round(len(bucket) * args.validation_fraction)
            if len(bucket) >= 10:
                n_val = max(1, n_val)
            else:
                n_val = 0
            ds_validation.extend(bucket[:n_val])
            ds_train.extend(bucket[n_val:])
        random.Random(stable_seed(args.seed, dataset, "train_order")).shuffle(ds_train)
        random.Random(stable_seed(args.seed, dataset, "validation_order")).shuffle(ds_validation)
        train_rows.extend(make_kev_record(row) for row in ds_train)
        validation_rows.extend(make_kev_record(row) for row in ds_validation)
        split_stats[dataset] = {"train": len(ds_train), "validation": len(ds_validation)}

    random.Random(stable_seed(args.seed, "all_train_order")).shuffle(train_rows)
    random.Random(stable_seed(args.seed, "all_validation_order")).shuffle(validation_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = OUT / "kev_bias_train.jsonl"
    validation_path = OUT / "kev_bias_validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    manifest = {
        "schema_version": "kev_bias_posttrain_data_v1.0",
        "seed": args.seed,
        "target_per_dataset": args.target_per_dataset,
        "validation_fraction": args.validation_fraction,
        "max_context_chars": args.max_context_chars,
        "sensitive_filter_enabled": bool(args.filter_sensitive),
        "selection": "stratified round-robin without replacement; deficits redistributed across nonempty strata",
        "choice_order": "deterministic random semantic-level to neutral-key mapping per record; KEV performs fresh order augmentation per epoch",
        "blacklist": str(args.blacklist.relative_to(ROOT)),
        "datasets": stats,
        "splits": split_stats,
        "totals": {"selected": len(train_rows) + len(validation_rows), "train": len(train_rows), "validation": len(validation_rows)},
        "files": {
            "train": {"path": train_path.relative_to(ROOT).as_posix(), "sha256": file_sha256(train_path), "bytes": train_path.stat().st_size},
            "validation": {"path": validation_path.relative_to(ROOT).as_posix(), "sha256": file_sha256(validation_path), "bytes": validation_path.stat().st_size},
        },
        "source_revisions": REVISIONS,
    }
    manifest_path = OUT / "prepared_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest["totals"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
