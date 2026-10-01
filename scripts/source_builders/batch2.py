#!/usr/bin/env python3
"""Pinned source adapters for the second ten public datasets."""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import gzip
import html
import json
import math
import re
import tarfile
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from . import batch1 as base


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ordinal_bias_batch2"
INPUT = ROOT / "data" / "ordinal_bias_batch2_10x5000.jsonl"
OUTPUT = ROOT / "results" / "legacy_batch2_answers.jsonl"
MANIFEST = ROOT / "results" / "legacy_batch2_manifest.json"
SEED = 20260927
VERSION = "ordinal_bias_builder_batch2_v1.0"
MAX_REQUEST_JSON_BYTES = 24_000


SOURCES = {
    "imdb_ratings": {
        "url": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        "revision": "aclImdb_v1",
        "label_policy": "Native IMDb 1-10 ratings; the source benchmark contains ratings 1-4 and 7-10.",
    },
    "goodreads_reviews": {
        "url": "https://huggingface.co/datasets/vngclinh/goodreads-reviews",
        "revision": "3880685167798bd9979269f545bf21d71abf8762",
        "label_policy": "Native 1-5 reviewer star rating; unrated rows (0) are excluded.",
    },
    "tripadvisor_hotel_reviews": {
        "url": "https://huggingface.co/datasets/argilla/tripadvisor-hotel-reviews",
        "revision": "533be43",
        "label_policy": "Native 1-5 overall hotel rating.",
    },
    "labr_arabic_book_reviews": {
        "url": "https://huggingface.co/datasets/mohamedadaly/labr",
        "revision": "170054b58c3dc352be53ab1649a5338a6e19282a",
        "label_policy": "Native 1-5 Arabic book-review rating.",
    },
    "beeradvocate_reviews": {
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/beer/beer_50000.json",
        "revision": "UCSD public 50,000-review snapshot",
        "label_policy": "Recorded overall score mapped to three fixed bands: low <=2.5, medium (2.5,3.5], high >3.5.",
    },
    "cmu_mosei_sentiment": {
        "url": "https://huggingface.co/datasets/vintp/CMU-Mosei-text",
        "revision": "915b96e",
        "label_policy": "Native mean sentiment score mapped to five fixed bands at -1.5, -0.5, 0.5, and 1.5.",
    },
    "ibm_argument_quality": {
        "url": "https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k",
        "revision": "590726b",
        "label_policy": "Crowd-derived WA quality score mapped to five empirical quintile bands computed on the full public corpus before sampling.",
    },
    "helpsteer2_correctness": {
        "url": "https://huggingface.co/datasets/nvidia/HelpSteer2",
        "revision": "990b2711a36180dd19d9c94b8627844866f8982a",
        "label_policy": "Native human correctness rating 0-4.",
    },
    "ultrafeedback_quality": {
        "url": "https://huggingface.co/datasets/argilla/ultrafeedback-critique",
        "revision": "edf2dcc03c41e35e6e762e15f111d8e681bbcb64",
        "label_policy": "Evaluator overall score mapped to five fixed-width intervals over [0,10].",
    },
    "dynasent": {
        "url": "https://github.com/cgpotts/dynasent",
        "revision": "dynasent-v1.1",
        "label_policy": "Native crowd-validated negative/neutral/positive label.",
    },
}


def configure_base():
    base.RAW = RAW
    base.INPUT = INPUT
    base.OUTPUT = OUTPUT
    base.MANIFEST = MANIFEST
    base.SEED = SEED
    base.VERSION = VERSION
    base.PER_DATASET = 5000
    base.TOTAL = 50000
    base.request_body = request_body


def request_body(row):
    keys = base.ALL_KEYS[:len(row["choices"])]
    body = {"state": row["context"], "model": base.MODEL, "questions": {"decision": {
        "type": "choice", "instructions": row["question"], "criteria": dict(zip(keys, row["choices"]))
    }}}
    if len(json.dumps(body, ensure_ascii=True).encode("utf-8")) <= MAX_REQUEST_JSON_BYTES:
        return body
    original = body["state"]
    marker = "\n...[middle truncated to satisfy provider request-size limit]...\n"
    low, high = 0, len(original)
    while low < high:
        keep = (low + high + 1) // 2
        left = (keep + 1) // 2
        right = keep // 2
        body["state"] = original[:left] + marker + (original[-right:] if right else "")
        if len(json.dumps(body, ensure_ascii=True).encode("utf-8")) <= MAX_REQUEST_JSON_BYTES:
            low = keep
        else:
            high = keep - 1
    left = (low + 1) // 2
    right = low // 2
    body["state"] = original[:left] + marker + (original[-right:] if right else "")
    return body


def hf(repo, revision, remote, filename):
    path = RAW / filename
    base.hf_download(repo, revision, remote, path)
    return path


def parquet_rows(path, columns=None):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        yield from batch.to_pylist()


def compact_text(value):
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def row(source, original_id, split, context, question, labels, gold, stratum, revision, extra=None):
    return base.make_row(source, original_id, split, compact_text(context), question,
                         labels, gold, stratum, revision, extra)


def iter_imdb():
    path = RAW / "aclImdb_v1.tar.gz"
    base.download(SOURCES["imdb_ratings"]["url"], path)
    labels = ["1/10", "2/10", "3/10", "4/10", "7/10", "8/10", "9/10", "10/10"]
    rating_to_pos = {rating: i for i, rating in enumerate((1, 2, 3, 4, 7, 8, 9, 10))}
    pattern = re.compile(r"aclImdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt$")
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            match = pattern.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            rating = int(match.group(4))
            if rating not in rating_to_pos:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            text = handle.read().decode("utf-8", errors="replace")
            yield row("imdb_ratings", f"{match.group(1)}:{match.group(2)}:{match.group(3)}", match.group(1), text,
                      "What rating did the reviewer assign to this movie review?",
                      labels, rating_to_pos[rating], f"rating={rating}", SOURCES["imdb_ratings"]["revision"])


def iter_goodreads():
    rev = SOURCES["goodreads_reviews"]["revision"]
    path = hf("vngclinh/goodreads-reviews", rev, "data/train-00001-of-00032.parquet",
              "goodreads_train_00001.parquet")
    labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    for i, x in enumerate(parquet_rows(path, ["review_id", "rating", "review_text"])):
        rating = int(x.get("rating") or 0)
        if 1 <= rating <= 5:
            yield row("goodreads_reviews", x.get("review_id") or i, "train", x.get("review_text"),
                      "What star rating did the reviewer assign to this book review?",
                      labels, rating - 1, f"rating={rating}", rev)


def iter_tripadvisor():
    rev = SOURCES["tripadvisor_hotel_reviews"]["revision"]
    remote = "data/train-00000-of-00001-0e99e58b23dccc25.parquet"
    path = hf("argilla/tripadvisor-hotel-reviews", rev, remote, "tripadvisor_hotel_reviews.parquet")
    labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    for i, x in enumerate(parquet_rows(path)):
        value = x.get("label", x.get("rating", x.get("overall")))
        if value is None and x.get("prediction"):
            value = x["prediction"][0].get("label")
        if value is None:
            continue
        rating = int(round(float(value)))
        text = x.get("review", x.get("text"))
        if 1 <= rating <= 5:
            yield row("tripadvisor_hotel_reviews", x.get("id") or i, "train", text,
                      "What overall star rating did the reviewer assign to this hotel review?",
                      labels, rating - 1, f"rating={rating}", rev)


def iter_labr():
    rev = SOURCES["labr_arabic_book_reviews"]["revision"]
    labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    for split in ("train", "test"):
        remote = f"plain_text/{split}-00000-of-00001.parquet"
        path = hf("mohamedadaly/labr", rev, remote, f"labr_{split}.parquet")
        values = list(parquet_rows(path))
        raw_labels = [int(x["label"]) for x in values if x.get("label") is not None]
        zero_based = bool(raw_labels) and min(raw_labels) == 0 and max(raw_labels) <= 4
        for i, x in enumerate(values):
            if x.get("label") is None:
                continue
            rating = int(x["label"]) + (1 if zero_based else 0)
            if 1 <= rating <= 5:
                yield row("labr_arabic_book_reviews", f"{split}:{i}", split, x.get("text"),
                          "What star rating did the reviewer assign to this Arabic book review?",
                          labels, rating - 1, f"rating={rating}", rev)


def iter_beeradvocate():
    path = RAW / "beeradvocate_50000.json"
    base.download(SOURCES["beeradvocate_reviews"]["url"], path)
    labels = ["Low rating (1-2.5)", "Medium rating (3-3.5)", "High rating (4-5)"]
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            x = ast.literal_eval(line)
            score = x.get("review/overall", x.get("review_overall", x.get("overall")))
            text = x.get("review/text", x.get("review_text", x.get("text")))
            if score is None:
                continue
            score = float(score)
            pos = 0 if score <= 2.5 else (1 if score <= 3.5 else 2)
            yield row("beeradvocate_reviews", i, "snapshot", text,
                      "Which overall rating band best matches the recorded beer-review score?",
                      labels, pos, f"rating_band={pos}", SOURCES["beeradvocate_reviews"]["revision"],
                      {"recorded_score": score})


def iter_mosei():
    rev = SOURCES["cmu_mosei_sentiment"]["revision"]
    labels = ["Strongly negative", "Weakly negative", "Neutral", "Weakly positive", "Strongly positive"]
    thresholds = [-1.5, -0.5, 0.5, 1.5]
    for split in ("train", "validation", "test"):
        path = hf("vintp/CMU-Mosei-text", rev, f"{split}.csv", f"mosei_{split}.csv")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for i, x in enumerate(csv.DictReader(handle)):
                try:
                    score = float(x["sentiment"])
                except (KeyError, TypeError, ValueError):
                    continue
                pos = bisect.bisect_right(thresholds, score)
                yield row("cmu_mosei_sentiment", f"{split}:{i}", split, x.get("text"),
                          "What sentiment intensity label was assigned to this utterance?",
                          labels, pos, f"sentiment_band={pos}", rev, {"mean_sentiment_score": score})


def iter_argument_quality():
    rev = SOURCES["ibm_argument_quality"]["revision"]
    labels = ["Q1 - Lowest quality", "Q2 - Low quality", "Q3 - Medium quality", "Q4 - High quality", "Q5 - Highest quality"]
    records = []
    for split, remote in (("train", "train.csv"), ("validation", "dev.csv"), ("test", "test.csv")):
        path = hf("ibm-research/argument_quality_ranking_30k", rev, remote, f"argument_quality_{remote}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for i, x in enumerate(csv.DictReader(handle)):
                try:
                    score = float(x["WA"])
                except (KeyError, TypeError, ValueError):
                    continue
                records.append((split, i, x, score))
    ordered = sorted(score for _, _, _, score in records)
    thresholds = [ordered[math.ceil(len(ordered) * q / 5) - 1] for q in range(1, 5)]
    for split, i, x, score in records:
        pos = bisect.bisect_right(thresholds, score)
        context = f"Topic: {x.get('topic', '')}\nArgument: {x.get('argument', '')}"
        yield row("ibm_argument_quality", f"{split}:{i}", split, context,
                  "Which crowd-derived argument-quality quintile was assigned to this argument?",
                  labels, pos, f"quality_quintile={pos + 1}", rev,
                  {"weighted_average_score": score, "quintile_thresholds": thresholds})


def iter_helpsteer2():
    rev = SOURCES["helpsteer2_correctness"]["revision"]
    labels = ["0 - Incorrect", "1 - Mostly incorrect", "2 - Partly correct", "3 - Mostly correct", "4 - Fully correct"]
    for split in ("train", "validation"):
        path = hf("nvidia/HelpSteer2", rev, f"{split}.jsonl.gz", f"helpsteer2_{split}.jsonl.gz")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                x = json.loads(line)
                score = int(x["correctness"])
                context = f"User prompt:\n{x['prompt']}\n\nAssistant response:\n{x['response']}"
                yield row("helpsteer2_correctness", f"{split}:{i}", split, context,
                          "What correctness rating was assigned to this assistant response?",
                          labels, score, f"correctness={score}", rev)


def iter_ultrafeedback():
    rev = SOURCES["ultrafeedback_quality"]["revision"]
    labels = ["Very low (0-2]", "Low (2-4]", "Medium (4-6]", "High (6-8]", "Very high (8-10]"]
    remotes = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
    for shard, remote in enumerate(remotes):
        path = hf("argilla/ultrafeedback-critique", rev, remote, f"ultrafeedback_{shard}.parquet")
        for i, x in enumerate(parquet_rows(path, ["instruction", "response", "overall_score"])):
            try:
                score = float(x["overall_score"])
            except (KeyError, TypeError, ValueError):
                continue
            pos = max(0, min(4, math.ceil(score / 2) - 1))
            context = f"User instruction:\n{x.get('instruction', '')}\n\nAssistant response:\n{x.get('response', '')}"
            yield row("ultrafeedback_quality", f"shard{shard}:{i}", "train", context,
                      "Which overall quality band was assigned by the evaluator?",
                      labels, pos, f"quality_band={pos}", rev, {"overall_score": score})


def iter_dynasent():
    path = RAW / "dynasent-v1.1.zip"
    base.download("https://github.com/cgpotts/dynasent/raw/refs/heads/main/dynasent-v1.1.zip", path,
                  use_curl=True)
    labels = ["Negative", "Neutral", "Positive"]
    positions = {"negative": 0, "neutral": 1, "positive": 2}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".jsonl") or "sst-dev" in name or name.startswith("__MACOSX/"):
                continue
            split_match = re.search(r"-(train|dev|test)\.jsonl$", name)
            split = split_match.group(1) if split_match else "unknown"
            with archive.open(name) as raw:
                for i, binary in enumerate(raw):
                    x = json.loads(binary.decode("utf-8"))
                    label = x.get("gold_label")
                    if label not in positions:
                        continue
                    original_id = x.get("text_id") or f"{Path(name).stem}:{i}"
                    yield row("dynasent", original_id, split, x.get("sentence"),
                              "What sentiment label was assigned to this sentence?",
                              labels, positions[label], f"sentiment={label}", SOURCES["dynasent"]["revision"],
                              {"round": "r2" if "round02" in name else "r1"})


LOADERS = [
    ("imdb_ratings", iter_imdb),
    ("goodreads_reviews", iter_goodreads),
    ("tripadvisor_hotel_reviews", iter_tripadvisor),
    ("labr_arabic_book_reviews", iter_labr),
    ("beeradvocate_reviews", iter_beeradvocate),
    ("cmu_mosei_sentiment", iter_mosei),
    ("ibm_argument_quality", iter_argument_quality),
    ("helpsteer2_correctness", iter_helpsteer2),
    ("ultrafeedback_quality", iter_ultrafeedback),
    ("dynasent", iter_dynasent),
]


def enrich_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["dataset_documentation"] = SOURCES
    manifest["batch_relationship"] = "Second independent 10-dataset ordinal-bias batch; no overlap with batch-1 source names."
    manifest["selection_rule"] = (
        "Text-bearing public datasets with at least 5,000 usable labeled examples and an ordered label scale. "
        "Native discrete labels are preserved; continuous labels use fixed intervals declared before model inference."
    )
    manifest["request_state_truncation"] = {
        "maximum_serialized_request_bytes": MAX_REQUEST_JSON_BYTES,
        "strategy": "Keep equal-size beginning and ending spans with an explicit middle-truncation marker.",
        "reason": "The intermediary rejected a 33 KB Arabic request with a provider size error; the cap is applied before inference and is independent of labels or predictions.",
    }
    base.atomic_json(MANIFEST, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1 or args.retries < 0 or args.limit < 0:
        parser.error("Invalid worker/timeout/retry/limit settings.")
    configure_base()
    base.LOADERS = LOADERS
    with base.exclusive_run():
        if args.prepare:
            base.prepare()
            enrich_manifest()
            return 0
        return base.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
