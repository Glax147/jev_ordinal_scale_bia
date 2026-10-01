#!/usr/bin/env python3
"""Pinned source adapters for the first ten public datasets.

Public reconstruction uses only ``LOADERS``.  The legacy runner functions are
retained as inert provenance helpers and require an explicitly configured
System One-compatible endpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import csv
import ctypes
import getpass
import gzip
import hashlib
import io
import json
import math
import os
import random
import subprocess
import threading
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ordinal_bias"
INPUT = ROOT / "data" / "ordinal_bias_10x5000.jsonl"
OUTPUT = ROOT / "results" / "legacy_batch1_answers.jsonl"
MANIFEST = ROOT / "results" / "legacy_batch1_manifest.json"
ENDPOINT = os.environ.get("JEV_API_URL", "")
MODEL = "jev-1.13"
SEED = 20260926
PER_DATASET = 5000
TOTAL = 50000
VERSION = "ordinal_bias_builder_batch1_v1.0"
ALL_KEYS = "ABCDEFGHIJ"
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
LOCAL = threading.local()

HF = {
    "sst5": ("SetFit/sst5", "e51bdcd8cd3a30da231967c1a249ba59361279a3"),
    "tweet_eval": ("cardiffnlp/tweet_eval", "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"),
    "app_reviews": ("sealuzh/app_reviews", "9eaa95f66364367e8752b0f34c00f67aafa95d15"),
    "helpsteer": ("nvidia/HelpSteer", "3ca5d59c1bc1080af195b4254e7407db60b6f450"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".writing")
    if staging.exists():
        raise RuntimeError(f"Unfinished metadata write exists: {staging}")
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def download(url, path, expected_size=None, use_curl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        if expected_size is not None and path.stat().st_size != expected_size:
            raise RuntimeError(f"Unexpected existing size for {path}: {path.stat().st_size}")
        return
    if path.exists():
        path.unlink()
    part = path.with_name(path.name + ".part")
    print(f"Downloading {path.name} ...", flush=True)
    if use_curl:
        subprocess.run(["curl.exe", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(part), url], check=True)
    else:
        last_error = None
        for attempt in range(1, 7):
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if offset > 0 and not append:
                        print("  server did not honor Range; restarting this file", flush=True)
                    with part.open("ab" if append else "wb") as f:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                f.write(block)
                if expected_size is None or part.stat().st_size == expected_size:
                    last_error = None
                    break
                last_error = RuntimeError(
                    f"incomplete file after attempt {attempt}: {part.stat().st_size}/{expected_size}")
            except requests.RequestException as exc:
                last_error = exc
            if attempt < 6:
                print(f"  download interrupted; retry {attempt}/6 from byte {part.stat().st_size if part.exists() else 0}",
                      flush=True)
                time.sleep(min(20, 2 ** attempt))
        if last_error is not None:
            raise last_error
    if expected_size is not None and part.stat().st_size != expected_size:
        raise RuntimeError(f"Downloaded size mismatch for {path}: {part.stat().st_size}")
    os.replace(part, path)


def hf_download(repo, revision, remote, local):
    if Path(local).exists() and Path(local).stat().st_size:
        return
    errors = []
    configured = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    hosts = [configured] if configured else []
    if "https://huggingface.co" not in hosts:
        hosts.append("https://huggingface.co")
    for host in hosts:
        try:
            download(f"{host}/datasets/{repo}/resolve/{revision}/{remote}", local)
            return
        except Exception as exc:
            errors.append(f"{host}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Hugging Face download failed: " + " | ".join(errors))


def choices(labels):
    return list(labels)


def make_row(source, original_id, split, context, question, label_choices, gold_position,
             stratum, source_revision, extra=None):
    context = str(context).strip()
    if not context:
        return None
    label_choices = choices(label_choices)
    gold_position = int(gold_position)
    if not 0 <= gold_position < len(label_choices):
        return None
    row = {
        "sample_id": f"{source}::{original_id}",
        "source": source,
        "original_id": str(original_id),
        "split": str(split),
        "task_type": "ordinal_choice",
        "decision_format": "multiple_choice",
        "context": context,
        "question": question,
        "choices": label_choices,
        "gold_position": gold_position,
        "gold_label_text": label_choices[gold_position],
        "sampling_seed": SEED,
        "sampling_stratum": str(stratum),
        "source_revision": source_revision,
    }
    if extra:
        row.update(extra)
    return row


def stratified_reservoir(rows, n, seed):
    """Uniform reservoir within each stratum, followed by capped-balanced allocation."""
    rng = random.Random(seed)
    reservoirs = defaultdict(list)
    seen = Counter()
    ids = set()
    signatures = set()
    rejected_blank = duplicate_ids = duplicate_inputs = 0
    for row in rows:
        if row is None or not row.get("context", "").strip():
            rejected_blank += 1
            continue
        sid = row["sample_id"]
        if sid in ids:
            duplicate_ids += 1
            continue
        ids.add(sid)
        signature = hashlib.sha256((row["question"] + "\0" + row["context"]).encode("utf-8")).digest()
        if signature in signatures:
            duplicate_inputs += 1
            continue
        signatures.add(signature)
        key = row["sampling_stratum"]
        seen[key] += 1
        bucket = reservoirs[key]
        if len(bucket) < n:
            bucket.append(row)
        else:
            j = rng.randrange(seen[key])
            if j < n:
                bucket[j] = row
    if sum(seen.values()) < n:
        raise RuntimeError(f"Only {sum(seen.values())} unique usable rows; need {n}.")
    allocation = Counter()
    active = sorted(seen)
    remaining = n
    cursor = 0
    while remaining:
        if not active:
            raise RuntimeError("Stratified allocation exhausted unexpectedly.")
        key = active[cursor % len(active)]
        if allocation[key] < seen[key]:
            allocation[key] += 1
            remaining -= 1
        if allocation[key] >= seen[key]:
            active.remove(key)
            if active:
                cursor %= len(active)
        else:
            cursor += 1
    selected = []
    for key in sorted(allocation):
        selected.extend(rng.sample(reservoirs[key], allocation[key]))
    rng.shuffle(selected)
    return selected, {
        "usable_unique_rows": sum(seen.values()),
        "available_by_stratum": dict(sorted(seen.items())),
        "selected_by_stratum": dict(sorted(allocation.items())),
        "rejected_blank": rejected_blank,
        "duplicate_ids_removed": duplicate_ids,
        "duplicate_inputs_removed": duplicate_inputs,
    }


def iter_sst5():
    repo, rev = HF["sst5"]
    labels = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
    question = "What sentiment label was assigned to this sentence?"
    for split in ("train", "dev", "test"):
        path = RAW / f"sst5_{split}.jsonl"
        hf_download(repo, rev, f"{split}.jsonl", path)
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                label = int(x["label"])
                yield make_row("sst5", f"{split}:{i}", split, x["text"], question, labels,
                               label, f"sentiment={label}", rev)


def iter_tweet_eval():
    import pyarrow.parquet as pq
    repo, rev = HF["tweet_eval"]
    labels = ["Negative", "Neutral", "Positive"]
    question = "What sentiment label was assigned to this tweet?"
    for split in ("train", "validation", "test"):
        path = RAW / f"tweet_eval_sentiment_{split}.parquet"
        hf_download(repo, rev, f"sentiment/{split}-00000-of-00001.parquet", path)
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=["text", "label"], batch_size=8192):
            for x in batch.to_pylist():
                label = int(x["label"])
                yield make_row("tweet_eval_sentiment", f"{split}:{index}", split, x["text"],
                               question, labels, label, f"sentiment={label}", rev)
                index += 1


def iter_semeval():
    path = RAW / "semeval2018_task1_all_data.zip"
    download("http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip",
             path, expected_size=5975590)
    labels = ["Intensity 0 (none)", "Intensity 1 (low)", "Intensity 2 (moderate)", "Intensity 3 (high)"]
    rev = digest(path)
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if "/English/EI-oc/" in n and n.endswith(".txt")
                       and not n.startswith("__MACOSX/"))
        for name in names:
            split = "train" if "/training/" in name else "validation" if "/development/" in name else "test"
            text = z.read(name).decode("utf-8-sig")
            for x in csv.DictReader(io.StringIO(text), delimiter="\t"):
                emotion = x["Affect Dimension"].strip().lower()
                label = int(x["Intensity Class"].split(":", 1)[0])
                question = f"How strongly does this tweet express {emotion}?"
                yield make_row("semeval2018_ei_oc", f"{split}:{emotion}:{x['ID']}", split,
                               x["Tweet"], question, labels, label,
                               f"emotion={emotion}|intensity={label}", rev,
                               {"target_emotion": emotion})


def iter_meld():
    rev = "e8cedf27b5d2877e198332c957127e16eb214afe"
    labels = ["Negative", "Neutral", "Positive"]
    question = "What sentiment label was assigned to this dialogue utterance?"
    for split, remote in (("train", "train_sent_emo.csv"), ("validation", "dev_sent_emo.csv"),
                          ("test", "test_sent_emo.csv")):
        path = RAW / f"meld_{split}.csv"
        download(f"https://raw.githubusercontent.com/declare-lab/MELD/{rev}/data/MELD/{remote}", path)
        with path.open(encoding="utf-8-sig", newline="") as f:
            for x in csv.DictReader(f):
                sentiment = x["Sentiment"].strip().lower()
                label = {"negative": 0, "neutral": 1, "positive": 2}[sentiment]
                oid = f"{split}:dialogue{x['Dialogue_ID']}:utterance{x['Utterance_ID']}"
                yield make_row("meld_sentiment", oid, split, x["Utterance"], question, labels,
                               label, f"sentiment={label}", rev)


def iter_amazon():
    path = RAW / "amazon_reviews_2023_all_beauty.jsonl.gz"
    url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"
    download(url, path, expected_size=94441517)
    labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    question = "What star rating did the reviewer assign?"
    rev = digest(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            x = json.loads(line)
            label = int(float(x["rating"])) - 1
            context = "Review title: " + str(x.get("title") or "") + "\n\nReview text: " + str(x.get("text") or "")
            oid = f"all_beauty:{i}:{x.get('parent_asin','')}:{x.get('user_id','')}"
            yield make_row("amazon_reviews_2023_all_beauty", oid, "all", context, question,
                           labels, label, f"stars={label + 1}", rev)


def iter_app_reviews():
    import pyarrow.parquet as pq
    repo, rev = HF["app_reviews"]
    path = RAW / "app_reviews.parquet"
    hf_download(repo, rev, "data/train-00000-of-00001.parquet", path)
    labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    question = "What star rating did the reviewer assign to this app review?"
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["package_name", "review", "date", "star"], batch_size=8192):
        for x in batch.to_pylist():
            label = int(x["star"]) - 1
            extra = {"package_name": str(x.get("package_name") or "")}
            yield make_row("app_reviews", f"row:{index}", "train", x["review"], question,
                           labels, label, f"stars={label + 1}", rev, extra)
            index += 1


def iter_drugs_com():
    path = RAW / "drugs_com_raw.zip"
    download("http://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip",
             path, expected_size=42989872)
    labels = [f"Rating {i}" for i in range(1, 11)]
    question = "What 1-to-10 satisfaction rating did the reviewer assign to this drug review?"
    rev = digest(path)
    with zipfile.ZipFile(path) as z:
        for split, name in (("train", "drugsComTrain_raw.tsv"), ("test", "drugsComTest_raw.tsv")):
            with io.TextIOWrapper(z.open(name), encoding="utf-8-sig", newline="") as f:
                for i, x in enumerate(csv.DictReader(f, delimiter="\t")):
                    label = int(float(x["rating"])) - 1
                    context = (f"Drug: {x.get('drugName','')}\nCondition: {x.get('condition','')}\n\n"
                               f"Review: {x.get('review','')}")
                    oid = str(x.get("Unnamed: 0") or f"{split}:{i}")
                    yield make_row("uci_drugs_com", f"{split}:{oid}", split, context, question,
                                   labels, label, f"rating={label + 1}", rev)


def iter_asap2():
    path = RAW / "asap2_train.zip"
    url = "https://raw.githubusercontent.com/scrosseye/ASAP_2.0/060ab225ca0984c0bf43ff222bf6bb2bc6195b6f/ASAP_2_Final_github_train.zip"
    download(url, path, expected_size=11240696)
    labels = [f"Score {i}" for i in range(1, 7)]
    question = "What holistic quality score (1 to 6) did the human raters assign to this essay?"
    rev = "060ab225ca0984c0bf43ff222bf6bb2bc6195b6f"
    with zipfile.ZipFile(path) as z, io.TextIOWrapper(z.open("ASAP_2_Final_github_train.csv"),
                                                      encoding="utf-8-sig", newline="") as f:
        for x in csv.DictReader(f):
            label = int(x["score"]) - 1
            context = f"Writing assignment:\n{x['assignment']}\n\nStudent essay:\n{x['full_text']}"
            yield make_row("asap2", x["essay_id"], "train", context, question, labels,
                           label, f"score={label + 1}", rev,
                           {"prompt_name": x.get("prompt_name"), "grade_level": x.get("grade_level")})


def iter_persuade2():
    path = RAW / "persuade2_train.csv"
    url = "https://drive.usercontent.google.com/download?id=13phHyDzIsb0MHyJr6q-B-qIa9P2tM135&export=download&confirm=t"
    download(url, path, expected_size=616894963)
    labels = ["Ineffective", "Adequate", "Effective"]
    question = "What effectiveness label did the human annotators assign to the target argument component?"
    rev = "google-drive:13phHyDzIsb0MHyJr6q-B-qIa9P2tM135"
    with path.open(encoding="utf-8-sig", newline="") as f:
        for i, x in enumerate(csv.DictReader(f)):
            name = x.get("discourse_effectiveness", "").strip()
            if name not in labels:
                continue
            label = labels.index(name)
            context = (f"Writing assignment:\n{x.get('assignment','')}\n\nFull essay:\n{x.get('full_text','')}\n\n"
                       f"Target component type: {x.get('discourse_type','')}\n"
                       f"Target argument component:\n{x.get('discourse_text','')}")
            # The official CSV serializes very large discourse IDs in scientific notation,
            # which collapses distinct IDs after decimal rounding.  Essay ID + character
            # span + source row is stable and collision-free within this file.
            oid = (f"essay:{x.get('essay_id_comp','')}|start:{x.get('discourse_start','')}|"
                   f"end:{x.get('discourse_end','')}|row:{i}")
            yield make_row("persuade2_argument_effectiveness", oid, x.get("competition_set") or "train",
                           context, question, labels, label, f"effectiveness={label}", rev,
                           {"discourse_type": x.get("discourse_type"), "prompt_name": x.get("prompt_name")})


def iter_helpsteer():
    repo, rev = HF["helpsteer"]
    labels = [f"Helpfulness {i}" for i in range(5)]
    question = "What helpfulness rating (0 to 4) did the human annotators assign to this response?"
    for split in ("train", "validation"):
        path = RAW / f"helpsteer_{split}.jsonl.gz"
        hf_download(repo, rev, f"{split}.jsonl.gz", path)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                label = int(x["helpfulness"])
                context = f"User prompt:\n{x['prompt']}\n\nAssistant response:\n{x['response']}"
                yield make_row("helpsteer", f"{split}:{i}", split, context, question, labels,
                               label, f"helpfulness={label}", rev)


LOADERS = [
    ("sst5", iter_sst5),
    ("tweet_eval_sentiment", iter_tweet_eval),
    ("semeval2018_ei_oc", iter_semeval),
    ("meld_sentiment", iter_meld),
    ("amazon_reviews_2023_all_beauty", iter_amazon),
    ("app_reviews", iter_app_reviews),
    ("uci_drugs_com", iter_drugs_com),
    ("asap2", iter_asap2),
    ("persuade2_argument_effectiveness", iter_persuade2),
    ("helpsteer", iter_helpsteer),
]


def prepare():
    if INPUT.exists():
        rows = load_rows()
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
        if manifest.get("input_sha256") != digest(INPUT):
            raise RuntimeError("Frozen input exists but manifest/hash does not match.")
        print(f"Reusing verified frozen input: {len(rows)} rows.", flush=True)
        return
    if OUTPUT.exists() and OUTPUT.stat().st_size:
        raise RuntimeError("Answer file exists without frozen input; refusing to resample.")
    RAW.mkdir(parents=True, exist_ok=True)
    all_rows = []
    sampling = {}
    for dataset_index, (name, loader) in enumerate(LOADERS):
        print(f"Preparing {name} ({dataset_index + 1}/10) ...", flush=True)
        rows, stats = stratified_reservoir(loader(), PER_DATASET, SEED + dataset_index * 1009)
        if len(rows) != PER_DATASET or any(r["source"] != name for r in rows):
            raise RuntimeError(f"Invalid prepared rows for {name}")
        all_rows.extend(rows)
        sampling[name] = stats
        print(f"  selected={len(rows)} usable={stats['usable_unique_rows']} strata={stats['selected_by_stratum']}", flush=True)
    if len(all_rows) != TOTAL or len({r["sample_id"] for r in all_rows}) != TOTAL:
        raise RuntimeError("Combined sample count or global IDs are invalid.")
    random.Random(SEED).shuffle(all_rows)
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    with INPUT.open("x", encoding="utf-8", newline="\n") as f:
        for row in all_rows:
            f.write(json_bytes(row).decode("utf-8") + "\n")
    raw_files = {}
    for p in sorted(RAW.iterdir()):
        if p.is_file() and not p.name.endswith(".part"):
            raw_files[p.name] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    atomic_json(MANIFEST, {
        "runner_version": VERSION,
        "created_at": now(),
        "state": "prepared",
        "input": str(INPUT),
        "input_rows": TOTAL,
        "input_sha256": digest(INPUT),
        "datasets": [name for name, _ in LOADERS],
        "per_dataset": PER_DATASET,
        "sampling": sampling,
        "raw_files": raw_files,
        "sampling_seed": SEED,
        "sampling_method": "capped-balanced without replacement by declared stratum; per-stratum reservoir sampling",
        "endpoint": ENDPOINT,
        "requested_model": MODEL,
        "output": str(OUTPUT),
        "gold_sent_to_model": False,
        "converted_requests_persisted": False,
        "latency_definition": "client elapsed time including retries/backoff; not model-only time",
    })
    print(f"Frozen {TOTAL} rows at {INPUT}", flush=True)


def load_rows():
    with INPUT.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != TOTAL or len({r["sample_id"] for r in rows}) != TOTAL:
        raise RuntimeError("Expected exactly 50,000 unique frozen rows.")
    counts = Counter(r["source"] for r in rows)
    if counts != Counter({name: PER_DATASET for name, _ in LOADERS}):
        raise RuntimeError(f"Dataset counts mismatch: {counts}")
    for r in rows:
        if not (3 <= len(r["choices"]) <= 10 and 0 <= r["gold_position"] < len(r["choices"])):
            raise RuntimeError(f"Invalid choices/gold: {r['sample_id']}")
        if r["gold_label_text"] != r["choices"][r["gold_position"]]:
            raise RuntimeError(f"Gold text mismatch: {r['sample_id']}")
    return rows


def request_body(row):
    keys = ALL_KEYS[:len(row["choices"])]
    return {"state": row["context"], "model": MODEL, "questions": {"decision": {
        "type": "choice", "instructions": row["question"], "criteria": dict(zip(keys, row["choices"]))
    }}}


def request_hash(row):
    return hashlib.sha256(json_bytes(request_body(row))).hexdigest()


class CallFailure(Exception):
    def __init__(self, message, fatal=False, retryable=False, retry_after=None):
        super().__init__(message)
        self.fatal = fatal
        self.retryable = retryable
        self.retry_after = retry_after


def normalize_response(data, row):
    if not isinstance(data, dict):
        raise CallFailure("Response is not a JSON object.")
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    answer = payload.get("answers", {}).get("decision")
    keys = ALL_KEYS[:len(row["choices"])]
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise CallFailure("Malformed choice answer: " + json.dumps(data, ensure_ascii=False)[:1200])
    key = answer.get("choice")
    if key not in keys:
        raise CallFailure(f"Unknown choice key: {key!r}")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
        raise CallFailure(f"Missing/incomplete {len(keys)}-choice probability distribution.")
    for value in probabilities.values():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise CallFailure("Invalid choice probability.")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.02):
        raise CallFailure("Choice probabilities do not sum to approximately one.")
    pos = keys.index(key)
    usage = payload.get("usage") or data.get("usage") or {}
    return {
        "predicted_key": key,
        "predicted_position": pos,
        "predicted_label_text": row["choices"][pos],
        "probabilities": probabilities,
        "confidence": answer.get("confidence"),
        "returned_model": payload.get("model") or data.get("model"),
        "usage": usage if isinstance(usage, dict) else {},
        "provider_latency_ms": payload.get("latency_ms"),
    }


def run_item(row, api_key, input_sha, args, stop):
    started = time.perf_counter()
    result = {
        "schema_version": VERSION,
        "sample_id": row["sample_id"],
        "source": row["source"],
        "requested_model": MODEL,
        "endpoint": ENDPOINT,
        "input_sha256": input_sha,
        "request_sha256": request_hash(row),
        "gold_position": row["gold_position"],
        "gold_label_text": row["gold_label_text"],
        "choices": row["choices"],
    }
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(1, args.retries + 2):
        result["request_attempts"] = attempt
        try:
            if stop.is_set():
                raise CallFailure("Run stopped before request.", fatal=True)
            with LOCAL.session.post(ENDPOINT, json=request_body(row),
                                    headers={"Authorization": "Bearer " + api_key},
                                    timeout=(20, args.timeout)) as response:
                result["http_status"] = response.status_code
                result["provider_request_id"] = response.headers.get("X-Request-ID") or response.headers.get("x-typesafe-request-id")
                if response.status_code >= 400:
                    try:
                        after = float(response.headers.get("Retry-After", ""))
                    except ValueError:
                        after = None
                    raise CallFailure(f"HTTP {response.status_code}: {response.text[:1200]}",
                                      fatal=response.status_code in {401, 402, 403},
                                      retryable=response.status_code in RETRYABLE,
                                      retry_after=after)
                result.update(normalize_response(response.json(), row))
            result.update(status="ok", correct=result["predicted_position"] == row["gold_position"])
            break
        except (requests.RequestException, ValueError, CallFailure, AttributeError, TypeError) as exc:
            fatal = isinstance(exc, CallFailure) and exc.fatal
            retryable = (isinstance(exc, CallFailure) and exc.retryable) or isinstance(exc, requests.RequestException)
            if fatal:
                stop.set()
            if retryable and attempt <= args.retries and not stop.is_set():
                retry_after = getattr(exc, "retry_after", None)
                delay = min(60.0, max(0.0, retry_after)) if retry_after is not None else min(30.0, 2 ** attempt) + random.random()
                if not stop.wait(delay):
                    continue
            result.update(status="error", error=(type(exc).__name__ + ": " + str(exc)).replace(api_key, "[REDACTED]")[:1600], stop_run=fatal)
            break
    result.update(latency_ms=round((time.perf_counter() - started) * 1000, 3), created_at=now())
    return result


def read_successes(rows, input_sha):
    by_id = {r["sample_id"]: r for r in rows}
    successes = {}
    statuses = Counter()
    if not OUTPUT.exists():
        return successes, statuses
    with OUTPUT.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            sid = x["sample_id"]
            if sid not in by_id or x.get("input_sha256") != input_sha:
                raise RuntimeError("Existing answer does not belong to current frozen input.")
            statuses[x["status"]] += 1
            if x["status"] == "ok":
                successes[sid] = x
    return successes, statuses


def summarize(successes):
    out = {"unique_successes": len(successes), "correct": 0, "accuracy": None, "by_dataset": {}}
    grouped = defaultdict(list)
    for x in successes.values():
        grouped[x["source"]].append(x)
    for source, values in sorted(grouped.items()):
        size = len(values[0]["choices"])
        matrix = [[0] * size for _ in range(size)]
        for x in values:
            matrix[x["gold_position"]][x["predicted_position"]] += 1
        correct = sum(matrix[i][i] for i in range(size))
        out["correct"] += correct
        out["by_dataset"][source] = {
            "count": len(values),
            "correct": correct,
            "accuracy": correct / len(values),
            "mean_absolute_position_error": sum(abs(x["predicted_position"] - x["gold_position"]) for x in values) / len(values),
            "gold_counts": [sum(row) for row in matrix],
            "predicted_counts": [sum(row[i] for row in matrix) for i in range(size)],
            "confusion_matrix_gold_rows_predicted_columns": matrix,
        }
    if successes:
        out["accuracy"] = out["correct"] / len(successes)
    return out


@contextlib.contextmanager
def exclusive_run():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    name = "Local\\JevOrdinalBias_" + hashlib.sha256(str(OUTPUT.resolve()).lower().encode()).hexdigest()
    ctypes.set_last_error(0)
    handle = kernel.CreateMutexW(None, False, name)
    error = ctypes.get_last_error()
    if not handle:
        raise ctypes.WinError(error)
    try:
        if error == 183:
            raise RuntimeError("Another runner for this output is already active.")
        yield
    finally:
        kernel.CloseHandle(handle)


def run(args):
    rows = load_rows()
    input_sha = digest(INPUT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("input_sha256") != input_sha:
        raise RuntimeError("Frozen input hash mismatch.")
    successes, statuses = read_successes(rows, input_sha)
    pending = [r for r in rows if r["sample_id"] not in successes]
    if args.limit:
        pending = pending[:args.limit]
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "input_rows": len(rows),
                      "existing_successes": len(successes), "pending_this_run": len(pending),
                      "endpoint": ENDPOINT, "model": MODEL, "workers": args.workers,
                      "gold_sent_to_model": False}, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        for row in rows:
            body = request_body(row)
            if set(body) != {"state", "model", "questions"}:
                raise RuntimeError("Request whitelist failed.")
        print("All 50,000 requests validated; no API calls made.", flush=True)
        return 0
    if not pending:
        print("All 50,000 items already have successful answers.", flush=True)
        return 0
    key = os.environ.get("JEV_API_KEY", "").strip()
    if not key:
        manifest.update(state="awaiting_api_key", updated_at=now(), workers=args.workers)
        atomic_json(MANIFEST, manifest)
        key = getpass.getpass("Enter System One API key (input hidden): ").strip()
    if not key:
        raise RuntimeError("No API key supplied.")
    stop = threading.Event()
    manifest.update(state="running", started_at=now(), workers=args.workers,
                    timeout_seconds=args.timeout, request_retries=args.retries)

    def save_progress(state="running"):
        manifest.update(state=state, updated_at=now(), metrics=summarize(successes),
                        status_rows=dict(statuses), remaining=TOTAL - len(successes))
        atomic_json(MANIFEST, manifest)

    save_progress()
    completed = errors = consecutive_errors = 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8", newline="\n", buffering=1) as output:
        def accept(result):
            nonlocal completed, errors, consecutive_errors
            output.write(json_bytes(result).decode("utf-8") + "\n")
            output.flush()
            completed += 1
            statuses[result["status"]] += 1
            if result["status"] == "ok":
                successes[result["sample_id"]] = result
                consecutive_errors = 0
            else:
                errors += 1
                consecutive_errors += 1
                if consecutive_errors >= max(20, args.workers):
                    stop.set()
            if completed <= 10 or completed % 25 == 0 or result["status"] == "error":
                metrics = summarize(successes)
                print(f"[{completed}/{len(pending)}] status={result['status']} unique_ok={len(successes)}/{TOTAL} "
                      f"errors_this_run={errors} accuracy={metrics['accuracy']} sample={result['sample_id']}", flush=True)
                if result["status"] == "error":
                    print(result["error"], flush=True)
                save_progress()

        print("Preflight: one pending item (saved to the same answer file).", flush=True)
        first = run_item(pending[0], key, input_sha, args, stop)
        accept(first)
        if first["status"] != "ok":
            save_progress("preflight_failed")
            return 2
        print(f"Preflight passed. Continuing with {args.workers} workers.", flush=True)
        iterator = iter(pending[1:])
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = set()

            def fill():
                while len(futures) < args.workers and not stop.is_set():
                    row = next(iterator, None)
                    if row is None:
                        break
                    futures.add(pool.submit(run_item, row, key, input_sha, args, stop))

            fill()
            while futures:
                done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    accept(future.result())
                fill()
    state = "complete" if len(successes) == TOTAL else "stopped" if stop.is_set() else "incomplete"
    save_progress(state)
    print(json.dumps({"state": state, **summarize(successes), "remaining": TOTAL - len(successes)},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if state == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Download, sample, and freeze 50,000 items only.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1 or args.retries < 0 or args.limit < 0:
        parser.error("Invalid worker/timeout/retry/limit settings.")
    with exclusive_run():
        if args.prepare:
            prepare()
            return 0
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
