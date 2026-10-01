#!/usr/bin/env python3
"""Run a post-trained KEV checkpoint on the exact frozen main40 panel.

The append journal is resumable only when input, per-item, checkpoint-tree and
run-configuration hashes match. A successful run is atomically canonicalized
to one OK row per input ID in frozen input order. Unresolved errors return a
non-zero exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from sklearn.metrics import cohen_kappa_score

from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

SCHEMA = "kev-ba-lora-main40-result-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_tree(path: Path) -> dict:
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and ".cache" not in item.relative_to(path).parts:
            files.append({
                "path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            })
    if not files:
        raise RuntimeError("checkpoint directory contains no files")
    digest = hashlib.sha256()
    for item in files:
        digest.update(f"{item['sha256']}  {item['bytes']}  {item['path']}\n".encode())
    return {"files": len(files), "bytes": sum(item["bytes"] for item in files), "sha256": digest.hexdigest()}


def request_record(row: dict) -> dict:
    keys = [chr(ord("a") + i) for i in range(len(row["choices"]))]
    body = {
        "state": row["context"],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": row["question"],
                "criteria": dict(zip(keys, row["choices"])),
            }
        },
    }
    record, _ = to_record(SystemOneRequest.model_validate(body))
    return record


def infer_encoded(model, encoded: list):
    try:
        return model.forward_batch(encoded, shared_prefix=False)
    except torch.OutOfMemoryError:
        if len(encoded) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(encoded) // 2
        return infer_encoded(model, encoded[:midpoint]) + infer_encoded(model, encoded[midpoint:])


def distribution_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    k = rows[0]["option_count"]
    if any(row["option_count"] != k for row in rows):
        raise RuntimeError("a dataset group contains inconsistent option counts")
    gold = Counter(row["gold_position"] for row in rows)
    pred = Counter(row["predicted_position"] for row in rows)
    n = len(rows)
    gold_p = [gold[index] / n for index in range(k)]
    pred_p = [pred[index] / n for index in range(k)]
    accuracy = sum(row["correct"] for row in rows) / n
    chance = sum(a * b for a, b in zip(gold_p, pred_p))
    entropy = -sum(p * math.log(p) for p in pred_p if p > 0)
    ordinal = rows[0]["task_type"] == "ordinal_choice"
    y_true = [row["gold_position"] for row in rows]
    y_pred = [row["predicted_position"] for row in rows]
    return {
        "rows": n,
        "option_count": k,
        "task_type": rows[0]["task_type"],
        "accuracy": accuracy,
        "chance_accuracy": chance,
        "chance_corrected_accuracy": (accuracy - chance) / (1 - chance) if chance < 1 else 0.0,
        "total_variation_distance": 0.5 * sum(abs(a - b) for a, b in zip(gold_p, pred_p)),
        "normalized_prediction_entropy": entropy / math.log(k) if k > 1 else 1.0,
        "active_labels_1pct": sum(count >= 0.01 * n for count in pred.values()),
        "label_space_utilization_1pct": sum(count >= 0.01 * n for count in pred.values()) / k,
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic") if ordinal else None,
        "normalized_mae": (
            sum(abs(a - b) for a, b in zip(y_true, y_pred)) / (n * (k - 1))
            if ordinal and k > 1 else None
        ),
        "gold_counts": [gold[index] for index in range(k)],
        "prediction_counts": [pred[index] for index in range(k)],
        "gold_distribution": gold_p,
        "prediction_distribution": pred_p,
    }


def load_input_metadata(path: Path) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    item_hashes: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in item_hashes:
                raise RuntimeError(f"duplicate input sample ID at line {line_number}: {sample_id}")
            choices = row.get("choices")
            if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
                raise RuntimeError(f"invalid choice count at line {line_number}: {sample_id}")
            if not 0 <= int(row["gold_position"]) < len(choices):
                raise RuntimeError(f"invalid gold position at line {line_number}: {sample_id}")
            order.append(sample_id)
            item_hashes[sample_id] = canonical_sha256(row)
    if not order:
        raise RuntimeError("evaluation input is empty")
    return order, item_hashes


def provenance_fields(input_sha: str, item_sha: str, checkpoint_sha: str, config_sha: str, run_name: str) -> dict:
    return {
        "schema_version": SCHEMA,
        "input_sha256": input_sha,
        "item_sha256": item_sha,
        "checkpoint_tree_sha256": checkpoint_sha,
        "run_config_sha256": config_sha,
        "run_name": run_name,
    }


def load_existing(
    path: Path,
    item_hashes: dict[str, str],
    input_sha: str,
    checkpoint_sha: str,
    config_sha: str,
    run_name: str,
) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                sample_id = str(row["sample_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(f"invalid resumable output row at line {line_number}") from exc
            if sample_id not in item_hashes:
                raise RuntimeError(f"output contains an ID absent from the current input: {sample_id}")
            expected = provenance_fields(
                input_sha, item_hashes[sample_id], checkpoint_sha, config_sha, run_name
            )
            for key, value in expected.items():
                if row.get(key) != value:
                    raise RuntimeError(f"resume provenance mismatch at line {line_number}: {sample_id}/{key}")
            latest[sample_id] = row
    return latest


def atomic_canonicalize(path: Path, order: list[str], latest: dict[str, dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for sample_id in order:
                row = latest[sample_id]
                if row.get("status") != "ok":
                    raise RuntimeError(f"cannot canonicalize unresolved row: {sample_id}")
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_summary(
    summary: Path,
    output: Path,
    order: list[str],
    latest: dict[str, dict],
    run_config: dict,
    run_config_sha: str,
    elapsed: float,
) -> None:
    statuses = Counter(str(latest[sample_id].get("status", "missing")) if sample_id in latest else "missing" for sample_id in order)
    good = [latest[sample_id] for sample_id in order if sample_id in latest and latest[sample_id].get("status") == "ok"]
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in good:
        by_dataset[row["source"]].append(row)
    ordinal = [row for row in good if row["task_type"] == "ordinal_choice"]
    nominal = [row for row in good if row["task_type"] == "nominal_choice"]
    payload = {
        "schema_version": "kev-ba-lora-main40-summary-v2",
        "checkpoint": run_config["run_name"],
        "output": output.name,
        "evaluation_input_sha256": run_config["input_sha256"],
        "checkpoint_tree_sha256": run_config["checkpoint_tree_sha256"],
        "run_config_sha256": run_config_sha,
        "run_config": run_config,
        "statuses": dict(sorted(statuses.items())),
        "elapsed_seconds_this_run": elapsed,
        "overall_accuracy": sum(row["correct"] for row in good) / len(good) if good else None,
        "ordinal_accuracy": sum(row["correct"] for row in ordinal) / len(ordinal) if ordinal else None,
        "nominal_accuracy": sum(row["correct"] for row in nominal) / len(nominal) if nominal else None,
        "datasets": {name: distribution_metrics(rows) for name, rows in sorted(by_dataset.items())},
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary.with_suffix(summary.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, summary)
    print(json.dumps({key: value for key, value in payload.items() if key not in {"datasets", "run_config"}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True, help="portable model label stored in results")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.batch < 1 or args.progress_every < 1:
        parser.error("--batch and --progress-every must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    order, item_hashes = load_input_metadata(args.input)
    input_sha = sha256_file(args.input)
    checkpoint = Checkpoint(args.checkpoint)
    checkpoint_info = checkpoint_tree(Path(checkpoint.path))
    device = torch.device(args.device)
    run_config = {
        "schema_version": SCHEMA,
        "run_name": args.run_name,
        "input_sha256": input_sha,
        "checkpoint_tree_sha256": checkpoint_info["sha256"],
        "checkpoint_files": checkpoint_info["files"],
        "checkpoint_bytes": checkpoint_info["bytes"],
        "batch": args.batch,
        "device_type": device.type,
        "dtype": "bfloat16" if device.type == "cuda" else "float32",
        "backend": "torch",
        "temperature": 1.0,
        "max_state": SERVE_MAX_STATE,
        "max_branch": SERVE_MAX_BRANCH,
    }
    run_config_sha = canonical_sha256(run_config)
    latest = load_existing(
        args.output, item_hashes, input_sha, checkpoint_info["sha256"], run_config_sha, args.run_name
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = checkpoint.load(
        device,
        LoadOptions(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                    merge=True, temperature=1.0, backend="torch"),
    )
    model.eval()
    limits = {"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH}
    started = time.time()
    processed = ok = errors = 0
    pending: list[dict] = []

    def write_result(handle, row: dict, result: dict) -> None:
        nonlocal processed, ok, errors
        sample_id = str(row["sample_id"])
        result.update(provenance_fields(
            input_sha, item_hashes[sample_id], checkpoint_info["sha256"], run_config_sha, args.run_name
        ))
        handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
        latest[sample_id] = result
        processed += 1
        if result["status"] == "ok":
            ok += 1
        else:
            errors += 1

    def flush_batch(handle) -> None:
        nonlocal pending
        if not pending:
            return
        encodings, kept = [], []
        for row in pending:
            try:
                encodings.append(model.encode(tokenizer, request_record(row), strict=True, **limits))
                kept.append(row)
            except Exception as exc:
                write_result(handle, row, {
                    "sample_id": row["sample_id"], "source": row["source"], "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        if encodings:
            try:
                with torch.inference_mode():
                    logits_batch = infer_encoded(model, encodings)
                if len(logits_batch) != len(kept):
                    raise RuntimeError("model returned the wrong batch length")
                for row, logits in zip(kept, logits_batch):
                    values = logits[0].float()
                    probabilities = torch.softmax(values, -1).cpu().tolist()
                    prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
                    write_result(handle, row, {
                        "sample_id": row["sample_id"], "source": row["source"],
                        "task_type": row["task_type"], "option_count": len(row["choices"]),
                        "gold_position": row["gold_position"], "predicted_position": prediction,
                        "predicted_label_text": row["choices"][prediction],
                        "correct": prediction == row["gold_position"],
                        "probabilities": probabilities, "status": "ok",
                    })
            except Exception as exc:
                for row in kept:
                    write_result(handle, row, {
                        "sample_id": row["sample_id"], "source": row["source"], "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        handle.flush()
        if processed and processed % args.progress_every < args.batch:
            print(f"processed={processed} ok={ok} error={errors}", flush=True)
        pending = []

    with args.output.open("a", encoding="utf-8", newline="\n", buffering=1) as output_handle:
        with args.input.open(encoding="utf-8-sig") as input_handle:
            for line in input_handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                if latest.get(sample_id, {}).get("status") == "ok":
                    continue
                pending.append(row)
                if len(pending) >= args.batch:
                    flush_batch(output_handle)
            flush_batch(output_handle)

    unresolved = [sample_id for sample_id in order if latest.get(sample_id, {}).get("status") != "ok"]
    if not unresolved:
        atomic_canonicalize(args.output, order, latest)
    write_summary(
        args.summary, args.output, order, latest, run_config, run_config_sha, time.time() - started
    )
    if unresolved:
        print(f"unresolved errors: {len(unresolved)}; first={unresolved[:5]}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
