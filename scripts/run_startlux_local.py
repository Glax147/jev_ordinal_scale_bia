#!/usr/bin/env python3
"""Run a downloaded StartLux-Decision checkpoint on an evaluation JSONL file.

The model's bundled ``startlux_decision`` package supplies the prompt renderer
and option-letter readout. Gold labels are used only after inference for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.integrity import tree_manifest
from jev_bias.io import canonical_json_sha256, iter_jsonl, sha256_file

SIZES = ("0.8b", "2b", "4b", "9b", "27b")


def request_for(row: dict) -> tuple[str, dict]:
    choices = row["choices"]
    if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
        raise ValueError(f"Expected 2–26 choices for {row.get('sample_id')}")
    return row["context"], {"decision": {
        "type": "choice",
        "instructions": row["question"],
        "criteria": {chr(ord("a") + i): choice for i, choice in enumerate(choices)},
    }}


def parse_answer(row: dict, answer: dict) -> tuple[int, list[float]]:
    decision = answer["decision"]
    keys = [chr(ord("a") + i) for i in range(len(row["choices"]))]
    probabilities = [float(decision["probabilities"][key]) for key in keys]
    key = decision["choice"]
    if (key not in keys or len(probabilities) != len(keys)
            or any(not math.isfinite(p) or p < 0 for p in probabilities)):
        raise ValueError(f"Invalid StartLux answer for {row['sample_id']}")
    return keys.index(key), probabilities


def downloaded_checkpoint(model_root: Path, model_name: str) -> tuple[Path, dict]:
    config = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))["models"][model_name]
    manifest = json.loads((model_root / "download_manifest.json").read_text(encoding="utf-8"))
    matches = [entry for entry in manifest["models"]
               if entry["model"] == model_name and entry["role"] == "weights"]
    if len(matches) != 1:
        raise RuntimeError(f"Download {model_name} with scripts/download_models.py first")
    entry = matches[0]
    if entry["repo_id"] != config["repo_id"]:
        raise RuntimeError(f"Downloaded model repository mismatch for {model_name}")
    if config.get("revision") and entry["resolved_revision"] != config["revision"]:
        raise RuntimeError(f"Downloaded model revision mismatch for {model_name}")
    checkpoint = (model_root / entry["cache_path"]).resolve()
    if not checkpoint.is_dir() or tree_manifest(checkpoint)["tree_sha256"] != entry["tree_sha256"]:
        raise RuntimeError(f"Downloaded model files changed after manifest creation: {model_name}")
    return checkpoint, entry


def completed_ids(path: Path, input_hash: str, config_hash: str) -> dict[str, str]:
    done: dict[str, str] = {}
    if not path.is_file():
        return done
    for row in iter_jsonl(path):
        if row.get("input_sha256") != input_hash or row.get("run_config_sha256") != config_hash:
            raise RuntimeError("Existing output was produced with a different input or model configuration")
        if row.get("status") == "ok":
            sample_id = row["sample_id"]
            if sample_id in done:
                raise RuntimeError(f"Duplicate successful result: {sample_id}")
            done[sample_id] = row["item_sha256"]
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=[f"startlux-{size}" for size in SIZES])
    parser.add_argument("--model-root", type=Path, default=ROOT / "artifacts" / "models")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batch-tokens", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_batch_tokens < 1:
        raise SystemExit("Batch size and maximum batch tokens must be positive")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    model_root = args.model_root.resolve()
    checkpoint, entry = downloaded_checkpoint(model_root, args.model)
    input_hash = sha256_file(args.input)
    config_hash = canonical_json_sha256({
        "model": args.model, "checkpoint_sha256": entry["tree_sha256"],
        "revision": entry["resolved_revision"], "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens, "device": args.device,
        "decision_type": "choice", "option_keys": "a..z",
    })
    done = completed_ids(args.output, input_hash, config_hash)
    seen_input: set[str] = set()
    pending_count = 0
    for row in iter_jsonl(args.input):
        sample_id = row.get("sample_id")
        if not sample_id or sample_id in seen_input:
            raise ValueError(f"Missing or duplicate input sample ID: {sample_id!r}")
        seen_input.add(sample_id)
        gold = row.get("gold_position")
        if not isinstance(gold, int) or not 0 <= gold < len(row.get("choices", [])):
            raise ValueError(f"Invalid gold choice for {sample_id}")
        if sample_id in done:
            if done[sample_id] != canonical_json_sha256(row):
                raise RuntimeError(f"Existing result refers to a different input item: {sample_id}")
        else:
            pending_count += 1
    if not seen_input or not done.keys() <= seen_input:
        raise RuntimeError("Existing output contains IDs outside this input, or the input is empty")
    if not pending_count:
        print(f"All {len(done)} input items already have successful results")
        return

    sys.path.insert(0, str(checkpoint))
    from startlux_decision import StartLuxDecision

    model = StartLuxDecision(str(checkpoint), device=args.device,
                             max_batch_tokens=args.max_batch_tokens, graphs=False, images=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending: list[dict] = []
    completed = 0

    def flush(handle) -> None:
        nonlocal completed
        if not pending:
            return
        answers = model.decide_batch([request_for(row) for row in pending])
        if len(answers) != len(pending):
            raise RuntimeError("StartLux returned fewer answers than requests")
        for row, answer in zip(pending, answers):
            pred, probabilities = parse_answer(row, answer)
            gold = int(row["gold_position"])
            result = {
                "sample_id": row["sample_id"], "source": row["source"],
                "status": "ok", "model": args.model,
                "input_sha256": input_hash, "item_sha256": canonical_json_sha256(row),
                "run_config_sha256": config_hash,
                "option_count": len(row["choices"]),
                "gold_position": gold, "gold_label_text": row["choices"][gold],
                "predicted_position": pred,
                "predicted_label_text": row["choices"][pred],
                "correct": pred == gold, "probabilities": probabilities,
            }
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed += 1
        handle.flush()
        pending.clear()
        if completed % 1000 < args.batch_size:
            print(f"completed_this_run={completed}", flush=True)

    with args.output.open("a", encoding="utf-8", newline="\n", buffering=1) as handle:
        for row in iter_jsonl(args.input):
            if row["sample_id"] in done:
                continue
            pending.append(row)
            if len(pending) == args.batch_size:
                flush(handle)
        flush(handle)
    print(f"Completed {completed} new items; {len(done)} were already successful")


if __name__ == "__main__":
    main()
