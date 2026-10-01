#!/usr/bin/env python3
"""Run a downloaded KEV checkpoint on a frozen JSONL input."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_bias.io import canonical_json_sha256, iter_jsonl, sha256_file
from jev_bias.integrity import tree_manifest
from jev_bias.kev_provenance import verify_installed_kev
from jev_bias.model_integrity import load_and_verify_downloaded_model


def request_record(row: dict) -> dict:
    from kev.api import SystemOneRequest, to_record

    keys = [chr(ord("a") + i) for i in range(len(row["choices"]))]
    payload = {"state": row["context"], "questions": {"decision": {
        "type": "choice", "instructions": row["question"], "criteria": dict(zip(keys, row["choices"]))
    }}}
    record, _ = to_record(SystemOneRequest.model_validate(payload))
    return record


def infer(model, encoded: list):
    import torch

    try:
        return model.forward_batch(encoded, shared_prefix=False)
    except torch.OutOfMemoryError:
        if len(encoded) == 1:
            raise
        torch.cuda.empty_cache(); midpoint = len(encoded) // 2
        return infer(model, encoded[:midpoint]) + infer(model, encoded[midpoint:])


def validate_input(path: Path) -> tuple[set[str], dict[str, str]]:
    ids: set[str] = set()
    hashes: dict[str, str] = {}
    for line_number, row in enumerate(iter_jsonl(path), 1):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in ids:
            raise ValueError(f"Missing or duplicate sample_id at input row {line_number}: {sample_id!r}")
        choices = row.get("choices")
        gold = row.get("gold_position")
        if not isinstance(choices, list) or not (2 <= len(choices) <= 26):
            raise ValueError(f"Invalid choices at {sample_id}")
        if not isinstance(gold, int) or not (0 <= gold < len(choices)):
            raise ValueError(f"Invalid gold_position at {sample_id}")
        ids.add(sample_id)
        hashes[sample_id] = canonical_json_sha256(row)
    if not ids:
        raise ValueError("Input is empty")
    return ids, hashes


def validate_existing(path: Path, *, input_sha256: str, model: str, run_config_sha256: str,
                      input_ids: set[str], item_hashes: dict[str, str]) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for line_number, row in enumerate(iter_jsonl(path), 1):
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in input_ids:
            raise RuntimeError(f"Existing output row {line_number} has an ID outside this input: {sample_id}")
        if row.get("input_sha256") != input_sha256:
            raise RuntimeError(f"Existing output belongs to a different input at row {line_number}")
        if row.get("item_sha256") != item_hashes[sample_id]:
            raise RuntimeError(f"Existing output item hash mismatch for {sample_id}")
        if row.get("model") != model:
            raise RuntimeError(f"Existing output model mismatch for {sample_id}")
        if row.get("run_config_sha256") != run_config_sha256:
            raise RuntimeError(f"Existing output inference configuration mismatch for {sample_id}")
        if row.get("status") in {"ok", "rejected"}:
            done.add(sample_id)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", choices=(
        "kev-0.8b", "kev-4b", "kev-9b",
        "kev-0.8b-ba-lora", "kev-4b-ba-lora",
    ),
                             help="Use the pinned Hub adapter in configs/models.json")
    model_group.add_argument("--checkpoint", help="Explicit local checkpoint directory or Hub ID")
    parser.add_argument("--model-root", type=Path, default=ROOT / "artifacts" / "models",
                        help="Hugging Face home populated by download_models.py")
    parser.add_argument("--offline", action="store_true", help="Forbid network access after models have been downloaded")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--allow-unverified-checkpoint", action="store_true",
                        help="Permit a non-local --checkpoint that cannot be covered by the model hash manifest")
    parser.add_argument("--allow-unverified-kev-code", action="store_true",
                        help="Permit installed KEV code without the pinned PEP 610 Git revision")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive (1.0 preserves the checkpoint's raw logit scale)")
    input_ids, item_hashes = validate_input(args.input)
    input_sha256 = sha256_file(args.input)
    args.model_root = args.model_root.resolve()
    os.environ["HF_HOME"] = str(args.model_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(args.model_root / "hub")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import ContextOverflow, SERVE_MAX_BRANCH, SERVE_MAX_STATE

    checkpoint = args.checkpoint
    model_label = args.checkpoint
    model_artifacts = []
    if args.model:
        spec = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))["models"][args.model]
        if not spec.get("revision"):
            raise SystemExit(f"{args.model} has no recorded adapter revision; pass an explicit --checkpoint instead")
        model_artifacts = load_and_verify_downloaded_model(args.model_root, args.model)
        adapter = next(item for item in model_artifacts if item["role"] == "adapter")
        base = next(item for item in model_artifacts if item["role"] == "base")
        if adapter["repo_id"] != spec["repo_id"] or adapter["resolved_revision"] != spec["revision"]:
            raise SystemExit(f"Downloaded adapter does not match the pinned {args.model} revision")
        if base["repo_id"] != spec["base_repo_id"] or base["resolved_revision"] != spec["base_revision"]:
            raise SystemExit(f"Downloaded base model does not match the pinned {args.model} revision")
        checkpoint = str(args.model_root / adapter["cache_path"])
        model_label = args.model
        checkpoint_display = f"{adapter['repo_id']}@{adapter['resolved_revision']}"
    else:
        explicit_path = Path(str(args.checkpoint)).expanduser()
        if explicit_path.is_dir():
            checkpoint = str(explicit_path.resolve())
            explicit_tree = tree_manifest(checkpoint)
            model_label = explicit_path.name
            checkpoint_display = explicit_path.name
            model_artifacts = [{
                "role": "adapter-or-merged-checkpoint",
                "tree_sha256": explicit_tree["tree_sha256"],
                "file_count": explicit_tree["file_count"],
            }]
        elif not args.allow_unverified_checkpoint:
            raise SystemExit(
                "--checkpoint must be a local directory so its files can be hashed; "
                "use --allow-unverified-checkpoint only for a deliberately non-reproducible Hub run"
            )
        else:
            checkpoint_display = str(args.checkpoint)
            model_label = str(args.checkpoint)
    device = torch.device(args.device)
    dtype_name = "bfloat16" if device.type == "cuda" else "float32"
    model_config = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))
    kev_code = verify_installed_kev(
        model_config["kev_code"]["revision"],
        allow_unverified=args.allow_unverified_kev_code,
    )
    run_config_sha256 = canonical_json_sha256({
        "model": str(model_label),
        "checkpoint": checkpoint_display,
        "verified_model_artifacts": model_artifacts,
        "kev_code_revision": model_config["kev_code"]["revision"],
        "kev_code": kev_code,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "backend": "torch",
        "dtype": dtype_name,
        "device_type": device.type,
    })
    done = validate_existing(
        args.output,
        input_sha256=input_sha256,
        model=str(model_label),
        run_config_sha256=run_config_sha256,
        input_ids=input_ids,
        item_hashes=item_hashes,
    )
    tokenizer, model = Checkpoint(checkpoint).load(
        device, LoadOptions(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                            merge=True, temperature=args.temperature, backend="torch")
    )
    model.eval(); args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    processed = 0

    def flush(handle):
        nonlocal processed, pending
        if not pending:
            return
        accepted = []
        for row in pending:
            try:
                encoded = model.encode(tokenizer, request_record(row), strict=True,
                                       max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
                accepted.append((row, encoded))
            except ContextOverflow as exc:
                result = {
                    "schema_version": "kev-answer-public-v1",
                    "sample_id": row["sample_id"],
                    "source": row["source"],
                    "status": "rejected",
                    "model": model_label,
                    "run_config_sha256": run_config_sha256,
                    "input_sha256": input_sha256,
                    "item_sha256": item_hashes[row["sample_id"]],
                    "option_count": len(row["choices"]),
                    "gold_position": row["gold_position"],
                    "gold_label_text": row.get("gold_label_text", row["choices"][row["gold_position"]]),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1600],
                }
                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                processed += 1
        if not accepted:
            handle.flush()
            pending = []
            return
        started = time.perf_counter()
        with torch.inference_mode():
            logits_batch = infer(model, [encoded for _, encoded in accepted])
        elapsed = (time.perf_counter() - started) * 1000
        for (row, _), logits in zip(accepted, logits_batch):
            probs = torch.softmax(logits[0].float(), -1).cpu().tolist()
            pred = max(range(len(probs)), key=probs.__getitem__)
            result = {"schema_version": "kev-answer-public-v1", "sample_id": row["sample_id"],
                      "source": row["source"], "status": "ok", "model": model_label,
                      "run_config_sha256": run_config_sha256,
                      "input_sha256": input_sha256, "item_sha256": canonical_json_sha256(row),
                      "option_count": len(row["choices"]), "predicted_position": pred,
                      "predicted_label_text": row["choices"][pred],
                      "gold_position": row["gold_position"],
                      "gold_label_text": row.get("gold_label_text", row["choices"][row["gold_position"]]),
                      "correct": pred == row["gold_position"],
                      "probabilities": probs, "batch_latency_ms": elapsed}
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            processed += 1
        handle.flush(); pending = []
        if processed % 100 < args.batch_size:
            print(f"processed={processed}", flush=True)

    with args.output.open("a", encoding="utf-8", newline="\n", buffering=1) as handle:
        for row in iter_jsonl(args.input):
            if row["sample_id"] in done:
                continue
            pending.append(row)
            if len(pending) >= args.batch_size:
                flush(handle)
        flush(handle)
    statuses = Counter()
    rows = 0
    for row in iter_jsonl(args.output):
        rows += 1
        statuses[str(row.get("status", "missing"))] += 1
    manifest = {
        "schema_version": "kev-local-run-manifest-v1",
        "input": args.input.name,
        "input_sha256": input_sha256,
        "model": model_label,
        "checkpoint": checkpoint_display,
        "verified_model_artifacts": model_artifacts,
        "run_config_sha256": run_config_sha256,
        "kev_code_revision": model_config["kev_code"]["revision"],
        "verified_kev_code": kev_code,
        "batch_size": args.batch_size,
        "device_type": device.type,
        "temperature": args.temperature,
        "output": args.output.name,
        "output_sha256": sha256_file(args.output),
        "rows_in_append_log": rows,
        "statuses_in_append_log": dict(sorted(statuses.items())),
    }
    args.output.with_suffix(args.output.suffix + ".run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    main()
