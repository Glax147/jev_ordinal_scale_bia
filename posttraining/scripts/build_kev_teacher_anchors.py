#!/usr/bin/env python3
"""Build or verify the exact frozen-KEV teacher anchors used by BA-LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import load_records, materialize
from kev.model import resolve_model_source, training_context

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RECORDS = 32_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_model_tree(directory: Path) -> dict:
    manifest = json.loads((ROOT / "manifests/model_files_manifest.json").read_text(encoding="utf-8"))
    prefix = f"models/{directory.name}/"
    entries = sorted((item for item in manifest["files"] if item["path"].startswith(prefix)), key=lambda x: x["path"])
    if not entries:
        raise RuntimeError(f"model directory is not represented in the pinned manifest: {directory.name}")
    expected = {Path(item["path"]).relative_to(Path("models") / directory.name).as_posix() for item in entries}
    actual = {
        item.relative_to(directory).as_posix()
        for item in directory.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(directory).parts
    }
    if actual != expected:
        raise RuntimeError(
            f"model file-set mismatch for {directory.name}: "
            f"missing={sorted(expected - actual)[:5]} unexpected={sorted(actual - expected)[:5]}"
        )
    tree = hashlib.sha256()
    total = 0
    for entry in entries:
        relative = Path(entry["path"]).relative_to(Path("models") / directory.name)
        path = directory / relative
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"model file integrity mismatch: {entry['path']}")
        tree.update(f"{entry['sha256']}  {entry['bytes']}  {relative.as_posix()}\n".encode())
        total += int(entry["bytes"])
    return {"directory": directory.name, "files": len(entries), "bytes": total, "tree_sha256": tree.hexdigest()}


def record_ids(records: list[dict]) -> list[str]:
    ids = [str(record["_meta"]["id"]) for record in records]
    if len(ids) != EXPECTED_RECORDS or len(set(ids)) != EXPECTED_RECORDS:
        raise RuntimeError(f"teacher input must contain {EXPECTED_RECORDS} unique records; got {len(ids)} rows/{len(set(ids))} IDs")
    return ids


def provenance(args: argparse.Namespace, checkpoint: Checkpoint, records: list[dict]) -> dict:
    ids = record_ids(records)
    adapter_tree = verified_model_tree(Path(checkpoint.path))
    base_path = Path(resolve_model_source(checkpoint.meta.base))
    if not base_path.is_dir():
        raise RuntimeError("the pinned teacher base must be present in the bundle-local model directory")
    base_tree = verified_model_tree(base_path)
    context = training_context(args.max_state)
    return {
        "schema_version": "kev-ba-lora-teacher-anchors-v2",
        "teacher_checkpoint": args.checkpoint,
        "teacher_weights_sha256": checkpoint.weights_sha256(),
        "teacher_head_sha256": sha256_file(checkpoint.file("head.pt")),
        "teacher_adapter_tree": adapter_tree,
        "teacher_base_repository": checkpoint.meta.base,
        "teacher_base_revision": checkpoint.meta.base_revision,
        "teacher_base_tree": base_tree,
        "teacher_temperature": args.temperature,
        "train_input": args.data.name,
        "train_input_bytes": args.data.stat().st_size,
        "train_input_sha256": sha256_file(args.data),
        "train_record_ids_ordered_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "records": EXPECTED_RECORDS,
        "dropped": 0,
        "max_state": context["max_state"],
        "max_branch": context["max_branch"],
        "readout": "original KEV pointer probabilities",
    }


def validate_payload(payload: dict, expected_meta: dict, expected_ids: set[str]) -> None:
    if payload.get("_meta") != expected_meta:
        raise RuntimeError("teacher-anchor provenance does not match the current data/model/temperature configuration")
    if payload.get("dropped") != []:
        raise RuntimeError("teacher-anchor file contains dropped records")
    targets = payload.get("targets")
    if not isinstance(targets, dict) or set(targets) != expected_ids:
        raise RuntimeError("teacher-anchor target IDs do not match the exact 32,000-row training set")
    for record_id, questions in targets.items():
        if not isinstance(questions, dict) or not questions:
            raise RuntimeError(f"teacher anchor has no questions: {record_id}")
        for qid, distribution in questions.items():
            values = list(distribution.values()) if isinstance(distribution, dict) else []
            if not values or any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
                raise RuntimeError(f"invalid teacher probabilities: {record_id}/{qid}")
            if abs(sum(float(value) for value in values) - 1.0) > 1e-4:
                raise RuntimeError(f"teacher probabilities do not sum to one: {record_id}/{qid}")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="e.g. jaredpalmer/kev-0.8b")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_state", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.temperature <= 0 or args.batch < 1:
        parser.error("--temperature and --batch must be positive")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    records = load_records(args.data)
    ids = record_ids(records)
    checkpoint = Checkpoint(args.checkpoint)
    expected_meta = provenance(args, checkpoint, records)
    if args.verify_existing:
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        validate_payload(payload, expected_meta, set(ids))
        print(json.dumps({"status": "ok", **expected_meta}, ensure_ascii=False, indent=2))
        return
    if args.out.exists():
        raise FileExistsError(args.out)

    device = torch.device(args.device)
    tokenizer, model = checkpoint.load(
        device,
        LoadOptions(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                    merge=True, temperature=1.0, backend="torch"),
    )
    model.eval()
    limits = {"max_state": expected_meta["max_state"], "max_branch": expected_meta["max_branch"]}
    targets: dict[str, dict] = {}
    with torch.no_grad():
        for start in range(0, len(records), args.batch):
            requests = records[start:start + args.batch]
            encodings, internals = [], []
            for request in requests:
                record_id = str(request["_meta"]["id"])
                internal = materialize(request)
                try:
                    encoded = model.encode(tokenizer, internal, strict=True, **limits)
                except Exception as exc:
                    raise RuntimeError(f"teacher encoding failed for {record_id}: {type(exc).__name__}: {exc}") from exc
                encodings.append(encoded)
                internals.append((record_id, internal))
            logits_batch = model.forward_batch(encodings, shared_prefix=False)
            if len(logits_batch) != len(internals):
                raise RuntimeError("teacher forward pass returned the wrong batch length")
            for (record_id, internal), logits in zip(internals, logits_batch):
                if record_id in targets or len(logits) != len(internal["questions"]):
                    raise RuntimeError(f"invalid teacher output structure for {record_id}")
                by_question = {}
                for question, values in zip(internal["questions"], logits):
                    probabilities = torch.softmax(values.float() / args.temperature, -1).cpu().tolist()
                    by_question[question["qid"]] = dict(zip(question["keys"], probabilities))
                targets[record_id] = by_question
            if start % (args.batch * 50) == 0:
                print(f"anchors {start}/{len(records)}", flush=True)

    payload = {"_meta": expected_meta, "targets": targets, "dropped": []}
    validate_payload(payload, expected_meta, set(ids))
    atomic_json(args.out, payload)
    print(json.dumps({"status": "ok", **expected_meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
