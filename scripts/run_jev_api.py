#!/usr/bin/env python3
"""Resumable System One choice runner for JEV-compatible hosted endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import getpass
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_bias.io import canonical_json_sha256, iter_jsonl, sha256_file

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
FATAL = {401, 402, 403}
LOCAL = threading.local()


class RequestFailure(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, fatal: bool = False,
                 retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.fatal = fatal
        self.retry_after = retry_after


def body(row: dict, model: str) -> dict:
    keys = [chr(ord("A") + i) for i in range(len(row["choices"]))]
    return {"state": row["context"], "model": model, "questions": {"decision": {
        "type": "choice", "instructions": row["question"], "criteria": dict(zip(keys, row["choices"]))
    }}}


def parse_response(data: dict, row: dict) -> dict:
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    answer = payload.get("answers", {}).get("decision")
    keys = [chr(ord("A") + i) for i in range(len(row["choices"]))]
    if not isinstance(answer, dict) or answer.get("type") != "choice" or answer.get("choice") not in keys:
        raise ValueError("Malformed choice response: " + json.dumps(data, ensure_ascii=False)[:1000])
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(keys):
        raise ValueError("Missing or incomplete probability distribution")
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1
               for v in probs.values()):
        raise ValueError("Invalid probability value")
    if not math.isclose(sum(probs.values()), 1.0, abs_tol=0.02):
        raise ValueError("Probabilities do not sum to approximately one")
    position = keys.index(answer["choice"])
    return {"predicted_key": answer["choice"], "predicted_position": position,
            "predicted_label_text": row["choices"][position], "probabilities": probs,
            "confidence": answer.get("confidence"), "returned_model": payload.get("model") or data.get("model"),
            "usage": payload.get("usage") or data.get("usage") or {}}


def call(row: dict, args, api_key: str, stop: threading.Event) -> dict:
    started = time.perf_counter()
    result = {"schema_version": "jev-answer-public-v1", "sample_id": row["sample_id"],
              "source": row["source"], "requested_model": args.model,
              "input_sha256": args.input_sha256, "item_sha256": canonical_json_sha256(row),
              "option_count": len(row["choices"]), "gold_position": row["gold_position"],
              "gold_label_text": row.get("gold_label_text", row["choices"][row["gold_position"]])}
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(args.retries + 1):
        try:
            if stop.is_set():
                raise RequestFailure("Run stopped before request", fatal=True)
            response = LOCAL.session.post(args.endpoint, json=body(row, args.model),
                                          headers={"Authorization": f"Bearer {api_key}"},
                                          timeout=(20, args.timeout))
            if response.status_code >= 400:
                try:
                    retry_after = float(response.headers.get("Retry-After", ""))
                except ValueError:
                    retry_after = None
                raise RequestFailure(
                    f"HTTP {response.status_code}: {response.text[:1000]}",
                    retryable=response.status_code in RETRYABLE,
                    fatal=response.status_code in FATAL,
                    retry_after=retry_after,
                )
            parsed = parse_response(response.json(), row)
            returned_model = parsed.get("returned_model")
            if (
                args.expected_returned_model
                and returned_model != args.expected_returned_model
                and not args.allow_model_drift
            ):
                raise RequestFailure(
                    "Returned model identity mismatch: "
                    f"observed={returned_model!r}, expected={args.expected_returned_model!r}",
                    fatal=True,
                )
            result.update(parsed)
            result.update(status="ok", correct=result["predicted_position"] == row["gold_position"])
            break
        except (requests.RequestException, RequestFailure, ValueError, TypeError, AttributeError) as exc:
            fatal = isinstance(exc, RequestFailure) and exc.fatal
            retryable = isinstance(exc, requests.RequestException) or (
                isinstance(exc, RequestFailure) and exc.retryable
            )
            exhausted = retryable and attempt >= args.retries
            if fatal or exhausted:
                stop.set()
            if retryable and not fatal and attempt < args.retries and not stop.is_set():
                delay = getattr(exc, "retry_after", None)
                delay = min(60, max(0.0, delay)) if delay is not None else min(30, 2 ** (attempt + 1)) + random.random()
                if not stop.wait(delay):
                    continue
            error = f"{type(exc).__name__}: {exc}".replace(api_key, "[REDACTED]")
            # Hosted relays are operational details, not experiment metadata.
            # requests exceptions often echo the full URL, so redact both it
            # and its authority before persisting a failure row.
            error = error.replace(str(args.endpoint), "[ENDPOINT]")
            authority = urlsplit(str(args.endpoint)).netloc
            if authority:
                error = error.replace(authority, "[ENDPOINT]")
            error = error[:1600]
            result.update(status="error", error=error, stop_run=fatal or exhausted)
            break
    result.update(request_attempts=attempt + 1, latency_ms=round((time.perf_counter() - started) * 1000, 3),
                  created_at=datetime.now(timezone.utc).isoformat())
    return result


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


def validate_existing(path: Path, *, input_sha256: str, model: str,
                      expected_returned_model: str | None, allow_model_drift: bool,
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
        if row.get("requested_model") != model:
            raise RuntimeError(f"Existing output model mismatch for {sample_id}")
        if (
            row.get("status") == "ok"
            and expected_returned_model
            and row.get("returned_model") != expected_returned_model
            and not allow_model_drift
        ):
            raise RuntimeError(f"Existing output returned-model mismatch for {sample_id}")
        if row.get("status") == "ok":
            done.add(sample_id)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("JEV_API_URL"))
    parser.add_argument("--model", default=os.environ.get("JEV_MODEL", "jev-1.13"))
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected-returned-model",
                        help="Exact hosted model identity required in successful responses")
    parser.add_argument("--allow-model-drift", action="store_true",
                        help="Permit responses routed to a different/unspecified hosted model")
    args = parser.parse_args()
    if not args.endpoint:
        raise SystemExit("Set JEV_API_URL or pass --endpoint")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1 second")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    if not args.expected_returned_model:
        model_config = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))
        for spec in model_config.get("models", {}).values():
            if spec.get("access") == "hosted-api" and spec.get("requested_model") == args.model:
                args.expected_returned_model = spec.get("returned_model")
                break
    input_ids, item_hashes = validate_input(args.input)
    args.input_sha256 = sha256_file(args.input)
    api_key = os.environ.get("JEV_API_KEY") or getpass.getpass("JEV API key (hidden): ")
    if not api_key:
        raise SystemExit("JEV_API_KEY is empty")
    done = validate_existing(
        args.output,
        input_sha256=args.input_sha256,
        model=args.model,
        expected_returned_model=args.expected_returned_model,
        allow_model_drift=args.allow_model_drift,
        input_ids=input_ids,
        item_hashes=item_hashes,
    )

    def pending_rows():
        yielded = 0
        for row in iter_jsonl(args.input):
            if row["sample_id"] in done:
                continue
            if args.limit and yielded >= args.limit:
                break
            yielded += 1
            yield row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    completed = 0
    run_failed = False
    iterator = iter(pending_rows())
    with args.output.open("a", encoding="utf-8", newline="\n", buffering=1) as handle:
        # A single synchronous request prevents a bad credential, endpoint, or
        # exhausted account from being multiplied by the worker pool.
        try:
            first = next(iterator)
        except StopIteration:
            first = None
        if first is not None:
            result = call(first, args, api_key, stop)
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed = 1
            if result.get("status") != "ok":
                run_failed = True
                print(f"preflight failed: {result.get('error', 'unknown error')}", file=sys.stderr, flush=True)
            else:
                with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    inflight: dict[futures.Future, str] = {}

                    def submit_one() -> bool:
                        if stop.is_set():
                            return False
                        try:
                            row = next(iterator)
                        except StopIteration:
                            return False
                        inflight[pool.submit(call, row, args, api_key, stop)] = row["sample_id"]
                        return True

                    for _ in range(args.workers):
                        if not submit_one():
                            break
                    while inflight:
                        finished, _ = futures.wait(inflight, return_when=futures.FIRST_COMPLETED)
                        for future in finished:
                            inflight.pop(future)
                            result = future.result()
                            completed += 1
                            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                            if result.get("status") != "ok":
                                run_failed = True
                            if completed % 50 == 0:
                                print(
                                    f"completed={completed} status={result['status']} sample={result['sample_id']}",
                                    flush=True,
                                )
                            if not stop.is_set():
                                submit_one()
                    if stop.is_set():
                        run_failed = True
        print(f"completed={completed}", flush=True)
    statuses = Counter()
    returned_models = Counter()
    rows = 0
    for row in iter_jsonl(args.output):
        rows += 1
        statuses[str(row.get("status", "missing"))] += 1
        if row.get("status") == "ok":
            returned_models[str(row.get("returned_model") or "missing")] += 1
    manifest = {
        "schema_version": "jev-api-run-manifest-v1",
        "input": args.input.name,
        "input_sha256": args.input_sha256,
        "requested_model": args.model,
        "expected_returned_model": args.expected_returned_model,
        "allow_model_drift": args.allow_model_drift,
        "observed_returned_models": dict(sorted(returned_models.items())),
        "workers": args.workers,
        "timeout_seconds": args.timeout,
        "retries": args.retries,
        "output": args.output.name,
        "output_sha256": sha256_file(args.output),
        "rows_in_append_log": rows,
        "statuses_in_append_log": dict(sorted(statuses.items())),
        "note": "Endpoint and credential are intentionally omitted. Canonicalize before comparing reruns.",
    }
    args.output.with_suffix(args.output.suffix + ".run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    if run_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
