#!/usr/bin/env python3
"""Download and hash-check pinned KEV checkpoints and Qwen bases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "models"
MANIFEST_PATH = ROOT / "manifests" / "model_files_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_files(manifest: dict, directory: str) -> list[dict]:
    prefix = f"models/{directory}/"
    return [entry for entry in manifest["files"] if entry["path"].startswith(prefix)]


def verify_directory(path: Path, entries: list[dict], full_hash: bool = True) -> tuple[bool, str]:
    expected_paths = {
        Path(entry["path"]).relative_to(Path("models") / path.name).as_posix()
        for entry in entries
    }
    actual_paths = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(path).parts
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        return False, f"file-set mismatch missing={missing[:3]} unexpected={unexpected[:3]}"
    for entry in entries:
        relative = Path(entry["path"]).relative_to(Path("models") / path.name)
        item = path / relative
        if not item.is_file():
            return False, f"missing {relative}"
        if item.stat().st_size != entry["bytes"]:
            return False, f"size mismatch {relative}"
        if full_hash and sha256(item) != entry["sha256"]:
            return False, f"sha256 mismatch {relative}"
    return True, "ok"


def hf_download(model: dict, target: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model["repository"],
        revision=model["revision"],
        local_dir=target,
    )


def download_one(model: dict, entries: list[dict]) -> None:
    target = MODELS_ROOT / model["directory"]
    if target.exists():
        valid, reason = verify_directory(target, entries)
        if valid:
            print(f"verified existing {model['directory']}", flush=True)
            return
        print(f"existing {model['directory']} is incomplete ({reason}); redownloading", flush=True)
        shutil.rmtree(target)

    partial = MODELS_ROOT / f".{model['directory']}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        print(f"downloading pinned revision: {model['repository']}@{model['revision']}", flush=True)
        hf_download(model, partial)
        valid, reason = verify_directory(partial, entries)
        if not valid:
            raise RuntimeError(reason)
        partial.replace(target)
        print(f"downloaded and SHA-256 verified: {model['directory']}", flush=True)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="*", default=["0.8b", "4b"], choices=["0.8b", "4b"])
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    wanted = set(args.sizes or ["0.8b", "4b"])
    directories = {f"kev-{size}" for size in wanted} | {f"qwen3.5-{size}-base" for size in wanted}
    models = [model for model in manifest["models"] if model["directory"] in directories]
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    for model in models:
        download_one(model, expected_files(manifest, model["directory"]))


if __name__ == "__main__":
    main()
