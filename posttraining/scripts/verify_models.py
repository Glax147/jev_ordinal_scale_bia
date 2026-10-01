#!/usr/bin/env python3
"""Build or verify the manifest for bundled KEV and Qwen model files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = [
    {"directory": "kev-0.8b", "repository": "jaredpalmer/kev-0.8b", "revision": "9a45d25eb2ab761841196625383fa1dff0e56c1e"},
    {"directory": "kev-4b", "repository": "jaredpalmer/kev-4b", "revision": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"},
    {"directory": "qwen3.5-0.8b-base", "repository": "Qwen/Qwen3.5-0.8B-Base", "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"},
    {"directory": "qwen3.5-4b-base", "repository": "Qwen/Qwen3.5-4B-Base", "revision": "1001bb4d826a52d1f399e183466143f4da7b741b"},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(path: Path) -> dict:
    files = []
    for model in MODEL_SPECS:
        directory = ROOT / "models" / model["directory"]
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for item in sorted(directory.rglob("*")):
            if item.is_file() and ".cache" not in item.parts:
                files.append({
                    "path": item.relative_to(ROOT).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": sha256(item),
                })
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": MODEL_SPECS,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-hash", action="store_true", help="also hash every weight shard (slower)")
    parser.add_argument("--write-manifest", action="store_true", help="hash bundled files and create MODEL_MANIFEST.json")
    args = parser.parse_args()
    manifest_path = ROOT / "manifests" / "model_files_manifest.json"
    manifest = build_manifest(manifest_path) if args.write_manifest else json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = {entry["path"] for entry in manifest["files"]}
    model_root = ROOT / "models"
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in model_root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(model_root).parts
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise RuntimeError(f"model file-set mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    checked_bytes = 0
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"size mismatch: {path}: {size} != {entry['bytes']}")
        checked_bytes += size
        if args.full_hash and sha256(path) != entry["sha256"]:
            raise ValueError(f"sha256 mismatch: {path}")
    print(json.dumps({
        "status": "ok",
        "models": manifest["models"],
        "files": len(manifest["files"]),
        "bytes": checked_bytes,
        "full_hash": args.full_hash,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
