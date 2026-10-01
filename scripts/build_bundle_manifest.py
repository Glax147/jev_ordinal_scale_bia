#!/usr/bin/env python3
"""Build the top-level code/config/environment integrity manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.io import sha256_file


TRACKED_DIRS = (
    "configs", "requirements", "scripts", "src", "tests",
    "posttraining/configs", "posttraining/manifests", "posttraining/patches",
)
ROOT_FILES = (
    ".gitattributes", ".gitignore", ".python-version",
    "LICENSE", "pyproject.toml", "README.md", "PUBLIC_RELEASE.json",
)
ASSET_MANIFESTS = (
    "data/frozen_inputs/MANIFEST.json",
    "data/indexes/MANIFEST.json",
    "results/MANIFEST.json",
    "posttraining/MANIFEST.json",
)
GENERATED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def is_generated(path: Path) -> bool:
    relative_parts = path.relative_to(ROOT).parts
    return any(
        part in GENERATED_DIRS or part.endswith(".egg-info") or part.endswith(".dist-info")
        for part in relative_parts
    )


def build_bundle_manifest() -> dict:
    candidates = [ROOT / name for name in ROOT_FILES]
    for directory in TRACKED_DIRS:
        candidates.extend((ROOT / directory).rglob("*"))
    files = []
    for path in sorted({p for p in candidates if p.is_file() and not is_generated(p)}):
        relative = path.relative_to(ROOT).as_posix()
        files.append({"file": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    digest = hashlib.sha256()
    for item in files:
        digest.update(f"{item['sha256']}  {item['bytes']}  {item['file']}\n".encode("utf-8"))
    assets = []
    for relative in ASSET_MANIFESTS:
        path = ROOT / relative
        assets.append({"file": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema_version": "jev-bias-evaluation-bundle-v1",
        "python": (ROOT / ".python-version").read_text(encoding="utf-8").strip(),
        "hash_algorithm": "sha256",
        "code_config_environment_tree_sha256": digest.hexdigest(),
        "files": files,
        "asset_manifests": assets,
    }


def main() -> None:
    manifest = build_bundle_manifest()
    path = ROOT / "BUNDLE_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"Wrote {path}")
    print(f"Code/config/environment tree SHA-256: {manifest['code_config_environment_tree_sha256']}")


if __name__ == "__main__":
    main()
