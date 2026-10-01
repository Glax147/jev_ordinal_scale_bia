from __future__ import annotations

import json
from pathlib import Path

from .integrity import tree_manifest


def load_and_verify_downloaded_model(model_root: str | Path, model: str) -> list[dict]:
    """Verify the downloaded adapter/base trees and return portable provenance.

    The download manifest is deliberately stored beside the model cache rather
    than committed: model weights are large runtime artifacts.  A local run is
    refused if either tree differs from the SHA-256 inventory produced by the
    downloader.
    """
    model_root = Path(model_root).resolve()
    manifest_path = model_root / "download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run scripts/download_models.py before inference"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [entry for entry in manifest.get("models", []) if entry.get("model") == model]
    by_role = {entry.get("role"): entry for entry in entries}
    if set(by_role) != {"adapter", "base"}:
        raise RuntimeError(f"Download manifest must contain adapter and base entries for {model}")

    verified = []
    for role in ("adapter", "base"):
        entry = by_role[role]
        path = (model_root / str(entry["cache_path"])).resolve()
        try:
            path.relative_to(model_root)
        except ValueError as exc:
            raise RuntimeError(f"Model cache path escapes model root: {path}") from exc
        if not path.is_dir():
            raise FileNotFoundError(f"Downloaded {model} {role} tree is missing: {path}")
        observed = tree_manifest(path)
        if (
            observed["tree_sha256"] != entry.get("tree_sha256")
            or observed["file_count"] != entry.get("file_count")
            or observed["files"] != entry.get("files")
        ):
            raise RuntimeError(
                f"Downloaded {model} {role} failed SHA-256 verification; "
                "delete/redownload that artifact instead of continuing"
            )
        verified.append({
            "role": role,
            "repo_id": entry["repo_id"],
            "requested_revision": entry.get("requested_revision"),
            "resolved_revision": entry["resolved_revision"],
            "cache_path": entry["cache_path"],
            "tree_sha256": entry["tree_sha256"],
            "file_count": entry["file_count"],
            "exact_paper_revision": bool(entry.get("exact_paper_revision")),
        })
    return verified
