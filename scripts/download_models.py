#!/usr/bin/env python3
"""Download pinned KEV checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.integrity import tree_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="kev-0.8b,kev-4b",
                        help=("Comma-separated KEV adapters, including kev-0.8b-ba-lora and "
                              "kev-4b-ba-lora; exact defaults exclude the unpinned 9B adapter"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "models",
                        help="Self-contained Hugging Face home used by run_kev_local.py")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT") or None)
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="Required for KEV-9B until its paper-run revision is identified")
    args = parser.parse_args()
    args.output = args.output.resolve()
    os.environ["HF_HOME"] = str(args.output)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(args.output / "hub")
    from huggingface_hub import HfApi, snapshot_download

    models = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))["models"]
    selected = [x.strip() for x in args.models.split(",") if x.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "download_manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_by_key = {
            (entry["model"], entry["role"]): entry for entry in prior.get("models", [])
        }
    else:
        manifest_by_key = {}
    api = HfApi(endpoint=args.hf_endpoint)
    for name in selected:
        if name not in models or "repo_id" not in models[name]:
            raise SystemExit(f"Unknown downloadable KEV model: {name}")
        spec = models[name]
        revision = spec.get("revision")
        if not revision and not args.allow_unpinned:
            raise SystemExit(f"{name} has no recorded paper-run revision; rerun with --allow-unpinned or pin configs/models.json")
        artifacts = [
            ("adapter", spec["repo_id"], revision),
            ("base", spec["base_repo_id"], spec["base_revision"]),
        ]
        for role, repo_id, requested_revision in artifacts:
            info = api.model_info(repo_id, revision=requested_revision)
            resolved_revision = info.sha
            path = Path(snapshot_download(
                repo_id=repo_id,
                revision=resolved_revision,
                cache_dir=args.output / "hub",
                endpoint=args.hf_endpoint,
            ))
            tree = tree_manifest(path)
            try:
                portable_path = path.relative_to(args.output).as_posix()
            except ValueError:
                portable_path = f"hub/{repo_id.replace('/', '--')}/snapshots/{resolved_revision}"
            manifest_by_key[(name, role)] = {
                "model": name,
                "role": role,
                "repo_id": repo_id,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
                "cache_path": portable_path,
                "tree_sha256": tree["tree_sha256"],
                "file_count": tree["file_count"],
                "files": tree["files"],
                "exact_paper_revision": bool(revision) if role == "adapter" else True,
            }
            print(f"Downloaded {name} {role}: {repo_id}@{resolved_revision} ({tree['file_count']} files)")
    manifest_path.write_text(
        json.dumps({
            "schema_version": "jev-model-download-v2",
            "hash_algorithm": "sha256",
            "models": [manifest_by_key[key] for key in sorted(manifest_by_key)],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
