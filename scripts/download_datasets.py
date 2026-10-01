#!/usr/bin/env python3
"""Download all pinned upstream dataset snapshots without sampling them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_bias.integrity import tree_manifest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        append = bool(offset and response.status_code == 206)
        total = int(response.headers.get("content-length", 0)) + (offset if append else 0)
        with part.open("ab" if append else "wb") as handle, tqdm(
            total=total or None, initial=offset if append else 0, unit="B", unit_scale=True, desc=target.name
        ) as bar:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))
    os.replace(part, target)


def git_snapshot(url: str, revision: str, target: Path) -> None:
    if not target.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--tags", "origin"], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="pinned",
                        help="pinned (default), all, or comma-separated source IDs")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "datasets")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT") or None)
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="Allow sources whose exact original revision was not recorded; resolved state is still hashed")
    args = parser.parse_args()

    config = json.loads((ROOT / "configs" / "sources.json").read_text(encoding="utf-8"))["sources"]
    if args.sources == "all":
        selected = sorted(config)
    elif args.sources == "pinned":
        selected = sorted(name for name, spec in config.items()
                          if spec.get("pin_status") != "unresolved-original-commit")
    else:
        selected = [x.strip() for x in args.sources.split(",") if x.strip()]
    unknown = sorted(set(selected) - set(config))
    if unknown:
        raise SystemExit(f"Unknown sources: {unknown}")
    unresolved = [name for name in selected if config[name].get("pin_status") == "unresolved-original-commit"]
    if unresolved and not args.allow_unpinned:
        raise SystemExit(
            "Exact original revisions were not recorded for: " + ", ".join(unresolved) +
            ". Released metrics can be recomputed without prompt text; pass --allow-unpinned to download and hash "
            "the currently resolved snapshots. No downloads were started."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, name in enumerate(selected, 1):
        spec = config[name]
        target = args.output / name
        print(f"[{index}/{len(selected)}] {name} ({spec['kind']})", flush=True)
        if spec["kind"] == "huggingface":
            path = snapshot_download(
                repo_id=spec["repo_id"], repo_type="dataset", revision=spec.get("revision"),
                local_dir=target, endpoint=args.hf_endpoint,
            )
            from huggingface_hub import HfApi
            resolved = HfApi(endpoint=args.hf_endpoint).dataset_info(
                spec["repo_id"], revision=spec.get("revision")
            ).sha
            tree = tree_manifest(path, exclude_names=("download_manifest.json",))
            records.append({"source": name, "path": name, "repo_id": spec["repo_id"],
                            "requested_revision": spec.get("revision"), "resolved_revision": resolved,
                            "tree_sha256": tree["tree_sha256"], "file_count": tree["file_count"]})
        elif spec["kind"] == "direct":
            target.mkdir(parents=True, exist_ok=True)
            path = target / spec["filename"]
            if not path.exists():
                download(spec["url"], path)
            expected = spec.get("sha256")
            observed = sha256(path)
            if expected and observed.lower() != expected.lower():
                raise RuntimeError(f"SHA-256 mismatch for {name}: {observed} != {expected}")
            records.append({"source": name, "path": f"{name}/{path.name}", "revision": spec.get("revision"),
                            "bytes": path.stat().st_size, "sha256": observed})
        elif spec["kind"] == "git":
            git_snapshot(spec["url"], spec["revision"], target)
            head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
            tree = tree_manifest(target, exclude_names=("download_manifest.json",))
            records.append({"source": name, "path": name, "requested_revision": spec["revision"],
                            "resolved_revision": head, "tree_sha256": tree["tree_sha256"],
                            "file_count": tree["file_count"],
                            "exact_original_revision": spec.get("pin_status") != "unresolved-original-commit"})
        elif spec["kind"] == "convokit":
            from convokit import download as convokit_download
            path = convokit_download(spec["corpus"], data_dir=str(target))
            path = Path(path)
            tree = tree_manifest(path)
            records.append({"source": name, "path": path.name, "revision": "convokit-managed-unpinned",
                            "tree_sha256": tree["tree_sha256"], "file_count": tree["file_count"],
                            "exact_original_revision": False})
        else:
            raise RuntimeError(f"Unsupported source kind for {name}: {spec['kind']}")
    manifest = {"schema_version": "upstream-download-v2", "hash_algorithm": "sha256", "sources": records}
    (args.output / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Downloaded {len(records)} source snapshots to {args.output}")


if __name__ == "__main__":
    main()
