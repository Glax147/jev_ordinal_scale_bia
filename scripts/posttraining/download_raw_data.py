#!/usr/bin/env python3
"""Download the pinned raw training sources used by the KEV bias study.

The script intentionally downloads source training data only.  It does not
download model weights.  Hugging Face repositories are pinned to the same
revisions used to build the frozen evaluation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2] / "posttraining"
RAW = ROOT / "data" / "raw"

HF_SOURCES = {
    "ibm_argument_quality": {
        "repo": "ibm-research/argument_quality_ranking_30k",
        "revision": "590726b3765b1b90c5e53a17e3b1f77d92d3aa8a",
        "files": ["train.csv", "dev.csv", "test.csv"],
    },
    "helpsteer2_correctness": {
        "repo": "nvidia/HelpSteer2",
        "revision": "990b2711a36180dd19d9c94b8627844866f8982a",
        "files": ["train.jsonl.gz"],
    },
    "civil_comments_toxicity": {
        "repo": "google/civil_comments",
        "revision": "f2970eb3a55777454c94069077cc8d9b5866312d",
        "files": [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
        ],
    },
    "wine_quality": {
        "repo": "spawn99/wine-reviews",
        "revision": "e6b10f4db3091a6fed8c5b294c0cc885e7f6e99d",
        "files": ["data/train-00000-of-00001.parquet"],
    },
    "word_concreteness": {
        "repo": "StephanAkkerman/concreteness-ratings",
        "revision": "10f460a4d535800d89c579db7cd2664dbaf6b1d6",
        "files": ["concreteness_ratings.csv"],
    },
    "wmt20_translation_quality": {
        "repo": "wmt/wmt20_mlqe_task1",
        "revision": "0783ed2bd75f44835df4ea664f9ccb85812c8563",
        "files": [
            "en-de/train-00000-of-00001.parquet",
            "en-zh/train-00000-of-00001.parquet",
            "et-en/train-00000-of-00001.parquet",
            "ne-en/train-00000-of-00001.parquet",
            "ro-en/train-00000-of-00001.parquet",
            "si-en/train-00000-of-00001.parquet",
        ],
    },
    "realtoxicity_continuation": {
        "repo": "allenai/real-toxicity-prompts",
        "revision": "f21629712ffd6a3d13a54fd2807ccd521c55ef74",
        "files": ["prompts.jsonl"],
    },
    "measuring_hate_speech": {
        "repo": "ucberkeley-dlab/measuring-hate-speech",
        "revision": "5468f6e118396646b02a2f691e771f6b6d9502ea",
        "files": ["data/train-00000-of-00001.parquet"],
    },
}



def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_hf(repo: str, revision: str, filename: str, dest: Path) -> None:
    cached = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
def main() -> None:
    argparse.ArgumentParser().parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    for dataset, spec in HF_SOURCES.items():
        for filename in spec["files"]:
            copy_hf(spec["repo"], spec["revision"], filename, RAW / dataset / filename)

    files = []
    for dataset, spec in sorted(HF_SOURCES.items()):
        for filename in sorted(spec["files"]):
            path = RAW / dataset / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append({
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "schema_version": "kev-bias-raw-sources-v1",
        "hf_sources": HF_SOURCES,
        "files": files,
    }
    expected_path = ROOT / "manifests" / "raw_source_manifest.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if manifest != expected:
        raise RuntimeError("downloaded raw source files do not match the bundled bytes/SHA-256 manifest")
    runtime_manifest = RAW / "raw_source_manifest.json"
    temporary = runtime_manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(runtime_manifest)
    print(json.dumps({"status": "ok", "files": len(files), "bytes": sum(x["bytes"] for x in files)}, indent=2))


if __name__ == "__main__":
    main()
