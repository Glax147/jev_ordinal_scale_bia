#!/usr/bin/env python3
"""Build deterministic integrity manifests for the private full bundle.

The public release keeps the resulting audit manifests but intentionally omits
the upstream-text ZIPs. Public users should run ``verify_artifacts.py`` rather
than attempting to regenerate the private input manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.integrity import scan_gzip_jsonl, scan_jsonl_binary
from jev_bias.io import selection_digest, sha256_file


INPUT_LAYOUT = (
    (
        "data/frozen_inputs/ordinal_bias_20datasets_100000.zip",
        (
            ("ordinal_bias_10x5000.jsonl", "main40"),
            ("ordinal_bias_batch2_10x5000.jsonl", "main40"),
        ),
    ),
    (
        "data/frozen_inputs/ordinal_bias_batch3_20x5000.zip",
        (("ordinal_bias_batch3_20x5000.jsonl", "main40"),),
    ),
    (
        "data/frozen_inputs/core9_two_random_orders_90000.zip",
        (("core9_two_random_orders_90000.jsonl", "position9"),),
    ),
    (
        "data/frozen_inputs/controlled_k_curve_127400.zip",
        (("controlled_k_curve_127400.jsonl", "kcurve4"),),
    ),
)


def validate_archive_members(names: list[str], expected: set[str], archive_name: str) -> None:
    if len(names) != len(set(names)):
        raise RuntimeError(f"{archive_name} contains duplicate ZIP member names")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise RuntimeError(f"{archive_name} contains an unsafe ZIP member path: {name!r}")
    observed = set(names)
    if observed != expected:
        raise RuntimeError(
            f"{archive_name} member set mismatch; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _experiment_for_result(name: str) -> str:
    if "__main-40datasets__" in name:
        return "main40"
    if "__position-ablation-9datasets__" in name:
        return "position9"
    if "__choice-count-ablation-4datasets-k2to14__" in name:
        return "kcurve4"
    raise ValueError(f"Unrecognized result filename: {name}")


def build_input_manifest() -> dict:
    archives = []
    experiment_members: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for relative, members in INPUT_LAYOUT:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = {
            "file": relative,
            "archive_bytes": path.stat().st_size,
            "archive_sha256": sha256_file(path),
            "members": [],
        }
        with zipfile.ZipFile(path) as archive:
            expected_members = {member for member, _ in members}
            validate_archive_members(archive.namelist(), expected_members, relative)
            available = set(archive.namelist())
            for member, experiment in members:
                if member not in available:
                    raise RuntimeError(f"{relative} is missing {member}")
                with archive.open(member) as handle:
                    scan = scan_jsonl_binary(handle)
                if scan["duplicate_ids"]:
                    raise RuntimeError(f"Duplicate IDs in {relative}:{member}: {scan['duplicate_ids']}")
                scan.update({"name": member, "experiment": experiment})
                entry["members"].append(scan)
                experiment_members[experiment].append((relative, member))
        archives.append(entry)

    # Hash the exact byte sequence produced when each experiment is extracted.
    experiments = {}
    for experiment in ("main40", "position9", "kcurve4"):
        content = hashlib.sha256()
        ordered = hashlib.sha256()
        ids: list[str] = []
        rows = 0
        byte_count = 0
        member_refs = []
        for relative, member in experiment_members[experiment]:
            with zipfile.ZipFile(ROOT / relative) as archive, archive.open(member) as handle:
                for line_number, raw in enumerate(handle, 1):
                    content.update(raw)
                    byte_count += len(raw)
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    sample_id = str(row.get("sample_id") or row.get("row_uid"))
                    if sample_id == "None":
                        raise ValueError(f"Missing sample_id in {relative}:{member}:{line_number}")
                    rows += 1
                    ids.append(sample_id)
                    ordered.update(sample_id.encode("utf-8") + b"\n")
            member_refs.append({"archive": relative, "member": member})
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"Duplicate IDs across {experiment} members")
        experiments[experiment] = {
            "output": f"artifacts/frozen_inputs/{experiment}.jsonl",
            "members": member_refs,
            "rows": rows,
            "bytes": byte_count,
            "content_sha256": content.hexdigest(),
            "ordered_ids_sha256": ordered.hexdigest(),
            "selection_sha256": selection_digest(ids),
        }
    return {
        "schema_version": "jev-frozen-input-manifest-v1",
        "hash_algorithm": "sha256",
        "archives": archives,
        "experiments": experiments,
    }


def build_result_manifest() -> dict:
    files = []
    released = ROOT / "results" / "released"
    for path in sorted(released.glob("*.jsonl.gz")):
        scan = scan_gzip_jsonl(path, predictions=True)
        if scan["duplicate_ids"]:
            raise RuntimeError(f"Duplicate IDs in {path.name}: {scan['duplicate_ids']}")
        files.append({
            "file": path.relative_to(ROOT).as_posix(),
            "model": path.name.split("__", 1)[0],
            "experiment": _experiment_for_result(path.name),
            **scan,
        })
    if len(files) != 12:
        raise RuntimeError(f"Expected 12 released result files, found {len(files)}")
    return {
        "schema_version": "jev-released-results-manifest-v2",
        "hash_algorithm": "sha256",
        "note": "archive_sha256 checks the .gz object; content_sha256 checks exact decompressed JSONL; prediction_digest is order-independent, excludes timing/usage/timestamps, normalizes historical schemas, and quantizes probabilities to the shared four-decimal recorded precision; decision_digest excludes probability values.",
        "files": files,
    }


def write_manifest(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {path.relative_to(ROOT)}")


def main() -> None:
    if (ROOT / "PUBLIC_RELEASE.json").exists():
        raise SystemExit(
            "Private frozen-input ZIPs are intentionally absent from this public release; "
            "use scripts/verify_artifacts.py, or rebuild prompts with scripts/rebuild_inputs.py."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    args = parser.parse_args()
    if args.inputs_only and args.results_only:
        parser.error("--inputs-only and --results-only are mutually exclusive")
    if not args.results_only:
        write_manifest(ROOT / "data" / "frozen_inputs" / "MANIFEST.json", build_input_manifest())
    if not args.inputs_only:
        write_manifest(ROOT / "results" / "MANIFEST.json", build_result_manifest())


if __name__ == "__main__":
    main()
