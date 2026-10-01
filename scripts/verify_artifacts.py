#!/usr/bin/env python3
"""Verify the full local evaluation bundle against committed SHA-256 manifests."""

from __future__ import annotations

import argparse
import itertools
import json
import runpy
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from jev_bias.io import canonical_json_sha256, iter_jsonl, sha256_file
from jev_bias.integrity import scan_gzip_jsonl
from scripts.build_manifests import build_input_manifest, build_result_manifest
from scripts.build_bundle_manifest import build_bundle_manifest


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_indexes() -> dict[str, dict]:
    manifest = load(ROOT / "data" / "indexes" / "MANIFEST.json")
    ids_by_experiment = {}
    for experiment in ("main40", "position9", "kcurve4"):
        spec = manifest[experiment]
        path = ROOT / spec["file"]
        observed_hash = sha256_file(path)
        if observed_hash != spec["sha256"]:
            raise RuntimeError(f"Index archive hash mismatch for {experiment}: {observed_hash} != {spec['sha256']}")
        scan = scan_gzip_jsonl(path)
        if scan["rows"] != spec["rows"] or scan["rows"] != scan["unique_ids"]:
            raise RuntimeError(f"Index row/uniqueness failure for {experiment}")
        observed_selection = scan["selection_sha256"]
        if observed_selection != spec["selection_sha256"]:
            raise RuntimeError(
                f"Index selection hash mismatch for {experiment}: {observed_selection} != {spec['selection_sha256']}"
            )
        for field, observed_field in (
            ("content_sha256", scan["content_sha256"]),
            ("ordered_ids_sha256", scan["ordered_ids_sha256"]),
        ):
            if field in spec and spec[field] != observed_field:
                raise RuntimeError(f"Index {field} mismatch for {experiment}")
        ids_by_experiment[experiment] = {
            "rows": scan["rows"],
            "selection_sha256": observed_selection,
            "file": spec["file"],
        }
        print(f"OK index {experiment}: rows={scan['rows']:,} selection={observed_selection}")
    return ids_by_experiment


def verify_input_rows(expected_inputs: dict, index_specs: dict[str, dict]) -> None:
    """Cross-check every exact prompt against the independently stored row hash."""
    for experiment, spec in expected_inputs["experiments"].items():
        expected = {
            str(row["sample_id"]): str(row["row_sha256"])
            for row in iter_jsonl(ROOT / index_specs[experiment]["file"])
        }
        observed_ids: set[str] = set()
        for member_spec in spec["members"]:
            archive_path = ROOT / member_spec["archive"]
            with zipfile.ZipFile(archive_path) as archive, archive.open(member_spec["member"]) as handle:
                for line_number, raw in enumerate(handle, 1):
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    sample_id = str(row.get("sample_id") or row.get("row_uid"))
                    row_hash = canonical_json_sha256(row)
                    if expected.get(sample_id) != row_hash:
                        raise RuntimeError(
                            f"Frozen input/index row hash mismatch for {experiment}:{sample_id} "
                            f"at {member_spec['member']}:{line_number}"
                        )
                    observed_ids.add(sample_id)
        if observed_ids != set(expected):
            raise RuntimeError(f"Frozen input/index row set mismatch for {experiment}")
        print(f"OK row hashes {experiment}: {len(observed_ids):,}")


def verify_rebuilt_rows(index_specs: dict[str, dict]) -> None:
    """Validate optional locally rebuilt prompts without requiring private archives."""
    rebuilt_root = ROOT / "artifacts" / "frozen_inputs"
    paths = {name: rebuilt_root / f"{name}.jsonl" for name in index_specs}
    present = {name for name, path in paths.items() if path.is_file()}
    if not present:
        print("INFO rebuilt prompts are absent; metadata/results verification only")
        return
    if present != set(paths):
        raise RuntimeError(f"Partial rebuilt-input set: present={sorted(present)}")
    sentinel = object()
    for experiment, path in paths.items():
        index_path = ROOT / index_specs[experiment]["file"]
        rows = 0
        for row, meta in itertools.zip_longest(
            iter_jsonl(path), iter_jsonl(index_path), fillvalue=sentinel
        ):
            if row is sentinel or meta is sentinel:
                raise RuntimeError(f"Rebuilt/index length mismatch for {experiment}")
            rows += 1
            if str(row.get("sample_id")) != str(meta.get("sample_id")):
                raise RuntimeError(f"Rebuilt/index order mismatch for {experiment} row {rows}")
            if canonical_json_sha256(row) != str(meta.get("row_sha256")):
                raise RuntimeError(f"Rebuilt row hash mismatch for {experiment}:{row.get('sample_id')}")
        if rows != index_specs[experiment]["rows"]:
            raise RuntimeError(f"Rebuilt row count mismatch for {experiment}: {rows}")
        print(f"OK rebuilt row hashes {experiment}: {rows:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-result-id-crosscheck", action="store_true",
                        help="Skip comparing every prediction ID set with its frozen index")
    args = parser.parse_args()

    public_mode = (ROOT / "PUBLIC_RELEASE.json").is_file()
    expected_inputs = load(ROOT / "data" / "frozen_inputs" / "MANIFEST.json")
    if public_mode:
        unexpected_archives = list((ROOT / "data" / "frozen_inputs").glob("*.zip"))
        if unexpected_archives:
            raise RuntimeError("Public release must not contain frozen dataset archives")
        print("OK public release: private frozen-text archives are intentionally absent")
    else:
        observed_inputs = build_input_manifest()
        if observed_inputs != expected_inputs:
            raise RuntimeError("Frozen-input manifest mismatch; run scripts/build_manifests.py only after auditing the changed assets")
        print(f"OK frozen inputs: {len(observed_inputs['archives'])} archives")

    index_specs = verify_indexes()
    for experiment, spec in expected_inputs["experiments"].items():
        if spec["selection_sha256"] != index_specs[experiment]["selection_sha256"]:
            raise RuntimeError(f"Frozen-input/index selection mismatch for {experiment}")
    if public_mode:
        verify_rebuilt_rows(index_specs)
    else:
        verify_input_rows(expected_inputs, index_specs)

    expected_results = load(ROOT / "results" / "MANIFEST.json")
    observed_results = build_result_manifest()
    if observed_results != expected_results:
        raise RuntimeError("Released-result manifest mismatch; a result file is missing, changed or corrupted")
    print(f"OK released predictions: {len(observed_results['files'])} files")

    posttraining_builder = runpy.run_path(str(ROOT / "scripts/posttraining/build_manifest.py"))["build"]
    observed_posttraining = posttraining_builder()
    expected_posttraining = load(ROOT / "posttraining/MANIFEST.json")
    if observed_posttraining != expected_posttraining:
        raise RuntimeError("Post-training manifest mismatch; optional BA-LoRA assets changed or are corrupted")
    print(f"OK post-training release: {len(observed_posttraining['released_results'])} result files")

    if not args.skip_result_id_crosscheck:
        for spec in observed_results["files"]:
            expected = index_specs[spec["experiment"]]
            if spec["selection_sha256"] != expected["selection_sha256"] or spec["unique_ids"] != expected["rows"]:
                raise RuntimeError(f"Result/index ID-set digest mismatch in {Path(spec['file']).name}")
            print(f"OK IDs {Path(spec['file']).name}: {spec['unique_ids']:,}")

    expected_bundle = load(ROOT / "BUNDLE_MANIFEST.json")
    observed_bundle = build_bundle_manifest()
    if observed_bundle != expected_bundle:
        raise RuntimeError(
            "Code/config/environment bundle hash mismatch. If changes were intentional and reviewed, "
            "run scripts/build_bundle_manifest.py and verify again."
        )
    print(f"OK bundle tree: {observed_bundle['code_config_environment_tree_sha256']}")

    if public_mode:
        print("Public metadata, indexes and released predictions passed SHA-256 and selection checks.")
    else:
        print("All frozen inputs, indexes and released predictions passed SHA-256 and selection checks.")


if __name__ == "__main__":
    main()
