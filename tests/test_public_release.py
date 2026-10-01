import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_release_omits_declared_archives():
    release = json.loads((ROOT / "PUBLIC_RELEASE.json").read_text(encoding="utf-8"))
    omitted = [ROOT / item for item in release["omitted_assets"]]
    assert omitted
    assert all(path.suffix == ".zip" for path in omitted)
    assert all(not path.exists() for path in omitted)
    assert not list((ROOT / "data" / "frozen_inputs").glob("*.zip"))


def test_private_manifest_still_audits_every_omitted_archive():
    release = json.loads((ROOT / "PUBLIC_RELEASE.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / "data" / "frozen_inputs" / "MANIFEST.json").read_text(encoding="utf-8")
    )
    manifest_paths = {entry["file"] for entry in manifest["archives"]}
    assert set(release["omitted_assets"]) == manifest_paths
