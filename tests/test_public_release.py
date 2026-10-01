import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scripts_and_requirements_are_centralized():
    assert not list(ROOT.glob("*.sh"))
    assert not list(ROOT.glob("*.ps1"))
    assert not list(ROOT.glob("requirements*.txt"))
    assert not (ROOT / "posttraining" / "scripts").exists()
    assert {path.name for path in (ROOT / "requirements").glob("*.txt")} == {
        "base.txt", "data.txt", "kev.txt", "posttraining.txt"
    }
    assert (ROOT / "scripts" / "setup.sh").is_file()
    assert (ROOT / "scripts" / "setup.ps1").is_file()
    assert (ROOT / "scripts" / "posttraining" / "run.sh").is_file()


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
