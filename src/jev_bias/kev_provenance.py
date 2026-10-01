from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from .integrity import tree_manifest


def verify_installed_kev(expected_revision: str, *, allow_unverified: bool = False) -> dict:
    """Verify PEP 610 VCS provenance and inventory the imported KEV package."""
    import kev

    distribution = importlib.metadata.distribution("kev")
    raw = distribution.read_text("direct_url.json")
    direct_url = json.loads(raw) if raw else {}
    vcs_info = direct_url.get("vcs_info") or {}
    commit = vcs_info.get("commit_id")
    if commit != expected_revision and not allow_unverified:
        raise RuntimeError(
            "Installed KEV code is not verified at the configured evaluation commit: "
            f"observed={commit!r}, expected={expected_revision!r}. Install requirements-kev.txt "
            "or explicitly use --allow-unverified-kev-code for a non-paper run."
        )
    package_root = Path(kev.__file__).resolve().parent
    tree = tree_manifest(package_root)
    return {
        "distribution_version": distribution.version,
        "repository_url": direct_url.get("url"),
        "vcs_commit_id": commit,
        "expected_revision": expected_revision,
        "revision_verified": commit == expected_revision,
        "package_tree_sha256": tree["tree_sha256"],
        "package_file_count": tree["file_count"],
    }
