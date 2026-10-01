import pytest

from scripts.build_manifests import validate_archive_members


def test_archive_member_set_must_be_exact():
    validate_archive_members(["data.jsonl"], {"data.jsonl"}, "x.zip")
    with pytest.raises(RuntimeError, match="unexpected"):
        validate_archive_members(["data.jsonl", "local_manifest.json"], {"data.jsonl"}, "x.zip")


@pytest.mark.parametrize("name", ["../data.jsonl", "/data.jsonl", "C:/data.jsonl", "a\\data.jsonl"])
def test_archive_member_paths_must_be_safe(name):
    with pytest.raises(RuntimeError, match="unsafe"):
        validate_archive_members([name], {name}, "x.zip")


def test_duplicate_archive_member_names_are_rejected():
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_archive_members(["data.jsonl", "data.jsonl"], {"data.jsonl"}, "x.zip")
