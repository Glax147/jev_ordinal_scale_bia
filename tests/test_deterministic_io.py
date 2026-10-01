from jev_bias.io import sha256_file, write_jsonl


def test_gzip_output_is_deterministic(tmp_path):
    path = tmp_path / "rows.jsonl.gz"
    second_path = tmp_path / "same-content-different-name.jsonl.gz"
    rows = [{"sample_id": "b", "value": 2}, {"sample_id": "a", "value": 1}]
    write_jsonl(path, rows)
    first = sha256_file(path)
    write_jsonl(path, rows)
    assert sha256_file(path) == first
    write_jsonl(second_path, rows)
    assert sha256_file(second_path) == first
