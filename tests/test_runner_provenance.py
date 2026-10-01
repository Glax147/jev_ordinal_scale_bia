import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kev_run_hash_includes_batch_size() -> None:
    """Changing batch semantics must invalidate resumable KEV output."""
    tree = ast.parse((ROOT / "scripts" / "run_kev_local.py").read_text(encoding="utf-8"))
    run_hash_dicts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "canonical_json_sha256":
            if isinstance(node.args[0], ast.Dict):
                run_hash_dicts.append(node.args[0])

    assert len(run_hash_dicts) == 1
    keys = {
        key.value
        for key in run_hash_dicts[0].keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "batch_size" in keys
