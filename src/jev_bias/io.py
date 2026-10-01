from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Iterator


def open_text(path: str | Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        binary_mode = mode.replace("t", "")
        if "b" not in binary_mode:
            binary_mode += "b"
        if any(flag in mode for flag in "wax"):
            # gzip.GzipFile(filename=str(path)) stores the output filename in
            # the header.  Supplying an empty logical filename plus mtime=0
            # makes identical content byte-identical across paths and runs.
            raw = path.open(binary_mode)
            stream = gzip.GzipFile(filename="", mode=binary_mode, fileobj=raw, mtime=0)
            stream._jev_raw_file = raw  # keep the underlying handle alive
        else:
            stream = gzip.GzipFile(filename=str(path), mode=binary_mode)
        return stream if "b" in mode else io.TextIOWrapper(stream, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="") if "b" not in mode else path.open(mode)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open_text(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selection_digest(sample_ids: Iterable[str]) -> str:
    """Hash a sample set independently of file order."""
    payload = "".join(f"{sample_id}\n" for sample_id in sorted(sample_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_selection_digest(sample_ids: Iterable[str]) -> str:
    """Hash sample IDs in file order (unlike :func:`selection_digest`)."""
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_stream(handle) -> tuple[str, int]:
    """Return SHA-256 and byte count for a binary stream from its current position."""
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size
