#!/usr/bin/env python3
"""Remove non-evaluation sidecars from a historical frozen-input archive.

The exact JSONL member is copied byte-for-byte through ZIP streams.  Only the
obsolete builder sidecar (which contained a machine-local path) is omitted.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "frozen_inputs" / "ordinal_bias_batch3_20x5000.zip"
KEEP = "ordinal_bias_batch3_20x5000.jsonl"
REMOVE = "ordinal_bias_batch3_20x5000_manifest.json"


def main() -> None:
    temporary_path = None
    with zipfile.ZipFile(ARCHIVE, "r") as source:
        names = source.namelist()
        if names == [KEEP]:
            print(f"Already clean: {ARCHIVE.relative_to(ROOT)}")
            return
        if set(names) != {KEEP, REMOVE}:
            raise RuntimeError(f"Refusing unexpected archive members: {names}")
        info = source.getinfo(KEEP)
        with tempfile.NamedTemporaryFile(
            prefix=ARCHIVE.stem + ".", suffix=".tmp", dir=ARCHIVE.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(temporary_path, "w") as target:
            with source.open(KEEP, "r") as reader, target.open(info, "w") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
    try:
        os.replace(temporary_path, ARCHIVE)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Removed {REMOVE}; retained the exact {KEEP} byte stream")


if __name__ == "__main__":
    main()
