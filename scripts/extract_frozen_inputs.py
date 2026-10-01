#!/usr/bin/env python3
"""Compatibility notice for releases that formerly contained frozen ZIPs."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "This public release intentionally omits frozen-input ZIP files. "
        "Install requirements-data.txt, then run "
        "`python scripts/rebuild_inputs.py --download --allow-unpinned`; "
        "the rebuild validates every prompt against its paper-v1 SHA-256."
    )


if __name__ == "__main__":
    main()
