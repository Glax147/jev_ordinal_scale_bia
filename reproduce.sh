#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
if [[ ! -d "$REPO/.venv" ]]; then
  "$PYTHON_BIN" -m venv "$REPO/.venv"
fi
"$REPO/.venv/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else "Python 3.13 is required")'
"$REPO/.venv/bin/python" -m pip install -r "$REPO/requirements/bootstrap.txt"
"$REPO/.venv/bin/python" -m pip install -r "$REPO/requirements.txt"
"$REPO/.venv/bin/python" -m pip install --no-deps -e "$REPO"
"$REPO/.venv/bin/python" "$REPO/scripts/one_click.py"
