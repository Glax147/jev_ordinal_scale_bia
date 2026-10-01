#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
PROFILE="${PROFILE:-analysis}"
MODELS="${MODELS:-}"
DOWNLOAD_UPSTREAM_DATA="${DOWNLOAD_UPSTREAM_DATA:-0}"
REBUILD_INPUTS="${REBUILD_INPUTS:-0}"
ALLOW_UNPINNED_DATA="${ALLOW_UNPINNED_DATA:-0}"

if [[ "$PROFILE" != "analysis" && "$PROFILE" != "kev" ]]; then
  echo "PROFILE must be analysis or kev" >&2
  exit 2
fi
if [[ -n "$MODELS" && "$PROFILE" != "kev" ]]; then
  echo "MODELS requires PROFILE=kev" >&2
  exit 2
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi
PY="$ROOT/.venv/bin/python"
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else "Python 3.13 is required")'
"$PY" -m pip install -r "$ROOT/requirements/bootstrap.txt"
if [[ "$PROFILE" == "kev" ]]; then
  export PIP_NO_BUILD_ISOLATION=1
  "$PY" -m pip install -r "$ROOT/requirements-kev.txt"
else
  "$PY" -m pip install -r "$ROOT/requirements.txt"
fi
"$PY" -m pip install --no-deps -e "$ROOT"
if [[ "$DOWNLOAD_UPSTREAM_DATA" == "1" || "$REBUILD_INPUTS" == "1" ]]; then
  "$PY" -m pip install -r "$ROOT/requirements-data.txt"
fi
if [[ "$REBUILD_INPUTS" == "1" ]]; then
  REBUILD_ARGS=("$ROOT/scripts/rebuild_inputs.py" --download)
  if [[ "$ALLOW_UNPINNED_DATA" == "1" ]]; then REBUILD_ARGS+=(--allow-unpinned); fi
  "$PY" "${REBUILD_ARGS[@]}"
elif [[ "$DOWNLOAD_UPSTREAM_DATA" == "1" ]]; then
  DATA_ARGS=("$ROOT/scripts/download_datasets.py" --sources pinned)
  if [[ "$ALLOW_UNPINNED_DATA" == "1" ]]; then
    DATA_ARGS=("$ROOT/scripts/download_datasets.py" --sources all --allow-unpinned)
  fi
  "$PY" "${DATA_ARGS[@]}"
fi
if [[ -n "$MODELS" ]]; then
  "$PY" "$ROOT/scripts/download_models.py" --models "$MODELS"
fi
"$PY" "$ROOT/scripts/verify_artifacts.py"
echo "Bundle ready. Python: $PY"
