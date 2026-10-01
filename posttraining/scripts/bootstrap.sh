#!/usr/bin/env bash
set -euo pipefail

PT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$PT_DIR/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PT_DIR/.cache/uv}"
export TMPDIR="${TMPDIR:-$PT_DIR/.cache/tmp}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR" "$PT_DIR/artifacts/anchors" "$PT_DIR/runs" "$PT_DIR/logs"

bash "$PT_DIR/scripts/setup_code.sh"
if ! command -v uv >/dev/null 2>&1 || [[ "$(uv --version | awk '{print $2}')" != "0.11.8" ]]; then
  python3.13 -m pip install --upgrade --force-reinstall uv==0.11.8
  hash -r
fi
[[ "$(uv --version | awk '{print $2}')" = "0.11.8" ]] || { echo "uv 0.11.8 is required" >&2; exit 3; }
cd "$PT_DIR/code/kev"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.13
fi
uv pip install --python .venv/bin/python -r "$PT_DIR/requirements.txt"
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .
uv run --no-sync python -c 'import torch, transformers, peft; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__, "peft", peft.__version__)'
