#!/usr/bin/env bash
set -euo pipefail

SIZE="${1:-0.8b}"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-$PKG_DIR/data/prepared/kev_bias_train.jsonl}"
ANCHOR_DIR="$PKG_DIR/artifacts/anchors"
RUN_DIR="$PKG_DIR/runs"

export HF_HOME="${HF_HOME:-$PKG_DIR/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PKG_DIR/.cache/uv}"
export TMPDIR="${TMPDIR:-$PKG_DIR/.cache/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export KEV_LOCAL_MODEL_ROOT="${KEV_LOCAL_MODEL_ROOT:-$PKG_DIR/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

case "$SIZE" in
  0.8b)
    BASE="Qwen/Qwen3.5-0.8B-Base"; INIT="jaredpalmer/kev-0.8b"
    BATCH="${BATCH:-32}"; ACCUM="${ACCUM:-2}"; LR="${LR:-2e-5}"; CKPT="${CKPT:-0}"
    ;;
  4b)
    BASE="Qwen/Qwen3.5-4B-Base"; INIT="jaredpalmer/kev-4b"
    BATCH="${BATCH:-8}"; ACCUM="${ACCUM:-8}"; LR="${LR:-1.5e-5}"; CKPT="${CKPT:-0}"
    ;;
  *) echo "SIZE must be one of: 0.8b, 4b" >&2; exit 2 ;;
esac

for required in "$KEV_LOCAL_MODEL_ROOT/kev-${SIZE}" "$KEV_LOCAL_MODEL_ROOT/qwen3.5-${SIZE}-base"; do
  if [[ ! -d "$required" ]]; then
    echo "Bundled model directory is missing: $required" >&2
    exit 4
  fi
done

EPOCHS="${EPOCHS:-2}"
MAX_STATE="${MAX_STATE:-2048}"
SEED="${SEED:-42}"
PERM_KL="${PERM_KL:-0}"
ANCHOR_W="${ANCHOR_W:-0.025}"
BA_DIV_W="${BA_DIV_W:-0.005}"
BA_SVD_W="${BA_SVD_W:-0.005}"
BA_SVD_K="${BA_SVD_K:-10}"
BA_TEMP="${BA_TEMP:-2.0}"
BA_CONSISTENCY_SCHEDULE="${BA_CONSISTENCY_SCHEDULE:-cosine}"
BA_FOCUS_SCHEDULE="${BA_FOCUS_SCHEDULE:-two_phase}"
BA_WARMUP_RATIO="${BA_WARMUP_RATIO:-0.2}"
BA_RAMP_UP_RATIO="${BA_RAMP_UP_RATIO:-0.05}"
BA_SVD_FROB_NORM="${BA_SVD_FROB_NORM:-1}"
ANCHOR="$ANCHOR_DIR/original-kev-${SIZE}-train-temp${BA_TEMP}.json"
RUN_TAG="${RUN_TAG:-ba-lora}"
OUT="${OUT:-$RUN_DIR/kev-${SIZE}-${RUN_TAG}-seed${SEED}}"

mkdir -p "$ANCHOR_DIR" "$RUN_DIR" "$PKG_DIR/logs"
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing output: $OUT" >&2
  exit 3
fi

cd "$PKG_DIR/code/kev"
if [[ "$ANCHOR_W" != "0" && ! -s "$ANCHOR" ]]; then
  echo "Building frozen original-KEV teacher anchors: $ANCHOR"
  uv run --no-sync python "$PKG_DIR/scripts/build_kev_teacher_anchors.py" \
    --checkpoint "$INIT" --data "$DATA" --out "$ANCHOR" \
    --batch "$BATCH" --max_state "$MAX_STATE" --device cuda --temperature "$BA_TEMP"
elif [[ "$ANCHOR_W" != "0" ]]; then
  echo "Verifying reusable teacher anchors against the exact data and model trees: $ANCHOR"
  uv run --no-sync python "$PKG_DIR/scripts/build_kev_teacher_anchors.py" \
    --checkpoint "$INIT" --data "$DATA" --out "$ANCHOR" \
    --batch "$BATCH" --max_state "$MAX_STATE" --device cuda --temperature "$BA_TEMP" \
    --verify-existing
fi

ANCHOR_ARGS=()
if [[ "$ANCHOR_W" != "0" ]]; then
  ANCHOR_ARGS=(--anchor "$ANCHOR" --anchor_w "$ANCHOR_W" --anchor_temp "$BA_TEMP")
fi

echo "Training $SIZE -> $OUT"
uv run --no-sync python -m kev.train \
  --data "$DATA" \
  --base "$BASE" \
  --init_from "$INIT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --batch "$BATCH" \
  --accum "$ACCUM" \
  --dtype bf16 \
  --weights_dtype bf16 \
  --checkpointing "$CKPT" \
  --max_state "$MAX_STATE" \
  --p_none 0 \
  --p_none_distract 0 \
  --p_distract 0 \
  --p_none_pair 0 \
  --perm_kl "$PERM_KL" \
  --perm_frac 1.0 \
  "${ANCHOR_ARGS[@]}" \
  --ba_div_w "$BA_DIV_W" \
  --ba_svd_w "$BA_SVD_W" \
  --ba_svd_k "$BA_SVD_K" \
  --ba_svd_frob_norm "$BA_SVD_FROB_NORM" \
  --ba_consistency_schedule "$BA_CONSISTENCY_SCHEDULE" \
  --ba_focus_schedule "$BA_FOCUS_SCHEDULE" \
  --ba_warmup_ratio "$BA_WARMUP_RATIO" \
  --ba_ramp_up_ratio "$BA_RAMP_UP_RATIO" \
  --seed "$SEED" \
  --device cuda \
  --out "$OUT"

echo "Finished: $OUT"
