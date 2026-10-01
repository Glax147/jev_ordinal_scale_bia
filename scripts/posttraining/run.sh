#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PT_DIR="$BUNDLE_DIR/posttraining"
SCRIPT_DIR="$BUNDLE_DIR/scripts/posttraining"
SIZES=("$@")
if [[ ${#SIZES[@]} -eq 0 ]]; then SIZES=(0.8b 4b); fi
for size in "${SIZES[@]}"; do
  [[ "$size" = "0.8b" || "$size" = "4b" ]] || { echo "Supported sizes: 0.8b 4b" >&2; exit 2; }
done
if [[ ${#SIZES[@]} -gt 1 && -n "${OUT:-}" ]]; then
  echo "OUT may only be set when training one model size; otherwise both sizes would share one directory." >&2
  exit 2
fi

bash "$SCRIPT_DIR/setup.sh"
cd "$PT_DIR/code/kev"
uv run --no-sync python "$SCRIPT_DIR/download_models.py" "${SIZES[@]}"
uv run --no-sync python "$SCRIPT_DIR/download_raw_data.py"
uv run --no-sync python "$SCRIPT_DIR/prepare_kev_training_data.py"
if [[ ! -f "$BUNDLE_DIR/artifacts/frozen_inputs/main40.jsonl" ]]; then
  echo "main40 is not reconstructed. From the repository root run:" >&2
  echo "  python -m pip install -r requirements/data.txt" >&2
  echo "  python scripts/rebuild_inputs.py --download --allow-unpinned" >&2
  exit 4
fi
uv run --no-sync python "$SCRIPT_DIR/verify_package.py"
uv run --no-sync python "$SCRIPT_DIR/verify_models.py" --full-hash

for size in "${SIZES[@]}"; do
  seed="${SEED:-42}"
  tag="${RUN_TAG:-ba-lora}"
  checkpoint="${OUT:-$PT_DIR/runs/kev-${size}-${tag}-seed${seed}}"
  if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    bash "$SCRIPT_DIR/train.sh" "$size" 2>&1 | tee "$PT_DIR/logs/kev-${size}-${tag}-seed${seed}.log"
  fi
  [[ -d "$checkpoint" ]] || { echo "Fine-tuned checkpoint is missing: $checkpoint" >&2; exit 5; }
  if [[ "$size" = "0.8b" ]]; then eval_batch="${EVAL_BATCH:-32}"; else eval_batch="${EVAL_BATCH:-8}"; fi
  eval_dir="$PT_DIR/results/generated/kev-${size}-${tag}-seed${seed}/main40"
  mkdir -p "$eval_dir"
  uv run --no-sync python "$SCRIPT_DIR/evaluate_kev_bias40.py" \
    --checkpoint "$checkpoint" \
    --run-name "kev-${size}-${tag}-seed${seed}" \
    --input "$BUNDLE_DIR/artifacts/frozen_inputs/main40.jsonl" \
    --output "$eval_dir/answers.jsonl" \
    --summary "$eval_dir/score_summary.json" \
    --batch "$eval_batch" --device cuda
done
