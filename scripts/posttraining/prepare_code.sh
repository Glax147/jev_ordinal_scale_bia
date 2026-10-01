#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PT_DIR="$REPO_ROOT/posttraining"
CODE_DIR="$PT_DIR/code"
KEV_REV="3e1cd3bb588a388a06827443380befece23e68c7"
BA_LORA_REV="1fe17ab3e39dcfefc630cdf386d69d942c61f16c"

clone_exact() {
  local url="$1" revision="$2" target="$3"
  if [[ ! -d "$target/.git" ]]; then
    mkdir -p "$(dirname "$target")"
    git clone --no-checkout "$url" "$target"
    git -C "$target" checkout --detach "$revision"
  fi
  if [[ "$(git -C "$target" rev-parse HEAD)" != "$revision" ]]; then
    echo "Existing checkout is not at the pinned revision: $target" >&2
    exit 4
  fi
}

require_clean() {
  local repo="$1"
  if [[ -n "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Unexpected local changes in pinned checkout: $repo" >&2
    git -C "$repo" status --short >&2
    exit 5
  fi
}

require_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "Patched source hash mismatch: $path" >&2
    exit 6
  fi
}

apply_once() {
  local repo="$1" patch="$2"
  if git -C "$repo" apply --reverse --check "$patch" >/dev/null 2>&1; then
    return
  fi
  git -C "$repo" apply --check "$patch"
  git -C "$repo" apply "$patch"
}

clone_exact https://github.com/jaredpalmer/kev.git "$KEV_REV" "$CODE_DIR/kev"
if [[ "${CLONE_BA_LORA_REFERENCE:-0}" = "1" ]]; then
  # Reference-only provenance checkout; the KEV runtime does not import it.
  clone_exact https://github.com/llm172/BA-LoRA.git "$BA_LORA_REV" "$CODE_DIR/BA-LoRA"
  require_clean "$CODE_DIR/BA-LoRA"
  test "$(git -C "$CODE_DIR/BA-LoRA" rev-parse HEAD)" = "$BA_LORA_REV"
fi
apply_once "$CODE_DIR/kev" "$PT_DIR/patches/kev_ba_lora_adapter.patch"
apply_once "$CODE_DIR/kev" "$PT_DIR/patches/kev_offline_model_bundle.patch"

test "$(git -C "$CODE_DIR/kev" rev-parse HEAD)" = "$KEV_REV"
git -C "$CODE_DIR/kev" diff --check
expected_status=$' M kev/checkpoint.py\n M kev/model.py\n M kev/train.py'
actual_status="$(git -C "$CODE_DIR/kev" status --porcelain=v1 --untracked-files=all | LC_ALL=C sort)"
if [[ "$actual_status" != "$expected_status" ]]; then
  echo "KEV checkout is not the exact expected patched tree." >&2
  git -C "$CODE_DIR/kev" status --short >&2
  exit 7
fi
git -C "$CODE_DIR/kev" apply --reverse --check "$PT_DIR/patches/kev_ba_lora_adapter.patch"
git -C "$CODE_DIR/kev" apply --reverse --check "$PT_DIR/patches/kev_offline_model_bundle.patch"
require_sha256 "$CODE_DIR/kev/kev/train.py" "4324bbcb28408650171d16db9ca95c530ce62271ca2b756241514e022f3649ee"
require_sha256 "$CODE_DIR/kev/kev/checkpoint.py" "3a0199cfaff2f30e585c11a666a9138951d9356921ee579065aa49d9aa0f7e24"
require_sha256 "$CODE_DIR/kev/kev/model.py" "a2e6d1a72b8931fa0b1f83f7cea72c794a9b5cd4edd4696626eca52789145885"
echo "Pinned KEV tree is ready (official BA-LoRA source remains reference-only)."
