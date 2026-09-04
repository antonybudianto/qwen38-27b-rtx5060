#!/usr/bin/env bash
set -euo pipefail

# Validate the patches whose order and hunk metadata are part of the vLLM 0.28.0
# contract. GNU patch is intentionally permissive about offsets/fuzz; git apply
# is the stricter format check that catches a hand-edited hunk header immediately.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SOURCE=${1:?usage: bash patches/check_vllm_series.sh /path/to/vllm-v0.28.0}

git -C "$VLLM_SOURCE" rev-parse --is-inside-work-tree >/dev/null

PATCHES=(
  vllm-pr50021-gdn-spec-bounds.patch
  dflash2-lookup-drafting.patch
  dflash2-ngram-chains.patch
  dflash2-prewarm.patch
  dflash2-z-adaptive-emitted.patch
)

for name in "${PATCHES[@]}"; do
  patch="$HERE/patches/$name"
  echo "== git apply --check $name"
  git -C "$VLLM_SOURCE" apply --check --whitespace=error -p1 < "$patch"
  git -C "$VLLM_SOURCE" apply --whitespace=error -p1 < "$patch"
done

git -C "$VLLM_SOURCE" diff --check
echo "patch integrity: OK"
