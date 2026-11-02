#!/bin/bash
# Local launcher for Phase 2 evals via the generic Lambda driver.
# Usage: bash scripts/phase2/kick_pi05_eval.sh [SUITE] [TRIALS] [TAG]
#   P2_MODE=plain  (default) RLinf's libero_eval.py, the M0 baseline path.
#   P2_MODE=guided scripts/phase2/libero_eval_guided.py; pass flags in P2_EXTRA_ARGS, e.g.
#     P2_MODE=guided P2_EXTRA_ARGS="--rho 0 --equivalence_checks 1000000 --equivalence_strict --task_ids 5,9 --dump /outputs/dumps/m1" \
#       bash scripts/phase2/kick_pi05_eval.sh libero_spatial 3 _m1equiv
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
SUITE="${1:-libero_spatial}"; TRIALS="${2:-10}"; TAG="${3:-}"
P2_MODE="${P2_MODE:-plain}"; P2_EXTRA_ARGS="${P2_EXTRA_ARGS:-}"
export BABYSIT_GPU_TYPES="${BABYSIT_GPU_TYPES:-gpu_1x_a10 gpu_1x_a100_sxm4 gpu_1x_h100_pcie gpu_1x_gh200}"
export BABYSIT_SYNC="outputs/phase2/logs outputs/phase2/dumps"
export BABYSIT_IDLE_POLLS=0   # LIBERO eval is simulator-bound; GPU reads 0% between inference bursts
# Env vars do not cross the ssh hop (the driver runs `bash $RUNNER`); mode and flags go positionally.
# EXTRA_ARGS travels single-quoted through the driver's ssh string, so it must not contain quotes.
case "$P2_EXTRA_ARGS" in *\'*|*\"*) echo "P2_EXTRA_ARGS must not contain quotes" >&2; exit 1;; esac
REMOTE="scripts/phase2/run_pi05_eval.sh $SUITE $TRIALS RLinf/RLinf-Pi05-LIBERO-SFT '$TAG' $P2_MODE '$P2_EXTRA_ARGS'"
bash scripts/cloud/kick_generic.sh "p2_eval_${SUITE}${TAG}" \
  "$REMOTE" \
  "=== PHASE2 EVAL done ===" \
  "outputs/phase2/pi05_eval_${SUITE}${TAG}.json" 4 6
