#!/bin/bash
# Local launcher for Phase 2 M0 via the generic Lambda driver.
# Usage: bash scripts/phase2/kick_pi05_eval.sh [SUITE] [TRIALS] [TAG]
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
SUITE="${1:-libero_spatial}"; TRIALS="${2:-10}"; TAG="${3:-}"
export BABYSIT_GPU_TYPES="${BABYSIT_GPU_TYPES:-gpu_1x_a100_sxm4 gpu_1x_h100_pcie gpu_1x_gh200}"
export BABYSIT_SYNC="outputs/phase2/logs"
bash scripts/cloud/kick_generic.sh "p2_eval_${SUITE}${TAG}" \
  "scripts/phase2/run_pi05_eval.sh $SUITE $TRIALS RLinf/RLinf-Pi05-LIBERO-SFT '$TAG'" \
  "=== PHASE2 EVAL done ===" \
  "outputs/phase2/pi05_eval_${SUITE}${TAG}.json" 4 6
