#!/bin/bash
# Launch the Phase 2 sampling-side arms (M2/M3), each on its own A10 box, paired with base50.
# Usage: bash scripts/phase2/launch_guided_arms.sh [ARM ...]     (default: all five)
#   g_raw002  guidance, raw metric,  rho 0.002   (last-step logit gain ~ +1.6, rotation dims only)
#   g_raw005  guidance, raw metric,  rho 0.005   (~ +3.9)
#   g_std05   guidance, std metric,  rho 0.05    (~ +1.7, every dim moved ~0.1 of its natural sd)
#   g_std10   guidance, std metric,  rho 0.1     (~ +3.4, ~0.2 sd)
#   bon8      best-of-8 by Q, no gradient
# Tasks 0,5,3,6,9 x 50 trials = 250 episodes per arm: task 0 is the harm control (49/50 in base50),
# tasks 5,3,6,9 hold 99 of the 116 base50 failures. Same seed, noise and init states as base50; an
# unguided rerun on another A10 reproduced base50 episodes exactly, so stay on A10 for exact pairing.
# Cost at $1.29/h (Nov 2026): ~3.2 h, ~$4 per arm, ~$20 for all five.
set -euo pipefail
cd "$(dirname "$0")/../.."
QC="/workspace/phase2/q/xfit_hi.pt,/workspace/phase2/q/xfit_lo.pt"
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(g_raw002 g_raw005 g_std05 g_std10 bon8)
for TAG in "${ARMS[@]}"; do
  case "$TAG" in
    g_raw002) FLAGS="--rho 0.002 --metric raw" ;;
    g_raw005) FLAGS="--rho 0.005 --metric raw" ;;
    g_std05)  FLAGS="--rho 0.05 --metric std" ;;
    g_std10)  FLAGS="--rho 0.1 --metric std" ;;
    bon8)     FLAGS="--bon 8" ;;
    *) echo "unknown arm $TAG" >&2; exit 1 ;;
  esac
  P2_MODE=guided P2_WATCH_HOURS=6 BABYSIT_GPU_TYPES="gpu_1x_a10" \
  P2_SYNC="outputs/phase2/logs/pi05_sft_libero_spatial_${TAG}.stdout outputs/phase2/dumps/${TAG}" \
  P2_EXTRA_ARGS="$FLAGS --task_ids 0,5,3,6,9 --q_ckpt $QC --dump /outputs/dumps/${TAG}" \
    bash scripts/phase2/kick_pi05_eval.sh libero_spatial 50 "_${TAG}"
  sleep 20
done
echo "monitor: bash scripts/phase2/monitor_jobs.sh ${ARMS[*]}"
echo "keep the Mac awake and on AC: caffeinate -s -i -m -t 28800 &"
