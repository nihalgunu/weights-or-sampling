#!/bin/bash
# Runs ON the Lambda box. Re-runs the two CLIP regimes at the larger GRPO group
# (FG_GROUP=8) that flipped PickScore from regression to gain. Sequential:
#   CLIP/DDRL-prompts run -> eval, then CLIP/OCR-prompts run -> eval.
# The checkpoint-copy bug that lost the first group=8 CLIP run is fixed in
# run_flowgrpo_clip.sh; clip_ocr already copied correctly.
#
# Usage: HF_TOKEN=... [FG_GROUP=8] [FG_NBPE=8] bash run_flowgrpo_clip_both.sh [WALL_HOURS_PER_REWARD]

set -uo pipefail   # not -e: one reward failing shouldn't skip the other

export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

: "${HF_TOKEN:?HF_TOKEN required (SD3.5-M is gated)}"
export FG_GROUP="${FG_GROUP:-8}"
export FG_NBPE="${FG_NBPE:-8}"
WALL_HOURS="${1:-4}"

log "=== CLIP-BOTH orchestrator: FG_GROUP=$FG_GROUP FG_NBPE=$FG_NBPE, ${WALL_HOURS}h per reward ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

log ">>> [1/4] CLIP / DDRL-prompts training (group=$FG_GROUP)"
HF_TOKEN="$HF_TOKEN" FG_GROUP="$FG_GROUP" FG_NBPE="$FG_NBPE" \
    bash scripts/cloud/run_flowgrpo_clip.sh "$WALL_HOURS" \
    || log "WARN: clip training exited nonzero"

log ">>> [2/4] CLIP / DDRL-prompts eval"
HF_TOKEN="$HF_TOKEN" bash scripts/cloud/run_flowgrpo_clip_eval.sh \
    || log "WARN: clip eval failed"

log ">>> [3/4] CLIP / OCR-prompts training (group=$FG_GROUP)"
HF_TOKEN="$HF_TOKEN" FG_GROUP="$FG_GROUP" FG_NBPE="$FG_NBPE" \
    bash scripts/cloud/run_flowgrpo_clip_ocr.sh "$WALL_HOURS" \
    || log "WARN: clip_ocr training exited nonzero"

log ">>> [4/4] CLIP / OCR-prompts eval"
HF_TOKEN="$HF_TOKEN" bash scripts/cloud/run_flowgrpo_clip_ocr_eval.sh \
    || log "WARN: clip_ocr eval failed"

log "=== CLIP-BOTH orchestrator done ==="
log "Results on box: outputs/flowgrpo_clip_eval.json, outputs/flowgrpo_clip_ocr_eval.json"
