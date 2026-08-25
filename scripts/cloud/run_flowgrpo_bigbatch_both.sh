#!/bin/bash
# Runs ON the Lambda box. Big-batch FlowGRPO per Haotian's suggestion: gradient
# accumulation + a much larger rollout/group size. Restores the GRPO group from
# our 40GB-crippled 2 back to upstream's general_ocr_sd3_1gpu value of 8
# (FG_GROUP=8), which on 1 GPU means train_batch_size=8 and needs an 80GB GPU.
#
# Sequential: PickScore run -> PickScore eval -> CLIP run -> CLIP eval. Each
# training run uses the SAME wall-clock budget as the prior failing group=2 runs,
# so the only thing that changed is the group size (controlled comparison).
#
# Usage: HF_TOKEN=... [FG_GROUP=8] [FG_NBPE=8] bash run_flowgrpo_bigbatch_both.sh [WALL_HOURS_PER_REWARD]

set -uo pipefail   # not -e: a failure in one reward shouldn't skip the other

export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

: "${HF_TOKEN:?HF_TOKEN required (SD3.5-M is gated)}"
export FG_GROUP="${FG_GROUP:-8}"
export FG_NBPE="${FG_NBPE:-8}"
WALL_HOURS="${1:-4}"

log "=== BIGBATCH orchestrator: FG_GROUP=$FG_GROUP FG_NBPE=$FG_NBPE, ${WALL_HOURS}h per reward ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

log ">>> [1/4] PickScore training (group=$FG_GROUP)"
HF_TOKEN="$HF_TOKEN" FG_GROUP="$FG_GROUP" FG_NBPE="$FG_NBPE" \
    bash scripts/cloud/run_flowgrpo_pickscore.sh "$WALL_HOURS" \
    || log "WARN: pickscore training exited nonzero"

log ">>> [2/4] PickScore eval"
HF_TOKEN="$HF_TOKEN" bash scripts/cloud/run_flowgrpo_pickscore_eval.sh \
    || log "WARN: pickscore eval failed"

log ">>> [3/4] CLIP training (group=$FG_GROUP)"
HF_TOKEN="$HF_TOKEN" FG_GROUP="$FG_GROUP" FG_NBPE="$FG_NBPE" \
    bash scripts/cloud/run_flowgrpo_clip.sh "$WALL_HOURS" \
    || log "WARN: clip training exited nonzero"

log ">>> [4/4] CLIP eval"
HF_TOKEN="$HF_TOKEN" bash scripts/cloud/run_flowgrpo_clip_eval.sh \
    || log "WARN: clip eval failed"

log "=== BIGBATCH orchestrator done ==="
log "Results on box: outputs/flowgrpo_pickscore_eval.json, outputs/flowgrpo_clip_eval.json"
