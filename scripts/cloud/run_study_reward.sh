#!/bin/bash
# Runs ON the Lambda box (80GB GPU required). One reward of the TFG-vs-GRPO
# study, end to end:
#   1. FlowGRPO group=8 training (existing per-reward runner, FG_GROUP=8),
#      matched 4h wall-clock — same regime as the June group=8 runs.
#   2. Full eval matrix (eval_study_matrix.py): base, TFG rho-sweep guided on
#      the same reward, RL checkpoints, TFG-stacked-on-RL — 3 paired seed
#      bases x 30 prompts, every image scored on CLIP + PickScore + jpeg.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_study_reward.sh <clip|clip_ocr|pickscore> [WALL_HOURS]

set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

: "${HF_TOKEN:?HF_TOKEN required}"
REWARD_KEY="${1:?usage: run_study_reward.sh <clip|clip_ocr|pickscore> [WALL_HOURS]}"
WALL_HOURS="${2:-4}"
export FG_GROUP="${FG_GROUP:-8}"
export FG_NBPE="${FG_NBPE:-8}"

case "$REWARD_KEY" in
  clip)
    TRAIN_SH=scripts/cloud/run_flowgrpo_clip.sh
    CKPT_DIR=outputs/checkpoints/flowgrpo_clip
    EVAL_REWARD=clip
    PROMPTS_SPEC=ddrl
    ;;
  clip_ocr)
    TRAIN_SH=scripts/cloud/run_flowgrpo_clip_ocr.sh
    CKPT_DIR=outputs/checkpoints/flowgrpo_clip_ocr
    EVAL_REWARD=clip
    PROMPTS_SPEC=ocr
    ;;
  pickscore)
    TRAIN_SH=scripts/cloud/run_flowgrpo_pickscore.sh
    CKPT_DIR=outputs/checkpoints/flowgrpo_pickscore
    EVAL_REWARD=pickscore
    PROMPTS_SPEC=ddrl
    ;;
  *) log "unknown reward key: $REWARD_KEY"; exit 1 ;;
esac

log "=== STUDY[$REWARD_KEY]: group=$FG_GROUP train (${WALL_HOURS}h) + full eval matrix ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

# --- 1. Train ---
log ">>> [1/2] FlowGRPO training ($REWARD_KEY, group=$FG_GROUP)"
HF_TOKEN="$HF_TOKEN" FG_GROUP="$FG_GROUP" FG_NBPE="$FG_NBPE" \
    bash "$TRAIN_SH" "$WALL_HOURS" \
    || log "WARN: training exited nonzero (eval proceeds with whatever checkpoints exist)"

N_CKPT=$(ls "$CKPT_DIR" 2>/dev/null | grep -c '^checkpoint-' || true)
log "checkpoints in $CKPT_DIR: $N_CKPT"

# --- 2. Eval prompts file ---
if [ "$PROMPTS_SPEC" = "ddrl" ]; then
    PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_ddrl.txt"
    python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
prompts = [d["prompt"] for d in json.loads(src.read_text())]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(prompts) + "\n")
print(f"wrote {len(prompts)} prompts -> {out}")
PY
else
    PROMPTS_FILE="$PROJECT_ROOT/repos/flow_grpo/dataset/ocr/test.txt"
    [ -f "$PROMPTS_FILE" ] || { log "FATAL: $PROMPTS_FILE missing"; exit 1; }
fi

# --- 3. Eval matrix ---
log ">>> [2/2] eval matrix ($EVAL_REWARD on $PROMPTS_SPEC prompts)"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward "$EVAL_REWARD" \
    --prompts-file "$PROMPTS_FILE" \
    --num-prompts 30 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/$CKPT_DIR" \
    --tfg-rhos 2,5,10,20 --stack-rhos 20 \
    --seed-bases 0,1000,2000 \
    --image-dir "$PROJECT_ROOT/outputs/study_${REWARD_KEY}_images" \
    --out "$PROJECT_ROOT/outputs/study_${REWARD_KEY}_matrix.json" \
    || { log "FATAL: eval matrix failed"; exit 1; }

log "=== STUDY[$REWARD_KEY] done ==="
log "Result: outputs/study_${REWARD_KEY}_matrix.json"
