#!/bin/bash
# Runs ON the box. Fair-baseline check: FlowGRPO with KL beta=0 on the CLIP
# setting (the KL ablation showed beta=0.04, our default, can be pathological).
#   role short: beta=0, seeds 42 + 1042, 4h each -> eval {base, tfg rho20, both RL}
#   role long:  beta=0, seed 42, 16h            -> eval {base, tfg rho20, RL}
# Usage: HF_TOKEN=... bash scripts/cloud/run_beta0.sh <short|long>

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"
ROLE="${1:?usage: run_beta0.sh <short|long>}"
export FG_GROUP=8 FG_NBPE=8

train_one() { # hours seed tag
  local HOURS=$1 SEED=$2 TAG=$3
  local DEST="$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip_${TAG}"
  if [ -d "$DEST" ] && [ -n "$(ls "$DEST" 2>/dev/null | grep '^checkpoint-')" ]; then
    log ">>> $TAG already trained"; return 0
  fi
  log ">>> training $TAG (beta=0, ${HOURS}h, seed $SEED)"
  rm -rf "$PROJECT_ROOT/repos/flow_grpo/logs/clip_score" "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip"
  HF_TOKEN="$HF_TOKEN" FG_GROUP=8 FG_NBPE=8 FG_SEED=$SEED FG_BETA=0 \
      bash scripts/cloud/run_flowgrpo_clip.sh "$HOURS" || log "WARN: training exited nonzero"
  [ -d "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip" ] && \
      mv "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip" "$DEST"
  log "$TAG: $(ls "$DEST" 2>/dev/null | grep -c '^checkpoint-') checkpoints"
}

PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" \
          "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY

lora_of() { ls -d "$PROJECT_ROOT"/outputs/checkpoints/flowgrpo_clip_$1/checkpoint-*/lora 2>/dev/null | sort -t- -k2 -V | tail -1; }

if [ "$ROLE" = "short" ]; then
  train_one 4 42 b0_s42
  train_one 4 1042 b0_s1042
  RL_ARGS=()
  L=$(lora_of b0_s42);   [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_b0_s42=$L")
  L=$(lora_of b0_s1042); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_b0_s1042=$L")
  OUT="study_beta0_clip_matrix.json"
else
  train_one 16 42 b0_16h_s42
  RL_ARGS=()
  L=$(lora_of b0_16h_s42); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_b0_16h_s42=$L")
  OUT="study_beta0_16h_matrix.json"
fi

log ">>> eval"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos 20 --stack-rhos "" "${RL_ARGS[@]}" \
    --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/$OUT" \
    || { log "FATAL: eval failed"; exit 1; }

log "=== BETA0[$ROLE] done ==="
