#!/bin/bash
# Runs ON the box. KL-coefficient ablation on the seed that regressed:
# PickScore, seed 2042, beta in {0, 0.2} (the existing runs are beta=0.04,
# the upstream default; the beta=0.04 seed-2042 LoRA is uploaded for a
# same-run eval comparison at outputs/checkpoints/flowgrpo_pickscore_pub_s2042/).
# Usage: HF_TOKEN=... bash scripts/cloud/run_kl.sh

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"
export FG_GROUP=8 FG_NBPE=8

train_beta() { # beta tag
  local BETA=$1 TAG=$2
  local DEST="$PROJECT_ROOT/outputs/checkpoints/flowgrpo_pickscore_${TAG}"
  if [ -d "$DEST" ] && [ -n "$(ls "$DEST" 2>/dev/null | grep '^checkpoint-')" ]; then
    log ">>> $TAG already trained"; return 0
  fi
  log ">>> training beta=$BETA (seed 2042, 4h)"
  rm -rf "$PROJECT_ROOT/repos/flow_grpo/logs/pickscore" "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_pickscore"
  HF_TOKEN="$HF_TOKEN" FG_GROUP=8 FG_NBPE=8 FG_SEED=2042 FG_BETA=$BETA \
      bash scripts/cloud/run_flowgrpo_pickscore.sh 4 || log "WARN: training exited nonzero"
  [ -d "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_pickscore" ] && \
      mv "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_pickscore" "$DEST"
  log "$TAG: $(ls "$DEST" 2>/dev/null | grep -c '^checkpoint-') checkpoints"
}

train_beta 0 b0_s2042
train_beta 0.2 b02_s2042

PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" \
          "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY

lora_of() { ls -d "$PROJECT_ROOT"/outputs/checkpoints/$1/checkpoint-*/lora 2>/dev/null | sort -t- -k2 -V | tail -1; }
L0=$(lora_of flowgrpo_pickscore_b0_s2042)
L02=$(lora_of flowgrpo_pickscore_b02_s2042)
L004=$(lora_of flowgrpo_pickscore_pub_s2042)
RL_ARGS=()
[ -n "$L0" ] && RL_ARGS+=(--rl-lora "rl_beta0=$L0")
[ -n "$L004" ] && RL_ARGS+=(--rl-lora "rl_beta004=$L004")
[ -n "$L02" ] && RL_ARGS+=(--rl-lora "rl_beta02=$L02")

log ">>> KL ablation eval"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward pickscore --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos "" --stack-rhos "" "${RL_ARGS[@]}" \
    --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/study_kl_matrix.json" \
    || { log "FATAL: eval failed"; exit 1; }

log "=== KL done ==="
