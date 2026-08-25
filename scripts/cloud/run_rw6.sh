#!/bin/bash
# Runs ON the box. Slope-predictor expansion: aesthetic + imagereward.
# beta=0 RL (4h each, seed 42) + TFG rho dose-response {2,5,10,20,80} + RL arm.
# Usage: HF_TOKEN=... bash scripts/cloud/run_rw6.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"
export FG_GROUP=8 FG_NBPE=8
python3 -m pip install --quiet image-reward || log "WARN: image-reward pip failed"
PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY
lora_of() { ls -d "$PROJECT_ROOT"/outputs/checkpoints/$1/checkpoint-*/lora 2>/dev/null | sort -t- -k2 -V | tail -1; }
for KEY in aesthetic imagereward; do
  DEST="flowgrpo_${KEY}_b0_s42"
  if [ ! -d "$PROJECT_ROOT/outputs/checkpoints/$DEST" ]; then
    log ">>> training $KEY beta=0"
    rm -rf "$PROJECT_ROOT/repos/flow_grpo/logs/$KEY" "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_$KEY"
    HF_TOKEN="$HF_TOKEN" FG_GROUP=8 FG_NBPE=8 FG_SEED=42 FG_BETA=0 \
        bash scripts/cloud/run_flowgrpo_$KEY.sh 4 || log "WARN: $KEY train nonzero"
    [ -d "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_$KEY" ] && \
        mv "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_$KEY" "$PROJECT_ROOT/outputs/checkpoints/$DEST"
  fi
  L=$(lora_of $DEST)
  RL_ARGS=(); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_b0_s42=$L")
  log ">>> $KEY dose-response + RL eval"
  python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward $KEY --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos 2,5,10,20,80 --stack-rhos "" "${RL_ARGS[@]}" \
    --seed-bases 0,1000 \
    --out "$PROJECT_ROOT/outputs/study_rw6_${KEY}_matrix.json" \
    || { log "FATAL: $KEY eval failed"; exit 1; }
done
log "=== RW6 done ==="
