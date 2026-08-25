#!/bin/bash
# Runs ON the box. Aesthetic RL seed replication: beta=0, 4h, seeds 1042+2042
# (seed 42 already measured in study_rw6_aesthetic_matrix.json at +0.126 t37).
# Usage: HF_TOKEN=... bash scripts/cloud/run_aes_seeds.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"
export FG_GROUP=8 FG_NBPE=8

PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY
lora_of() { ls -d "$PROJECT_ROOT"/outputs/checkpoints/$1/checkpoint-*/lora 2>/dev/null | sort -t- -k2 -V | tail -1; }

for SEED in 1042 2042; do
  DEST="flowgrpo_aesthetic_b0_s$SEED"
  if [ ! -d "$PROJECT_ROOT/outputs/checkpoints/$DEST" ]; then
    log ">>> training aesthetic beta=0 seed $SEED"
    rm -rf "$PROJECT_ROOT/repos/flow_grpo/logs/aesthetic" "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_aesthetic"
    HF_TOKEN="$HF_TOKEN" FG_GROUP=8 FG_NBPE=8 FG_SEED=$SEED FG_BETA=0 \
        bash scripts/cloud/run_flowgrpo_aesthetic.sh 4 || log "WARN: seed $SEED train nonzero"
    [ -d "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_aesthetic" ] && \
        mv "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_aesthetic" "$PROJECT_ROOT/outputs/checkpoints/$DEST"
  fi
done

RL_ARGS=()
L=$(lora_of flowgrpo_aesthetic_b0_s1042); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_b0_s1042=$L")
L=$(lora_of flowgrpo_aesthetic_b0_s2042); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_b0_s2042=$L")
[ ${#RL_ARGS[@]} -gt 0 ] || { log "FATAL: no checkpoints trained"; exit 1; }

log ">>> eval (base + tfg rho20 + 2 RL seeds)"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward aesthetic --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos 20 --stack-rhos "" "${RL_ARGS[@]}" \
    --seed-bases 0,1000 \
    --out "$PROJECT_ROOT/outputs/study_aes_seeds_matrix.json" \
    || { log "FATAL: eval failed"; exit 1; }

log "=== AES-SEEDS done ==="
