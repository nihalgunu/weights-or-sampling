#!/bin/bash
# Runs ON the box. Rollout group-size curve at beta=0: g=2 and g=4 (4h each,
# seed 42, CLIP setting) + 150-prompt eval. Completes the "training helps only
# above a rollout-information threshold" axis with g={2,4,8} at fair beta.
# Usage: HF_TOKEN=... bash scripts/cloud/run_group.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"
for G in 2 4; do
  DEST="$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip_g${G}_b0_s42"
  if [ ! -d "$DEST" ]; then
    log ">>> training group=$G beta=0"
    rm -rf "$PROJECT_ROOT/repos/flow_grpo/logs/clip_score" "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip"
    HF_TOKEN="$HF_TOKEN" FG_GROUP=$G FG_NBPE=8 FG_SEED=42 FG_BETA=0 \
        bash scripts/cloud/run_flowgrpo_clip.sh 4 || log "WARN: g$G nonzero"
    [ -d "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip" ] && \
        mv "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_clip" "$DEST"
  fi
done
GP="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$GP" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY
lora_of() { ls -d "$PROJECT_ROOT"/outputs/checkpoints/$1/checkpoint-*/lora 2>/dev/null | sort -t- -k2 -V | tail -1; }
RL_ARGS=()
L=$(lora_of flowgrpo_clip_g2_b0_s42); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_g2_b0=$L")
L=$(lora_of flowgrpo_clip_g4_b0_s42); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_g4_b0=$L")
log ">>> group-curve eval"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$GP" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos "" --stack-rhos "" "${RL_ARGS[@]}" --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/study_group_matrix.json" \
    || { log "FATAL: group eval failed"; exit 1; }
log "=== GROUP done ==="
