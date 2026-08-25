#!/bin/bash
# Runs ON the box. Reward-ensemble experiment (0.5 CLIP-L + 0.5 PickScore),
# Haotian's individual-reward vs reward-ensemble distinction.
#
# Roles t42|t1042|t2042: one FlowGRPO training run (group=8, 4h) at that seed.
# Role eval: full matrix with TFG guided on the SAME ensemble (multi-predictor),
#            expecting the three trained LoRAs uploaded by the driver at
#            outputs/checkpoints/flowgrpo_ensemble_s{42,1042,2042}/.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_ens.sh <t42|t1042|t2042|eval>

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?HF_TOKEN required}"
ROLE="${1:?usage: run_ens.sh <t42|t1042|t2042|eval>}"
export FG_GROUP=8 FG_NBPE=8

case "$ROLE" in
  t42|t1042|t2042)
    SEED="${ROLE#t}"
    DEST="$PROJECT_ROOT/outputs/checkpoints/flowgrpo_ensemble_s${SEED}"
    if [ -d "$DEST" ] && [ -n "$(ls "$DEST" 2>/dev/null | grep '^checkpoint-')" ]; then
      log "seed $SEED already trained"
    else
      rm -rf "$PROJECT_ROOT/repos/flow_grpo/logs/ensemble" "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_ensemble"
      HF_TOKEN="$HF_TOKEN" FG_GROUP=8 FG_NBPE=8 FG_SEED=$SEED \
          bash scripts/cloud/run_flowgrpo_ensemble.sh 4 || log "WARN: training exited nonzero"
      [ -d "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_ensemble" ] && \
          mv "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_ensemble" "$DEST"
      log "seed $SEED: $(ls "$DEST" 2>/dev/null | grep -c '^checkpoint-') checkpoints"
    fi
    ;;
  eval)
    log ">>> deps"
    python3 -m pip install --quiet --upgrade pip
    python3 -m pip install --quiet --upgrade --force-reinstall pillow
    python3 -m pip install --quiet 'diffusers>=0.31,<0.40' 'transformers==4.49.0' \
        'peft>=0.12' 'accelerate>=0.33' sentencepiece protobuf safetensors torchvision tqdm
    python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
    PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
    python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" \
              "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY
    RL_ARGS=(); STACK_ARGS=(); S42_LORA=""
    for SEED in 42 1042 2042; do
      D="$PROJECT_ROOT/outputs/checkpoints/flowgrpo_ensemble_s${SEED}"
      F=$(ls "$D" 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1)
      [ -n "$F" ] || { log "WARN: no ckpt for seed $SEED"; continue; }
      RL_ARGS+=(--rl-lora "rl_s${SEED}=$D/$F/lora")
      [ "$SEED" = "42" ] && S42_LORA="$D/$F/lora"
    done
    if [ -n "$S42_LORA" ]; then
      STACK_ARGS+=(--stack "stack_s42_rho20=$S42_LORA:20" --stack "stack_s42_rho80=$S42_LORA:80")
    fi
    log ">>> ensemble eval matrix"
    python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
        --reward ensemble --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
        --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
        --tfg-rhos 20,80 --stack-rhos "" "${RL_ARGS[@]}" "${STACK_ARGS[@]}" \
        --seed-bases 0,1000,2000 \
        --image-dir "$PROJECT_ROOT/outputs/study_ensemble_images" \
        --out "$PROJECT_ROOT/outputs/study_ensemble_matrix.json" \
        || { log "FATAL: ensemble eval failed"; exit 1; }
    tar czf "$PROJECT_ROOT/outputs/study_ensemble_images.tar.gz" \
        -C "$PROJECT_ROOT/outputs" study_ensemble_images || log "WARN: tar failed"
    ;;
  *) log "unknown role $ROLE"; exit 1 ;;
esac

log "=== ENS[$ROLE] done ==="
