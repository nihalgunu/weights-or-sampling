#!/bin/bash
# Runs ON the box. Training-budget crossover experiment (Haotian's
# "under what situation should training win?" question), CLIP/DDRL setting.
#
# Role a: RL at 8h wall x 2 seeds (42, 1042) + eval {base, tfg rho20, both RL}
# Role b: RL at 16h wall x 1 seed (42)      + eval {base, tfg rho20, RL}
#         + PickScore rho-ceiling arms {160, 320}
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_xover.sh <a|b>

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?HF_TOKEN required}"
ROLE="${1:?usage: run_xover.sh <a|b|c>}"
export FG_GROUP=8 FG_NBPE=8

SAVE_DIR=repos/flow_grpo/logs/clip_score/sd3.5-M-1gpu-lora
CKPT_NAME=flowgrpo_clip

train_one() { # budget_hours seed tag
  local HOURS=$1 SEED=$2 TAG=$3
  local DEST="$PROJECT_ROOT/outputs/checkpoints/${CKPT_NAME}_${TAG}"
  if [ -d "$DEST" ] && [ -n "$(ls "$DEST" 2>/dev/null | grep '^checkpoint-')" ]; then
    log ">>> $TAG already trained; skipping"; return 0
  fi
  log ">>> training $TAG (${HOURS}h, seed $SEED)"
  rm -rf "$PROJECT_ROOT/$SAVE_DIR" "$PROJECT_ROOT/outputs/checkpoints/$CKPT_NAME"
  HF_TOKEN="$HF_TOKEN" FG_GROUP=8 FG_NBPE=8 FG_SEED=$SEED \
      bash scripts/cloud/run_flowgrpo_clip.sh "$HOURS" || log "WARN: $TAG training exited nonzero"
  [ -d "$PROJECT_ROOT/outputs/checkpoints/$CKPT_NAME" ] && \
      mv "$PROJECT_ROOT/outputs/checkpoints/$CKPT_NAME" "$DEST"
  log "$TAG: $(ls "$DEST" 2>/dev/null | grep -c '^checkpoint-') checkpoints"
}

final_lora() { # tag -> echoes lora path of final ckpt
  local D="$PROJECT_ROOT/outputs/checkpoints/${CKPT_NAME}_$1"
  local F=$(ls "$D" 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1)
  [ -n "$F" ] && echo "$D/$F/lora"
}

# prompts (identical construction to pub run)
PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" \
          "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY

if [ "$ROLE" = "a" ]; then
  train_one 8 42 8h_s42
  train_one 8 1042 8h_s1042
  L1=$(final_lora 8h_s42); L2=$(final_lora 8h_s1042)
  RL_ARGS=()
  [ -n "$L1" ] && RL_ARGS+=(--rl-lora "rl_8h_s42=$L1")
  [ -n "$L2" ] && RL_ARGS+=(--rl-lora "rl_8h_s1042=$L2")
  log ">>> eval (8h arms)"
  python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
      --reward clip --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
      --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
      --tfg-rhos 20 --stack-rhos "" "${RL_ARGS[@]}" \
      --seed-bases 0,1000,2000 \
      --out "$PROJECT_ROOT/outputs/study_xover8h_matrix.json" \
      || { log "FATAL: eval failed"; exit 1; }
elif [ "$ROLE" = "c" ]; then
  train_one 32 42 32h_s42
  L=$(final_lora 32h_s42)
  RL_ARGS=(); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_32h_s42=$L")
  log ">>> eval (32h arm)"
  python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
      --reward clip --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
      --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
      --tfg-rhos 20 --stack-rhos "" "${RL_ARGS[@]}" \
      --seed-bases 0,1000,2000 \
      --out "$PROJECT_ROOT/outputs/study_xover32h_matrix.json" \
      || { log "FATAL: eval failed"; exit 1; }
else
  train_one 16 42 16h_s42
  L=$(final_lora 16h_s42)
  RL_ARGS=(); [ -n "$L" ] && RL_ARGS+=(--rl-lora "rl_16h_s42=$L")
  log ">>> eval (16h arm)"
  python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
      --reward clip --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
      --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
      --tfg-rhos 20 --stack-rhos "" "${RL_ARGS[@]}" \
      --seed-bases 0,1000,2000 \
      --out "$PROJECT_ROOT/outputs/study_xover16h_matrix.json" \
      || { log "FATAL: eval failed"; exit 1; }
  log ">>> PickScore rho ceiling {160, 320}"
  mkdir -p "$PROJECT_ROOT/outputs/empty_ckpts"
  python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
      --reward pickscore --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
      --ckpt-dir "$PROJECT_ROOT/outputs/empty_ckpts" \
      --tfg-rhos 160,320 --stack-rhos "" \
      --seed-bases 0,1000,2000 \
      --out "$PROJECT_ROOT/outputs/study_pickscore_rhoceil_matrix.json" \
      || log "WARN: rho ceiling eval failed"
fi

log "=== XOVER[$ROLE] done ==="
