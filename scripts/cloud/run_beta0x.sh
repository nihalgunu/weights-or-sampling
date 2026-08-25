#!/bin/bash
# Runs ON the box. Closes the "OCR and ensemble not retrained at beta=0"
# limitation + adds a 2nd seed for the beta0 16h crossover point.
#   role a: OCR beta=0 seed42 4h + ensemble beta=0 seed42 4h + eval both
#   role b: CLIP beta=0 seed1042 16h + eval
# Usage: HF_TOKEN=... bash scripts/cloud/run_beta0x.sh <a|b>
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"; ROLE="${1:?a|b}"
export FG_GROUP=8 FG_NBPE=8

train() { # runner hours seed dest logdir srcdir
  local RUNNER=$1 HOURS=$2 SEED=$3 DEST=$4 LOGDIR=$5 SRC=$6
  if [ -d "$PROJECT_ROOT/outputs/checkpoints/$DEST" ] && [ -n "$(ls "$PROJECT_ROOT/outputs/checkpoints/$DEST" 2>/dev/null | grep '^checkpoint-')" ]; then
    log ">>> $DEST already trained"; return 0; fi
  log ">>> training $DEST"
  rm -rf "$PROJECT_ROOT/repos/flow_grpo/logs/$LOGDIR" "$PROJECT_ROOT/outputs/checkpoints/$SRC"
  HF_TOKEN="$HF_TOKEN" FG_GROUP=8 FG_NBPE=8 FG_SEED=$SEED FG_BETA=0 \
      bash scripts/cloud/$RUNNER "$HOURS" || log "WARN: nonzero"
  [ -d "$PROJECT_ROOT/outputs/checkpoints/$SRC" ] && mv "$PROJECT_ROOT/outputs/checkpoints/$SRC" "$PROJECT_ROOT/outputs/checkpoints/$DEST"
  log "$DEST: $(ls "$PROJECT_ROOT/outputs/checkpoints/$DEST" 2>/dev/null | grep -c '^checkpoint-') ckpts"
}
lora_of() { ls -d "$PROJECT_ROOT"/outputs/checkpoints/$1/checkpoint-*/lora 2>/dev/null | sort -t- -k2 -V | tail -1; }

PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY

if [ "$ROLE" = "a" ]; then
  train run_flowgrpo_clip_ocr.sh 4 42 flowgrpo_ocr_b0_s42 clip_score_ocr flowgrpo_clip_ocr
  train run_flowgrpo_ensemble.sh 4 42 flowgrpo_ens_b0_s42 ensemble flowgrpo_ensemble
  OCR_FILE="$PROJECT_ROOT/repos/flow_grpo/dataset/ocr/test.txt"
  L1=$(lora_of flowgrpo_ocr_b0_s42); L2=$(lora_of flowgrpo_ens_b0_s42)
  log ">>> eval OCR b0"
  [ -n "$L1" ] && python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$OCR_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" --tfg-rhos "" --stack-rhos "" \
    --rl-lora "rl_ocr_b0=$L1" --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/study_beta0_ocr_matrix.json" || log "WARN ocr eval"
  log ">>> eval ensemble b0"
  [ -n "$L2" ] && python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward ensemble --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" --tfg-rhos "" --stack-rhos "" \
    --rl-lora "rl_ens_b0=$L2" --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/study_beta0_ens_matrix.json" || log "WARN ens eval"
else
  train run_flowgrpo_clip.sh 16 1042 flowgrpo_clip_b0_16h_s1042 clip_score flowgrpo_clip
  L=$(lora_of flowgrpo_clip_b0_16h_s1042)
  log ">>> eval 16h b0 s1042"
  [ -n "$L" ] && python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward clip --prompts-file "$PROMPTS_FILE" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" --tfg-rhos "" --stack-rhos "" \
    --rl-lora "rl_b0_16h_s1042=$L" --seed-bases 0,1000,2000 \
    --out "$PROJECT_ROOT/outputs/study_beta0_16h_s1042_matrix.json" || log "WARN eval"
fi
log "=== BETA0X[$ROLE] done ==="
