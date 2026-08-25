#!/bin/bash
# Runs ON the Lambda box (80GB GPU). Publication-grade study for ONE reward:
#   1. THREE independent FlowGRPO group=8 trainings (FG_SEED 42/1042/2042),
#      4h wall each — the established matched regime, now with training-seed
#      replication.
#   2. Eval matrix on 150 prompts x 3 eval seeds: base, TFG (headline rho),
#      RL final ckpt per training seed, TFG-stacked per training seed.
#   3. External metrics (SigLIP everywhere; EasyOCR text-match on the OCR set).
#   4. Tar the full image tree for later GenEval / human eval.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_pub_reward.sh <clip|clip_ocr|pickscore> [WALL_HOURS_PER_SEED]

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?HF_TOKEN required}"

KEY="${1:?usage: run_pub_reward.sh <clip|clip_ocr|pickscore> [WALL_HOURS_PER_SEED]}"
WALL_HOURS="${2:-4}"
export FG_GROUP=8 FG_NBPE=8
SEEDS="42 1042 2042"

case "$KEY" in
  clip)
    TRAIN_SH=scripts/cloud/run_flowgrpo_clip.sh
    CKPT_NAME=flowgrpo_clip
    SAVE_DIR=repos/flow_grpo/logs/clip_score/sd3.5-M-1gpu-lora
    EVAL_REWARD=clip; PROMPTS_SPEC=ddrl
    TFG_RHOS="20"; STACK_RHO=20; EXTRA_STACK=""
    EXT_METRICS="siglip"
    ;;
  clip_ocr)
    TRAIN_SH=scripts/cloud/run_flowgrpo_clip_ocr.sh
    CKPT_NAME=flowgrpo_clip_ocr
    SAVE_DIR=repos/flow_grpo/logs/clip_score_ocr/sd3.5-M-1gpu-lora
    EVAL_REWARD=clip; PROMPTS_SPEC=ocr
    TFG_RHOS="20"; STACK_RHO=20; EXTRA_STACK=""
    EXT_METRICS="siglip,ocr"
    ;;
  pickscore)
    TRAIN_SH=scripts/cloud/run_flowgrpo_pickscore.sh
    CKPT_NAME=flowgrpo_pickscore
    SAVE_DIR=repos/flow_grpo/logs/pickscore/sd3.5-M-1gpu-lora
    EVAL_REWARD=pickscore; PROMPTS_SPEC=ddrl
    TFG_RHOS="20,80"; STACK_RHO=80; EXTRA_STACK="rho20_s42"
    EXT_METRICS="siglip"
    ;;
  *) log "unknown key $KEY"; exit 1 ;;
esac

log "=== PUB[$KEY] start: 3 training seeds x ${WALL_HOURS}h + 150-prompt matrix ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

# --- 1. Three trainings ---
for SEED in $SEEDS; do
  SEED_DIR="$PROJECT_ROOT/outputs/checkpoints/${CKPT_NAME}_s${SEED}"
  if [ -d "$SEED_DIR" ] && [ -n "$(ls "$SEED_DIR" 2>/dev/null | grep '^checkpoint-')" ]; then
    log ">>> seed $SEED already trained ($(ls "$SEED_DIR" | grep -c '^checkpoint-') ckpts) — skipping"
    continue
  fi
  log ">>> training seed $SEED"
  rm -rf "$PROJECT_ROOT/$SAVE_DIR" "$PROJECT_ROOT/outputs/checkpoints/$CKPT_NAME"
  HF_TOKEN="$HF_TOKEN" FG_GROUP=8 FG_NBPE=8 FG_SEED=$SEED \
      bash "$TRAIN_SH" "$WALL_HOURS" || log "WARN: seed $SEED training exited nonzero"
  if [ -d "$PROJECT_ROOT/outputs/checkpoints/$CKPT_NAME" ]; then
    mv "$PROJECT_ROOT/outputs/checkpoints/$CKPT_NAME" "$SEED_DIR"
    log "seed $SEED: $(ls "$SEED_DIR" | grep -c '^checkpoint-') checkpoints -> $SEED_DIR"
  else
    log "WARN: seed $SEED produced no checkpoint dir"
  fi
done

# --- 2. Prompts file (150) ---
if [ "$PROMPTS_SPEC" = "ddrl" ]; then
  PROMPTS_FILE="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
  python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" \
            "$PROJECT_ROOT/scripts/cloud/prompts_train.json" "$PROMPTS_FILE" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
extra = [p for p in train if p not in set(smoke)][:120]
out = smoke + extra
pathlib.Path(sys.argv[3]).write_text("\n".join(out) + "\n")
print(f"wrote {len(out)} prompts (first 30 = original eval set, order preserved)")
PY
  NPROMPTS=150
else
  PROMPTS_FILE="$PROJECT_ROOT/repos/flow_grpo/dataset/ocr/test.txt"
  [ -f "$PROMPTS_FILE" ] || { log "FATAL: $PROMPTS_FILE missing"; exit 1; }
  AVAIL=$(grep -c . "$PROMPTS_FILE")
  NPROMPTS=$(( AVAIL < 150 ? AVAIL : 150 ))
  log "OCR prompts available: $AVAIL, using $NPROMPTS"
fi

# --- 3. Build arm args: final ckpt per seed ---
RL_ARGS=(); STACK_ARGS=()
FIRST_SEED_LORA=""
for SEED in $SEEDS; do
  SEED_DIR="$PROJECT_ROOT/outputs/checkpoints/${CKPT_NAME}_s${SEED}"
  FINAL=$(ls "$SEED_DIR" 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1)
  if [ -z "$FINAL" ]; then log "WARN: no ckpt for seed $SEED — skipping its arms"; continue; fi
  LORA="$SEED_DIR/$FINAL/lora"
  [ -d "$LORA" ] || { log "WARN: $LORA missing — skipping seed $SEED"; continue; }
  [ -z "$FIRST_SEED_LORA" ] && FIRST_SEED_LORA="$LORA"
  RL_ARGS+=(--rl-lora "rl_s${SEED}=$LORA")
  STACK_ARGS+=(--stack "stack_s${SEED}_rho${STACK_RHO}=$LORA:$STACK_RHO")
  log "seed $SEED arm: $FINAL"
done
if [ "$EXTRA_STACK" = "rho20_s42" ] && [ -n "$FIRST_SEED_LORA" ]; then
  STACK_ARGS+=(--stack "stack_s42_rho20=$FIRST_SEED_LORA:20")
fi

# --- 4. Eval matrix ---
log ">>> eval matrix ($EVAL_REWARD, $NPROMPTS prompts, arms: base + tfg{$TFG_RHOS} + ${#RL_ARGS[@]}/2 RL seeds + stacks)"
python3 "$PROJECT_ROOT/scripts/cloud/eval_study_matrix.py" \
    --reward "$EVAL_REWARD" \
    --prompts-file "$PROMPTS_FILE" \
    --num-prompts "$NPROMPTS" --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos "$TFG_RHOS" --stack-rhos "" \
    "${RL_ARGS[@]}" "${STACK_ARGS[@]}" \
    --seed-bases 0,1000,2000 \
    --image-dir "$PROJECT_ROOT/outputs/study_${KEY}_pub_images" \
    --out "$PROJECT_ROOT/outputs/study_${KEY}_pub_matrix.json" \
    || { log "FATAL: eval matrix failed"; exit 1; }

# --- 5. External metrics (tolerant) ---
log ">>> external metrics ($EXT_METRICS)"
if echo "$EXT_METRICS" | grep -q ocr; then
    python3 -m pip install --quiet easyocr || log "WARN: easyocr install failed"
fi
python3 "$PROJECT_ROOT/scripts/cloud/external_metric.py" \
    --image-root "$PROJECT_ROOT/outputs/study_${KEY}_pub_images" \
    --prompts-file "$PROMPTS_FILE" --num-prompts "$NPROMPTS" \
    --metrics "$EXT_METRICS" \
    --out "$PROJECT_ROOT/outputs/study_${KEY}_pub_external.json" \
    || log "WARN: external metrics failed (matrix result unaffected)"

# --- 6. Tar images for pull ---
log ">>> tarring images"
tar czf "$PROJECT_ROOT/outputs/study_${KEY}_pub_images.tar.gz" \
    -C "$PROJECT_ROOT/outputs" "study_${KEY}_pub_images" \
    || log "WARN: tar failed"

log "=== PUB[$KEY] done ==="
