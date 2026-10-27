#!/bin/bash
# Runs ON the Lambda box (80GB GPU). The DRaFT arm for ONE reward setting,
# publication grade, mirroring run_pub_reward.sh:
#   1. THREE independent DRaFT-1 LoRA trainings (seeds 42/1042/2042),
#      WALL_HOURS each — the matched regime FlowGRPO used.
#   2. Eval matrix on 150 prompts x 3 eval seeds: base + the three DRaFT
#      checkpoints (+ optional TFG-stacked arms, + optional TFG-alone arms).
#      Base/TFG/GRPO numbers for this reward already exist in
#      outputs/study_<key>_pub_matrix.json with the same seed bases, so the
#      matrix is paired with them.
#   3. External metrics (SigLIP; EasyOCR on the OCR set); tar the images.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_draft_reward.sh <clip|clip_ocr|pickscore|aesthetic> [WALL_HOURS] [LR] ["SEEDS"] [SEEDBASES] [CKPT_ARMS] [TAG] [SAVE_MIN] [KL]
# Env:   DRAFT_STACK_RHO=20   add TFG rho=20 on top of each DRaFT seed (composition)
#        DRAFT_TFG_RHOS=20    also re-run TFG-alone arms in this matrix
#        DRAFT_GRAD_STEPS=1   DRaFT-K depth
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?HF_TOKEN required}"
KEY="${1:?usage: run_draft_reward.sh <clip|clip_ocr|pickscore|aesthetic> [WALL_HOURS] [LR]}"
WALL_HOURS="${2:-4}"
LR="${3:-1e-4}"
SEEDS="${4:-42 1042 2042}"
EVAL_SEEDBASES="${5:-0,1000,2000}"   # 0 alone = cheap 150x1 selection eval
CKPT_ARMS="${6:-final,best}"         # final,best | all (every saved checkpoint) | final
TAG="${7:-}"                         # suffix for checkpoint dirs and output files (parallel recipe runs)
SAVE_MIN="${8:-30}"                  # checkpoint cadence (minutes)
KL="${9:-0}"                         # velocity-KL coefficient (0 = DRaFT, no regulariser)
STACK_RHO="${DRAFT_STACK_RHO:-}"
TFG_RHOS="${DRAFT_TFG_RHOS:-}"
GRAD_STEPS="${DRAFT_GRAD_STEPS:-1}"

case "$KEY" in
  clip)      REWARD=clip;      PROMPTS_SPEC=ddrl; EXT_METRICS="siglip" ;;
  clip_ocr)  REWARD=clip;      PROMPTS_SPEC=ocr;  EXT_METRICS="siglip,ocr" ;;
  pickscore) REWARD=pickscore; PROMPTS_SPEC=ddrl; EXT_METRICS="siglip" ;;
  aesthetic) REWARD=aesthetic; PROMPTS_SPEC=ddrl; EXT_METRICS="siglip" ;;
  *) log "unknown key $KEY"; exit 1 ;;
esac

log "=== DRAFT[$KEY] start: 3 seeds x ${WALL_HOURS}h, lr $LR, K=$GRAD_STEPS + 150-prompt matrix ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

if [ ! -f .remote_setup_done ]; then
  HF_TOKEN="$HF_TOKEN" bash scripts/cloud/remote_setup.sh && touch .remote_setup_done \
    || { log "FATAL: remote_setup failed"; exit 1; }
fi
mkdir -p outputs/checkpoints logs
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
if [ "${GPU_MB:-0}" -lt 60000 ]; then
  MEM_ARGS="--t5_cpu --batch_size 2 --grad_accum 2"; log "small GPU (${GPU_MB}MB): $MEM_ARGS"
else
  MEM_ARGS="--batch_size 4 --grad_accum 1"
fi

# --- 0. Prompt files: train (what FlowGRPO trained on) and eval (150) ---
if [ "$PROMPTS_SPEC" = "ddrl" ]; then
  TRAIN_PROMPTS="scripts/cloud/prompts_smoke.json"
  EVAL_PROMPTS="outputs/eval_prompts_pub.txt"
  python3 - scripts/cloud/prompts_smoke.json scripts/cloud/prompts_train.json "$EVAL_PROMPTS" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
extra = [p for p in train if p not in set(smoke)][:120]
out = smoke + extra
pathlib.Path(sys.argv[3]).write_text("\n".join(out) + "\n")
print(f"wrote {len(out)} eval prompts (first 30 = training set, order preserved)")
PY
  NPROMPTS=150
else
  # flow_grpo ships dataset/ocr/{train,test}.txt; clone (no install) if absent.
  FG_DIR="repos/flow_grpo"
  if [ ! -f "$FG_DIR/dataset/ocr/train.txt" ]; then
    mkdir -p repos
    git clone --quiet --depth 1 https://github.com/yifan123/flow_grpo "$FG_DIR" \
      || { log "FATAL: could not clone flow_grpo for OCR prompts"; exit 1; }
  fi
  TRAIN_PROMPTS="$FG_DIR/dataset/ocr/train.txt"
  EVAL_PROMPTS="$FG_DIR/dataset/ocr/test.txt"
  AVAIL=$(grep -c . "$EVAL_PROMPTS")
  NPROMPTS=$(( AVAIL < 150 ? AVAIL : 150 ))
  log "OCR prompts: $(grep -c . "$TRAIN_PROMPTS") train, using $NPROMPTS of $AVAIL test"
fi

# --- 1. Three trainings ---
for SEED in $SEEDS; do
  OUT="outputs/checkpoints/draft_${KEY}${TAG}_s${SEED}"
  if [ -f "$OUT/summary.json" ]; then
    log ">>> seed $SEED already trained — skipping"
    continue
  fi
  log ">>> training seed $SEED (${WALL_HOURS}h)"
  python3 scripts/train_draft.py \
    --reward "$REWARD" --prompts_file "$TRAIN_PROMPTS" \
    --output_dir "$OUT" --seed "$SEED" \
    --wall_hours "$WALL_HOURS" --lr "$LR" \
    $MEM_ARGS --grad_steps "$GRAD_STEPS" --num_inference_steps 20 \
    --gradient_checkpointing --vae_slicing \
    --save_every_minutes "$SAVE_MIN" --kl_coef "$KL" --log_every 10 \
    2>&1 | tee -a "logs/draft_${KEY}${TAG}_s${SEED}.log" || log "WARN: seed $SEED exited nonzero"
done

# --- 2. Arms ---
RL_ARGS=(); STACK_ARGS=()
for SEED in $SEEDS; do
  OUT="outputs/checkpoints/draft_${KEY}${TAG}_s${SEED}"
  FINAL=$(ls "$OUT" 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1)
  [ -n "$FINAL" ] && [ -d "$OUT/$FINAL/lora" ] || { log "WARN: no checkpoint for seed $SEED"; continue; }
  python3 scripts/cloud/verify_lora_load.py "$OUT/$FINAL/lora" 2>&1 | tail -2 \
    || log "WARN: verify_lora_load failed for seed $SEED"
  RL_ARGS+=(--rl-lora "draft_s${SEED}=$OUT/$FINAL/lora")
  BEST=""
  case "$CKPT_ARMS" in
    *best*)
      # early-stopped arm: checkpoint nearest the peak of the smoothed training reward
      BEST=$(python3 scripts/cloud/pick_best_ckpt.py "$OUT" 2>>"logs/draft_${KEY}${TAG}_s${SEED}.log")
      [ -n "$BEST" ] && [ "$BEST" != "$OUT/$FINAL/lora" ] && RL_ARGS+=(--rl-lora "draft_s${SEED}_best=$BEST") ;;
    all)
      # the whole trajectory: every saved checkpoint except the final one (already an arm)
      for CK in $(ls "$OUT" | grep '^checkpoint-' | sort -t- -k2 -n); do
        [ "$CK" = "$FINAL" ] && continue
        [ -d "$OUT/$CK/lora" ] && RL_ARGS+=(--rl-lora "draft_s${SEED}_${CK}=$OUT/$CK/lora")
      done ;;
  esac
  [ -n "$STACK_RHO" ] && STACK_ARGS+=(--stack "draftstack_s${SEED}_rho${STACK_RHO}=$OUT/$FINAL/lora:$STACK_RHO")
  log "seed $SEED arms: final $FINAL, best ${BEST:-n/a}"
done
[ ${#RL_ARGS[@]} -gt 0 ] || { log "FATAL: no DRaFT checkpoints to evaluate"; exit 1; }

# --- 3. Eval matrix (paired seed bases with the pub matrices) ---
log ">>> eval matrix ($REWARD, $NPROMPTS prompts x seedbases[$EVAL_SEEDBASES]; $(( ${#RL_ARGS[@]} / 2 )) draft arms + $(( ${#STACK_ARGS[@]} / 2 )) stacks; tfg='${TFG_RHOS}')"
python3 scripts/cloud/eval_study_matrix.py \
  --reward "$REWARD" --prompts-file "$EVAL_PROMPTS" \
  --num-prompts "$NPROMPTS" --num-steps 40 \
  --ckpt-dir outputs/checkpoints \
  --tfg-rhos "$TFG_RHOS" --stack-rhos "" \
  "${RL_ARGS[@]}" "${STACK_ARGS[@]}" \
  --seed-bases "$EVAL_SEEDBASES" \
  --image-dir "outputs/study_${KEY}_draft${TAG}_images" \
  --out "outputs/study_${KEY}_draft${TAG}_matrix.json" \
  || { log "FATAL: eval matrix failed"; exit 1; }

# --- 4. External metrics (tolerant) ---
log ">>> external metrics ($EXT_METRICS)"
echo "$EXT_METRICS" | grep -q ocr && { python3 -m pip install --quiet easyocr || log "WARN: easyocr install failed"; }
python3 scripts/cloud/external_metric.py \
  --image-root "outputs/study_${KEY}_draft${TAG}_images" \
  --prompts-file "$EVAL_PROMPTS" --num-prompts "$NPROMPTS" \
  --metrics "$EXT_METRICS" \
  --out "outputs/study_${KEY}_draft${TAG}_external.json" \
  || log "WARN: external metrics failed (matrix unaffected)"

# --- 5. Tar images ---
tar czf "outputs/study_${KEY}_draft${TAG}_images.tar.gz" -C outputs "study_${KEY}_draft${TAG}_images" || log "WARN: tar failed"
# --- 6. Bundle final LoRA per seed + train logs (the pull is files-only) ---
BUNDLE=()
for SEED in $SEEDS; do
  OUT="draft_${KEY}${TAG}_s${SEED}"
  FINAL=$(ls "outputs/checkpoints/$OUT" 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1)
  [ -n "$FINAL" ] && BUNDLE+=("$OUT/$FINAL/lora" "$OUT/train_log.jsonl" "$OUT/summary.json" "$OUT/args.json")
  BESTDIR=$(python3 scripts/cloud/pick_best_ckpt.py "outputs/checkpoints/$OUT" 2>/dev/null | sed "s#^outputs/checkpoints/##")
  [ -n "$BESTDIR" ] && [ "$BESTDIR" != "$OUT/$FINAL/lora" ] && BUNDLE+=("$BESTDIR")
done
[ ${#BUNDLE[@]} -gt 0 ] && tar czf "outputs/draft_${KEY}${TAG}_ckpts.tar.gz" -C outputs/checkpoints "${BUNDLE[@]}" \
  || log "WARN: checkpoint bundle failed"
log "=== DRAFT[$KEY$TAG] done ==="
