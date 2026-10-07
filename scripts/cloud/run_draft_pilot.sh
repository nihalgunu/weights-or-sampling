#!/bin/bash
# Runs ON the Lambda box (80GB GPU). DRaFT learning-rate pilot on CLIP.
#
# Why: the earlier reward-gradient run (train_ddrl.py, lr 1e-5, velocity-KL
# 0.05) did not move CLIP. Before spending 3 seeds x 4h per reward we need the
# lr that actually trains. Three short runs, one eval, ~2h total.
#
#   1. remote_setup.sh (deps + SD3.5-M cache) if not already done
#   2. train_draft.py on the 30 FlowGRPO training prompts, PILOT_MINUTES each,
#      lr in {3e-5, 1e-4, 3e-4}, seed 42, batch 4, DRaFT-1
#   3. eval_study_matrix.py: base + the three DRaFT checkpoints on the first
#      30 prompts, 1 eval seed (paired with every prior run's seed base 0)
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_draft_pilot.sh [PILOT_MINUTES] [LRS]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?HF_TOKEN required}"
PILOT_MINUTES="${1:-25}"
LRS="${2:-3e-5 1e-4 3e-4}"
WALL_HOURS=$(python3 -c "print($PILOT_MINUTES/60)")

log "=== DRaFT pilot: CLIP, ${PILOT_MINUTES} min per lr, lrs: $LRS ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

if [ ! -f .remote_setup_done ]; then
  HF_TOKEN="$HF_TOKEN" bash scripts/cloud/remote_setup.sh && touch .remote_setup_done \
    || { log "FATAL: remote_setup failed"; exit 1; }
fi

mkdir -p outputs/checkpoints logs
# 40GB boxes: T5 on CPU, batch 2 x accum 2 (same effective batch as 80GB's batch 4).
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
if [ "${GPU_MB:-0}" -lt 60000 ]; then
  MEM_ARGS="--t5_cpu --batch_size 2 --grad_accum 2"; log "small GPU (${GPU_MB}MB): $MEM_ARGS"
else
  MEM_ARGS="--batch_size 4 --grad_accum 1"
fi
RL_ARGS=()
for LR in $LRS; do
  TAG="draft_pilot_clip_lr${LR}"
  OUT="outputs/checkpoints/$TAG"
  if [ -f "$OUT/summary.json" ]; then
    log ">>> $TAG already trained, skipping"
  else
    log ">>> training $TAG (${WALL_HOURS}h)"
    python3 scripts/train_draft.py \
      --reward clip --prompts_file scripts/cloud/prompts_smoke.json \
      --output_dir "$OUT" --seed 42 \
      --wall_hours "$WALL_HOURS" --lr "$LR" \
      $MEM_ARGS --grad_steps 1 --num_inference_steps 20 \
      --gradient_checkpointing --vae_slicing \
      --save_every_minutes 0 --log_every 10 \
      2>&1 | tee "logs/${TAG}.log" || log "WARN: $TAG exited nonzero"
  fi
  FINAL=$(ls "$OUT" 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1)
  if [ -n "$FINAL" ] && [ -d "$OUT/$FINAL/lora" ]; then
    python3 scripts/cloud/verify_lora_load.py "$OUT/$FINAL/lora" 2>&1 | tail -3 \
      || log "WARN: verify_lora_load failed for $TAG"
    RL_ARGS+=(--rl-lora "${TAG}=$OUT/$FINAL/lora")
    log "$TAG -> $FINAL ($(python3 -c "import json;d=json.load(open('$OUT/summary.json'));print(d['steps'],'steps, reward',d['reward_first10'],'->',d['reward_last10'])"))"
  else
    log "WARN: no checkpoint for $TAG"
  fi
done

[ ${#RL_ARGS[@]} -gt 0 ] || { log "FATAL: nothing to evaluate"; exit 1; }

log ">>> eval: base + ${#RL_ARGS[@]} DRaFT arms, 30 prompts x 1 seed"
python3 scripts/cloud/eval_study_matrix.py \
  --reward clip --prompts-file scripts/cloud/prompts_smoke.json \
  --num-prompts 30 --num-steps 40 \
  --ckpt-dir outputs/checkpoints --tfg-rhos "" --stack-rhos "" \
  "${RL_ARGS[@]}" --seed-bases 0 \
  --out outputs/study_draft_pilot_matrix.json \
  || { log "FATAL: eval failed"; exit 1; }
# Bundle train logs + summaries (small) for the pull; checkpoints stay on the box.
tar czf outputs/draft_pilot_logs.tar.gz -C outputs/checkpoints \
  $(cd outputs/checkpoints && ls -d draft_pilot_clip_lr* 2>/dev/null | sed 's#$#/train_log.jsonl#; p; s#/train_log.jsonl#/summary.json#; p; s#/summary.json#/args.json#' | sort -u) \
  || log "WARN: log tar failed"
log "=== DRaFT pilot done: outputs/study_draft_pilot_matrix.json ==="
