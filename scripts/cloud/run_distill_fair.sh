#!/bin/bash
# Runs ON the box (80GB). Fair-scale guidance distillation for the CLIP/GenEval
# setting. The old probe (200 teacher samples, 1 seed, rho-only teacher) banked
# <10% of the guidance gain; this gives distillation a real shot:
#   1. Teacher: TFG rho=20 + mu=20 (the paper's best guidance recipe), 28 steps,
#      N_SAMPLES (default 5000) GenEval-style prompts from flow_grpo's 50k pool,
#      eval prompts excluded. Resumable.
#   2. Student: 3 seeds x WALL_HOURS of flow-matching LoRA distillation (the
#      same wall budget as the RL and DRaFT arms).
#   3. Eval: base + 3 students (+ student+TFG rho20 stacks) on 150 prompts x 3
#      eval seeds, paired with the pub matrices.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_distill_fair.sh [WALL_HOURS] [N_SAMPLES] [LR]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"
WALL_HOURS="${1:-4}"; N_SAMPLES="${2:-5000}"; LR="${3:-1e-4}"
SEEDS="42 1042 2042"
STACK_RHO="${DISTILL_STACK_RHO:-20}"
log "=== DISTILL-FAIR start: ${N_SAMPLES} teacher samples (rho20+mu20), 3 x ${WALL_HOURS}h students, lr $LR ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

if [ ! -f .remote_setup_done ]; then
  HF_TOKEN="$HF_TOKEN" bash scripts/cloud/remote_setup.sh && touch .remote_setup_done || { log "FATAL: remote_setup failed"; exit 1; }
fi
mkdir -p outputs/checkpoints logs repos
[ -f repos/flow_grpo/dataset/geneval/train_metadata.jsonl ] || \
  git clone --quiet --depth 1 https://github.com/yifan123/flow_grpo repos/flow_grpo || { log "FATAL: clone"; exit 1; }

# eval prompts (150) = the pub set; excluded from the teacher pool
EVAL_PROMPTS="outputs/eval_prompts_pub.txt"
python3 - scripts/cloud/prompts_smoke.json scripts/cloud/prompts_train.json "$EVAL_PROMPTS" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY

# --- 1. Teacher latents (resumable) ---
DS="outputs/distill_fair_dataset"
HAVE=$(python3 -c "import json,os;print(len(json.load(open('$DS/manifest.json'))['items']) if os.path.exists('$DS/manifest.json') else 0)")
if [ "$HAVE" -lt "$N_SAMPLES" ]; then
  log ">>> teacher generation: have $HAVE / $N_SAMPLES"
  python3 scripts/cloud/generate_tfg_dataset.py \
    --prompts_file repos/flow_grpo/dataset/geneval/train_metadata.jsonl \
    --exclude_file "$EVAL_PROMPTS" --max_samples "$N_SAMPLES" \
    --output_dir "$DS" --tfg_rho 20 --tfg_mu 20 --tfg_every_n 1 --apply_range 0.2,0.8 \
    --height 512 --width 512 --num_inference_steps 28 --seed_base 100000 \
    2>&1 | grep -vE "it/s\]|s/it\]" | tee logs/distill_fair_teacher.log || { log "FATAL: teacher gen failed"; exit 1; }
else
  log ">>> teacher dataset complete ($HAVE samples)"
fi

# --- 2. Students ---
for SEED in $SEEDS; do
  OUT="outputs/checkpoints/distill_fair_s${SEED}"
  [ -f "$OUT/summary.json" ] && { log ">>> seed $SEED trained — skipping"; continue; }
  log ">>> student seed $SEED (${WALL_HOURS}h)"
  python3 scripts/train_distillation.py \
    --config configs/base_config.yaml --manifest "$DS/manifest.json" \
    --output_dir "$OUT" --seed "$SEED" --wall_hours "$WALL_HOURS" --max_steps 0 \
    --batch_size 8 --learning_rate "$LR" --lora_rank 64 --lora_alpha 128 --lora_fp32 \
    --save_every 0 --save_every_minutes 30 --log_every 20 --no_wandb \
    2>&1 | grep -vE "it/s\]|s/it\]" | tee -a "logs/distill_fair_s${SEED}.log" || log "WARN: seed $SEED nonzero"
done

# --- 3. Eval ---
RL_ARGS=(); STACK_ARGS=()
for SEED in $SEEDS; do
  OUT="outputs/checkpoints/distill_fair_s${SEED}"
  FINAL=$(ls "$OUT" 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1)
  [ -n "$FINAL" ] && [ -d "$OUT/$FINAL/lora" ] || { log "WARN: no student for seed $SEED"; continue; }
  python3 scripts/cloud/verify_lora_load.py "$OUT/$FINAL/lora" 2>&1 | tail -2 || log "WARN: verify failed"
  RL_ARGS+=(--rl-lora "distill_s${SEED}=$OUT/$FINAL/lora")
  [ -n "$STACK_RHO" ] && STACK_ARGS+=(--stack "distillstack_s${SEED}_rho${STACK_RHO}=$OUT/$FINAL/lora:$STACK_RHO")
done
[ ${#RL_ARGS[@]} -gt 0 ] || { log "FATAL: no students"; exit 1; }
log ">>> eval: base + ${#RL_ARGS[@]} students + ${#STACK_ARGS[@]} stacks, 150 prompts x 3 seeds"
python3 scripts/cloud/eval_study_matrix.py \
  --reward clip --prompts-file "$EVAL_PROMPTS" --num-prompts 150 --num-steps 40 \
  --ckpt-dir outputs/checkpoints --tfg-rhos "" --stack-rhos "" \
  "${RL_ARGS[@]}" "${STACK_ARGS[@]}" --seed-bases 0,1000,2000 \
  --image-dir outputs/study_clip_distill_images \
  --out outputs/study_clip_distill_matrix.json || { log "FATAL: eval failed"; exit 1; }
python3 scripts/cloud/external_metric.py --image-root outputs/study_clip_distill_images \
  --prompts-file "$EVAL_PROMPTS" --num-prompts 150 --metrics siglip \
  --out outputs/study_clip_distill_external.json || log "WARN: external metrics failed"

# --- 4. Bundle (teacher manifest + students + logs; latents stay on the box) ---
B=("distill_fair_dataset/manifest.json")
for SEED in $SEEDS; do
  OUT="distill_fair_s${SEED}"; FINAL=$(ls "outputs/checkpoints/$OUT" 2>/dev/null | grep '^checkpoint-' | sort -t- -k2 -n | tail -1)
  [ -n "$FINAL" ] && B+=("checkpoints/$OUT/$FINAL/lora" "checkpoints/$OUT/summary.json" "checkpoints/$OUT/loss_curve.json")
done
tar czf outputs/distill_fair_bundle.tar.gz -C outputs "${B[@]}" || log "WARN: bundle failed"
log "=== DISTILL-FAIR done ==="
