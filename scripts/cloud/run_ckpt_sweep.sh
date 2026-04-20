#!/bin/bash
# Training-compute scaling: evaluate Cell C (distilled, no TFG) and Cell D
# (distilled + TFG ρ=20) for multiple training-compute checkpoints of the
# same teacher-ρ=10 distillation. Results go to outputs/ckpt_sweep/ckpt_<N>/.
# Each checkpoint evaluated in its own run_rho_sweep.py invocation. Preamble:
# verify_lora_load.py on each checkpoint before the long eval.

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date +%H:%M:%S)] $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STEPS_LIST=(100 200 300 400 500)
BASE_CKPT_DIR=outputs/checkpoints/distilled

for STEPS in "${STEPS_LIST[@]}"; do
    if [ "$STEPS" = "500" ]; then
        CKPT="$BASE_CKPT_DIR/final"
    else
        CKPT="$BASE_CKPT_DIR/checkpoint-$STEPS"
    fi
    if [ ! -d "$CKPT" ]; then
        log "WARN: $CKPT missing — skipping"
        continue
    fi

    OUT="outputs/ckpt_sweep/ckpt_$STEPS"
    log "=== Stage 0: verify LoRA at $CKPT ==="
    python3 scripts/cloud/verify_lora_load.py "$CKPT" 2>&1 | tee "verify_ckpt_$STEPS.log" || {
        log "FATAL: verify failed at ckpt $STEPS — skipping"; continue;
    }

    log "=== Stage 3 eval at $STEPS training steps ==="
    python3 scripts/cloud/run_rho_sweep.py \
        --output_dir "$OUT" \
        --checkpoint "$CKPT" \
        --rhos 20 \
        --apply_every_n 1 \
        --height 512 --width 512 --num_inference_steps 28 \
        --text_encoders_cpu --vae_tiling 2>&1 | tee "eval_ckpt_$STEPS.log" || {
        log "WARN: eval failed at ckpt $STEPS — continuing"; continue;
    }
    log "=== ckpt $STEPS done ==="
done

log "=== CKPT SWEEP COMPLETE ==="
