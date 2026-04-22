#!/bin/bash
# Narrow re-eval: Cell C' + Cell D' only, with the fixed PeftModel loader.
# Assumes:
#   - outputs/checkpoints/distilled/final/ already on the instance (from prior run)
#   - Base eval (Cell A' = 0.268352) already measured; we don't redo it.

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date +%H:%M:%S)] $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=outputs/checkpoints/distilled/final

# Stage 0: pre-flight LoRA load verification. ~45s. If this fails, the eval
# would produce bogus numbers (we hit this last run); abort before burning $.
log "=== Stage 0: LoRA load verification ==="
python3 scripts/cloud/verify_lora_load.py "$CKPT" 2>&1 | tee verify_lora.log || {
    log "FATAL: LoRA load verification failed — aborting"; exit 1;
}
log "=== Stage 0 OK ==="

# Stage 3b (renamed): Cells C' (no TFG) + D' (TFG ρ=20) with distilled LoRA.
log "=== Stage 3b re-eval: Cell C' and D' with distilled LoRA (fixed loader) ==="
python3 scripts/cloud/run_rho_sweep.py \
    --output_dir outputs/distill_eval_student_fixed \
    --checkpoint "$CKPT" \
    --rhos 20 \
    --apply_every_n 1 \
    --height 512 --width 512 --num_inference_steps 28 \
    --text_encoders_cpu --vae_tiling 2>&1 | tee eval_student_fixed.log
log "=== Stage 3b done ==="
log "=== RE-EVAL COMPLETE ==="
