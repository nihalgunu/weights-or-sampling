#!/bin/bash
# TFG-distillation pipeline on Lambda. Three stages:
#   0. Dry-run: 3 teacher rollouts + 3 training steps. Catches OOM and API
#      shape bugs cheaply before the long runs.
#   1. Teacher dataset: ~200 TFG-guided rollouts at ρ=10, latents saved.
#   2. Distillation training: LoRA finetune with flow-matching loss on those
#      latents, 500 steps default.
#   3. Eval: Cell A (base no TFG — regenerated fresh as reference),
#      Cell C' (distilled no TFG) and Cell D' (distilled + TFG ρ=20) on
#      the held-out 30-prompt smoke set.

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date +%H:%M:%S)] $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fast dry runs to validate the shape of both pipelines before committing.
log "=== Stage 0: dry-run teacher gen + distillation train ==="
python3 scripts/cloud/generate_tfg_dataset.py \
    --prompts_file scripts/cloud/prompts_train.json \
    --output_dir outputs/distill_dry \
    --tfg_rho 10 --tfg_every_n 1 \
    --height 512 --width 512 --num_inference_steps 28 \
    --max_prompts 3 2>&1 | tee dryrun_stage1.log || {
    log "FATAL: Stage 1 dry-run failed"; exit 1;
}
python3 scripts/train_distillation.py \
    --config configs/base_config.yaml \
    --manifest outputs/distill_dry/manifest.json \
    --output_dir outputs/distill_dry_ckpt \
    --max_steps 3 --batch_size 1 --lora_rank 32 \
    --t5_cpu --no_wandb 2>&1 | tee dryrun_stage2.log || {
    log "FATAL: Stage 2 dry-run failed"; exit 2;
}
rm -rf outputs/distill_dry outputs/distill_dry_ckpt
log "=== Stage 0 OK ==="

# Stage 1: generate full teacher dataset.
log "=== Stage 1: teacher dataset (ρ=10, 200 prompts) ==="
python3 scripts/cloud/generate_tfg_dataset.py \
    --prompts_file scripts/cloud/prompts_train.json \
    --output_dir outputs/distill_dataset \
    --tfg_rho 10 --tfg_every_n 1 \
    --height 512 --width 512 --num_inference_steps 28 2>&1 | tee gen_teachers.log || {
    log "FATAL: teacher gen failed"; exit 3;
}

# Stage 2: distillation training.
log "=== Stage 2: distillation training (500 steps, LoRA r=64) ==="
python3 scripts/train_distillation.py \
    --config configs/base_config.yaml \
    --manifest outputs/distill_dataset/manifest.json \
    --output_dir outputs/checkpoints/distilled \
    --max_steps 500 --batch_size 4 --learning_rate 2e-5 \
    --lora_rank 64 --lora_alpha 128 \
    --save_every 100 --log_every 10 \
    --t5_cpu --no_wandb 2>&1 | tee distill_train.log || {
    log "FATAL: distillation training failed"; exit 4;
}

CKPT=outputs/checkpoints/distilled/final
if [ ! -d "$CKPT" ]; then
    log "WARN: checkpoint missing — skipping Stage 3"; exit 0;
fi

# Stage 3: eval on held-out eval prompts.
# Cell A' (fresh baseline, for same-run reference), Cell C' (distilled no TFG),
# Cell D' (distilled + TFG ρ=20). Done as two passes of run_rho_sweep.
log "=== Stage 3a: Cell A' fresh baseline (no checkpoint, no TFG) ==="
python3 scripts/cloud/run_rho_sweep.py \
    --output_dir outputs/distill_eval_base \
    --rhos 20 \
    --apply_every_n 1 \
    --height 512 --width 512 --num_inference_steps 28 \
    --text_encoders_cpu --vae_tiling 2>&1 | tee eval_base.log

log "=== Stage 3b: Cell C' and D' with distilled LoRA ==="
python3 scripts/cloud/run_rho_sweep.py \
    --output_dir outputs/distill_eval_student \
    --checkpoint "$CKPT" \
    --rhos 20 \
    --apply_every_n 1 \
    --height 512 --width 512 --num_inference_steps 28 \
    --text_encoders_cpu --vae_tiling 2>&1 | tee eval_student.log

log "=== ALL STAGES COMPLETE ==="
