#!/bin/bash
# Teacher-ρ scaling: vary the ρ used by TFG during teacher dataset generation.
# For each ρ ∈ {2, 5, 20} run Stage 1 (teacher dataset), Stage 2 (distillation
# training, 500 steps), Stage 3 (Cell C + Cell D at eval ρ=20 — the same fixed
# inference-time ρ). Produces a scaling curve: how does the 9/69/22
# decomposition vary with teacher strength.

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date +%H:%M:%S)] $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_RHOS=(2 5 20)

for RHO in "${TEACHER_RHOS[@]}"; do
    log "================================================================="
    log "=== Teacher ρ=$RHO: Stage 1 (teacher dataset, 200 prompts) ==="
    log "================================================================="
    DATA_DIR="outputs/distill_dataset_rho${RHO}"
    CKPT_DIR="outputs/checkpoints/distilled_rho${RHO}"
    EVAL_DIR="outputs/distill_eval_rho${RHO}"

    python3 scripts/cloud/generate_tfg_dataset.py \
        --prompts_file scripts/cloud/prompts_train.json \
        --output_dir "$DATA_DIR" \
        --tfg_rho "$RHO" --tfg_every_n 1 \
        --height 512 --width 512 --num_inference_steps 28 \
        2>&1 | tee "gen_rho${RHO}.log" || {
        log "FATAL: teacher gen at ρ=$RHO failed — continuing to next"; continue;
    }

    log "=== Teacher ρ=$RHO: Stage 2 (distillation training) ==="
    python3 scripts/train_distillation.py \
        --config configs/base_config.yaml \
        --manifest "$DATA_DIR/manifest.json" \
        --output_dir "$CKPT_DIR" \
        --max_steps 500 --batch_size 4 --learning_rate 2e-5 \
        --lora_rank 64 --lora_alpha 128 \
        --save_every 100 --log_every 10 \
        --t5_cpu --no_wandb \
        2>&1 | tee "distill_rho${RHO}.log" || {
        log "WARN: distillation at ρ=$RHO failed — continuing"; continue;
    }

    CKPT="$CKPT_DIR/final"
    if [ ! -d "$CKPT" ]; then
        log "WARN: checkpoint missing after training at ρ=$RHO — skip eval"; continue;
    fi

    log "=== Teacher ρ=$RHO: Stage 0 verify LoRA ==="
    python3 scripts/cloud/verify_lora_load.py "$CKPT" 2>&1 | tee "verify_rho${RHO}.log" || {
        log "FATAL: verify failed at ρ=$RHO — skip eval"; continue;
    }

    log "=== Teacher ρ=$RHO: Stage 3 (Cell C + D) ==="
    python3 scripts/cloud/run_rho_sweep.py \
        --output_dir "$EVAL_DIR" \
        --checkpoint "$CKPT" \
        --rhos 20 \
        --apply_every_n 1 \
        --height 512 --width 512 --num_inference_steps 28 \
        --text_encoders_cpu --vae_tiling \
        2>&1 | tee "eval_rho${RHO}.log" || {
        log "WARN: eval at ρ=$RHO failed — continuing"; continue;
    }
    log "=== Teacher ρ=$RHO done ==="
done

log "=== TEACHER ρ SWEEP COMPLETE ==="
