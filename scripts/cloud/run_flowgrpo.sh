#!/bin/bash
# FlowGRPO on A100. Three stages:
#   0. pre-flight — verify GPU, repo, and GenEval detector are present
#   1. train one epoch of FlowGRPO (GRPO RL on SD3.5-M with GenEval reward)
#   2. evaluate Cells C and D (FlowGRPO ± TFG @ ρ=20)
#      → results land in outputs/flowgrpo_sweep/ to mirror ddrl_real_sweep/

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date +%H:%M:%S)] $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ── Stage 0: pre-flight ───────────────────────────────────────────────────────
log "=== Stage 0: pre-flight checks ==="

# 1. GPU present
python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA GPU found'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  GPU: {name}  VRAM: {vram:.1f} GB')
assert vram >= 38, f'Need >=38 GB VRAM, got {vram:.1f} GB'
print('  GPU OK')
"

# 2. flow_grpo repo and training script present
if [ ! -f "repos/flow_grpo/train_grpo.py" ]; then
    log "FATAL: repos/flow_grpo/train_grpo.py not found."
    log "  Run setup.sh first, or: git clone https://github.com/yifan123/flow_grpo.git repos/flow_grpo"
    exit 1
fi
log "  flow_grpo repo OK"

# 3. GenEval detector models present (needed for reward during training)
if [ ! -d "repos/geneval/detector_models" ]; then
    log "FATAL: repos/geneval/detector_models not found."
    log "  Run: cd repos/geneval && bash evaluation/download_models.sh ./detector_models/"
    exit 1
fi
log "  GenEval detector models OK"

# 4. Base sweep results present (needed to pick best eval ρ)
if [ ! -f "outputs/base_sweep/sweep_summary.json" ]; then
    log "WARN: outputs/base_sweep/sweep_summary.json not found — will default to ρ=20 for eval"
fi

log "=== Stage 0 OK ==="

# ── Stage 1: FlowGRPO training ────────────────────────────────────────────────
log "=== Stage 1: FlowGRPO training (1 epoch, single A100) ==="

mkdir -p outputs/checkpoints/flowgrpo logs

cd repos/flow_grpo

accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    train_grpo.py \
    --model_name_or_path stabilityai/stable-diffusion-3.5-medium \
    --reward_model geneval \
    --output_dir "$(pwd)/../../outputs/checkpoints/flowgrpo" \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --kl_coef 0.1 \
    --clip_range 0.2 \
    --save_steps 100 \
    --logging_steps 10 \
    --bf16 2>&1 | tee "$(pwd)/../../logs/flowgrpo_train.log" || {
    log "FATAL: FlowGRPO training failed — see logs/flowgrpo_train.log"
    exit 2
}

cd ../..
log "=== Stage 1 done ==="

# ── Checkpoint validation (same guard as distillation runs) ──────────────────
CKPT=outputs/checkpoints/flowgrpo/final
if [ ! -d "$CKPT" ]; then
    # fall back to last saved checkpoint
    CKPT=$(ls -d outputs/checkpoints/flowgrpo/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "$CKPT" ]; then
        log "FATAL: no checkpoint found in outputs/checkpoints/flowgrpo/"
        exit 3
    fi
    log "WARN: 'final' not found, using last checkpoint: $CKPT"
fi

log "Validating LoRA checkpoint at $CKPT ..."
python3 scripts/cloud/verify_lora_load.py --checkpoint "$CKPT" || {
    log "FATAL: LoRA load validation failed — weights did not attach correctly"
    exit 4
}
log "  Checkpoint OK"

# ── Stage 2: evaluate Cells C and D ──────────────────────────────────────────
log "=== Stage 2: Cell C (FlowGRPO, no TFG) and Cell D (+ TFG ρ=20) ==="

BEST_RHO=$(python3 -c "
import json, sys
try:
    d = json.load(open('outputs/base_sweep/sweep_summary.json'))
    best = max(d['per_rho'], key=lambda r: r['delta_vs_baseline'])
    print(f\"{best['rho']:g}\")
except Exception:
    print('20')
" 2>/dev/null || echo "20")
log "  Using eval ρ=$BEST_RHO"

python3 scripts/cloud/run_rho_sweep.py \
    --output_dir outputs/flowgrpo_sweep \
    --checkpoint "$CKPT" \
    --rhos "$BEST_RHO" \
    --apply_every_n 1 \
    --height 512 --width 512 --num_inference_steps 28 \
    --text_encoders_cpu --vae_tiling 2>&1 | tee logs/flowgrpo_eval.log || {
    log "WARN: eval failed — see logs/flowgrpo_eval.log (training checkpoint saved)"
}

log "=== Stage 2 done ==="

# ── Summary ──────────────────────────────────────────────────────────────────
log "=== ALL STAGES COMPLETE ==="
log ""
log "Results:    outputs/flowgrpo_sweep/"
log "Checkpoint: $CKPT"
log "Train log:  logs/flowgrpo_train.log"
log "Eval log:   logs/flowgrpo_eval.log"
log ""
log "Compare against:"
log "  DDRL:         outputs/ddrl_real_sweep/sweep_summary.json"
log "  Distillation: outputs/ckpt_sweep/ckpt_500/sweep_summary.json"
log "  Base:         outputs/base_sweep/sweep_summary.json"
