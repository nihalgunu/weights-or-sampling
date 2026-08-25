#!/bin/bash
# Real DDRL on A100. Three stages:
#   0. dry-run (3 steps) — verifies SD3 + ref + LoRA + CLIP fit in GPU memory
#      and the training loop runs against real models without OOM / shape bugs.
#      If this fails, abort before spending on the long training.
#   1. train 300 steps with the ddrl_real.yaml config (ref KL on, batch 4).
#   2. evaluate Cells C and D (DDRL ± TFG @ ρ=20).

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date +%H:%M:%S)] $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Stage 0: dry-run ─────────────────────────────────────────────────────────
log "=== Stage 0: dry-run (3 training steps on real SD3.5-M) ==="
python3 scripts/train_ddrl.py \
    --config configs/ddrl_real.yaml \
    --max_steps 3 \
    --t5_cpu \
    --no_wandb \
    --prompts_file scripts/cloud/prompts_smoke.json \
    --output_dir outputs/checkpoints/dryrun 2>&1 | tee dryrun.log || {
    log "FATAL: dry-run failed — see dryrun.log. Aborting before the long training."
    exit 1
}
log "=== Stage 0 OK ==="
rm -rf outputs/checkpoints/dryrun

# ── Stage 1: full training ───────────────────────────────────────────────────
log "=== Stage 1: full DDRL training (300 steps) ==="
python3 scripts/train_ddrl.py \
    --config configs/ddrl_real.yaml \
    --max_steps 300 \
    --t5_cpu \
    --no_wandb \
    --prompts_file scripts/cloud/prompts_smoke.json \
    --output_dir outputs/checkpoints/ddrl_real 2>&1 | tee train_real.log || {
    log "FATAL: training failed — see train_real.log"
    exit 2
}
log "=== Stage 1 done ==="

# ── Stage 2: evaluate Cells C and D ──────────────────────────────────────────
CKPT=outputs/checkpoints/ddrl_real/final
if [ ! -d "$CKPT" ]; then
    log "WARN: checkpoint $CKPT missing — skipping eval"
    exit 0
fi

log "=== Stage 2: Cell C (DDRL-real, no TFG) and Cell D (+ TFG ρ=20) ==="
# Load best ρ from the earlier base sweep (should be 20 from our last run)
BEST_RHO=$(python3 -c "
import json
d = json.load(open('outputs/base_sweep/sweep_summary.json'))
best = max(d['per_rho'], key=lambda r: r['delta_vs_baseline'])
print(f\"{best['rho']:g}\")
" 2>/dev/null || echo "20")

python3 scripts/cloud/run_rho_sweep.py \
    --output_dir outputs/ddrl_real_sweep \
    --checkpoint "$CKPT" \
    --rhos "$BEST_RHO" \
    --apply_every_n 1 \
    --height 512 --width 512 --num_inference_steps 28 \
    --text_encoders_cpu --vae_tiling 2>&1 | tee eval_real.log
log "=== Stage 2 done ==="

log "=== ALL STAGES COMPLETE ==="
