#!/bin/bash
# Runs on the Lambda instance. Waits for Phase 1 to finish, then chains:
#   Phase 2 — DDRL training (LoRA, no ref, 5 epochs over 30 prompts ≈ 75 steps)
#   Phase 3 — Cells C (DDRL + no TFG) and D (DDRL + TFG at best ρ from Phase 1)
# Reads best ρ from outputs/base_sweep/sweep_summary.json (max delta_vs_baseline).

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── wait for phase 1 completion (sweep_summary.json is written last) ─────────
log "waiting for Phase 1's sweep_summary.json ..."
while [ ! -f outputs/base_sweep/sweep_summary.json ]; do
    sleep 10
done
log "Phase 1 complete — sweep_summary.json exists"

# ── pick best ρ ──────────────────────────────────────────────────────────────
BEST_RHO=$(python3 -c "
import json
d = json.load(open('outputs/base_sweep/sweep_summary.json'))
best = max(d['per_rho'], key=lambda r: r['delta_vs_baseline'])
print(f\"{best['rho']:g}\")
")
log "best ρ from Phase 1 = $BEST_RHO"

# ── Phase 2: DDRL training ───────────────────────────────────────────────────
log "=== Phase 2: DDRL training (LoRA, no-ref) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 scripts/train_ddrl.py \
    --config configs/ddrl_smoke.yaml \
    --max_steps 60 \
    --no_ref_model \
    --t5_cpu \
    --no_wandb \
    --prompts_file scripts/cloud/prompts_smoke.json \
    --output_dir outputs/checkpoints/ddrl 2>&1 | tee phase2.log || {
    log "WARN: DDRL training failed — see phase2.log. Skipping Phase 3."
    exit 0
}
log "=== Phase 2 done ==="

# Verify checkpoint exists
CKPT=outputs/checkpoints/ddrl/final
if [ ! -d "$CKPT" ] || [ -z "$(ls -A $CKPT 2>/dev/null)" ]; then
    log "WARN: checkpoint $CKPT missing — skipping Phase 3"
    exit 0
fi

# ── Phase 3: Cells C + D ─────────────────────────────────────────────────────
log "=== Phase 3: Cells C (DDRL, no TFG) and D (DDRL + TFG @ ρ=$BEST_RHO) ==="
python3 scripts/cloud/run_rho_sweep.py \
    --output_dir outputs/ddrl_sweep \
    --checkpoint "$CKPT" \
    --rhos "$BEST_RHO" \
    --apply_every_n 1 \
    --height 512 --width 512 --num_inference_steps 28 \
    --text_encoders_cpu --vae_tiling 2>&1 | tee phase3.log
log "=== Phase 3 done ==="

log "=== ALL PHASES COMPLETE ==="
