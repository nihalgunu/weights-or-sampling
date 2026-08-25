#!/bin/bash
# Full scaling sweep: training-compute axis + teacher-ρ axis, back to back.
# Expected wall: ~77 min (ckpt sweep) + ~5.4 hr (teacher-ρ) ≈ 6.7 hr on A10.
# All results fetched at the end; each sub-sweep logs independently so a
# failure in the second doesn't lose the first's results.

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "###################################################################"
log "# PHASE A — training-compute axis (5 checkpoints, eval only)     #"
log "###################################################################"
bash scripts/cloud/run_ckpt_sweep.sh 2>&1 | tee ckpt_sweep_phase.log || {
    log "WARN: Phase A had errors — continuing to Phase B with whatever landed"
}

log "###################################################################"
log "# PHASE B — teacher-ρ axis (3 full distillation runs)            #"
log "###################################################################"
bash scripts/cloud/run_teacher_rho_sweep.sh 2>&1 | tee teacher_rho_phase.log || {
    log "WARN: Phase B had errors — Phase A data still saved"
}

log "###################################################################"
log "# FULL SCALING SWEEP COMPLETE                                    #"
log "###################################################################"
