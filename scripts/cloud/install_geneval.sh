#!/bin/bash
# GenEval installer — time-boxed. If mmcv/mmdet install fails within 15 min,
# we skip GenEval and fall back to CLIP-only reporting. GenEval's mmdet 2.x
# stack is notoriously brittle vs modern PyTorch; this is a known tradeoff.

set -euo pipefail
cd "$(dirname "$0")/../.."
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GenEval install starting ==="

mkdir -p repos
if [ ! -d repos/geneval ]; then
    log "cloning geneval"
    git clone --depth 1 https://github.com/djghosh13/geneval.git repos/geneval
fi

# mmcv + mmdet matrix:
#   torch 2.7 → mmcv 2.x + mmdet 3.x (geneval's evaluate_images.py uses mmdet
#   2.x API though — may need a shim). Try mmengine-based mmdet 3.x first.
log "installing mmengine/mmcv/mmdet stack (2.x-flavor for geneval)"
python3 -m pip install --quiet openmim
python3 -m pip install --quiet 'mmengine>=0.10' 'mmcv>=2.1,<2.2' || {
    log "WARN: mmcv install failed — GenEval unavailable"
    exit 10
}
python3 -m mim install 'mmdet>=3.0,<3.4' || {
    log "WARN: mmdet install failed — GenEval unavailable"
    exit 11
}

# Detector models (~3GB). GenEval's download script uses wget and a hard-coded
# destination; run from repo root to avoid surprises.
if [ ! -d repos/geneval/detector_models ] || [ -z "$(ls -A repos/geneval/detector_models 2>/dev/null)" ]; then
    log "downloading detector models to repos/geneval/detector_models (~3GB)"
    mkdir -p repos/geneval/detector_models
    cd repos/geneval
    bash evaluation/download_models.sh ./detector_models/ 2>&1 | tail -10 || {
        log "WARN: detector download failed"
        cd - > /dev/null
        exit 12
    }
    cd - > /dev/null
fi

log "=== GENEVAL READY ==="
