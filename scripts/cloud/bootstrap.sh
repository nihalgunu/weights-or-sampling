#!/bin/bash
# Runs on a fresh Lambda instance (no persistent filesystem).
# Installs miniconda, clones repo, sets up conda envs, then runs RUN_SCRIPT.
#
# Expected env vars (passed by launch_and_run.sh):
#   HF_TOKEN, WANDB_API_KEY, REPO_URL, RUN_SCRIPT

set -euo pipefail
log() { echo "[$(date +%H:%M:%S)] $*"; }

HF_TOKEN="${HF_TOKEN:?}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE=disabled
REPO_URL="${REPO_URL:-https://github.com/nihalgunu/haotian_research.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/cloud/run_flowgrpo.sh}"

INSTALL_DIR="$HOME"
CONDA_DIR="$INSTALL_DIR/miniconda3"

# ── 1. Install miniconda ──────────────────────────────────────────────────────
log "=== Installing miniconda ==="
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi

export PATH="$CONDA_DIR/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
conda init bash
source ~/.bashrc || true
log "  conda $(conda --version)"

# conda 26+ requires accepting ToS before creating envs non-interactively
conda tos accept 2>/dev/null || true

# ── 2. Export tokens ──────────────────────────────────────────────────────────
export HF_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="$WANDB_API_KEY"

# Persist to .bashrc for any child shells
echo "export HF_TOKEN='$HF_TOKEN'" >> ~/.bashrc
echo "export WANDB_API_KEY='$WANDB_API_KEY'" >> ~/.bashrc
echo "export PATH='$CONDA_DIR/bin:/usr/local/bin:/usr/bin:/bin:\$PATH'" >> ~/.bashrc

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
log "=== Cloning repo ==="
REPO_DIR="$HOME/haotian_research"
if [ ! -d "$REPO_DIR" ]; then
    # Use token-authenticated URL if GITHUB_TOKEN is set, else try plain URL
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        AUTH_URL=$(echo "$REPO_URL" | sed "s|https://|https://${GITHUB_TOKEN}@|")
        git clone --branch "$REPO_BRANCH" "$AUTH_URL" "$REPO_DIR"
    else
        GIT_TERMINAL_PROMPT=0 git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
    fi
fi
cd "$REPO_DIR"
log "  Repo at $REPO_DIR ($(git rev-parse --short HEAD))"

# ── 4. Run setup.sh ───────────────────────────────────────────────────────────
log "=== Running setup.sh ==="
chmod +x setup.sh
./setup.sh

# ── 5. Fix geneval mmcv (pre-built CUDA 12.1 wheel) ──────────────────────────
log "=== Installing mmcv pre-built wheel ==="
conda run -n geneval pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html || \
    log "WARN: mmcv install failed — geneval env may not work"

# ── 6. Verify GPU ─────────────────────────────────────────────────────────────
log "=== Verifying GPU ==="
conda run -n flowgrpo python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  CUDA: {torch.version.cuda}')
print('  GPU OK')
"

# ── 7. Run target script ──────────────────────────────────────────────────────
log "=== Starting $RUN_SCRIPT ==="
conda run -n flowgrpo bash "$RUN_SCRIPT"

log "=== BOOTSTRAP COMPLETE ==="
