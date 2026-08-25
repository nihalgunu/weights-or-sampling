#!/bin/bash
# Master setup script for TFG + RL Post-Training experiments
# Run this once to set up the entire environment

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TFG + RL Post-Training Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v conda &> /dev/null; then
    print_error "conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

if ! command -v git &> /dev/null; then
    print_error "git not found. Please install git first."
    exit 1
fi

# Step 1: Clone repositories
print_status "Step 1/6: Cloning repositories..."

mkdir -p repos
cd repos

if [ ! -d "TFG" ]; then
    print_status "Cloning Training-Free-Guidance..."
    git clone https://github.com/YWolfeee/Training-Free-Guidance.git TFG
else
    print_warning "TFG already exists, skipping..."
fi

if [ ! -d "flow_grpo" ]; then
    print_status "Cloning FlowGRPO..."
    git clone https://github.com/yifan123/flow_grpo.git flow_grpo
else
    print_warning "flow_grpo already exists, skipping..."
fi

if [ ! -d "geneval" ]; then
    print_status "Cloning GenEval..."
    git clone https://github.com/djghosh13/geneval.git geneval
else
    print_warning "geneval already exists, skipping..."
fi

if [ ! -d "mmdetection" ]; then
    print_status "Cloning MMDetection..."
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    git checkout 2.x
    cd ..
else
    print_warning "mmdetection already exists, skipping..."
fi

cd "$SCRIPT_DIR"

# Step 2: Create conda environments
print_status "Step 2/6: Creating conda environments..."

# On Gilbreth (or any RCAC cluster), redirect conda envs/pkgs to scratch
# so they don't overflow the home directory quota (~25GB).
# Cold-start import latency (~30-60s) is acceptable for long training jobs.
if [ -n "$RCAC_SCRATCH" ]; then
    print_status "RCAC cluster detected — redirecting conda to scratch..."
    mkdir -p "$RCAC_SCRATCH/conda/envs" "$RCAC_SCRATCH/conda/pkgs"

    # Export env vars — these take priority over .condarc and any system-set
    # CONDA_PKGS_DIRS, which Gilbreth sets to ~/. conda/pkgs by default.
    export CONDA_PKGS_DIRS="$RCAC_SCRATCH/conda/pkgs"
    export CONDA_ENVS_PATH="$RCAC_SCRATCH/conda/envs"
    export PIP_CACHE_DIR="$RCAC_SCRATCH/pip/cache"
    mkdir -p "$PIP_CACHE_DIR"

    # Also write to .condarc so conda run / sbatch jobs inherit the setting
    conda config --add envs_dirs "$RCAC_SCRATCH/conda/envs" 2>/dev/null || true
    conda config --add pkgs_dirs "$RCAC_SCRATCH/conda/pkgs" 2>/dev/null || true

    # Clean any corrupted/partial packages left in home from failed runs
    print_status "Cleaning corrupted package cache from home (~/.conda/pkgs)..."
    conda clean --all --yes --quiet 2>/dev/null || rm -rf "$HOME/.conda/pkgs"

    print_status "Conda pkgs  → $CONDA_PKGS_DIRS"
    print_status "Conda envs  → $CONDA_ENVS_PATH"
fi

create_or_update_env() {
    local yml_file=$1
    local env_name=$2
    if conda env list | grep -q "^${env_name} "; then
        print_warning "${env_name} env already exists — updating..."
        conda env update -n "$env_name" -f "$yml_file" --prune || \
            print_warning "Could not update ${env_name} env"
    else
        conda env create -f "$yml_file" || \
            print_warning "Could not create ${env_name} env"
    fi
}

print_status "Creating flowgrpo environment..."
create_or_update_env envs/flowgrpo.yml flowgrpo

print_status "Creating geneval environment..."
create_or_update_env envs/geneval.yml geneval

print_status "Installing mmcv pre-built wheel (CUDA 12.1 + PyTorch 2.1.0)..."
conda run -n geneval pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html || \
    print_warning "mmcv install may have issues"

print_status "Creating tfg environment..."
create_or_update_env envs/tfg.yml tfg

# Step 3: Install FlowGRPO
print_status "Step 3/6: Installing FlowGRPO..."
conda run -n flowgrpo pip install -e repos/flow_grpo || \
    print_warning "FlowGRPO install may have issues"

# Step 4: Download GenEval detector models
print_status "Step 4/6: Downloading GenEval detector models..."
cd repos/geneval
if [ ! -d "detector_models" ]; then
    mkdir -p detector_models
    if [ -f "evaluation/download_models.sh" ]; then
        bash evaluation/download_models.sh "./detector_models/"
    else
        print_warning "GenEval download script not found. You may need to download models manually."
    fi
else
    print_warning "detector_models already exists, skipping..."
fi
cd "$SCRIPT_DIR"

# Step 5: Setup HuggingFace authentication
print_status "Step 5/6: Setting up HuggingFace authentication..."
echo ""
print_warning "SD3.5-M requires HuggingFace authentication."
print_warning "Please ensure you have:"
print_warning "  1. Accepted the license at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
print_warning "  2. Created a HuggingFace token at https://huggingface.co/settings/tokens"
echo ""

if [ -n "$HF_TOKEN" ]; then
    print_status "HF_TOKEN found in environment — logging in non-interactively..."
    conda run -n flowgrpo hf auth login --token "$HF_TOKEN" 2>/dev/null || \
    conda run -n flowgrpo huggingface-cli login --token "$HF_TOKEN"
else
    read -p "Do you want to login to HuggingFace now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda run -n flowgrpo huggingface-cli login
    fi
fi

# Step 6: Setup Weights & Biases
print_status "Step 6/6: Setting up Weights & Biases..."
if [ -n "$WANDB_API_KEY" ]; then
    print_status "WANDB_API_KEY found in environment — skipping wandb login (env var is used automatically)."
else
    print_warning "WANDB_API_KEY not set. Add it to ~/.bashrc to enable W&B logging:"
    print_warning "  echo 'export WANDB_API_KEY=\"your_key\"' >> ~/.bashrc"
fi

# Create output directories
print_status "Creating output directories..."
mkdir -p outputs/images/run1_baseline
mkdir -p outputs/images/run2_tfg
mkdir -p outputs/images/run3_ddrl
mkdir -p outputs/images/run4_ddrl_tfg
mkdir -p outputs/images/run5_flowgrpo
mkdir -p outputs/images/run6_flowgrpo_tfg
mkdir -p outputs/checkpoints/ddrl
mkdir -p outputs/checkpoints/flowgrpo
mkdir -p outputs/results
mkdir -p logs

# Summary
echo ""
echo "=============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Conda environments created:"
echo "  - flowgrpo (FlowGRPO training + DDRL training + inference)"
echo "  - geneval (evaluation)"
echo "  - tfg (TFG-Flow inference)"
echo ""
echo "Note: DDRL repo not yet public. DDRL training is implemented in"
echo "  scripts/train_ddrl.py (forward KL + reward maximization from paper)."
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate flowgrpo"
echo "  2. Verify GPU access: python -c 'import torch; print(torch.cuda.is_available())'"
echo "  3. Run experiments: ./run_all.sh"
echo ""
echo "Directory structure:"
ls -la
echo ""
