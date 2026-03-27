#!/bin/bash
#SBATCH --job-name=ddrl_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/ddrl_train_%j.out
#SBATCH --error=logs/ddrl_train_%j.err

# DDRL Training on SD3.5-M with GenEval reward
# Algorithm: forward KL divergence + reward maximization (adapted from DDRL paper)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "DDRL Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "=============================================="

# Load modules
module load cuda/11.8 2>/dev/null || true
module load anaconda 2>/dev/null || true

eval "$(conda shell.bash hook)"
conda activate flowgrpo

echo "GPU Info:"
nvidia-smi

# Initialize W&B
export WANDB_PROJECT=tfg-rl-baselines
export WANDB_RUN_NAME=ddrl_training
export WANDB_DIR=$PROJECT_ROOT/logs

echo ""
echo "Starting DDRL training..."
echo "Using $(nvidia-smi -L | wc -l) GPUs"

# Run DDRL training
# Note: train_ddrl.py implements the forward KL + reward maximization algorithm
# from the DDRL paper (repo not yet released; adapted from paper description).
python scripts/train_ddrl.py \
    --config configs/base_config.yaml \
    --output_dir outputs/checkpoints/ddrl

cd "$PROJECT_ROOT"

echo ""
echo "DDRL training completed at $(date)"
echo "Checkpoint saved to: outputs/checkpoints/ddrl"
