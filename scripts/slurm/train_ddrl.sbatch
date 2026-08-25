#!/bin/bash
#SBATCH --job-name=ddrl_train
#SBATCH --partition=a10
#SBATCH --account=bera89-k
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=120G
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

export RCAC_SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/${USER}}
export CONDA_ENVS_PATH="$RCAC_SCRATCH/conda/envs"
export CONDA_PKGS_DIRS="$RCAC_SCRATCH/conda/pkgs"
export PIP_CACHE_DIR="$RCAC_SCRATCH/pip/cache"
export HF_HOME="$RCAC_SCRATCH/.cache/huggingface"
export WANDB_PROJECT=tfg-rl-baselines
export WANDB_RUN_NAME=ddrl_training
export WANDB_DIR=$PROJECT_ROOT/logs

source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate flowgrpo

echo "GPU Info:"
nvidia-smi

echo ""
echo "Starting DDRL training..."
python scripts/train_ddrl.py \
    --config configs/base_config.yaml \
    --output_dir outputs/checkpoints/ddrl

echo ""
echo "DDRL training completed at $(date)"
echo "Checkpoint saved to: outputs/checkpoints/ddrl"
