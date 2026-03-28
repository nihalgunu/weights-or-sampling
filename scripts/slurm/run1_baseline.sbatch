#!/bin/bash
#SBATCH --job-name=sd35m_baseline
#SBATCH --partition=a100-40gb
#SBATCH --account=labi
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/run1_baseline_%j.out
#SBATCH --error=logs/run1_baseline_%j.err

# Run 1: Base SD3.5-M with standard ODE sampling (vanilla baseline)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Run 1: Baseline SD3.5-M"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "=============================================="

# Scratch paths
export RCAC_SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/${USER}}
export CONDA_ENVS_PATH="$RCAC_SCRATCH/conda/envs"
export CONDA_PKGS_DIRS="$RCAC_SCRATCH/conda/pkgs"
export PIP_CACHE_DIR="$RCAC_SCRATCH/pip/cache"
export HF_HOME="$RCAC_SCRATCH/.cache/huggingface"

# Initialize conda and activate env
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate flowgrpo

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "Starting image generation..."
python scripts/generate_images.py \
    --config configs/base_config.yaml \
    --run_name run1_baseline \
    --output_dir outputs/images/run1_baseline \
    --use_tfg false \
    --checkpoint null

echo ""
echo "Starting GenEval evaluation..."
python scripts/evaluate_geneval.py \
    --image_dir outputs/images/run1_baseline \
    --output_file outputs/results/run1_baseline.json \
    --detector_path repos/geneval/detector_models

echo ""
echo "Logging results to W&B..."
python scripts/log_results.py \
    --results_file outputs/results/run1_baseline.json \
    --run_name run1_baseline \
    --wandb_project tfg-rl-baselines

echo ""
echo "Run 1 completed at $(date)"
