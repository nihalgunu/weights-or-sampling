#!/bin/bash
#SBATCH --job-name=sd35m_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/run1_baseline_%j.out
#SBATCH --error=logs/run1_baseline_%j.err

# Run 1: Base SD3.5-M with standard ODE sampling (vanilla baseline)

set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Run 1: Baseline SD3.5-M"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "=============================================="

# Load modules (adjust based on your cluster)
module load cuda/11.8 2>/dev/null || true
module load anaconda 2>/dev/null || true

# Activate environment
eval "$(conda shell.bash hook)"
conda activate flowgrpo

# Verify GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run image generation
echo ""
echo "Starting image generation..."
python scripts/generate_images.py \
    --config configs/base_config.yaml \
    --run_name run1_baseline \
    --output_dir outputs/images/run1_baseline \
    --use_tfg false \
    --checkpoint null

# Run evaluation
echo ""
echo "Starting GenEval evaluation..."
python scripts/evaluate_geneval.py \
    --image_dir outputs/images/run1_baseline \
    --output_file outputs/results/run1_baseline.json \
    --detector_path repos/geneval/detector_models

# Log to W&B
echo ""
echo "Logging results to W&B..."
python scripts/log_results.py \
    --results_file outputs/results/run1_baseline.json \
    --run_name run1_baseline \
    --wandb_project tfg-rl-baselines

echo ""
echo "Run 1 completed at $(date)"
