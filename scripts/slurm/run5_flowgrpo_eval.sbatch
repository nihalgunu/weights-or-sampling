#!/bin/bash
#SBATCH --job-name=run5_flowgrpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/run5_flowgrpo_%j.out
#SBATCH --error=logs/run5_flowgrpo_%j.err

# Run 5: FlowGRPO-finetuned with standard ODE sampling (FlowGRPO-only gains)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Run 5: FlowGRPO-finetuned Model (Standard ODE)"
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
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check if checkpoint exists
CHECKPOINT_PATH="outputs/checkpoints/flowgrpo/final"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: FlowGRPO checkpoint not found at $CHECKPOINT_PATH"
    echo "Please run train_flowgrpo.sbatch first"
    exit 1
fi

# Run image generation with finetuned model
echo ""
echo "Starting image generation with FlowGRPO model..."
python scripts/generate_images.py \
    --config configs/base_config.yaml \
    --run_name run5_flowgrpo \
    --output_dir outputs/images/run5_flowgrpo \
    --use_tfg false \
    --checkpoint "$CHECKPOINT_PATH"

# Run evaluation
echo ""
echo "Starting GenEval evaluation..."
python scripts/evaluate_geneval.py \
    --image_dir outputs/images/run5_flowgrpo \
    --output_file outputs/results/run5_flowgrpo.json \
    --detector_path repos/geneval/detector_models

# Log to W&B
echo ""
echo "Logging results to W&B..."
python scripts/log_results.py \
    --results_file outputs/results/run5_flowgrpo.json \
    --run_name run5_flowgrpo \
    --wandb_project tfg-rl-baselines

echo ""
echo "Run 5 (FlowGRPO) completed at $(date)"
