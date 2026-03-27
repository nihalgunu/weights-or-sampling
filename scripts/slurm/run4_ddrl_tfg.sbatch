#!/bin/bash
#SBATCH --job-name=run4_ddrl_tfg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/run4_ddrl_tfg_%j.out
#SBATCH --error=logs/run4_ddrl_tfg_%j.err

# Run 4: DDRL-finetuned + TFG guidance (DDRL + TFG complementarity)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Run 4: DDRL-finetuned Model + TFG Guidance"
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
CHECKPOINT_PATH="outputs/checkpoints/ddrl/final"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: DDRL checkpoint not found at $CHECKPOINT_PATH"
    echo "Please run train_ddrl.sbatch first"
    exit 1
fi

# Run image generation with DDRL model + TFG guidance
echo ""
echo "Starting image generation with DDRL model + TFG..."
python scripts/generate_images.py \
    --config configs/base_config.yaml \
    --run_name run4_ddrl_tfg \
    --output_dir outputs/images/run4_ddrl_tfg \
    --use_tfg true \
    --tfg_guidance_scale 7.5 \
    --tfg_recurrence_steps 1 \
    --checkpoint "$CHECKPOINT_PATH"

# Run evaluation
echo ""
echo "Starting GenEval evaluation..."
python scripts/evaluate_geneval.py \
    --image_dir outputs/images/run4_ddrl_tfg \
    --output_file outputs/results/run4_ddrl_tfg.json \
    --detector_path repos/geneval/detector_models

# Log to W&B
echo ""
echo "Logging results to W&B..."
python scripts/log_results.py \
    --results_file outputs/results/run4_ddrl_tfg.json \
    --run_name run4_ddrl_tfg \
    --wandb_project tfg-rl-baselines

echo ""
echo "Run 4 (DDRL + TFG) completed at $(date)"
