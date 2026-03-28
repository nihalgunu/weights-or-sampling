#!/bin/bash
#SBATCH --job-name=run4_ddrl_tfg
#SBATCH --partition=gpu
#SBATCH --account=labi
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
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

export RCAC_SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/${USER}}
export CONDA_ENVS_PATH="$RCAC_SCRATCH/conda/envs"
export CONDA_PKGS_DIRS="$RCAC_SCRATCH/conda/pkgs"
export PIP_CACHE_DIR="$RCAC_SCRATCH/pip/cache"
export HF_HOME="$RCAC_SCRATCH/.cache/huggingface"

source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate flowgrpo

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

CHECKPOINT_PATH="outputs/checkpoints/ddrl/final"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: DDRL checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

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

echo ""
echo "Starting GenEval evaluation..."
python scripts/evaluate_geneval.py \
    --image_dir outputs/images/run4_ddrl_tfg \
    --output_file outputs/results/run4_ddrl_tfg.json \
    --detector_path repos/geneval/detector_models

echo ""
echo "Logging results to W&B..."
python scripts/log_results.py \
    --results_file outputs/results/run4_ddrl_tfg.json \
    --run_name run4_ddrl_tfg \
    --wandb_project tfg-rl-baselines

echo ""
echo "Run 4 (DDRL + TFG) completed at $(date)"
