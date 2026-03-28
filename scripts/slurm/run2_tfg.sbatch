#!/bin/bash
#SBATCH --job-name=sd35m_tfg
#SBATCH --partition=gpu
#SBATCH --account=labi
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/run2_tfg_%j.out
#SBATCH --error=logs/run2_tfg_%j.err

# Run 2: Base SD3.5-M with TFG guidance (TFG-only gains)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Run 2: SD3.5-M + TFG Guidance"
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

echo ""
echo "Starting image generation with TFG..."
python scripts/generate_images.py \
    --config configs/base_config.yaml \
    --run_name run2_tfg \
    --output_dir outputs/images/run2_tfg \
    --use_tfg true \
    --tfg_guidance_scale 7.5 \
    --tfg_recurrence_steps 1 \
    --checkpoint null

echo ""
echo "Starting GenEval evaluation..."
python scripts/evaluate_geneval.py \
    --image_dir outputs/images/run2_tfg \
    --output_file outputs/results/run2_tfg.json \
    --detector_path repos/geneval/detector_models

echo ""
echo "Logging results to W&B..."
python scripts/log_results.py \
    --results_file outputs/results/run2_tfg.json \
    --run_name run2_tfg \
    --wandb_project tfg-rl-baselines

echo ""
echo "Run 2 completed at $(date)"
