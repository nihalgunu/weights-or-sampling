#!/bin/bash
#SBATCH --job-name=flowgrpo_train
#SBATCH --partition=gpu
#SBATCH --account=labi
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=60
#SBATCH --output=logs/flowgrpo_train_%j.out
#SBATCH --error=logs/flowgrpo_train_%j.err

# FlowGRPO Training on SD3.5-M with GenEval reward

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "FlowGRPO Training"
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
export WANDB_RUN_NAME=flowgrpo_training
export WANDB_DIR=$PROJECT_ROOT/logs

# Distributed training env vars
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate flowgrpo

echo "GPU Info:"
nvidia-smi

echo ""
echo "Starting FlowGRPO training..."

cd repos/flow_grpo

accelerate launch --num_processes=3 --mixed_precision=bf16 \
    train_grpo.py \
    --model_name_or_path stabilityai/stable-diffusion-3.5-medium \
    --reward_model geneval \
    --output_dir $PROJECT_ROOT/outputs/checkpoints/flowgrpo \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --kl_coef 0.1 \
    --clip_range 0.2 \
    --save_steps 500 \
    --logging_steps 10 \
    --report_to wandb \
    --bf16

cd "$PROJECT_ROOT"

echo ""
echo "Training completed at $(date)"
echo "Checkpoint saved to: outputs/checkpoints/flowgrpo"
