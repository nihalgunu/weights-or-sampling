#!/bin/bash
#SBATCH --job-name=flowgrpo_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
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

# Load modules
module load cuda/11.8 2>/dev/null || true
module load anaconda 2>/dev/null || true

eval "$(conda shell.bash hook)"
conda activate flowgrpo

echo "GPU Info:"
nvidia-smi

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID

# Initialize W&B
export WANDB_PROJECT=tfg-rl-baselines
export WANDB_RUN_NAME=flowgrpo_training
export WANDB_DIR=$PROJECT_ROOT/logs

echo ""
echo "Starting FlowGRPO training..."
echo "Using $(nvidia-smi -L | wc -l) GPUs"

cd repos/flow_grpo

# Run training using FlowGRPO's script
# Adjust parameters as needed based on their config structure
accelerate launch --num_processes=1 --mixed_precision=bf16 \
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

# If the above command doesn't work, try their provided script:
# bash scripts/single_node/grpo.sh

cd "$PROJECT_ROOT"

echo ""
echo "Training completed at $(date)"
echo "Checkpoint saved to: outputs/checkpoints/flowgrpo"
