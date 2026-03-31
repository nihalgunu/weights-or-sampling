#!/bin/bash
#SBATCH --job-name=aggregate_results
#SBATCH --partition=a10
#SBATCH --account=bera89-k
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/aggregate_results_%j.out
#SBATCH --error=logs/aggregate_results_%j.err

# Aggregate all results into final table (CPU-only job)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Aggregating Results"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "=============================================="

export RCAC_SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/${USER}}
export CONDA_ENVS_PATH="$RCAC_SCRATCH/conda/envs"
export CONDA_PKGS_DIRS="$RCAC_SCRATCH/conda/pkgs"
export PIP_CACHE_DIR="$RCAC_SCRATCH/pip/cache"

source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate flowgrpo

echo "Aggregating results from all 6 runs..."
python scripts/aggregate_results.py \
    --results_dir outputs/results \
    --output_file outputs/results/final_table.csv \
    --wandb_project tfg-rl-baselines

echo ""
echo "=============================================="
echo "FINAL RESULTS"
echo "=============================================="
cat outputs/results/final_table.csv

echo ""
echo "Results saved to: outputs/results/final_table.csv"
echo "Aggregation completed at $(date)"
