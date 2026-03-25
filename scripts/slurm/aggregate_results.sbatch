#!/bin/bash
#SBATCH --job-name=aggregate_results
#SBATCH --partition=cpu
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/aggregate_results_%j.out
#SBATCH --error=logs/aggregate_results_%j.err

# Aggregate all results into final table

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Aggregating Results"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "=============================================="

# Load modules
module load anaconda 2>/dev/null || true

eval "$(conda shell.bash hook)"
conda activate flowgrpo

# Run aggregation
echo "Aggregating results from all runs..."
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
echo "View in W&B: https://wandb.ai/<your-entity>/tfg-rl-baselines"
echo ""
echo "Aggregation completed at $(date)"
