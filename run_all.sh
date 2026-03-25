#!/bin/bash
# Master script to run all TFG + RL experiments
# Submits all Slurm jobs with proper dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TFG + RL Post-Training Experiments"
echo "=============================================="
echo "Starting time: $(date)"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're on a Slurm cluster
if ! command -v sbatch &> /dev/null; then
    echo -e "${YELLOW}Warning: sbatch not found. Running in local mode.${NC}"
    echo ""
    LOCAL_MODE=true
else
    LOCAL_MODE=false
fi

# Ensure output directories exist
mkdir -p outputs/{images/run{1..4},checkpoints,results}
mkdir -p logs

if [ "$LOCAL_MODE" = true ]; then
    echo "Running in local mode (sequential execution)..."
    echo ""

    # Run 1: Baseline
    echo -e "${GREEN}[1/5]${NC} Running baseline..."
    bash scripts/slurm/run1_baseline.sbatch

    # Run 2: TFG
    echo -e "${GREEN}[2/5]${NC} Running TFG..."
    bash scripts/slurm/run2_tfg.sbatch

    # FlowGRPO Training
    echo -e "${GREEN}[3/5]${NC} Training FlowGRPO..."
    bash scripts/slurm/train_flowgrpo.sbatch

    # Run 3: FlowGRPO eval
    echo -e "${GREEN}[4/5]${NC} Evaluating FlowGRPO..."
    bash scripts/slurm/run3_flowgrpo_eval.sbatch

    # Run 4: FlowGRPO + TFG
    echo -e "${GREEN}[5/5]${NC} Running FlowGRPO + TFG..."
    bash scripts/slurm/run4_flowgrpo_tfg.sbatch

    # Aggregate results
    echo -e "${GREEN}[Done]${NC} Aggregating results..."
    bash scripts/slurm/aggregate_results.sbatch

else
    echo "Submitting Slurm jobs..."
    echo ""

    # Run 1: Baseline (no dependencies)
    JOB1=$(sbatch --parsable scripts/slurm/run1_baseline.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 1 (baseline): Job $JOB1"

    # Run 2: TFG (no dependencies, runs in parallel with Run 1)
    JOB2=$(sbatch --parsable scripts/slurm/run2_tfg.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 2 (TFG): Job $JOB2"

    # FlowGRPO Training (no dependencies, runs in parallel with Runs 1 & 2)
    JOB_TRAIN=$(sbatch --parsable scripts/slurm/train_flowgrpo.sbatch)
    echo -e "${GREEN}[Submitted]${NC} FlowGRPO training: Job $JOB_TRAIN"

    # Run 3: FlowGRPO eval (depends on training completion)
    JOB3=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN scripts/slurm/run3_flowgrpo_eval.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 3 (FlowGRPO): Job $JOB3 (depends on $JOB_TRAIN)"

    # Run 4: FlowGRPO + TFG (depends on training completion)
    JOB4=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN scripts/slurm/run4_flowgrpo_tfg.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 4 (FlowGRPO+TFG): Job $JOB4 (depends on $JOB_TRAIN)"

    # Results aggregation (depends on all evaluation jobs)
    JOB_AGG=$(sbatch --parsable --dependency=afterok:$JOB1:$JOB2:$JOB3:$JOB4 scripts/slurm/aggregate_results.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Results aggregation: Job $JOB_AGG"

    echo ""
    echo "=============================================="
    echo "All jobs submitted!"
    echo "=============================================="
    echo ""
    echo "Job dependency graph:"
    echo ""
    echo "  Run 1 (baseline) ────────┐"
    echo "  Run 2 (TFG) ─────────────┤"
    echo "                           ├──► Aggregate Results"
    echo "  FlowGRPO Training ──┬────┤"
    echo "                      │    │"
    echo "  Run 3 (FlowGRPO) ◄──┘    │"
    echo "  Run 4 (FlowGRPO+TFG) ◄───┘"
    echo ""
    echo "Monitor progress:"
    echo "  squeue -u \$USER"
    echo ""
    echo "View logs:"
    echo "  tail -f logs/*.out"
    echo ""
    echo "W&B dashboard:"
    echo "  wandb open"
    echo ""

    # Save job IDs for reference
    cat > .job_ids <<EOF
JOB1_BASELINE=$JOB1
JOB2_TFG=$JOB2
JOB_TRAIN=$JOB_TRAIN
JOB3_FLOWGRPO=$JOB3
JOB4_FLOWGRPO_TFG=$JOB4
JOB_AGGREGATE=$JOB_AGG
SUBMITTED_AT=$(date -Iseconds)
EOF

    echo "Job IDs saved to .job_ids"
fi

echo ""
echo "=============================================="
echo "Experiment pipeline started!"
echo "=============================================="
