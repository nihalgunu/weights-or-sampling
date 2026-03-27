#!/bin/bash
# Master script to run all TFG + RL experiments (6 runs)
# Submits all Slurm jobs with proper dependencies
#
# DAG:
#   Run 1 (baseline) ─────────────────────────────────┐
#   Run 2 (base+TFG) ─────────────────────────────────┤
#                                                       ├──► Aggregate
#   DDRL Training ──► Run 3 (DDRL) ──────────────────┤
#                └──► Run 4 (DDRL+TFG) ──────────────┤
#                                                       │
#   FlowGRPO Training ──► Run 5 (FlowGRPO) ──────────┤
#                    └──► Run 6 (FlowGRPO+TFG) ───────┘

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TFG + RL Post-Training Experiments (6 Runs)"
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
mkdir -p outputs/images/run1_baseline
mkdir -p outputs/images/run2_tfg
mkdir -p outputs/images/run3_ddrl
mkdir -p outputs/images/run4_ddrl_tfg
mkdir -p outputs/images/run5_flowgrpo
mkdir -p outputs/images/run6_flowgrpo_tfg
mkdir -p outputs/checkpoints/ddrl
mkdir -p outputs/checkpoints/flowgrpo
mkdir -p outputs/results
mkdir -p logs

if [ "$LOCAL_MODE" = true ]; then
    echo "Running in local mode (sequential execution)..."
    echo ""

    # Run 1: Baseline
    echo -e "${GREEN}[1/9]${NC} Running baseline..."
    bash scripts/slurm/run1_baseline.sbatch

    # Run 2: TFG
    echo -e "${GREEN}[2/9]${NC} Running base + TFG..."
    bash scripts/slurm/run2_tfg.sbatch

    # DDRL Training
    echo -e "${GREEN}[3/9]${NC} Training DDRL..."
    bash scripts/slurm/train_ddrl.sbatch

    # Run 3: DDRL eval
    echo -e "${GREEN}[4/9]${NC} Evaluating DDRL..."
    bash scripts/slurm/run3_ddrl_eval.sbatch

    # Run 4: DDRL + TFG
    echo -e "${GREEN}[5/9]${NC} Running DDRL + TFG..."
    bash scripts/slurm/run4_ddrl_tfg.sbatch

    # FlowGRPO Training
    echo -e "${GREEN}[6/9]${NC} Training FlowGRPO..."
    bash scripts/slurm/train_flowgrpo.sbatch

    # Run 5: FlowGRPO eval
    echo -e "${GREEN}[7/9]${NC} Evaluating FlowGRPO..."
    bash scripts/slurm/run5_flowgrpo_eval.sbatch

    # Run 6: FlowGRPO + TFG
    echo -e "${GREEN}[8/9]${NC} Running FlowGRPO + TFG..."
    bash scripts/slurm/run6_flowgrpo_tfg.sbatch

    # Aggregate results
    echo -e "${GREEN}[9/9]${NC} Aggregating all 6 runs..."
    bash scripts/slurm/aggregate_results.sbatch

else
    echo "Submitting Slurm jobs..."
    echo ""

    # Runs 1 & 2: no dependencies (start immediately)
    JOB1=$(sbatch --parsable scripts/slurm/run1_baseline.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 1 (baseline): Job $JOB1"

    JOB2=$(sbatch --parsable scripts/slurm/run2_tfg.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 2 (base+TFG): Job $JOB2"

    # DDRL training (no dependency, runs in parallel with Runs 1 & 2 and FlowGRPO training)
    JOB_TRAIN_DDRL=$(sbatch --parsable scripts/slurm/train_ddrl.sbatch)
    echo -e "${GREEN}[Submitted]${NC} DDRL training: Job $JOB_TRAIN_DDRL"

    # FlowGRPO training (no dependency, runs in parallel with everything above)
    JOB_TRAIN_FGRPO=$(sbatch --parsable scripts/slurm/train_flowgrpo.sbatch)
    echo -e "${GREEN}[Submitted]${NC} FlowGRPO training: Job $JOB_TRAIN_FGRPO"

    # Runs 3 & 4: depend on DDRL training
    JOB3=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN_DDRL scripts/slurm/run3_ddrl_eval.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 3 (DDRL): Job $JOB3 (depends on $JOB_TRAIN_DDRL)"

    JOB4=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN_DDRL scripts/slurm/run4_ddrl_tfg.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 4 (DDRL+TFG): Job $JOB4 (depends on $JOB_TRAIN_DDRL)"

    # Runs 5 & 6: depend on FlowGRPO training
    JOB5=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN_FGRPO scripts/slurm/run5_flowgrpo_eval.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 5 (FlowGRPO): Job $JOB5 (depends on $JOB_TRAIN_FGRPO)"

    JOB6=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN_FGRPO scripts/slurm/run6_flowgrpo_tfg.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Run 6 (FlowGRPO+TFG): Job $JOB6 (depends on $JOB_TRAIN_FGRPO)"

    # Aggregation: depends on all 6 run evaluation jobs
    JOB_AGG=$(sbatch --parsable \
        --dependency=afterok:$JOB1:$JOB2:$JOB3:$JOB4:$JOB5:$JOB6 \
        scripts/slurm/aggregate_results.sbatch)
    echo -e "${GREEN}[Submitted]${NC} Results aggregation: Job $JOB_AGG"

    echo ""
    echo "=============================================="
    echo "All jobs submitted!"
    echo "=============================================="
    echo ""
    echo "Job dependency graph:"
    echo ""
    echo "  Run 1 (baseline) ─────────────────────────────────┐"
    echo "  Run 2 (base+TFG) ─────────────────────────────────┤"
    echo "                                                      ├──► Aggregate"
    echo "  DDRL Training ──► Run 3 (DDRL) ──────────────────┤"
    echo "               └──► Run 4 (DDRL+TFG) ──────────────┤"
    echo "                                                      │"
    echo "  FlowGRPO Training ──► Run 5 (FlowGRPO) ──────────┤"
    echo "                   └──► Run 6 (FlowGRPO+TFG) ───────┘"
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
JOB_TRAIN_DDRL=$JOB_TRAIN_DDRL
JOB3_DDRL=$JOB3
JOB4_DDRL_TFG=$JOB4
JOB_TRAIN_FLOWGRPO=$JOB_TRAIN_FGRPO
JOB5_FLOWGRPO=$JOB5
JOB6_FLOWGRPO_TFG=$JOB6
JOB_AGGREGATE=$JOB_AGG
SUBMITTED_AT=$(date -Iseconds)
EOF

    echo "Job IDs saved to .job_ids"
fi

echo ""
echo "=============================================="
echo "Experiment pipeline started!"
echo "=============================================="
