#!/bin/bash
#
# Submit one SLURM job per regression dataset for GNN regression.
#
# Usage:
#   ./run_gnn_regression_fox.sh                        # all datasets, default settings
#   ./run_gnn_regression_fox.sh -m 5000                # subsample to 5000 graphs
#   ./run_gnn_regression_fox.sh -d "alchemy_full"      # specific datasets only
#   ./run_gnn_regression_fox.sh --all-targets --normalize -d "alchemy_full"
#   ./run_gnn_regression_fox.sh --dry-run              # preview sbatch commands
#
# Options:
#   -t, --timeout N        Timeout per experiment in seconds (default: 0 = no limit)
#   -m, --max-graphs N     Subsample to at most N graphs (default: 0 = use all)
#   -d, --datasets "D"     Space-separated list of datasets (default: all)
#       --target-index N   Target column index in single-target mode (default: 0)
#       --all-targets      Run every target column (alchemy_full = 12 targets)
#       --normalize        Z-score normalize each target. Requires --all-targets.
#       --device DEV       cpu|gpu|cuda (default: gpu)
#       --gpu-type TYPE    GPU type: rtx30|a100|a40|h100|l40s (default: rtx30)
#       --gpu-count N      Number of GPUs per job (default: 1)
#       --partition PART   SLURM partition: accel|ifi_accel|hudel (default: accel)
#       --quick            Quick mode: 2x2 CV instead of 10x10
#       --dry-run          Print sbatch commands without submitting

set -e

# Default values
TIMEOUT=0
MAX_GRAPHS=0
QUICK=false
DRY_RUN=false
DATASETS=""
TARGET_INDEX=""
ALL_TARGETS=false
NORMALIZE=false
DEVICE="gpu"
GPU_TYPE="rtx30"
GPU_COUNT=1
PARTITION="accel"

ALL_DATASETS=(
    alchemy_full
    aspirin
    benzene
    ethanol
    malonaldehyde
    naphthalene
    toluene
    uracil
    ZINC_full
    ZINC_test
    ZINC_train
    ZINC_val
)

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -m|--max-graphs)
            MAX_GRAPHS="$2"
            shift 2
            ;;
        -d|--datasets)
            DATASETS="$2"
            shift 2
            ;;
        --target-index)
            TARGET_INDEX="$2"
            shift 2
            ;;
        --all-targets)
            ALL_TARGETS=true
            shift
            ;;
        --normalize)
            NORMALIZE=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Submit one SLURM job per regression dataset (GNN regression)."
            echo ""
            echo "Options:"
            echo "  -t, --timeout N        Timeout per experiment in seconds (default: 0)"
            echo "  -m, --max-graphs N     Subsample to at most N graphs (default: 0 = all)"
            echo "  -d, --datasets \"D\"     Space-separated list of datasets"
            echo "      --target-index N   Target column index in single-target mode"
            echo "      --all-targets      Run every target column (alchemy_full = 12)"
            echo "      --normalize        Z-score normalize each target (needs --all-targets)"
            echo "      --device DEV       cpu|gpu|cuda (default: gpu)"
            echo "      --gpu-type TYPE    GPU type: rtx30|a100|a40|h100|l40s (default: rtx30)"
            echo "      --gpu-count N      Number of GPUs per job (default: 1)"
            echo "      --partition PART   SLURM partition: accel|ifi_accel|hudel (default: accel)"
            echo "      --quick            Quick mode: 2x2 CV instead of 10x10"
            echo "      --dry-run          Print sbatch commands without submitting"
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
done

# Validation — matches the Python script's own check, but fail fast here too.
if [[ "$NORMALIZE" == true && "$ALL_TARGETS" != true ]]; then
    echo "Error: --normalize requires --all-targets"
    exit 1
fi

# Use all datasets if none specified
if [[ -z "$DATASETS" ]]; then
    DATASET_LIST=("${ALL_DATASETS[@]}")
else
    read -ra DATASET_LIST <<< "$DATASETS"
fi

mkdir -p logs

echo "=========================================="
echo "GNN Regression SLURM Job Submitter"
echo "=========================================="
echo "Datasets:     ${#DATASET_LIST[@]}"
echo "Timeout:      $TIMEOUT seconds (0 = no limit)"
echo "Max graphs:   $MAX_GRAPHS (0 = use all)"
echo "Target mode:  $( [[ "$ALL_TARGETS" == true ]] && echo "all targets" || echo "single target (index ${TARGET_INDEX:-0})" )"
echo "Normalize:    $NORMALIZE"
echo "Device:       $DEVICE"
echo "Partition:    $PARTITION"
echo "GPU:          ${GPU_TYPE}:${GPU_COUNT}"
echo "Quick mode:   $QUICK"
echo "=========================================="
echo ""

for dataset in "${DATASET_LIST[@]}"; do
    # Build the python command
    PY_CMD="python run_gnn_regression_alchemy.py --timeout $TIMEOUT --device $DEVICE"
    if [[ "$MAX_GRAPHS" -gt 0 ]]; then
        PY_CMD="$PY_CMD --max-graphs $MAX_GRAPHS"
    fi
    if [[ -n "$TARGET_INDEX" ]]; then
        PY_CMD="$PY_CMD --target-index $TARGET_INDEX"
    fi
    if [[ "$ALL_TARGETS" == true ]]; then
        PY_CMD="$PY_CMD --all-targets"
    fi
    if [[ "$NORMALIZE" == true ]]; then
        PY_CMD="$PY_CMD --normalize"
    fi
    if [[ "$QUICK" == true ]]; then
        PY_CMD="$PY_CMD --quick"
    fi
    PY_CMD="$PY_CMD $dataset"

    # Build sbatch command
    SBATCH_CMD="sbatch --account=ec12 \
        --job-name=GNNr_${dataset} \
        --partition=${PARTITION} \
        --gpus=${GPU_TYPE}:${GPU_COUNT} \
        --ntasks=1 \
        --cpus-per-task=4 \
        --time=5-00:00:00 \
        --mem-per-cpu=8G \
        --output=logs/GNNr_${dataset}_%j.log \
        --error=logs/GNNr_${dataset}_%j.log \
        --wrap=\"set -e; cd /fp/homes01/u01/ec-claudm/different_datasets; source .different_datasets_venv/bin/activate; $PY_CMD\""

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] $SBATCH_CMD"
    else
        echo "Submitting $dataset..."
        eval "$SBATCH_CMD"
    fi
done

echo ""
echo "=========================================="
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN complete. No jobs submitted."
else
    echo "All ${#DATASET_LIST[@]} jobs submitted!"
    echo "Monitor with: squeue -u \$USER"
fi
echo "=========================================="