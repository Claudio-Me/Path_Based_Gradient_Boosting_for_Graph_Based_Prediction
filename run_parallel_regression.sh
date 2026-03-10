#!/bin/bash
#
# Run PathBoost regression evaluation in parallel across regression datasets.
#
# Each dataset runs as a separate process, writing to its own CSV file.
#
# Usage:
#   ./run_parallel_regression.sh                    # all datasets, 4 jobs
#   ./run_parallel_regression.sh -j 7               # all datasets, 7 jobs
#   ./run_parallel_regression.sh -j 3 -d "aspirin benzene toluene"
#   ./run_parallel_regression.sh --quick --dry-run   # preview quick-mode commands
#
# Options:
#   -j, --jobs N        Number of parallel jobs (default: 4)
#   -t, --timeout N     Timeout per experiment in seconds (default: 0 = no limit)
#   -d, --datasets "D"  Space-separated list of datasets (default: all)
#   --quick             Quick mode: 2x2 CV instead of 10x10
#   --dry-run           Print commands without executing

set -e

# Default values
JOBS=4
TIMEOUT=0
QUICK=false
DRY_RUN=false
DATASETS=""

ALL_DATASETS=(
    alchemy_full
    aspirin
    benzene
    ethanol
    malonaldehyde
    naphthalene
    salicylic_acid
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
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -d|--datasets)
            DATASETS="$2"
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
            echo "Options:"
            echo "  -j, --jobs N        Number of parallel jobs (default: 4)"
            echo "  -t, --timeout N     Timeout per experiment in seconds (default: 0 = no limit)"
            echo "  -d, --datasets \"D\"  Space-separated list of datasets"
            echo "  --quick             Quick mode: 2x2 CV instead of 10x10"
            echo "  --dry-run           Print commands without executing"
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
done

# Use all datasets if none specified
if [[ -z "$DATASETS" ]]; then
    DATASET_LIST=("${ALL_DATASETS[@]}")
else
    read -ra DATASET_LIST <<< "$DATASETS"
fi

# Build base command
BASE_CMD="python run_pathboost_regression_alchemy.py --timeout $TIMEOUT"
if [[ "$QUICK" == true ]]; then
    BASE_CMD="$BASE_CMD --quick"
fi

# Create log directory
LOG_DIR="logs/regression_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Parallel Regression Runner"
echo "=========================================="
echo "Jobs:       $JOBS"
echo "Timeout:    $TIMEOUT seconds (0 = no limit)"
echo "Quick mode: $QUICK"
echo "Datasets:   ${#DATASET_LIST[@]}"
echo "Log dir:    $LOG_DIR"
echo "=========================================="

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "DRY RUN - Commands that would be executed:"
    echo ""
    for dataset in "${DATASET_LIST[@]}"; do
        echo "$BASE_CMD $dataset > $LOG_DIR/${dataset}.log 2>&1"
    done
    exit 0
fi

# Function to run a single dataset
run_dataset() {
    local dataset="$1"
    local log_file="$LOG_DIR/${dataset}.log"

    echo "[$(date +%H:%M:%S)] Starting $dataset..."

    if $BASE_CMD "$dataset" > "$log_file" 2>&1; then
        echo "[$(date +%H:%M:%S)] Completed $dataset"
    else
        echo "[$(date +%H:%M:%S)] FAILED $dataset (see $log_file)"
    fi
}

export -f run_dataset
export BASE_CMD LOG_DIR

# Run in parallel using xargs
echo ""
echo "Starting parallel execution with $JOBS jobs..."
echo ""

printf '%s\n' "${DATASET_LIST[@]}" | xargs -P "$JOBS" -I {} bash -c 'run_dataset "$@"' _ {}

echo ""
echo "=========================================="
echo "All jobs completed!"
echo "Logs saved to: $LOG_DIR"
echo "Results saved to: PathBoost_results/Sequential_PathBoost_Regression_Performance_*.csv"
echo "=========================================="
