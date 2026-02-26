#!/bin/bash
#
# Run evaluation scripts in parallel across multiple datasets.
#
# Usage:
#   ./run_parallel.sh <script> [options]
#
# Scripts:
#   pathboost  - Run run_pathboost.py
#   gnn        - Run run_gnn.py
#   kernel     - Run run_kernel.py
#
# Options:
#   -j, --jobs N        Number of parallel jobs (default: 4)
#   -t, --timeout N     Timeout per dataset in seconds (default: 72000)
#   -d, --datasets "D"  Space-separated list of datasets (default: all)
#   --device DEV        Device for GNN: cpu/gpu (default: gpu)
#   --dry-run           Print commands without executing
#
# Examples:
#   ./run_parallel.sh pathboost -j 4
#   ./run_parallel.sh gnn -j 2 --device cpu
#   ./run_parallel.sh kernel -j 8 -d "MUTAG AIDS PTC_MR"
#   ./run_parallel.sh pathboost -j 4 --dry-run

set -e

# Default values
JOBS=4
TIMEOUT=0
DEVICE="gpu"
DRY_RUN=false
DATASETS=""

# All available datasets (from run_pathboost_all_datasets.py)
ALL_DATASETS=(
    MUTAG AIDS
    MCF-7 MCF-7H MOLT-4 MOLT-4H Mutagenicity NCI1 NCI109
    NCI-H23 NCI-H23H OVCAR-8 OVCAR-8H P388 P388H PC-3 PC-3H
    PTC_FM PTC_FR PTC_MM PTC_MR SF-295 SF-295H SN12C SN12CH
    SW-620 SW-620H
    Tox21_AhR_training Tox21_AhR_testing Tox21_AhR_evaluation
    Tox21_AR_training Tox21_AR_testing Tox21_AR_evaluation
    Tox21_AR-LBD_training Tox21_AR-LBD_testing Tox21_AR-LBD_evaluation
    Tox21_ARE_training Tox21_ARE_testing Tox21_ARE_evaluation
    Tox21_aromatase_training Tox21_aromatase_testing Tox21_aromatase_evaluation
    Tox21_ATAD5_training Tox21_ATAD5_testing Tox21_ATAD5_evaluation
    Tox21_ER_training Tox21_ER_testing Tox21_ER_evaluation
    Tox21_ER-LBD_training Tox21_ER-LBD_testing Tox21_ER-LBD_evaluation
    Tox21_HSE_training Tox21_HSE_testing Tox21_HSE_evaluation
    Tox21_MMP_training Tox21_MMP_testing Tox21_MMP_evaluation
    Tox21_p53_training Tox21_p53_testing Tox21_p53_evaluation
    Tox21_PPAR-gamma_training Tox21_PPAR-gamma_testing Tox21_PPAR-gamma_evaluation
    UACC257 UACC257H Yeast YeastH DD KKI OHSU Peking_1
    PROTEINS PROTEINS_full DBLP_v1 TWITTER-Real-Graph-Partial SYNTHETIC DHFR_MD BZR_MD COX2
)

# Parse script name
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <script> [options]"
    echo "Scripts: pathboost, gnn, kernel"
    echo "Run '$0 <script> --help' for more options"
    exit 1
fi

SCRIPT_NAME="$1"
shift

case "$SCRIPT_NAME" in
    pathboost)
        PYTHON_SCRIPT="run_pathboost.py"
        ;;
    gnn)
        PYTHON_SCRIPT="run_gnn.py"
        ;;
    kernel)
        PYTHON_SCRIPT="run_kernel.py"
        ;;
    -h|--help)
        echo "Usage: $0 <script> [options]"
        echo ""
        echo "Scripts:"
        echo "  pathboost  - Run run_pathboost.py"
        echo "  gnn        - Run run_gnn.py"
        echo "  kernel     - Run run_kernel.py"
        echo ""
        echo "Options:"
        echo "  -j, --jobs N        Number of parallel jobs (default: 4)"
        echo "  -t, --timeout N     Timeout per dataset in seconds (default: 72000)"
        echo "  -d, --datasets \"D\"  Space-separated list of datasets (default: all)"
        echo "  --device DEV        Device for GNN: cpu/gpu (default: gpu)"
        echo "  --dry-run           Print commands without executing"
        exit 0
        ;;
    *)
        echo "Error: Unknown script '$SCRIPT_NAME'"
        echo "Valid scripts: pathboost, gnn, kernel"
        exit 1
        ;;
esac

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
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 $SCRIPT_NAME [options]"
            echo ""
            echo "Options:"
            echo "  -j, --jobs N        Number of parallel jobs (default: 4)"
            echo "  -t, --timeout N     Timeout per dataset in seconds (default: 72000)"
            echo "  -d, --datasets \"D\"  Space-separated list of datasets"
            echo "  --device DEV        Device for GNN: cpu/gpu (default: gpu)"
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
BASE_CMD="python $PYTHON_SCRIPT --timeout $TIMEOUT"
if [[ "$SCRIPT_NAME" == "gnn" ]]; then
    BASE_CMD="$BASE_CMD --device $DEVICE"
fi

# Create log directory
LOG_DIR="logs/${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Parallel Evaluation Runner"
echo "=========================================="
echo "Script:     $PYTHON_SCRIPT"
echo "Jobs:       $JOBS"
echo "Timeout:    $TIMEOUT seconds ($(echo "scale=1; $TIMEOUT/3600" | bc) hours)"
echo "Datasets:   ${#DATASET_LIST[@]}"
echo "Log dir:    $LOG_DIR"
if [[ "$SCRIPT_NAME" == "gnn" ]]; then
    echo "Device:     $DEVICE"
fi
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
echo "=========================================="
