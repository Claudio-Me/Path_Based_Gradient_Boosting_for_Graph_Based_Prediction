#!/bin/bash
#
# Submit one SLURM job per regression dataset.
#
# Usage:
#   ./run_regression_fox.sh                        # all datasets, default settings
#   ./run_regression_fox.sh -m 5000                # subsample to 5000 graphs
#   ./run_regression_fox.sh -d "aspirin benzene"   # specific datasets only
#   ./run_regression_fox.sh --dry-run              # preview sbatch commands
#
# Options:
#   -t, --timeout N     Timeout per experiment in seconds (default: 0 = no limit)
#   -m, --max-graphs N  Subsample to at most N graphs (default: 0 = use all)
#   -d, --datasets "D"  Space-separated list of datasets (default: all)
#   --quick             Quick mode: 2x2 CV instead of 10x10
#   --dry-run           Print sbatch commands without submitting

set -e

# Default values
TIMEOUT=0
MAX_GRAPHS=0
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
            echo "Submit one SLURM job per regression dataset."
            echo ""
            echo "Options:"
            echo "  -t, --timeout N     Timeout per experiment in seconds (default: 0)"
            echo "  -m, --max-graphs N  Subsample to at most N graphs (default: 0 = all)"
            echo "  -d, --datasets \"D\"  Space-separated list of datasets"
            echo "  --quick             Quick mode: 2x2 CV instead of 10x10"
            echo "  --dry-run           Print sbatch commands without submitting"
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

mkdir -p logs

echo "=========================================="
echo "Regression SLURM Job Submitter"
echo "=========================================="
echo "Datasets:   ${#DATASET_LIST[@]}"
echo "Timeout:    $TIMEOUT seconds (0 = no limit)"
echo "Max graphs: $MAX_GRAPHS (0 = use all)"
echo "Quick mode: $QUICK"
echo "=========================================="
echo ""

for dataset in "${DATASET_LIST[@]}"; do
    # Build the python command
    PY_CMD="python run_pathboost_regression_alchemy.py --timeout $TIMEOUT"
    if [[ "$MAX_GRAPHS" -gt 0 ]]; then
        PY_CMD="$PY_CMD --max-graphs $MAX_GRAPHS"
    fi
    if [[ "$QUICK" == true ]]; then
        PY_CMD="$PY_CMD --quick"
    fi
    PY_CMD="$PY_CMD $dataset"

    # Build sbatch command
    SBATCH_CMD="sbatch --account=ec12 \
        --job-name=PBr_${dataset} \
        --partition=normal \
        --ntasks=1 \
        --cpus-per-task=4 \
        --time=5-00:00:00 \
        --mem-per-cpu=8G \
        --output=logs/PBr_${dataset}_%j.log \
        --error=logs/PBr_${dataset}_%j.log \
        --wrap=\"set -e; cd /fp/homes01/u01/ec-claudm/Different_datasets/different_datasets; source .different_datasets_venv/bin/activate; $PY_CMD\""

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
