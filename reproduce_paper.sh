#!/bin/bash
#
# One entry point for reproducing the paper
#   "Path-Based Gradient Boosting for Graph-Level Prediction"
#
# Usage:
#   ./reproduce_paper.sh <target> [options]
#
# Targets:
#   setup          Download the 12 classification datasets + alchemy_full
#   smoke          Quick end-to-end check on MUTAG (all three methods, --quick)
#   table1         Dataset characteristics            -> dataset_summary.csv
#   classification Tables 2 and 3 (12 datasets x 3 methods)
#   regression     Table 4 (alchemy_full, complete vs restricted)
#   figure1        Learning curve on PROTEINS_full
#   tables         Rebuild paper_results/ from stored runs (no training)
#   all            setup + classification + regression + figure1 + tables
#
# Options:
#   -j, --jobs N      Parallel datasets for the classification target (default: 4)
#   -t, --timeout N   Seconds per dataset, 0 = no limit (default: 0)
#   --device DEV      Device for the GINE baseline: cpu or gpu (default: cpu)
#   --dry-run         Print the commands without running them
#
# Runtimes are dominated by the classification target: the paper allowed a
# 20-hour budget per dataset per method, and the full sweep takes days. Run
# 'smoke' first.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
JOBS=4
TIMEOUT=0
DEVICE="cpu"
DRY_RUN=false

# The 12 balanced datasets of Tables 1-3. Kept in sync with
# shared/constants.py:PAPER_DATASETS.
PAPER_DATASETS="AIDS BZR DHFR MUTAG PROTEINS_full PTC_FM SYNTHETIC \
Tox21_ARE_evaluation Tox21_ARE_testing Tox21_ARE_training \
Tox21_MMP_testing Tox21_MMP_training"

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

[[ $# -lt 1 ]] && { usage; exit 1; }
TARGET="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)    JOBS="$2"; shift 2 ;;
        -t|--timeout) TIMEOUT="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

run() {
    # Print the command the way it would have to be typed, so the dry run is
    # copy-pasteable rather than merely indicative.
    local shown=""
    local a
    for a in "$@"; do
        case "$a" in
            *[[:space:]]*) shown+=" \"$a\"" ;;
            *)             shown+=" $a" ;;
        esac
    done
    echo "+${shown}"
    $DRY_RUN || "$@"
}

banner() { printf '\n=== %s ===\n' "$1"; }

do_setup() {
    banner "Downloading datasets"
    run "$PYTHON" download_datasets.py --paper --regression
}

do_smoke() {
    banner "Smoke test on MUTAG (PathBoost, GINE, graph kernels)"
    echo "Checks that the pipeline runs end to end. It uses --quick: 2 repetitions,"
    echo "3 folds and a single hyperparameter configuration, so the accuracies land"
    echo "roughly in the 75-88% range rather than reproducing Table 2."
    echo
    echo "Reference values (Table 2, full 10x10 CV over the complete grid):"
    echo "  PathBoost 89.11 | GINE 82.83 | WL 76.35 | Graphlet 85.32 | Shortest-path 85.32"
    echo
    echo "Takes about a minute on a laptop CPU."
    run "$PYTHON" run_pathboost.py MUTAG --quick --timeout "$TIMEOUT"
    run "$PYTHON" run_gnn.py MUTAG --quick --device "$DEVICE" --timeout "$TIMEOUT"
    run "$PYTHON" run_kernel.py MUTAG --quick --timeout "$TIMEOUT"
    echo
    echo "Results written to PathBoost_results/, GNN_results/ and Kernel_results/."
}

do_table1() {
    banner "Table 1: dataset characteristics"
    run "$PYTHON" dataset_analysis.py
    echo "Wrote dataset_summary.csv"
}

do_classification() {
    banner "Tables 2 and 3: 12 datasets x {PathBoost, GINE, graph kernels}"
    echo "This is the long one. Budget days, not hours."
    local args=(-j "$JOBS" -t "$TIMEOUT" -d "$PAPER_DATASETS")
    local kernel_args=(-j 1 -t "$TIMEOUT" -d "$PAPER_DATASETS")
    if $DRY_RUN; then
        args+=(--dry-run)
        kernel_args+=(--dry-run)
    fi
    run ./run_parallel.sh pathboost "${args[@]}"
    run ./run_parallel.sh gnn "${args[@]}" --device "$DEVICE"
    # The C++ kernels already use every core, so they run one dataset at a time.
    run ./run_parallel.sh kernel "${kernel_args[@]}"
}

do_regression() {
    banner "Table 4: regression on alchemy_full (complete vs restricted)"
    run "$PYTHON" run_pathboost_regression_alchemy.py alchemy_full
}

do_figure1() {
    banner "Figure 1: learning curve on PROTEINS_full"
    run "$PYTHON" subsample_dataset/run_evaluation.py \
        --dataset PROTEINS_full --device "$DEVICE" --plot
}

do_tables() {
    banner "Rebuilding paper_results/ from stored runs"
    run "$PYTHON" tools/collect_paper_results.py --write
}

case "$TARGET" in
    setup)          do_setup ;;
    smoke)          do_smoke ;;
    table1)         do_table1 ;;
    classification) do_classification ;;
    regression)     do_regression ;;
    figure1)        do_figure1 ;;
    tables)         do_tables ;;
    all)            do_setup; do_table1; do_classification; do_regression; do_figure1; do_tables ;;
    *) echo "Unknown target: $TARGET" >&2; usage; exit 1 ;;
esac

banner "Done: $TARGET"
