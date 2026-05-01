#!/usr/bin/env bash
# Submit one job per (dataset, algorithm) for the four missing datasets.
# Non-linear kernels only: WL_subtree, Graphlet, Shortest_path, WLOA
#
# Usage (on Fox, from ~/different_datasets):
#   bash submit_missing_datasets.sh
#   bash submit_missing_datasets.sh --dry-run   # print commands without submitting

set -e

DRYRUN=false
[[ "${1:-}" == "--dry-run" ]] && DRYRUN=true

submit() {
    if $DRYRUN; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

WDIR=/fp/homes01/u01/ec-claudm/different_datasets
VENV="$WDIR/.different_datasets_venv/bin/activate"
MODULES="module load Python/3.13.5-GCCcore-14.3.0 CUDA/12.8.0"
SETUP="set -e; $MODULES; cd $WDIR; source $VENV"

DATASETS=(OHSU Mutagenicity BZR_MD DHFR_MD)

# ── PathBoost (CPU, 5 days) ────────────────────────────────────────────────
for DS in "${DATASETS[@]}"; do
    submit sbatch \
        --account=ec12 \
        --job-name="pb_${DS}" \
        --partition=normal \
        --ntasks=1 \
        --cpus-per-task=4 \
        --time=5-00:00:00 \
        --mem-per-cpu=8G \
        --output="logs/pb_${DS}_%j.log" \
        --error="logs/pb_${DS}_%j.log" \
        --wrap="$SETUP; python run_pathboost.py $DS"
done

# ── GNN (GPU RTX3090, 24 h) ───────────────────────────────────────────────
for DS in "${DATASETS[@]}"; do
    submit sbatch \
        --account=ec12 \
        --job-name="gnn_${DS}" \
        --partition=accel \
        --gpus=rtx30:1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --time=1-00:00:00 \
        --mem-per-cpu=8G \
        --output="logs/gnn_${DS}_%j.log" \
        --error="logs/gnn_${DS}_%j.log" \
        --wrap="$SETUP; python run_gnn.py $DS"
done

# ── Kernels — non-linear only (CPU, 5 days) ───────────────────────────────
for DS in "${DATASETS[@]}"; do
    submit sbatch \
        --account=ec12 \
        --job-name="ker_${DS}" \
        --partition=normal \
        --ntasks=1 \
        --cpus-per-task=4 \
        --time=5-00:00:00 \
        --mem-per-cpu=8G \
        --output="logs/ker_${DS}_%j.log" \
        --error="logs/ker_${DS}_%j.log" \
        --wrap="$SETUP; python run_kernel.py $DS --kernels WL_subtree Graphlet Shortest_path WLOA"
done
