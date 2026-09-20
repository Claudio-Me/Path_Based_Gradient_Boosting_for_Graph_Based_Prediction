#!/bin/bash
# SLURM wrapper around download_datasets.py (Fox HPC).
# For a normal machine just run:  python download_datasets.py --paper --regression
#SBATCH --account=ec12
#SBATCH --job-name=download_data
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -e
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
if [ -f .different_datasets_venv/bin/activate ]; then
    source .different_datasets_venv/bin/activate
elif [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

python download_datasets.py --paper --regression "$@"
