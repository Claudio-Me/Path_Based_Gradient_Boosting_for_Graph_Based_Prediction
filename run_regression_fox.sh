#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=PB_regression
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=55
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -e
mkdir -p logs

cd $SLURM_SUBMIT_DIR
source .different_datasets_venv/bin/activate

./run_parallel_regression.sh -j 7
