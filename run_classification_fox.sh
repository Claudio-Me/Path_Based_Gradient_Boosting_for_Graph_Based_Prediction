#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=PB_classify
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=55
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -e
mkdir -p logs

cd $SLURM_SUBMIT_DIR
source .different_datasets_venv/bin/activate

# Run all three in parallel within the same job
./run_parallel.sh pathboost -j 7 &
./run_parallel.sh gnn -j 7 --device cpu &
./run_parallel.sh kernel -j 1 &

wait
echo "All classification runs complete."
