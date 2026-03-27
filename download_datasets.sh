#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=download_data
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -e
mkdir -p logs

cd $SLURM_SUBMIT_DIR
source .different_datasets_venv/bin/activate

python -c "
from torch_geometric.datasets import TUDataset
datasets = ['aspirin','benzene','ethanol','malonaldehyde','naphthalene','toluene','uracil','ZINC_full','ZINC_test','ZINC_train','ZINC_val']
for name in datasets:
    print(f'Downloading {name}...', end=' ', flush=True)
    try:
        TUDataset(root=f'tudataset/tud_benchmark/datasets/{name}', name=name)
        print('OK')
    except Exception as e:
        print(f'FAILED: {e}')
print('All done.')
"
