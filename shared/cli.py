"""CLI argument parsing for evaluation scripts."""
import argparse
import os
from typing import List

from .constants import DEFAULT_TIMEOUT, ALL_DATASETS, get_base_dir


def create_argument_parser(script_name: str, supports_device: bool = False) -> argparse.ArgumentParser:
    """
    Create standardized argument parser for evaluation scripts.

    Args:
        script_name: Name of the script (pathboost, gnn, kernel)
        supports_device: Whether to add --device argument (GNN only)

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=f'Run {script_name} evaluation on TU datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python run_{script_name.lower()}.py MUTAG PTC_MR NCI1
  python run_{script_name.lower()}.py MUTAG --timeout 36000
  python run_{script_name.lower()}.py  # runs on all datasets
        """
    )

    parser.add_argument(
        'datasets',
        nargs='*',
        default=[],
        help='Dataset names (positional). If none provided, runs on all datasets.'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f'Timeout per dataset in seconds (default: {DEFAULT_TIMEOUT}, 0 = no limit)'
    )

    if supports_device:
        parser.add_argument(
            '--device',
            type=str,
            choices=['cpu', 'gpu', 'cuda'],
            default='gpu',
            help='Device to run on: cpu or gpu (default: gpu)'
        )

    parser.add_argument(
        '--repetitions',
        type=int,
        default=10,
        help='Number of CV repetitions (default: 10)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: reduced CV and hyperparameter grid, for checking that '
             'the pipeline runs. Does NOT reproduce the published tables.'
    )

    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Do not download missing datasets; skip them instead '
             '(useful for batch/SLURM runs on a shared dataset directory).'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser


def validate_datasets(dataset_names: List[str], download: bool = True) -> List[str]:
    """
    Validate dataset names, downloading any that are not present yet.

    TU datasets are not distributed with this repository; they are fetched from
    the TUDatasets collection through PyTorch Geometric on first use. A fresh
    clone therefore has an empty dataset directory, so by default a missing
    dataset is downloaded rather than skipped.

    Args:
        dataset_names: List of dataset names to validate.
                      If empty, returns all known datasets.
        download: Download missing datasets (default). When False, missing
                  datasets are reported and skipped.

    Returns:
        List of valid dataset names. Warns about ones that could not be obtained.
    """
    base_dir = get_base_dir()

    if not dataset_names:
        # No explicit selection: only run on what is already available locally.
        # Downloading all 80+ datasets implicitly would be a surprising side effect.
        valid = []
        for name in ALL_DATASETS:
            dataset_path = os.path.join(base_dir, name)
            if os.path.exists(dataset_path):
                valid.append(name)
        if not valid:
            print(
                "No datasets found locally. Download them first, e.g.:\n"
                "    python download_datasets.py --paper\n"
                "or name the datasets explicitly to have them downloaded on demand."
            )
        return valid

    valid = []
    for name in dataset_names:
        dataset_path = os.path.join(base_dir, name)
        if os.path.exists(dataset_path):
            valid.append(name)
            continue

        if not download:
            print(f"WARNING: Dataset '{name}' not found at {dataset_path}, skipping.")
            continue

        try:
            from utils import ensure_dataset_downloaded
            ensure_dataset_downloaded(name, verbose=True)
            valid.append(name)
        except Exception as exc:
            print(f"WARNING: Dataset '{name}' could not be downloaded ({exc}), skipping.")

    return valid
