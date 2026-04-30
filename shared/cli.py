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
        help=f'Timeout per dataset in seconds (default: {DEFAULT_TIMEOUT} = 20 hours)'
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
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser


def validate_datasets(dataset_names: List[str]) -> List[str]:
    """
    Validate dataset names exist in the datasets directory.

    Args:
        dataset_names: List of dataset names to validate.
                      If empty, returns all known datasets.

    Returns:
        List of valid dataset names. Warns about invalid ones.
    """
    base_dir = get_base_dir()

    if not dataset_names:
        # Return all datasets that exist
        valid = []
        for name in ALL_DATASETS:
            dataset_path = os.path.join(base_dir, name)
            if os.path.exists(dataset_path):
                valid.append(name)
        return valid

    valid = []
    for name in dataset_names:
        dataset_path = os.path.join(base_dir, name)
        if os.path.exists(dataset_path):
            valid.append(name)
        else:
            print(f"WARNING: Dataset '{name}' not found at {dataset_path}, skipping.")

    return valid
