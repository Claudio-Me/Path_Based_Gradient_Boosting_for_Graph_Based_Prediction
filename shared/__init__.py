"""Shared utilities for PathBoost, GNN, and Kernel evaluation scripts."""

from .cli import create_argument_parser, validate_datasets
from .csv_utils import ResultsCSVWriter, get_timestamped_path
from .timeout import run_with_timeout, TimeoutException
from .logging_config import setup_logging
from .constants import (
    DEFAULT_TIMEOUT,
    CV_SEED,
    NUM_REPETITIONS,
    NUM_FOLDS,
    NX_GRAPHS_DIR,
    get_base_dir,
    PATHBOOST_METRICS,
    GNN_METRICS,
    KERNEL_METRICS,
    KERNEL_METHODS,
    ALL_DATASETS,
)

__all__ = [
    # CLI
    'create_argument_parser',
    'validate_datasets',
    # CSV
    'ResultsCSVWriter',
    'get_timestamped_path',
    # Timeout
    'run_with_timeout',
    'TimeoutException',
    # Logging
    'setup_logging',
    # Constants
    'DEFAULT_TIMEOUT',
    'CV_SEED',
    'NUM_REPETITIONS',
    'NUM_FOLDS',
    'NX_GRAPHS_DIR',
    'get_base_dir',
    'PATHBOOST_METRICS',
    'GNN_METRICS',
    'KERNEL_METRICS',
    'KERNEL_METHODS',
    'ALL_DATASETS',
]
