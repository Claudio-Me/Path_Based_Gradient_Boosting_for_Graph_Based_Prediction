"""Dataset splitting utilities for creating train/test partitions."""
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_or_build_nx_graphs
from shared.constants import CV_SEED, NX_GRAPHS_DIR, get_base_dir
from tudataset.tud_benchmark.auxiliarymethods.datasets import get_dataset


def create_dataset_splits(dataset_name, n_splits, test_size=0.1, base_seed=CV_SEED):
    """
    Load a TU dataset and create 4 train/test splits with different seeds.

    Returns:
        splits: List of 4 dicts with keys: train_graphs, test_graphs,
                train_labels, test_labels, train_idx, test_idx, seed
        graphs: Full list of NetworkX graphs
        labels: Full array of labels

    Example:
        splits, graphs, labels = create_dataset_splits('PROTEINS_full')
        train_graphs = splits[0]['train_graphs']
        test_graphs = splits[0]['test_graphs']
        train_labels = splits[0]['train_labels']
        test_labels = splits[0]['test_labels']
    """
    base_dir = get_base_dir()
    dataset_path = os.path.join(base_dir, dataset_name)

    nx_graphs = load_or_build_nx_graphs(dataset_name, dataset_path, NX_GRAPHS_DIR)
    labels = np.array(get_dataset(dataset_name, regression=False)).flatten()

    graphs_array = np.array(nx_graphs, dtype=object)
    indices = np.arange(len(nx_graphs))

    splits = []
    for i in range(n_splits):
        seed = base_seed + i
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed
        )
        splits.append({
            'train_graphs': list(graphs_array[train_idx]),
            'test_graphs': list(graphs_array[test_idx]),
            'train_labels': labels[train_idx],
            'test_labels': labels[test_idx],
            'train_idx': train_idx,
            'test_idx': test_idx,
            'seed': seed
        })

    return splits, nx_graphs, labels


def subsample_train(train_graphs, train_labels, seed=CV_SEED):
    """
    Subsample training data at 10%, 20%, ..., 100%.

    Returns:
        List of 10 dicts with keys: graphs, labels, percentage

    Example:
        subsamples = subsample_train(splits[0]['train_graphs'], splits[0]['train_labels'])
        for s in subsamples:
            print(f"{s['percentage']}%: {len(s['graphs'])} graphs")
    """
    rng = np.random.default_rng(seed)
    n = len(train_graphs)
    graphs_array = np.array(train_graphs, dtype=object)
    labels_array = np.array(train_labels)

    # Shuffle indices once
    indices = np.arange(n)
    rng.shuffle(indices)

    subsamples = []
    for pct in range(10, 101, 10):
        size = int(n * pct / 100)
        subset_idx = indices[:size]
        subsamples.append({
            'graphs': list(graphs_array[subset_idx]),
            'labels': labels_array[subset_idx],
            'percentage': pct
        })

    return subsamples
