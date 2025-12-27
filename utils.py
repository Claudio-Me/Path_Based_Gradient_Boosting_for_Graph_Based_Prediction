"""
Utility functions for dataset loading and preprocessing.

This module provides core utilities for:
- Loading and caching NetworkX graphs from TU datasets
- Finding categorical node attributes for PathBoost
- Preprocessing labels for binary classification

Note: Timeout and CSV utilities have been moved to the `shared` module.
"""
import os
import pickle
import numpy as np
import networkx as nx
from typing import List, Optional
from collections import defaultdict
from collections.abc import Hashable

from tudataset.tud_benchmark.auxiliarymethods.reader import tud_to_networkx


# -----------------------------
# Generic helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Dataset conversion / loading
# -----------------------------
def convert_to_networkx(dataset_name: Optional[str] = None, path_to_dataset: Optional[str] = None) -> List[nx.Graph]:
    """
    Convert graphs to NetworkX format via tud_to_networkx.

    For each node, if an attribute is a list, create new attributes for each value.
    This expands list-valued attributes into individual scalar attributes.

    Args:
        dataset_name: Name of the TU dataset
        path_to_dataset: Path to the dataset directory

    Returns:
        List of NetworkX Graph objects
    """
    nx_graphs = tud_to_networkx(ds_name=dataset_name, path_to_dataset=path_to_dataset)
    for graph in nx_graphs:
        for _, node_data in graph.nodes(data=True):
            attrs_to_add = {}
            attrs_to_remove = []
            for attr, value in node_data.items():
                if isinstance(value, list):
                    for idx, v in enumerate(value):
                        attrs_to_add[f"{attr}_{idx}"] = v
                    attrs_to_remove.append(attr)
            for attr in attrs_to_remove:
                del node_data[attr]
            node_data.update(attrs_to_add)
    return nx_graphs


def load_or_build_nx_graphs(dataset_name: str, dataset_path: str, nx_graphs_dir: str = "nx_graphs") -> List[nx.Graph]:
    """
    Load NetworkX graphs from cache or build from TU dataset.

    Graphs are cached as pickle files for faster subsequent loading.

    Args:
        dataset_name: Name of the TU dataset
        dataset_path: Path to the dataset directory
        nx_graphs_dir: Directory for caching NetworkX graphs

    Returns:
        List of NetworkX Graph objects
    """
    ensure_dir(nx_graphs_dir)
    nx_graphs_file = os.path.join(nx_graphs_dir, f"{dataset_name}_nx_graphs.pkl")
    if os.path.exists(nx_graphs_file):
        with open(nx_graphs_file, "rb") as f:
            nx_graphs = pickle.load(f)
        print(f"Loaded NetworkX graphs from {nx_graphs_file}")
        return nx_graphs
    nx_graphs = convert_to_networkx(dataset_name=dataset_name, path_to_dataset=dataset_path)
    with open(nx_graphs_file, "wb") as f:
        pickle.dump(nx_graphs, f)
    print(f"Saved NetworkX graphs to {nx_graphs_file}")
    return nx_graphs


# -----------------------------
# Features / labels utils
# -----------------------------
def find_categorical_node_attributes(graphs: List[nx.Graph], max_unique_values: int = 200) -> Optional[str]:
    """
    Select the categorical node attribute having the most unique values (classes).

    Used by PathBoost to find suitable anchor node labels.

    Args:
        graphs: List of NetworkX graphs
        max_unique_values: Maximum number of unique values for an attribute to be considered categorical

    Returns:
        Name of the selected attribute, or None if no suitable attribute found
    """
    attr_values = defaultdict(list)
    for graph in graphs:
        for _, node_data in graph.nodes(data=True):
            for attr, value in node_data.items():
                attr_values[attr].append(value)

    selected_attr = None
    max_classes = 0
    for attr, values in attr_values.items():
        if not all(isinstance(v, Hashable) for v in values):
            continue
        unique_vals = set(values)
        if (all(isinstance(v, (str, int)) for v in unique_vals)
                and 1 < len(unique_vals) <= max_unique_values):
            if len(unique_vals) > max_classes:
                max_classes = len(unique_vals)
                selected_attr = attr
    return selected_attr


def preprocess_labels(labels: np.ndarray) -> Optional[np.ndarray]:
    """
    Ensure labels are binary (0/1).

    If labels are binary but not {0,1}, map them to {0,1}.
    If labels are not binary, return None.

    Args:
        labels: Array of class labels

    Returns:
        Preprocessed labels as {0,1} array, or None if not binary
    """
    unique = np.unique(labels)
    if len(unique) != 2:
        return None
    if set(unique) != {0, 1}:
        mapping = {unique[0]: 0, unique[1]: 1}
        labels = np.array([mapping[l] for l in labels])
    return labels


# -----------------------------
# Legacy compatibility
# -----------------------------
# The following imports are kept for backward compatibility with existing code
# that may import these from utils. New code should import from shared module.

def get_unique_csv_path(prefix: str = "combined_dataset_summary", ext: str = ".csv") -> str:
    """
    Generate unique CSV path with UUID.

    DEPRECATED: Use shared.csv_utils.get_timestamped_path instead for new code.
    """
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{unique_id}{ext}"


def append_row_to_csv(csv_path: str, row: dict) -> None:
    """
    Append a row to CSV file.

    DEPRECATED: Use shared.csv_utils.ResultsCSVWriter instead for new code.
    """
    import csv
    header = row.keys()
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
