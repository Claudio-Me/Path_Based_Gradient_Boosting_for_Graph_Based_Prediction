import os
import pickle
import uuid
import signal
import numpy as np
import networkx as nx
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from collections.abc import Hashable
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


from tudataset.tud_benchmark.auxiliarymethods.reader import tud_to_networkx
from multiprocessing import Process, Queue


# -----------------------------
# Generic helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_unique_csv_path(prefix: str = "combined_dataset_summary", ext: str = ".csv") -> str:
    unique_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{unique_id}{ext}"


def append_row_to_csv(csv_path: str,  row: Dict) -> None:
    import csv
    header = row.keys()
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------
# Timeouts (process-based, robust against sklearn swallowing exceptions)
# -----------------------------


class TimeoutException(Exception):
    pass

def _run_in_process(q, func, args, kwargs):
    try:
        res = func(*args, **(kwargs or {}))
        q.put(("OK", res))
    except Exception as e:
        q.put(("ERR", f"{type(e).__name__}: {e}"))

def run_with_timeout(func, args=(), kwargs=None, timeout_sec: int = 36000):
    """
    Execute func(*args, **kwargs) in a separate process and enforce a hard timeout.
    Returns (result, False) on success, or (None, True) on timeout/error.
    """
    q = Queue()
    p = Process(target=_run_in_process, args=(q, func, args, kwargs))
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"Timeout: {func.__name__} exceeded {timeout_sec/3600:.2f}h")
        return None, True

    if q.empty():
        # treat as error
        return None, True

    status, payload = q.get()
    if status == "OK":
        return payload, False
    else:
        # payload is a stringified error
        print(f"{func.__name__} failed: {payload}")
        return None, True


# -----------------------------
# Dataset conversion / loading
# -----------------------------
def convert_to_networkx(dataset_name: Optional[str] = None, path_to_dataset: Optional[str] = None) -> List[nx.Graph]:
    """
    Convert graphs to NetworkX format via tud_to_networkx.
    For each node, if an attribute is a list, create new attributes for each value.
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
    Select the categorical node attribute having the most unique values (classes),
    among attributes with 1 < unique_vals <= max_unique_values.
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
    Ensure labels are binary (0/1). If not, return None.
    If binary but not {0,1}, map to {0,1}.
    """
    unique = np.unique(labels)
    if len(unique) != 2:
        return None
    if set(unique) != {0, 1}:
        mapping = {unique[0]: 0, unique[1]: 1}
        labels = np.array([mapping[l] for l in labels])
    return labels


# -----------------------------
# Baselines
# -----------------------------
def pathboost_baseline(nx_graphs: List[nx.Graph], labels: np.ndarray, dataset_name: Optional[str] = None) -> float:
    """
    10-fold CV PathBoost baseline on NetworkX graphs.
    Saves a model per fold.
    Returns mean accuracy, or -1 if not applicable.
    """
    # Lazy imports to avoid cycles
    from dataset_analysis import dataset_prescreening
    from save_and_load_models import save_pathboost_model
    from extended_path_boost import SequentialPathBoostClassifier

    # Pre-screening: class distribution (non-blocking / no save)
    dataset_prescreening(labels, dataset_name=dataset_name, show_plot=False, save_plot=False)

    processed_labels = preprocess_labels(labels)
    if processed_labels is None:
        print("Skipped: Dataset is not binary classification.")
        return -1
    labels = processed_labels

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_scores: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(nx_graphs)):
        X_train = [nx_graphs[i] for i in train_idx]
        y_train = labels[train_idx]
        X_test = [nx_graphs[i] for i in test_idx]
        y_test = labels[test_idx]

        categorical_attr = find_categorical_node_attributes(X_train)
        if not categorical_attr:
            print("Skipped: No categorical node attributes found.")
            return -1

        # Collect distinct anchor labels from training graphs
        anchor_labels = []
        for g in X_train:
            for _, node_data in g.nodes(data=True):
                if categorical_attr in node_data:
                    anchor_labels.append(node_data[categorical_attr])
        anchor_labels = list(set(anchor_labels))

        kwargs_for_base_learner = {
            'random_state': 0,
            'splitter': 'best',
            'criterion': "squared_error",
            'max_leaf_nodes': 10
        }

        model = SequentialPathBoostClassifier(
            n_iter=1000,
            learning_rate=0.01,
            parameters_variable_importance=None,
            BaseLearnerClass=DecisionTreeRegressor,
            SelectorClass=DecisionTreeClassifier,
            kwargs_for_base_learner=kwargs_for_base_learner,
            kwargs_for_selector={},
            verbose=True,
            use_tree_boost=False
        )

        model.fit(
            X_train, y_train,
            anchor_nodes_label_name=categorical_attr,
            list_anchor_nodes_labels=anchor_labels,
            eval_set=[(X_test, y_test)]
        )
        accuracy_scores.append(model.eval_sets_accuracy_[0][-1])

        # Save the model for each fold
        if dataset_name:
            save_pathboost_model(model, f"{dataset_name}_fold{fold_idx}")

    return float(np.mean(accuracy_scores)) if accuracy_scores else -1.0





def evaluate_pathboost_with_timeout(nx_graphs: List[nx.Graph], labels: np.ndarray,
                                    dataset_name: str, timeout_sec: int = 36000,
                                    cv_seed: Optional[int] = None) -> Dict:
    """
    Run pathboost_gridcv_baseline with a timeout. Always return a 3-tuple:
      (accuracy, std_10, std_100)
    On timeout or failure, return {}.
    """
    res, timed_out = run_with_timeout(
        pathboost_gridcv_baseline,
        args=(nx_graphs, labels, dataset_name),
        kwargs={'cv_seed': cv_seed},  # pass the seed
        timeout_sec=timeout_sec
    )
    if timed_out:
        return {}


    else:
        return res


def evaluate_gnn_with_timeout(dataset_name: str, timeout_sec: int = 36000,
                              cv_seed: Optional[int] = None) -> Dict:
    """
    Run run_gnn_baseline with a timeout.

    Returns dict metric -> (mean, std_10, std_100) or {} on failure.
    """
    from GNN_baseline import run_gnn_baseline
    res, timed_out = run_with_timeout(
        run_gnn_baseline,
        args=(dataset_name,),
        kwargs={'cv_seed': cv_seed},
        timeout_sec=timeout_sec
    )
    if timed_out or res is None:
        return {}
    if isinstance(res, dict):
        return res



def pathboost_gridcv_baseline(
    nx_graphs: List[nx.Graph],
    labels: np.ndarray,
    dataset_name: Optional[str] = None,
    param_grid: Optional[Dict] = None,
    n_repeats: int = 10,
    cv_seed: Optional[int] = None,
) -> Dict:
    """
    Run GridSearchCV over the param_grid, repeating a 10-fold CV `n_repeats` times
    (each repeat uses a freshly shuffled KFold with a different random_state).

    Returns a tuple: (avg_accuracy, std_of_means, std_across_all_folds)
      - avg_accuracy: mean of the best mean-accuracies across repeats
      - std_of_means: std of those best-mean accuracies (across repeats)
      - std_across_all_folds: std computed over every single fold accuracy across all repeats
    Returns (-1.0, -1.0, -1.0) on failure or if dataset is not suitable.
    """
    # Lazy imports to avoid circular imports / heavy deps at module import
    from sklearn.model_selection import KFold, GridSearchCV
    from extended_path_boost import SequentialPathBoostClassifier
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from dataset_analysis import dataset_prescreening

    # Pre-screening: class distribution (silent)
    dataset_prescreening(labels, dataset_name=dataset_name, show_plot=False, save_plot=False)

    # Ensure binary labels mapped to {0,1}
    processed_labels = preprocess_labels(labels)
    if processed_labels is None:
        print("Skipped: Dataset is not binary classification.")
        return {}
    labels = processed_labels




    # Discover categorical node attribute from all graphs (constant for GridSearch)
    categorical_attr = find_categorical_node_attributes(nx_graphs)
    if not categorical_attr:
        print("Skipped: No categorical node attributes found.")
        return {}

    # Anchor labels from all graphs (constant fit_param for all CV runs)
    anchor_labels = []
    for g in nx_graphs:
        for _, node_data in g.nodes(data=True):
            if categorical_attr in node_data:
                anchor_labels.append(node_data[categorical_attr])
    anchor_labels = list(set(anchor_labels))
    if len(anchor_labels) < 2:
        print("Skipped: Not enough distinct anchor labels.")
        return {}

    # Choose a base seed for reproducibility
    base_seed = 42 if cv_seed is None else int(cv_seed)

    # Base estimator
    base_estimator = SequentialPathBoostClassifier(
        n_iter=1000, #1000
        learning_rate=0.01,
        parameters_variable_importance=None,
        BaseLearnerClass=DecisionTreeRegressor,
        SelectorClass=DecisionTreeClassifier,
        kwargs_for_base_learner={'random_state': base_seed, 'splitter': 'best', 'criterion': "squared_error", 'max_leaf_nodes': 10},
        kwargs_for_selector={},
        verbose=False,
        use_tree_boost=False
    )

    # Default grid if none provided
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.1, 0.02, 0.01],
            'max_path_length': [3, 4, 6],
            'kwargs_for_base_learner': [{'max_depth': 3}, {'max_depth': 4}],
            'n_iter': [500, 1000, 1500]
        }



    best_means: List[float] = []
    all_fold_scores: List[float] = []

    try:
        """
        Repeated K-Fold Cross-Validation with GridSearchCV for Hyperparameter Tuning

        This code performs a robust model evaluation using repeated cross-validation combined
        with grid search to find optimal hyperparameters and assess model performance stability.

        METHODOLOGY:
        ------------
        The evaluation follows a nested approach:
        1. Outer loop: Repeats the entire cross-validation process n_repeats times with 
           different random seeds to assess performance variability
        2. Inner loop: Each repetition uses 10-fold cross-validation within GridSearchCV
        3. Total evaluations: n_repeats × 10 folds (e.g., 10 repetitions = 100 fold evaluations)

        OPTIMIZATION STRATEGY:
        ---------------------
        - GridSearchCV explores all combinations in param_grid
        - Models are optimized based on 'balanced_accuracy' (better for imbalanced datasets)
        - Five metrics are computed for every parameter combination and fold:
          * accuracy: Standard classification accuracy
          * balanced_accuracy: Average recall per class (handles class imbalance)
          * f1: Binary F1 score (harmonic mean of precision and recall)
          * f1_macro: F1 averaged across classes (equal weight per class)
          * recall: Sensitivity/true positive rate

        RANDOMIZATION:
        --------------
        Each repetition uses a different random seed (base_seed + rep) for the KFold split,
        ensuring diverse train/test partitions and robust performance estimates.

        OUTPUT STATISTICS:
        ------------------
        The results_dict contains three key statistics for each metric:
        1. mean_score: Average of the best scores across all repetitions
           - Represents overall expected performance

        2. std_10: Standard deviation of best scores across repetitions
           - Measures performance variability due to different data splits
           - Lower values indicate more stable performance across different train/test splits

        3. std_100: Standard deviation of all individual fold scores
           - Measures total variability at the fold level (across all repetitions)
           - Captures both cross-validation variance and repetition variance
           - Generally higher than std_10 as it includes more sources of variability

        INTERPRETATION:
        ---------------
        - mean_score: Use this as your primary performance estimate
        - std_10: Use this to report performance stability (e.g., "accuracy: 0.85 ± 0.02")
        - std_100: Use this to understand total prediction variance across all data splits

        Example:
        If accuracy = (0.850, 0.015, 0.032):
        - Expected accuracy: 85.0%
        - Variability across repetitions: ±1.5%
        - Total variability across all folds: ±3.2%

        BEST PRACTICES:
        ---------------
        - Use balanced_accuracy for imbalanced datasets
        - Use 10+ repetitions for stable estimates (more is better but computationally expensive)
        - Compare std_10 across models: lower values indicate more robust models
        - If std_10 >> std_100 unexpectedly, investigate potential data leakage or issues
        """
        all_metrics_results = {
            'accuracy': {'best_scores': [], 'fold_scores': []},
            'balanced_accuracy': {'best_scores': [], 'fold_scores': []},
            'f1': {'best_scores': [], 'fold_scores': []},
            'f1_macro': {'best_scores': [], 'fold_scores': []},
            'recall': {'best_scores': [], 'fold_scores': []}
        }

        # Repeat cross-validation with different random seeds
        for rep in range(n_repeats):
            # Create 10-fold CV with unique random seed for this repetition
            cv = KFold(n_splits=10, shuffle=True, random_state=base_seed + rep)

            # Define multiple scoring metrics to track
            scoring = {
                'accuracy': 'accuracy',
                'balanced_accuracy': 'balanced_accuracy',
                'f1': make_scorer(f1_score),
                'f1_macro': make_scorer(f1_score, average='macro'),
                'recall': make_scorer(recall_score)
            }

            # Grid search with cross-validation
            grid = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring=scoring,  # Compute all metrics
                refit='balanced_accuracy',  # Optimize based on balanced_accuracy
                cv=cv,
                n_jobs=1,
                verbose=0,
                error_score='raise'
            )

            # Fit the grid search
            grid.fit(
                nx_graphs,
                labels,
                anchor_nodes_label_name=categorical_attr,
                list_anchor_nodes_labels=anchor_labels
            )

            # Skip if grid search failed
            if not hasattr(grid, "best_score_") or grid.best_score_ is None:
                print(f"Repeat {rep}: GridSearchCV returned no best_score_; skipping repeat.")
                continue

            # Extract results for the best parameters
            best_index = grid.best_index_

            # Collect scores for all metrics
            for metric in all_metrics_results.keys():
                # Best score = mean across the 10 folds for this repetition
                best_score = grid.cv_results_[f'mean_test_{metric}'][best_index]
                all_metrics_results[metric]['best_scores'].append(best_score)

                # Individual fold scores (all 10 folds)
                for fold_idx in range(10):
                    score_key = f'split{fold_idx}_test_{metric}'
                    all_metrics_results[metric]['fold_scores'].append(
                        grid.cv_results_[score_key][best_index]
                    )

        # Calculate final statistics for each metric
        results_dict = {}
        for metric in all_metrics_results.keys():
            mean_score = np.mean(all_metrics_results[metric]['best_scores'])
            std_10 = np.std(all_metrics_results[metric]['best_scores'], ddof=1)
            std_100 = np.std(all_metrics_results[metric]['fold_scores'], ddof=1)

            # Store as tuple: (mean, std across repetitions, std across all folds)
            results_dict[metric] = (mean_score, std_10, std_100)



        return results_dict

    except Exception as e:
        print(f"GridSearchCV repeated runs failed: {e}")
        return {}
