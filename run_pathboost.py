#!/usr/bin/env python3
"""
Sequential PathBoost evaluation script with CLI interface.

Runs PathBoost on specified TU datasets with 10x10 fold cross-validation,
computing accuracy, F1, F1-macro, balanced accuracy, recall, and AUC.

Usage:
    python run_pathboost.py MUTAG PTC_MR NCI1
    python run_pathboost.py --timeout 36000 MUTAG
    python run_pathboost.py  # runs on all datasets

Output:
    PathBoost_results/Sequential_PathBoost_Performance_<timestamp>.csv
"""
import os
import sys
import numpy as np
from typing import Dict, List, Optional

from shared import (
    create_argument_parser,
    validate_datasets,
    ResultsCSVWriter,
    get_timestamped_path,
    run_with_timeout,
    setup_logging,
    CV_SEED,
    NX_GRAPHS_DIR,
    PATHBOOST_METRICS,
    get_base_dir,
)

from tudataset.tud_benchmark.auxiliarymethods.datasets import get_dataset
from utils import load_or_build_nx_graphs, find_categorical_node_attributes, preprocess_labels


def pathboost_evaluation_with_auc(
    nx_graphs: List,
    labels: np.ndarray,
    dataset_name: Optional[str] = None,
    param_grid: Optional[Dict] = None,
    n_repeats: int = 10,
    cv_seed: Optional[int] = None,
) -> Dict:
    """
    Run PathBoost GridSearchCV with ROC-AUC metric included.

    This is an enhanced version of pathboost_gridcv_baseline that adds
    AUC computation using predict_proba.

    Args:
        nx_graphs: List of NetworkX graphs
        labels: Array of graph labels
        dataset_name: Name of dataset (for logging)
        param_grid: Hyperparameter grid for GridSearchCV
        n_repeats: Number of CV repetitions (default: 10)
        cv_seed: Random seed for reproducibility

    Returns:
        Dict mapping metric name to (mean, std_top10, std_all100) tuple.
        Returns empty dict {} on failure.
    """
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import make_scorer, f1_score, recall_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from extended_path_boost import SequentialPathBoostClassifier
    from dataset_analysis import dataset_prescreening

    # Pre-screening (silent)
    dataset_prescreening(labels, dataset_name=dataset_name, show_plot=False, save_plot=False)

    # Ensure binary labels mapped to {0,1}
    processed_labels = preprocess_labels(labels)
    if processed_labels is None:
        print("Skipped: Dataset is not binary classification.")
        return {}
    labels = processed_labels

    # Discover categorical node attribute
    categorical_attr = find_categorical_node_attributes(nx_graphs)
    if not categorical_attr:
        print("Skipped: No categorical node attributes found.")
        return {}

    # Anchor labels from all graphs
    anchor_labels = set()
    for g in nx_graphs:
        for _, node_data in g.nodes(data=True):
            if categorical_attr in node_data:
                anchor_labels.add(node_data[categorical_attr])
    anchor_labels = list(anchor_labels)
    if len(anchor_labels) < 2:
        print("Skipped: Not enough distinct anchor labels.")
        return {}

    base_seed = 42 if cv_seed is None else int(cv_seed)

    # Base estimator
    base_estimator = SequentialPathBoostClassifier(
        n_iter=1000,
        learning_rate=0.01,
        parameters_variable_importance=None,
        BaseLearnerClass=DecisionTreeRegressor,
        SelectorClass=DecisionTreeClassifier,
        kwargs_for_base_learner={
            'random_state': base_seed,
            'splitter': 'best',
            'criterion': "squared_error",
            'max_leaf_nodes': 10
        },
        kwargs_for_selector={},
        verbose=False,
        use_tree_boost=False
    )

    # Default grid if none provided
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.1, 0.02],
            'max_path_length': [3, 5],
            'kwargs_for_base_learner': [ {'max_depth': 4}],
            'n_iter': [500,  1500, 2000]
        }

    # Custom scorer for ROC-AUC that handles predict_proba
    def roc_auc_scorer(estimator, X, y):
        """Custom scorer for ROC-AUC that works with PathBoost."""
        try:
            if hasattr(estimator, 'predict_proba'):
                y_proba = estimator.predict_proba(X)
                if hasattr(y_proba, 'ndim') and y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    return roc_auc_score(y, y_proba[:, 1])
                return roc_auc_score(y, y_proba)
            else:
                # Fall back to decision function or predictions
                if hasattr(estimator, 'decision_function'):
                    y_scores = estimator.decision_function(X)
                    return roc_auc_score(y, y_scores)
                y_pred = estimator.predict(X)
                return roc_auc_score(y, y_pred)
        except Exception:
            return 0.0

    # Metrics tracking - includes ROC_AUC
    all_metrics_results = {
        'accuracy': {'best_scores': [], 'fold_scores': []},
        'balanced_accuracy': {'best_scores': [], 'fold_scores': []},
        'f1': {'best_scores': [], 'fold_scores': []},
        'f1_macro': {'best_scores': [], 'fold_scores': []},
        'recall': {'best_scores': [], 'fold_scores': []},
        'roc_auc': {'best_scores': [], 'fold_scores': []},
    }

    # Timing tracking for training time per hyperparameter config
    timing_results = {'fit_times': [], 'fit_time_stds': []}

    # Define scoring dict
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'f1': make_scorer(f1_score),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'recall': make_scorer(recall_score),
        'roc_auc': roc_auc_scorer,
    }

    try:
        # Repeat cross-validation with different random seeds
        for rep in range(n_repeats):
            cv = KFold(n_splits=10, shuffle=True, random_state=base_seed + rep)

            grid = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring=scoring,
                refit='balanced_accuracy',
                cv=cv,
                n_jobs=1,
                verbose=0,
                error_score='raise'
            )

            grid.fit(
                nx_graphs,
                labels,
                anchor_nodes_label_name=categorical_attr,
                list_anchor_nodes_labels=anchor_labels
            )

            if not hasattr(grid, "best_score_") or grid.best_score_ is None:
                print(f"Repeat {rep}: GridSearchCV returned no best_score_; skipping.")
                continue

            best_index = grid.best_index_

            # Collect timing for best config
            timing_results['fit_times'].append(grid.cv_results_['mean_fit_time'][best_index])
            timing_results['fit_time_stds'].append(grid.cv_results_['std_fit_time'][best_index])

            # Collect scores for all metrics
            for metric in all_metrics_results.keys():
                best_score = grid.cv_results_[f'mean_test_{metric}'][best_index]
                all_metrics_results[metric]['best_scores'].append(best_score)

                for fold_idx in range(10):
                    score_key = f'split{fold_idx}_test_{metric}'
                    all_metrics_results[metric]['fold_scores'].append(
                        grid.cv_results_[score_key][best_index]
                    )

        # Calculate final statistics for each metric
        results_dict = {}
        for metric in all_metrics_results.keys():
            best_scores = all_metrics_results[metric]['best_scores']
            fold_scores = all_metrics_results[metric]['fold_scores']

            if not best_scores:
                results_dict[metric] = (-1.0, -1.0, -1.0)
                continue

            mean_score = np.mean(best_scores)
            std_top10 = np.std(best_scores, ddof=1) if len(best_scores) > 1 else 0.0
            std_all100 = np.std(fold_scores, ddof=1) if len(fold_scores) > 1 else 0.0

            results_dict[metric] = (mean_score, std_top10, std_all100)

        # Calculate timing statistics (time per config)
        if timing_results['fit_times']:
            time_mean = np.mean(timing_results['fit_times'])
            time_std = np.mean(timing_results['fit_time_stds'])
        else:
            time_mean, time_std = None, None
        results_dict['_timing'] = (time_mean, time_std)

        return results_dict

    except Exception as e:
        print(f"GridSearchCV failed: {e}")
        return {}


def main():
    parser = create_argument_parser('PathBoost', supports_device=False)
    args = parser.parse_args()

    logger = setup_logging('pathboost', args.verbose)
    datasets = validate_datasets(args.datasets)

    if not datasets:
        logger.error("No valid datasets to process")
        sys.exit(1)

    logger.info(f"Processing {len(datasets)} dataset(s): {', '.join(datasets)}")
    logger.info(f"Timeout per dataset: {args.timeout} seconds ({args.timeout/3600:.1f} hours)")

    # Setup output directory and CSV
    csv_path = get_timestamped_path('PathBoost_results', 'Sequential_PathBoost')
    csv_writer = ResultsCSVWriter(csv_path)
    logger.info(f"Results will be saved to: {csv_path}")

    base_dir = get_base_dir()

    for dataset_name in datasets:
        logger.info(f"Processing {dataset_name}...")

        try:
            # Load labels
            labels = get_dataset(dataset_name, regression=False)
            labels = np.array(labels).flatten()

            # Load NetworkX graphs
            dataset_path = os.path.join(base_dir, dataset_name)
            nx_graphs = load_or_build_nx_graphs(dataset_name, dataset_path, NX_GRAPHS_DIR)

            logger.info(f"{dataset_name}: {len(nx_graphs)} graphs, {len(np.unique(labels))} classes")

            # Run evaluation with timeout
            result, timed_out, error = run_with_timeout(
                pathboost_evaluation_with_auc,
                args=(nx_graphs, labels, dataset_name),
                kwargs={'cv_seed': CV_SEED, 'n_repeats': args.repetitions},
                timeout_sec=args.timeout
            )

            if timed_out:
                logger.warning(f"{dataset_name}: TIMEOUT after {args.timeout}s")
                csv_writer.write_failure(dataset_name, PATHBOOST_METRICS, "TIMEOUT")
            elif error:
                logger.error(f"{dataset_name}: {error}")
                csv_writer.write_failure(dataset_name, PATHBOOST_METRICS, "FAILED")
            elif result:
                timing_data = result.pop('_timing', None)
                csv_writer.write_results(dataset_name, result, timing_data=timing_data)
                # Log summary
                if 'accuracy' in result:
                    acc, s10, s100 = result['accuracy']
                    logger.info(f"{dataset_name}: accuracy={acc:.4f} (std10={s10:.4f}, std100={s100:.4f})")
                else:
                    logger.info(f"{dataset_name}: completed successfully")
            else:
                logger.warning(f"{dataset_name}: No results returned")
                csv_writer.write_failure(dataset_name, PATHBOOST_METRICS, "FAILED")

        except Exception as e:
            logger.error(f"{dataset_name}: {e}")
            csv_writer.write_failure(dataset_name, PATHBOOST_METRICS, "FAILED")

    logger.info(f"All done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
