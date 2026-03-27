#!/usr/bin/env python3
"""
Sequential PathBoost regression evaluation on regression datasets.

Runs two side-by-side experiments per dataset with 10×10 fold cross-validation:
  1. Full: Use all node attributes as-is
  2. Categorical-only: Strip all node attributes except the single categorical
     attribute used for anchor node selection (ablation study)

Usage:
    python run_pathboost_regression_alchemy.py alchemy_full
    python run_pathboost_regression_alchemy.py aspirin benzene toluene
    python run_pathboost_regression_alchemy.py  # runs all regression datasets

Output:
    PathBoost_results/Sequential_PathBoost_Regression_Performance_<timestamp>.csv
"""
import os
import sys
import argparse
import traceback
import numpy as np
from typing import Dict, List, Optional

from shared import (
    ResultsCSVWriter,
    get_timestamped_path,
    run_with_timeout,
    setup_logging,
    CV_SEED,
    NX_GRAPHS_DIR,
    DEFAULT_TIMEOUT,
    get_base_dir,
)

from tudataset.tud_benchmark.auxiliarymethods.datasets import get_dataset
from utils import load_or_build_nx_graphs, find_categorical_node_attributes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Which regression target to use (alchemy_full has 12 targets, 0-indexed)
TARGET_INDEX = 0

REGRESSION_METRICS = ['mae', 'mse', 'r2']

ALL_REGRESSION_DATASETS = [
    "alchemy_full",
    "aspirin",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "toluene",
    "uracil",
    "ZINC_full",
    "ZINC_test",
    "ZINC_train",
    "ZINC_val",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_to_categorical_only(graphs, categorical_attr):
    """Return copies of graphs with only the categorical attribute on each node."""
    stripped = []
    for g in graphs:
        g_copy = g.copy()
        for node in g_copy.nodes():
            attrs_to_remove = [k for k in g_copy.nodes[node] if k != categorical_attr]
            for attr in attrs_to_remove:
                del g_copy.nodes[node][attr]
        stripped.append(g_copy)
    return stripped


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------

def pathboost_regression_evaluation(
    nx_graphs: List,
    labels: np.ndarray,
    categorical_attr: str,
    anchor_labels: List,
    experiment_name: Optional[str] = None,
    param_grid: Optional[Dict] = None,
    n_repeats: int = 10,
    n_folds: int = 10,
    cv_seed: Optional[int] = None,
) -> Dict:
    """
    Run SequentialPathBoost (regressor) with GridSearchCV over 10×10 CV.

    Args:
        nx_graphs: List of NetworkX graphs
        labels: Continuous target values (1-D array)
        categorical_attr: Node attribute name used for anchor selection
        anchor_labels: All distinct values of categorical_attr across graphs
        experiment_name: Label for log messages
        param_grid: Hyperparameter grid for GridSearchCV
        n_repeats: Number of CV repetitions
        n_folds: Number of CV folds
        cv_seed: Base random seed

    Returns:
        Dict mapping metric name to (mean, std_top10, std_all100) tuple, plus
        '_timing' key with (time_mean, time_std).  Returns {} on failure.
    """
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.tree import DecisionTreeRegressor
    from extended_path_boost import SequentialPathBoost

    base_seed = 42 if cv_seed is None else int(cv_seed)
    tag = experiment_name or "experiment"

    base_estimator = SequentialPathBoost(
        n_iter=1000,
        learning_rate=0.01,
        BaseLearnerClass=DecisionTreeRegressor,
        SelectorClass=DecisionTreeRegressor,
        kwargs_for_base_learner={
            'random_state': base_seed,
            'splitter': 'best',
            'criterion': 'squared_error',
            'max_leaf_nodes': 10,
        },
        kwargs_for_selector={
            'max_depth': 1,
            'random_state': base_seed,
            'criterion': 'squared_error',
        },
        verbose=False,
    )

    if param_grid is None:
        param_grid = {
            'learning_rate': [0.1, 0.02],
            'max_path_length': [3, 5],
            'kwargs_for_base_learner': [{'max_depth': 4}],
            'n_iter': [500, 1500, 2000],
        }

    # Scorer keys used in GridSearchCV
    scoring = {
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'r2': 'r2',
    }

    # Internal accumulator keyed by scorer name
    _scorer_to_metric = {
        'neg_mean_absolute_error': 'mae',
        'neg_mean_squared_error': 'mse',
        'r2': 'r2',
    }
    # Whether to negate the scorer value when reporting
    _negate = {
        'neg_mean_absolute_error': True,
        'neg_mean_squared_error': True,
        'r2': False,
    }

    all_metrics_results = {
        'mae': {'best_scores': [], 'fold_scores': []},
        'mse': {'best_scores': [], 'fold_scores': []},
        'r2':  {'best_scores': [], 'fold_scores': []},
    }
    timing_results = {'fit_times': [], 'fit_time_stds': []}

    try:
        for rep in range(n_repeats):
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed + rep)

            grid = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring=scoring,
                refit='neg_mean_absolute_error',
                cv=cv,
                n_jobs=1,
                verbose=0,
                error_score='raise',
            )

            grid.fit(
                nx_graphs,
                labels,
                anchor_nodes_label_name=categorical_attr,
                list_anchor_nodes_labels=anchor_labels,
            )

            if not hasattr(grid, "best_index_") or grid.best_index_ is None:
                print(f"[{tag}] Repeat {rep}: no best_index_; skipping.")
                continue

            best_index = grid.best_index_

            timing_results['fit_times'].append(grid.cv_results_['mean_fit_time'][best_index])
            timing_results['fit_time_stds'].append(grid.cv_results_['std_fit_time'][best_index])

            for scorer_key, metric_key in _scorer_to_metric.items():
                sign = -1 if _negate[scorer_key] else 1
                best_score = sign * grid.cv_results_[f'mean_test_{scorer_key}'][best_index]
                all_metrics_results[metric_key]['best_scores'].append(best_score)

                for fold_idx in range(n_folds):
                    score_key = f'split{fold_idx}_test_{scorer_key}'
                    all_metrics_results[metric_key]['fold_scores'].append(
                        sign * grid.cv_results_[score_key][best_index]
                    )

            print(f"[{tag}] Repeat {rep + 1}/{n_repeats} done.")

        # Aggregate
        results_dict = {}
        for metric_key, data in all_metrics_results.items():
            best_scores = data['best_scores']
            fold_scores = data['fold_scores']

            if not best_scores:
                results_dict[metric_key] = (-1.0, -1.0, -1.0)
                continue

            mean_score = np.mean(best_scores)
            std_top10 = np.std(best_scores, ddof=1) if len(best_scores) > 1 else 0.0
            std_all = np.std(fold_scores, ddof=1) if len(fold_scores) > 1 else 0.0
            results_dict[metric_key] = (mean_score, std_top10, std_all)

        if timing_results['fit_times']:
            time_mean = np.mean(timing_results['fit_times'])
            time_std = np.mean(timing_results['fit_time_stds'])
        else:
            time_mean, time_std = None, None
        results_dict['_timing'] = (time_mean, time_std)

        return results_dict

    except Exception as e:
        print(f"[{tag}] GridSearchCV failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset_name, csv_writer, logger, args, n_repeats, n_folds):
    """Run both experiments (full + categorical-only) for a single dataset."""
    base_dir = get_base_dir()

    # ------------------------------------------------------------------
    # Load targets
    # ------------------------------------------------------------------
    logger.info(f"Loading targets for {dataset_name}...")
    try:
        targets = get_dataset(dataset_name, multi_target_regression=True)
    except Exception:
        # Some datasets use single-target regression
        targets = get_dataset(dataset_name, regression=True)
    targets = np.array(targets)

    if targets.ndim == 1:
        n_graphs = len(targets)
        n_targets = 1
        labels = targets.astype(float)
    else:
        n_graphs, n_targets = targets.shape
        labels = targets[:, TARGET_INDEX].astype(float)

    print(
        f"{dataset_name} has {n_graphs} graphs, {n_targets} target(s). "
        f"Using target index {TARGET_INDEX}."
    )
    logger.info(
        f"Target stats: min={labels.min():.4f}, max={labels.max():.4f}, "
        f"mean={labels.mean():.4f}, std={labels.std():.4f}"
    )

    # ------------------------------------------------------------------
    # Load NetworkX graphs
    # ------------------------------------------------------------------
    dataset_path = os.path.join(base_dir, dataset_name)
    logger.info(f"Loading NetworkX graphs for {dataset_name}...")
    nx_graphs = load_or_build_nx_graphs(dataset_name, dataset_path, NX_GRAPHS_DIR)
    logger.info(f"Loaded {len(nx_graphs)} graphs.")

    # ------------------------------------------------------------------
    # Subsample if requested
    # ------------------------------------------------------------------
    if args.max_graphs and len(nx_graphs) > args.max_graphs:
        rng = np.random.RandomState(CV_SEED)
        indices = rng.choice(len(nx_graphs), args.max_graphs, replace=False)
        indices.sort()
        nx_graphs = [nx_graphs[i] for i in indices]
        labels = labels[indices]
        logger.info(f"Subsampled to {len(nx_graphs)} graphs (from {n_graphs}).")

    # ------------------------------------------------------------------
    # Find categorical attribute & anchor labels
    # ------------------------------------------------------------------
    categorical_attr = find_categorical_node_attributes(nx_graphs)
    if not categorical_attr:
        logger.warning(f"{dataset_name}: No categorical node attribute found. Skipping.")
        csv_writer.write_failure(f"{dataset_name}_full", REGRESSION_METRICS, "FAILED")
        csv_writer.write_failure(f"{dataset_name}_categorical_only", REGRESSION_METRICS, "FAILED")
        return

    anchor_labels = set()
    for g in nx_graphs:
        for _, node_data in g.nodes(data=True):
            if categorical_attr in node_data:
                anchor_labels.add(node_data[categorical_attr])
    anchor_labels = list(anchor_labels)

    if len(anchor_labels) < 2:
        logger.warning(f"{dataset_name}: Not enough distinct anchor labels. Skipping.")
        csv_writer.write_failure(f"{dataset_name}_full", REGRESSION_METRICS, "FAILED")
        csv_writer.write_failure(f"{dataset_name}_categorical_only", REGRESSION_METRICS, "FAILED")
        return

    print(
        f"Categorical attribute: '{categorical_attr}' "
        f"({len(anchor_labels)} distinct anchor labels)"
    )

    eval_kwargs = dict(
        categorical_attr=categorical_attr,
        anchor_labels=anchor_labels,
        n_repeats=n_repeats,
        n_folds=n_folds,
        cv_seed=CV_SEED,
    )

    # ------------------------------------------------------------------
    # Experiment 1: Full attributes
    # ------------------------------------------------------------------
    exp1_name = f"{dataset_name}_full"
    logger.info(f"=== Experiment 1: {exp1_name} ===")

    result1, timed_out1, error1 = run_with_timeout(
        pathboost_regression_evaluation,
        args=(nx_graphs, labels),
        kwargs=dict(experiment_name=exp1_name, **eval_kwargs),
        timeout_sec=args.timeout,
    )

    if timed_out1:
        logger.warning(f"{exp1_name}: TIMEOUT after {args.timeout}s")
        csv_writer.write_failure(exp1_name, REGRESSION_METRICS, "TIMEOUT")
    elif error1:
        logger.error(f"{exp1_name}: {error1}")
        csv_writer.write_failure(exp1_name, REGRESSION_METRICS, "FAILED")
    elif result1:
        timing_data1 = result1.pop('_timing', None)
        csv_writer.write_results(exp1_name, result1, timing_data=timing_data1)
        if 'mae' in result1:
            mae, s10, s100 = result1['mae']
            logger.info(f"{exp1_name}: MAE={mae:.4f} (std10={s10:.4f}, std100={s100:.4f})")
    else:
        logger.warning(f"{exp1_name}: No results returned")
        csv_writer.write_failure(exp1_name, REGRESSION_METRICS, "FAILED")

    # ------------------------------------------------------------------
    # Experiment 2: Categorical-only (ablation)
    # ------------------------------------------------------------------
    exp2_name = f"{dataset_name}_categorical_only"
    logger.info(f"=== Experiment 2: {exp2_name} ===")
    logger.info(f"Stripping all node attributes except '{categorical_attr}'...")

    stripped_graphs = strip_to_categorical_only(nx_graphs, categorical_attr)

    result2, timed_out2, error2 = run_with_timeout(
        pathboost_regression_evaluation,
        args=(stripped_graphs, labels),
        kwargs=dict(experiment_name=exp2_name, **eval_kwargs),
        timeout_sec=args.timeout,
    )

    if timed_out2:
        logger.warning(f"{exp2_name}: TIMEOUT after {args.timeout}s")
        csv_writer.write_failure(exp2_name, REGRESSION_METRICS, "TIMEOUT")
    elif error2:
        logger.error(f"{exp2_name}: {error2}")
        csv_writer.write_failure(exp2_name, REGRESSION_METRICS, "FAILED")
    elif result2:
        timing_data2 = result2.pop('_timing', None)
        csv_writer.write_results(exp2_name, result2, timing_data=timing_data2)
        if 'mae' in result2:
            mae, s10, s100 = result2['mae']
            logger.info(f"{exp2_name}: MAE={mae:.4f} (std10={s10:.4f}, std100={s100:.4f})")
    else:
        logger.warning(f"{exp2_name}: No results returned")
        csv_writer.write_failure(exp2_name, REGRESSION_METRICS, "FAILED")


def main():
    parser = argparse.ArgumentParser(
        description="PathBoost regression evaluation (full vs categorical-only)"
    )
    parser.add_argument(
        'datasets', nargs='*', default=None,
        help="Dataset names to evaluate (default: all regression datasets)"
    )
    parser.add_argument(
        '--timeout', type=int, default=DEFAULT_TIMEOUT,
        help=f"Timeout per experiment in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        '--max-graphs', type=int, default=0,
        help="Subsample to at most N graphs (0 = use all, default: 0)"
    )
    parser.add_argument(
        '--quick', action='store_true',
        help="Quick mode: 2x2 CV instead of 10x10 (for testing)"
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Verbose logging"
    )
    args = parser.parse_args()

    logger = setup_logging('pathboost_regression', args.verbose)

    datasets = args.datasets if args.datasets else ALL_REGRESSION_DATASETS

    n_repeats = 2 if args.quick else 10
    n_folds = 2 if args.quick else 10
    if args.quick:
        logger.info("Quick mode: using 2x2 CV")
    if args.max_graphs:
        logger.info(f"Subsampling to max {args.max_graphs} graphs per dataset")

    logger.info(f"Processing {len(datasets)} dataset(s): {', '.join(datasets)}")
    logger.info(f"Timeout per experiment: {args.timeout}s ({args.timeout/3600:.1f}h)")

    # ------------------------------------------------------------------
    # Output CSV
    # ------------------------------------------------------------------
    csv_path = get_timestamped_path(
        'PathBoost_results',
        'Sequential_PathBoost_Regression'
    )
    csv_writer = ResultsCSVWriter(csv_path)
    logger.info(f"Results will be saved to: {csv_path}")

    for dataset_name in datasets:
        logger.info(f"{'='*60}")
        logger.info(f"Processing {dataset_name}...")
        logger.info(f"{'='*60}")
        try:
            run_dataset(dataset_name, csv_writer, logger, args, n_repeats, n_folds)
        except Exception as e:
            logger.error(f"{dataset_name}: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            csv_writer.write_failure(f"{dataset_name}_full", REGRESSION_METRICS, "FAILED")
            csv_writer.write_failure(f"{dataset_name}_categorical_only", REGRESSION_METRICS, "FAILED")

    logger.info(f"All done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
