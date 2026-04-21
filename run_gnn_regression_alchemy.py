#!/usr/bin/env python3
"""
GNN regression evaluation on molecule datasets.

Supports single-target and all-target execution with optional per-target z-score
normalization (all-target mode only).
"""
import argparse
import os
import traceback
import numpy as np

from shared import (
    ResultsCSVWriter,
    get_timestamped_path,
    run_with_timeout,
    setup_logging,
    CV_SEED,
    DEFAULT_TIMEOUT,
    REGRESSION_METRICS,
    get_base_dir,
)
from tudataset.tud_benchmark.auxiliarymethods.datasets import get_dataset


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


def _load_targets_2d(dataset_name):
    try:
        targets = get_dataset(dataset_name, multi_target_regression=True)
    except Exception:
        targets = get_dataset(dataset_name, regression=True)
    targets = np.array(targets)
    if targets.ndim == 1:
        return targets.reshape(-1, 1)
    return targets


def _evaluate_single_target(dataset_name, labels, use_gine, n_repeats, n_folds, cv_seed, sample_indices=None):
    from tudataset.tud_benchmark.auxiliarymethods.gnn_regression_evaluation import gnn_regression_evaluation
    from tudataset.tud_benchmark.gnn_baselines.gnn_architectures_regression import (
        GINRegression,
        GINERegression,
    )

    gnn_layer = GINERegression if use_gine else GINRegression
    return gnn_regression_evaluation(
        gnn_layer,
        dataset_name,
        [1, 2, 3, 4, 5],
        [32, 64, 128],
        labels,
        max_num_epochs=200,
        batch_size=64,
        start_lr=0.01,
        num_repetitions=n_repeats,
        n_folds=n_folds,
        cv_seed=cv_seed,
        sample_indices=sample_indices,
    )


def run_gnn_regression_dataset(dataset_name, csv_writer, logger, args, n_repeats, n_folds):
    logger.info(f"Loading targets for {dataset_name}...")
    targets_2d = _load_targets_2d(dataset_name)
    n_graphs, n_targets = targets_2d.shape

    if args.normalize:
        mean = targets_2d.mean(axis=0)
        std = targets_2d.std(axis=0)
        std[std == 0] = 1.0
        targets_2d = (targets_2d - mean) / std
        logger.info(f"Normalized {n_targets} target(s); means={mean}, stds={std}")

    if args.all_targets:
        target_indices = list(range(n_targets))
    else:
        if args.target_index < 0 or args.target_index >= n_targets:
            logger.warning(
                f"{dataset_name}: target_index {args.target_index} out of range "
                f"(have {n_targets} targets); skipping"
            )
            return
        target_indices = [args.target_index]

    subsample_indices = None
    if args.max_graphs and n_graphs > args.max_graphs:
        rng = np.random.RandomState(CV_SEED)
        subsample_indices = rng.choice(n_graphs, args.max_graphs, replace=False)
        subsample_indices.sort()
        logger.info(f"Subsampled to {len(subsample_indices)} graphs (from {n_graphs}).")

    dataset_dir = os.path.join(get_base_dir(), dataset_name, dataset_name, "raw")
    has_edge_labels = os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_edge_labels.txt"))
    has_edge_attributes = os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_edge_attributes.txt"))
    use_gine = has_edge_labels or has_edge_attributes
    logger.info(f"{dataset_name}: using {'GINERegression' if use_gine else 'GINRegression'}")

    for target_idx in target_indices:
        labels = targets_2d[:, target_idx].astype(float)
        labels_for_stats = labels[subsample_indices] if subsample_indices is not None else labels

        tag = f"{dataset_name}_t{target_idx}"
        logger.info(
            f"{tag}: target stats min={labels_for_stats.min():.4f}, max={labels_for_stats.max():.4f}, "
            f"mean={labels_for_stats.mean():.4f}, std={labels_for_stats.std():.4f}"
        )

        result, timed_out, error = run_with_timeout(
            _evaluate_single_target,
            args=(dataset_name, labels, use_gine, n_repeats, n_folds, CV_SEED, subsample_indices),
            timeout_sec=args.timeout,
        )

        if timed_out:
            logger.warning(f"{tag}: TIMEOUT after {args.timeout}s")
            csv_writer.write_failure(tag, REGRESSION_METRICS, "TIMEOUT")
        elif error:
            logger.error(f"{tag}: {error}")
            csv_writer.write_failure(tag, REGRESSION_METRICS, "FAILED")
        elif result:
            timing_data = result.pop('_timing', None)
            csv_writer.write_results(tag, result, timing_data=timing_data)
            if 'mae' in result:
                mae, s10, s100 = result['mae']
                logger.info(f"{tag}: MAE={mae:.4f} (std10={s10:.4f}, std100={s100:.4f})")
        else:
            logger.warning(f"{tag}: No results returned")
            csv_writer.write_failure(tag, REGRESSION_METRICS, "FAILED")


def main():
    parser = argparse.ArgumentParser(description="GNN regression evaluation on molecule datasets")
    parser.add_argument(
        'datasets', nargs='*', default=None,
        help="Dataset names to evaluate (default: alchemy_full)"
    )
    parser.add_argument(
        '--target-index', type=int, default=0,
        help="Which target column to use in single-target mode (default: 0)"
    )
    parser.add_argument(
        '--all-targets', action='store_true',
        help="Run every target column; for alchemy_full this is 12 targets"
    )
    parser.add_argument(
        '--normalize', action='store_true',
        help="Z-score normalize each target before CV. Requires --all-targets."
    )
    parser.add_argument(
        '--timeout', type=int, default=DEFAULT_TIMEOUT,
        help=f"Timeout per target experiment in seconds (default: {DEFAULT_TIMEOUT})"
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
        '--device', choices=['cpu', 'gpu', 'cuda'], default='cpu',
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Verbose logging"
    )
    args = parser.parse_args()

    if args.normalize and not args.all_targets:
        parser.error("--normalize requires --all-targets")

    logger = setup_logging('gnn_regression', args.verbose)
    datasets = args.datasets if args.datasets else ["alchemy_full"]

    device = 'cuda' if args.device in ('gpu', 'cuda') else 'cpu'
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    n_repeats = 2 if args.quick else 10
    n_folds = 2 if args.quick else 10

    target_mode = f"single target (index {args.target_index})"
    if args.all_targets:
        total_targets = None
        if len(datasets) == 1:
            try:
                total_targets = _load_targets_2d(datasets[0]).shape[1]
            except Exception:
                total_targets = None
        target_mode = (
            f"all targets ({total_targets} total)"
            if total_targets is not None
            else "all targets"
        )

    norm_text = "ENABLED (z-score per target)" if args.normalize else "disabled"
    cv_text = "2x2 (quick)" if args.quick else "10x10"
    banner = (
        "=" * 60 + "\n"
        "  GNN REGRESSION\n"
        f"  Datasets:      {', '.join(datasets)}\n"
        f"  Target mode:   {target_mode}\n"
        f"  Normalization: {norm_text}\n"
        f"  Device:        {device}\n"
        f"  CV:            {cv_text}\n"
        + "=" * 60
    )
    print(banner)
    logger.info("\n" + banner)

    csv_path = get_timestamped_path('GNN_results', 'GNN_Regression')
    csv_writer = ResultsCSVWriter(csv_path)
    logger.info(f"Results will be saved to: {csv_path}")

    for dataset_name in datasets:
        logger.info(f"{'='*60}")
        logger.info(f"Processing {dataset_name}...")
        logger.info(f"{'='*60}")
        try:
            run_gnn_regression_dataset(dataset_name, csv_writer, logger, args, n_repeats, n_folds)
        except Exception as e:
            logger.error(f"{dataset_name}: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            if args.all_targets:
                try:
                    n_targets = _load_targets_2d(dataset_name).shape[1]
                except Exception:
                    n_targets = 1
                for target_idx in range(n_targets):
                    csv_writer.write_failure(f"{dataset_name}_t{target_idx}", REGRESSION_METRICS, "FAILED")
            elif args.target_index >= 0:
                csv_writer.write_failure(f"{dataset_name}_t{args.target_index}", REGRESSION_METRICS, "FAILED")

    logger.info(f"All done. Results saved to: {csv_path}")


if __name__ == '__main__':
    main()

