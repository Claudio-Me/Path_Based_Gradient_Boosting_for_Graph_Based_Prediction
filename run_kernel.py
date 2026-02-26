#!/usr/bin/env python3
"""
Kernel methods evaluation script with CLI interface.

Runs various graph kernels (WL, Graphlet, Shortest-path, WLOA) on specified
TU datasets with 10x10 fold cross-validation, computing accuracy.

Note: Kernel methods only compute accuracy (C++ implementation).

Usage:
    python run_kernel.py MUTAG PTC_MR NCI1
    python run_kernel.py --timeout 72000 MUTAG

Output:
    Kernel_results/Kernel_Performance_<timestamp>.csv
"""
import os
import sys
import time

from shared import (
    create_argument_parser,
    validate_datasets,
    ResultsCSVWriter,
    get_timestamped_path,
    run_with_timeout,
    setup_logging,
    KERNEL_METHODS,
    get_base_dir,
)

import tudataset.tud_benchmark.auxiliarymethods.datasets as dp
import tudataset.tud_benchmark.auxiliarymethods.auxiliary_methods as aux
from tudataset.tud_benchmark.auxiliarymethods.kernel_evaluation import (
    kernel_svm_evaluation,
    linear_svm_evaluation
)
import tudataset.tud_benchmark.kernel_baselines as kb


# Kernel evaluation functions (extracted from kernel_baseline_all_datasets.py)

def _wl_dense_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    """Weisfeiler-Lehman subtree kernel (dense gram matrix)."""
    wl_gram_matrix_collection = []
    for iterations in range(1, 6):
        wl_gram_matrix = kb.compute_wl_1_dense(dataset_name, int(iterations), use_labels, has_edge_labels)
        wl_gram_matrix = aux.normalize_gram_matrix(wl_gram_matrix)
        wl_gram_matrix_collection.append(wl_gram_matrix)
    return kernel_svm_evaluation(
        wl_gram_matrix_collection, dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )


def _wl_sparse_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    """Weisfeiler-Lehman subtree kernel (sparse linear)."""
    wl_feature_vector_collection = []
    for iterations in range(1, 6):
        wl_feature_vectors = kb.compute_wl_1_sparse(dataset_name, iterations, use_labels, has_edge_labels)
        wl_feature_vectors = aux.normalize_feature_vector(wl_feature_vectors)
        wl_feature_vector_collection.append(wl_feature_vectors)
    return linear_svm_evaluation(
        wl_feature_vector_collection, dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )


def _graphlet_dense_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    """Graphlet kernel (dense gram matrix)."""
    gr_gram_matrix = kb.compute_graphlet_dense(dataset_name, use_labels, has_edge_labels)
    return kernel_svm_evaluation(
        [gr_gram_matrix], dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )


def _graphlet_sparse_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    """Graphlet kernel (sparse linear)."""
    gr_feature_vectors = kb.compute_graphlet_sparse(dataset_name, use_labels, has_edge_labels)
    return linear_svm_evaluation(
        [gr_feature_vectors], dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )


def _sp_dense_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    """Shortest-path kernel (dense gram matrix)."""
    sp_gram_matrix = kb.compute_shortestpath_dense(dataset_name, use_labels)
    return kernel_svm_evaluation(
        [sp_gram_matrix], dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )


def _sp_sparse_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    """Shortest-path kernel (sparse linear)."""
    sp_feature_vectors = kb.compute_shortestpath_sparse(dataset_name, use_labels)
    return linear_svm_evaluation(
        [sp_feature_vectors], dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )


def _wloa_dense_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    """Weisfeiler-Lehman optimal assignment kernel."""
    wlpa_matrices = []
    for iteration in range(1, 6):
        # compute_wloa_dense requires: (dataset_name, iteration, use_labels, has_edge_labels)
        gram = kb.compute_wloa_dense(dataset_name, iteration, use_labels, has_edge_labels)
        gram = aux.normalize_gram_matrix(gram)
        wlpa_matrices.append(gram)
    return kernel_svm_evaluation(
        wlpa_matrices, dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )


# Mapping from kernel method names to evaluation functions
KERNEL_FUNCS = [
    ("WL_subtree", _wl_dense_eval),
    ("WL_subtree_linear", _wl_sparse_eval),
    ("Graphlet", _graphlet_dense_eval),
    ("Graphlet_linear", _graphlet_sparse_eval),
    ("Shortest_path", _sp_dense_eval),
    ("Shortest_path_linear", _sp_sparse_eval),
    ("WLOA", _wloa_dense_eval),
]


def run_all_kernels(dataset_name: str, timeout_per_kernel: int = 10286) -> dict:
    """
    Run all kernel methods on a dataset.

    Each kernel method is run with its own timeout to prevent one slow
    kernel from blocking all others.

    Args:
        dataset_name: Name of the TU dataset
        timeout_per_kernel: Timeout per individual kernel (default: ~2.86 hours)

    Returns:
        Dict mapping kernel_name to (accuracy, std_top10, std_all100)
        Failed kernels are marked with ("FAILED"/"TIMEOUT", ..., ...)
    """
    # Ensure dataset is loaded
    dp.get_dataset(dataset_name)

    # Check for edge labels/attributes
    dataset_dir = os.path.join(
        get_base_dir(), dataset_name, dataset_name, "raw"
    )
    has_edge_labels = os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_edge_labels.txt"))
    has_edge_labels = has_edge_labels or os.path.exists(
        os.path.join(dataset_dir, f"{dataset_name}_edge_attributes.txt")
    )

    num_reps = 10
    use_labels = True

    results = {}
    for kernel_name, kernel_func in KERNEL_FUNCS:
        print(f"  Running {kernel_name}...")

        start_time = time.time()
        result, timed_out, error = run_with_timeout(
            kernel_func,
            args=(dataset_name, num_reps, use_labels, has_edge_labels),
            timeout_sec=timeout_per_kernel
        )
        elapsed_time = time.time() - start_time

        if timed_out:
            print(f"  {kernel_name}: TIMEOUT")
            results[kernel_name] = ("TIMEOUT", "TIMEOUT", "TIMEOUT", None)
        elif error:
            print(f"  {kernel_name}: FAILED - {error}")
            results[kernel_name] = ("FAILED", "FAILED", "FAILED", None)
        elif result is None:
            print(f"  {kernel_name}: No result")
            results[kernel_name] = ("FAILED", "FAILED", "FAILED", None)
        else:
            # result is (accuracy, std_10, std_100), add elapsed_time
            print(f"  {kernel_name}: accuracy={result[0]:.4f} (time={elapsed_time:.2f}s)")
            results[kernel_name] = (*result, elapsed_time)

    return results


def main():
    parser = create_argument_parser('Kernel', supports_device=False)
    args = parser.parse_args()

    logger = setup_logging('kernel', args.verbose)
    datasets = validate_datasets(args.datasets)

    if not datasets:
        logger.error("No valid datasets to process")
        sys.exit(1)

    logger.info(f"Processing {len(datasets)} dataset(s): {', '.join(datasets)}")
    logger.info(f"Total timeout per dataset: {args.timeout} seconds ({args.timeout/3600:.1f} hours)")

    # Setup output directory and CSV
    csv_path = get_timestamped_path('Kernel_results', 'Kernel')
    csv_writer = ResultsCSVWriter(csv_path)
    logger.info(f"Results will be saved to: {csv_path}")

    # Calculate timeout per kernel (divide total by 7 kernels)
    timeout_per_kernel = args.timeout // 7
    logger.info(f"Timeout per kernel: {timeout_per_kernel} seconds ({timeout_per_kernel/3600:.1f} hours)")

    for dataset_name in datasets:
        logger.info(f"Processing {dataset_name}...")

        try:
            results = run_all_kernels(dataset_name, timeout_per_kernel)

            # Write results - one row per kernel method
            for kernel_name, result_tuple in results.items():
                acc, std10, std100, elapsed_time = result_tuple
                metric_name = f"accuracy_{kernel_name}"
                if isinstance(acc, str):  # "FAILED" or "TIMEOUT"
                    csv_writer.write_failure(dataset_name, [metric_name], acc)
                else:
                    # Divide total elapsed time by number of folds (10 reps × 10 folds = 100)
                    # to get per-fold time, consistent with PathBoost and GNN reporting
                    timing_data = (elapsed_time / 100, 0.0) if elapsed_time is not None else None
                    csv_writer.write_results(dataset_name, {metric_name: (acc, std10, std100)}, timing_data=timing_data)

            logger.info(f"{dataset_name}: completed all kernels")

        except Exception as e:
            logger.error(f"{dataset_name}: {e}")
            # Write failure for all kernels
            for kernel_name in KERNEL_METHODS:
                csv_writer.write_failure(dataset_name, [f"accuracy_{kernel_name}"], "FAILED")

    logger.info(f"All done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
