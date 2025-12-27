#!/usr/bin/env python3
"""
GNN (Graph Neural Network) evaluation script with CLI interface.

Runs GIN/GINE on specified TU datasets with 10x10 fold cross-validation,
computing accuracy, F1, F1-macro, balanced accuracy, recall, and AUC.

Usage:
    python run_gnn.py MUTAG PTC_MR NCI1
    python run_gnn.py --device cpu MUTAG
    python run_gnn.py --timeout 36000 --device gpu MUTAG

Output:
    GNN_results/GNN_Performance_<timestamp>.csv
"""
import os
import sys

from shared import (
    create_argument_parser,
    validate_datasets,
    ResultsCSVWriter,
    get_timestamped_path,
    run_with_timeout,
    setup_logging,
    CV_SEED,
    GNN_METRICS,
    get_base_dir,
)


def run_gnn_evaluation(dataset_name: str, device: str, cv_seed: int = CV_SEED):
    """
    Run GNN evaluation with device selection.

    This wraps the existing GNN evaluation infrastructure, adding explicit
    device control via environment variables.

    Args:
        dataset_name: Name of the TU dataset
        device: 'cpu', 'gpu', or 'cuda'
        cv_seed: Random seed for reproducibility

    Returns:
        Dict mapping metric to (mean, std_top10, std_all100)
    """
    import torch

    # Set device via environment variable BEFORE importing torch-geometric
    # This ensures the device is properly selected
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif device in ('gpu', 'cuda'):
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            print(f"WARNING: GPU requested but CUDA not available, falling back to CPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import tudataset.tud_benchmark.auxiliarymethods.datasets as dp
    from tudataset.tud_benchmark.auxiliarymethods.gnn_evaluation import gnn_evaluation
    from tudataset.tud_benchmark.gnn_baselines.gnn_architectures import GIN, GINE

    # Load dataset to ensure it exists
    dp.get_dataset(dataset_name)

    # Check for edge labels or attributes to decide GIN vs GINE
    dataset_dir = os.path.join(
        get_base_dir(), dataset_name, dataset_name, "raw"
    )
    has_edge_labels = os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_edge_labels.txt"))
    has_edge_attributes = os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_edge_attributes.txt"))
    use_gine = has_edge_labels or has_edge_attributes
    GNNLayer = GINE if use_gine else GIN

    # Run evaluation
    result = gnn_evaluation(
        GNNLayer,
        dataset_name,
        [1, 2, 3, 4, 5],       # layers
        [32, 64, 128],         # hidden dimensions
        max_num_epochs=200,
        batch_size=64,
        start_lr=0.01,
        num_repetitions=10,
        all_std=True,
        cv_seed=cv_seed
    )

    return result


def main():
    parser = create_argument_parser('GNN', supports_device=True)
    args = parser.parse_args()

    logger = setup_logging('gnn', args.verbose)
    datasets = validate_datasets(args.datasets)

    if not datasets:
        logger.error("No valid datasets to process")
        sys.exit(1)

    # Normalize device argument
    device = 'cuda' if args.device in ('gpu', 'cuda') else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Processing {len(datasets)} dataset(s): {', '.join(datasets)}")
    logger.info(f"Timeout per dataset: {args.timeout} seconds ({args.timeout/3600:.1f} hours)")

    # Setup output directory and CSV
    csv_path = get_timestamped_path('GNN_results', 'GNN')
    csv_writer = ResultsCSVWriter(csv_path)
    logger.info(f"Results will be saved to: {csv_path}")

    for dataset_name in datasets:
        logger.info(f"Processing {dataset_name}...")

        try:
            # Run evaluation with timeout
            result, timed_out, error = run_with_timeout(
                run_gnn_evaluation,
                args=(dataset_name, device),
                kwargs={'cv_seed': CV_SEED},
                timeout_sec=args.timeout
            )

            if timed_out:
                logger.warning(f"{dataset_name}: TIMEOUT after {args.timeout}s")
                csv_writer.write_failure(dataset_name, GNN_METRICS, "TIMEOUT")
            elif error:
                logger.error(f"{dataset_name}: {error}")
                csv_writer.write_failure(dataset_name, GNN_METRICS, "FAILED")
            elif result:
                csv_writer.write_results(dataset_name, result)
                # Log summary
                if 'accuracy' in result:
                    acc, s10, s100 = result['accuracy']
                    logger.info(f"{dataset_name}: accuracy={acc:.4f} (std10={s10:.4f}, std100={s100:.4f})")
                else:
                    logger.info(f"{dataset_name}: completed successfully")
            else:
                logger.warning(f"{dataset_name}: No results returned")
                csv_writer.write_failure(dataset_name, GNN_METRICS, "FAILED")

        except Exception as e:
            logger.error(f"{dataset_name}: {e}")
            csv_writer.write_failure(dataset_name, GNN_METRICS, "FAILED")

    logger.info(f"All done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
