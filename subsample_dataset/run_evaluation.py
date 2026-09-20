#!/usr/bin/env python3
"""Learning-curve experiment: PathBoost vs GIN on increasing training-set sizes.

This reproduces Figure 1 of the paper, which uses PROTEINS_full and varies the
training set from 10% to 100% of the original sample size in 10% increments.

Usage:
    python subsample_dataset/run_evaluation.py --dataset PROTEINS_full --plot
    python subsample_dataset/run_evaluation.py --dataset SYNTHETIC --n-splits 4 --device cpu

Output:
    subsample_dataset/<dataset>_results.json   (accuracies per subsample size)
    subsample_dataset/<dataset>_learning_curve.png   (with --plot)
"""
import argparse
import json
import os
import sys

import numpy as np

# Make the repository root and this directory importable regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trainers import run_pathboost_evaluation, run_gin_evaluation  # noqa: E402


def run_evaluation(dataset='PROTEINS_full', n_splits=4, device='cpu', output=None):
    """
    Run PathBoost and GIN evaluation on subsampled training data.

    Args:
        dataset: Name of TU dataset
        n_splits: Number of train/test splits
        device: Device for GIN ('cpu' or 'cuda')
        output: Output JSON file path (default: subsample_dataset/<dataset>_results.json)

    Returns:
        dict with results
    """
    if output is None:
        output = os.path.join(_HERE, f'{dataset}_results.json')

    print(f"Running evaluation on {dataset} with {n_splits} splits...")

    print("\nRunning PathBoost evaluation...")
    pb_results = run_pathboost_evaluation(dataset, n_splits=n_splits)

    print("\nRunning GIN evaluation...")
    gin_results = run_gin_evaluation(dataset, n_splits=n_splits, device=device)

    results = {
        'dataset': dataset,
        'n_splits': n_splits,
        'pathboost': pb_results,
        'gin': gin_results
    }

    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output}")
    return results


def plot_results(results, save_path=None, show=False):
    """
    Plot PathBoost and GIN performance across subsample sizes with std error bands.

    Args:
        results: dict with 'pathboost' and 'gin' keys, each containing
                 {10: [acc1, acc2, ...], 20: [...], ...}
        save_path: If given, write the figure to this path
        show: Open an interactive window (off by default so the script stays headless)
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    percentages = sorted(int(k) for k in results['pathboost'].keys())

    def stats(key):
        means = [np.mean(results[key][str(p)]) for p in percentages]
        stds = [np.std(results[key][str(p)]) for p in percentages]
        return np.array(means), np.array(stds)

    pb_means, pb_stds = stats('pathboost')
    gin_means, gin_stds = stats('gin')

    plt.figure()
    plt.plot(percentages, pb_means, 'o-', label='PathBoost')
    plt.fill_between(percentages, pb_means - pb_stds, pb_means + pb_stds, alpha=0.2)
    plt.plot(percentages, gin_means, 's-', label='GNN')
    plt.fill_between(percentages, gin_means - gin_stds, gin_means + gin_stds, alpha=0.2)
    plt.xlabel('Training data (%)')
    plt.ylabel('Accuracy')
    plt.title(results.get('dataset', ''))
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Learning-curve experiment (Figure 1 of the paper).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', default='PROTEINS_full',
                        help='TU dataset name (default: PROTEINS_full, as in Figure 1)')
    parser.add_argument('--n-splits', type=int, default=4,
                        help='Number of train/test splits (default: 4)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device for the GIN baseline (default: cpu)')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: subsample_dataset/<dataset>_results.json)')
    parser.add_argument('--plot', action='store_true',
                        help='Also write <dataset>_learning_curve.png')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training and plot an existing results JSON')
    args = parser.parse_args()

    output = args.output or os.path.join(_HERE, f'{args.dataset}_results.json')

    if args.plot_only:
        with open(output) as f:
            results = json.load(f)
    else:
        results = run_evaluation(
            dataset=args.dataset,
            n_splits=args.n_splits,
            device=args.device,
            output=output,
        )

    if args.plot or args.plot_only:
        plot_results(
            results,
            save_path=os.path.join(_HERE, f'{args.dataset}_learning_curve.png'),
        )


if __name__ == '__main__':
    main()
