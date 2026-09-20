#!/usr/bin/env python3
"""
Download the TUDatasets used in the paper.

The TU benchmark datasets are not redistributed with this repository; they are
fetched from https://www.graphlearning.io through PyTorch Geometric and stored
under ``tudataset/tud_benchmark/datasets/``. Running this script once up front
means every later experiment starts from a complete local copy.

Usage:
    python download_datasets.py --paper        # 12 classification datasets (Tables 1-3)
    python download_datasets.py --regression   # alchemy_full (Table 4)
    python download_datasets.py --paper --regression
    python download_datasets.py MUTAG BZR      # named datasets only
    python download_datasets.py --all          # every dataset known to the codebase (large!)
"""
import argparse
import sys

from shared import ALL_DATASETS, PAPER_DATASETS, PAPER_REGRESSION_DATASET
from utils import ensure_dataset_downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download TU datasets used by the experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=[],
        help="Explicit dataset names to download.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=f"Download the {len(PAPER_DATASETS)} classification datasets reported in the paper.",
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        help=f"Download the regression dataset ({PAPER_REGRESSION_DATASET}). Large: ~200k graphs.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every dataset listed in shared/constants.py (tens of GB).",
    )
    args = parser.parse_args()

    selected = list(args.datasets)
    if args.paper:
        selected += PAPER_DATASETS
    if args.regression:
        selected.append(PAPER_REGRESSION_DATASET)
    if args.all:
        selected += ALL_DATASETS

    # Preserve order, drop duplicates
    seen = set()
    selected = [d for d in selected if not (d in seen or seen.add(d))]

    if not selected:
        parser.print_help()
        print("\nNothing selected. Use --paper for the datasets reported in the paper.")
        return 1

    print(f"Downloading {len(selected)} dataset(s)...\n")
    failed = []
    for i, name in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {name}")
        try:
            ensure_dataset_downloaded(name, verbose=True)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed.append((name, exc))

    print()
    print(f"Done: {len(selected) - len(failed)}/{len(selected)} dataset(s) available.")
    if failed:
        print("Failed:")
        for name, exc in failed:
            print(f"  {name}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
