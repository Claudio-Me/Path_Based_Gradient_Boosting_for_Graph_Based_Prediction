# integrate_f1_into_merged_results.py
# -*- coding: utf-8 -*-
"""
Script: integrate_f1_into_merged_results.py

Purpose
-------
Integrate metrics from multiple CSVs located in the folder "F1_scores/" into a
main CSV file (default: "merged results.csv" or "merged_results.csv").

What it does
------------
1) Adds any missing columns found in the F1_scores CSVs to the main file,
   matching rows by the key column 'Dataset'.
2) If any F1_scores file has a higher PB_accuracy for a given Dataset than the
   main file's PathBoostAccuracy, the script replaces PathBoostAccuracy with
   that better PB_accuracy.
3) If 'PathBoostAccuracy_times_100' exists in the main file, it is updated to
   100 * PathBoostAccuracy after the replacement.

Assumptions
-----------
- All CSVs (main and F1_scores files) contain a 'Dataset' column used as the key.
- PB_accuracy and PathBoostAccuracy are numeric accuracies in the same scale
  ([0, 1]). If your main file stores percentages, adjust the scaling logic below.

Usage
-----
Run from the directory containing the main CSV and the 'F1_scores/' folder:

    python integrate_f1_into_merged_results.py

Optional args:

    python integrate_f1_into_merged_results.py \
        --main "path/to/merged results.csv" \
        --f1_dir "path/to/F1_scores" \
        --out "path/to/merged_results_enriched.csv"

Outputs
-------
- A new CSV file (default: 'merged_results_enriched.csv') with added columns and
  any PB_accuracy-based replacements applied to PathBoostAccuracy.
- Console summary of how many columns were added and how many rows had
  PathBoostAccuracy improved.
"""

import argparse
import os
import sys
from typing import List, Tuple

import pandas as pd


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read a CSV with reasonable defaults and helpful errors."""
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to read CSV '{path}': {e}")
        raise


def find_main_csv(explicit_main: str = None) -> str:
    """Locate the main CSV file.
    Priority:
    1) explicit_main if provided
    2) 'merged results.csv'
    3) 'merged_results.csv'
    """
    if explicit_main:
        if os.path.exists(explicit_main):
            return explicit_main
        else:
            raise FileNotFoundError(f"Specified main CSV not found: {explicit_main}")

    candidates = ["merged result.csv", "merged_result.csv"]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Main CSV not found. Looked for 'merged results.csv' and 'merged_result.csv'."
    )


def list_f1_csvs(f1_dir: str) -> List[str]:
    """List all CSV files in the F1_scores directory."""
    if not os.path.isdir(f1_dir):
        raise FileNotFoundError(
            f"F1_scores directory not found: {f1_dir}. Ensure the folder exists."
        )
    csvs = [os.path.join(f1_dir, f) for f in os.listdir(f1_dir) if f.lower().endswith('.csv')]
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in '{f1_dir}'. Add CSVs and retry."
        )
    # Sort for deterministic processing order
    csvs.sort()
    return csvs


def concat_f1_data(csv_paths: List[str]) -> pd.DataFrame:
    """Concatenate all F1_scores CSVs into a single DataFrame."""
    frames = []
    for p in csv_paths:
        df = read_csv_robust(p)
        if 'Dataset' not in df.columns:
            raise KeyError(f"CSV '{p}' does not contain a 'Dataset' column.")
        # Standardize column names minimal processing (strip whitespace)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    f1_all = pd.concat(frames, axis=0, ignore_index=True)
    return f1_all


def select_best_rows_by_pb(f1_all: pd.DataFrame) -> pd.DataFrame:
    """Select one row per Dataset, preferring the row with highest PB_accuracy.

    Fallback: if PB_accuracy is entirely missing/NaN for a Dataset, take the first
    occurrence (by the concatenation order).
    """
    if 'PB_accuracy' not in f1_all.columns:
        # If PB_accuracy doesn't exist at all, just take first occurrence per Dataset
        return f1_all.drop_duplicates(subset=['Dataset'], keep='first')

    # Ensure PB_accuracy is numeric
    f1_all['PB_accuracy'] = pd.to_numeric(f1_all['PB_accuracy'], errors='coerce')

    def pick_row(group: pd.DataFrame) -> pd.Series:
        if group['PB_accuracy'].notna().any():
            # idx of max PB_accuracy (NaNs are ignored by idxmax after filling -inf)
            temp = group.copy()
            temp['__pb__'] = temp['PB_accuracy'].fillna(float('-inf'))
            idx = temp['__pb__'].idxmax()
            return group.loc[idx]
        else:
            # no PB values -> take first
            return group.iloc[0]

    best_rows = f1_all.groupby('Dataset', as_index=False, group_keys=False).apply(pick_row)
    # Drop helper column if present
    if '__pb__' in best_rows.columns:
        best_rows = best_rows.drop(columns=['__pb__'])
    return best_rows.reset_index(drop=True)


def add_missing_columns(main_df: pd.DataFrame, f1_best: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add columns that exist in f1_best but not in main_df (left join on Dataset)."""
    main_cols = set(main_df.columns)
    f1_cols = set(f1_best.columns)
    new_cols = sorted(list(f1_cols - main_cols))

    if not new_cols:
        return main_df, []

    # Only merge the new columns + Dataset (key)
    to_merge = f1_best[['Dataset'] + new_cols]
    enriched = main_df.merge(to_merge, on='Dataset', how='left')
    return enriched, new_cols


def replace_pathboost_with_pb(main_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """If PB_accuracy is better than PathBoostAccuracy, replace it.

    Also updates PathBoostAccuracy_times_100 if present.
    """
    if 'PB_accuracy' not in main_df.columns:
        # Nothing to compare
        return main_df, 0

    # Coerce to numeric
    pb = pd.to_numeric(main_df['PB_accuracy'], errors='coerce')
    if 'PathBoostAccuracy' in main_df.columns:
        pba = pd.to_numeric(main_df['PathBoostAccuracy'], errors='coerce')
    else:
        # If PathBoostAccuracy missing, we simply create it from PB_accuracy
        main_df['PathBoostAccuracy'] = pd.to_numeric(main_df['PB_accuracy'], errors='coerce')
        pba = pd.to_numeric(main_df['PathBoostAccuracy'], errors='coerce')

    improved_mask = pb.notna() & pba.notna() & (pb > pba)
    # If PathBoostAccuracy is NaN but PB_accuracy isn't, treat that as improvement
    improved_mask = improved_mask | (pb.notna() & pba.isna())

    improvements = int(improved_mask.sum())
    # Apply replacements where improved
    main_df.loc[improved_mask, 'PathBoostAccuracy'] = pb.loc[improved_mask]

    # Update times_100 if present
    if 'PathBoostAccuracy_times_100' in main_df.columns:
        main_df['PathBoostAccuracy_times_100'] = 100.0 * pd.to_numeric(
            main_df['PathBoostAccuracy'], errors='coerce'
        )

    return main_df, improvements


def run(main_path: str, f1_dir: str, out_path: str):
    print("[INFO] Loading main CSV ...")
    main_df = read_csv_robust(main_path)
    if 'Dataset' not in main_df.columns:
        raise KeyError("Main CSV does not contain a 'Dataset' column.")

    print("[INFO] Discovering F1_scores CSVs ...")
    f1_csvs = list_f1_csvs(f1_dir)
    print(f"[INFO] Found {len(f1_csvs)} CSV(s) in '{f1_dir}'.")

    print("[INFO] Concatenating F1_scores data ...")
    f1_all = concat_f1_data(f1_csvs)

    print("[INFO] Selecting best rows per Dataset (by PB_accuracy) ...")
    f1_best = select_best_rows_by_pb(f1_all)

    print("[INFO] Adding missing columns from F1_scores to main ...")
    enriched, added_cols = add_missing_columns(main_df, f1_best)
    print(f"[INFO] Added {len(added_cols)} new column(s): {added_cols}")

    print("[INFO] Applying PB_accuracy-based improvement to PathBoostAccuracy ...")
    enriched, n_improved = replace_pathboost_with_pb(enriched)
    print(f"[INFO] Improved PathBoostAccuracy for {n_improved} row(s).")

    print(f"[INFO] Writing output CSV to '{out_path}' ...")
    enriched.to_csv(out_path, index=False)
    print("[SUCCESS] Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Integrate F1_scores into merged results.")
    parser.add_argument('--main', type=str, default=None, help="Path to main CSV. Defaults to 'merged result.csv' or 'merged_result.csv'.")
    parser.add_argument('--f1_dir', type=str, default='f1_scores', help="Directory containing F1_scores CSVs.")
    parser.add_argument('--out', type=str, default='merged_results_enriched.csv', help="Output CSV path.")

    args = parser.parse_args()

    try:
        main_csv = find_main_csv(args.main)
        run(main_csv, args.f1_dir, args.out)
    except Exception as e:
        print(f"[FAILED] {e}")
        sys.exit(1)
