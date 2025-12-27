import pandas as pd
import glob
import os
import re

# Get all CSV files in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = glob.glob(os.path.join(script_dir, "*.csv"))

print(f"Found {len(csv_files)} CSV files")

# Lists to store dataframes
dfs_with_pathboost = []
dfs_without_pathboost = []

# Load and categorize CSV files
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)

        # Check if "Dataset" column exists
        if "Dataset" not in df.columns:
            print(f"Warning: '{csv_file}' does not contain 'Dataset' column. Skipping.")
            continue

        # Categorize based on presence of PathBoostAccuracy
        if "PathBoostAccuracy" in df.columns:
            dfs_with_pathboost.append(df)
            print(f"Added '{os.path.basename(csv_file)}' to PathBoost group")
        else:
            dfs_without_pathboost.append(df)
            print(f"Added '{os.path.basename(csv_file)}' to non-PathBoost group")

    except Exception as e:
        print(f"Error loading '{csv_file}': {e}")

# Merge datasets with PathBoostAccuracy
if dfs_with_pathboost:
    merged_with_pathboost = pd.concat(dfs_with_pathboost, ignore_index=True)
    print(f"\nMerged {len(dfs_with_pathboost)} files with PathBoostAccuracy")
    print(f"Shape: {merged_with_pathboost.shape}")
else:
    print("\nNo files with PathBoostAccuracy found")
    merged_with_pathboost = None

# Merge datasets without PathBoostAccuracy
if dfs_without_pathboost:
    merged_without_pathboost = pd.concat(dfs_without_pathboost, ignore_index=True)
    print(f"Merged {len(dfs_without_pathboost)} files without PathBoostAccuracy")
    print(f"Shape: {merged_without_pathboost.shape}")
else:
    print("No files without PathBoostAccuracy found")
    merged_without_pathboost = None


def process_dataset_for_algorithms(df):
    """
    Process dataset to create separate tables for each algorithm,
    group by Dataset and take maximum according to accuracy column.
    """
    if df is None:
        return None

    # Find all columns that contain "_accuracy"
    accuracy_cols = [col for col in df.columns if "_accuracy" in col]

    if not accuracy_cols:
        print("No accuracy columns found in dataset")
        return None

    print(f"\nFound {len(accuracy_cols)} accuracy columns")

    # Extract algorithm names (text before "_accuracy")
    algorithms = {}
    for col in accuracy_cols:
        algo_name = col.replace("_accuracy", "")
        algorithms[algo_name] = {
            "accuracy": col,
            "std_10": f"{algo_name}_std_10" if f"{algo_name}_std_10" in df.columns else None,
            "std_100": f"{algo_name}_std_100" if f"{algo_name}_std_100" in df.columns else None
        }

    print(f"Identified {len(algorithms)} algorithms: {list(algorithms.keys())}")

    # Create a table for each algorithm
    algo_tables = []

    for algo_name, cols in algorithms.items():
        # Select relevant columns for this algorithm
        columns_to_select = ["Dataset", cols["accuracy"]]

        if cols["std_10"] and cols["std_10"] in df.columns:
            columns_to_select.append(cols["std_10"])
        if cols["std_100"] and cols["std_100"] in df.columns:
            columns_to_select.append(cols["std_100"])

        # Create algorithm table
        algo_df = df[columns_to_select].copy()

        # Group by Dataset and take maximum according to accuracy column
        algo_df_grouped = algo_df.sort_values(by=cols["accuracy"], ascending=False).groupby("Dataset",
                                                                                            as_index=False).first()

        print(f"  {algo_name}: {algo_df_grouped.shape[0]} unique datasets")

        algo_tables.append(algo_df_grouped)

    # Merge all algorithm tables by joining on "Dataset"
    if algo_tables:
        merged_algos = algo_tables[0]
        for i in range(1, len(algo_tables)):
            merged_algos = pd.merge(merged_algos, algo_tables[i], on="Dataset", how="outer")

        print(f"\nMerged algorithm tables shape: {merged_algos.shape}")
        return merged_algos

    return None


# Process the dataset without PathBoostAccuracy
print("\n" + "=" * 60)
print("Processing datasets WITHOUT PathBoostAccuracy")
print("=" * 60)
processed_without_pathboost = process_dataset_for_algorithms(merged_without_pathboost)

# Process the dataset with PathBoostAccuracy
print("\n" + "=" * 60)
print("Processing dataset WITH PathBoostAccuracy")
print("=" * 60)

if merged_with_pathboost is not None:
    # Remove columns containing "_accuracy", "_std_10", or "_std_100"
    cols_to_remove = [col for col in merged_with_pathboost.columns
                      if "_accuracy" in col or "_std_10" in col or "_std_100" in col]

    print(f"Removing {len(cols_to_remove)} columns from PathBoost dataset:")
    print(f"  {cols_to_remove}")

    pathboost_cleaned = merged_with_pathboost.drop(columns=cols_to_remove)
    print(f"Cleaned PathBoost dataset shape: {pathboost_cleaned.shape}")
else:
    pathboost_cleaned = None

# Perform inner join between processed datasets
print("\n" + "=" * 60)
print("Performing final inner join")
print("=" * 60)

if processed_without_pathboost is not None and pathboost_cleaned is not None:
    joined_result = pd.merge(
        pathboost_cleaned,
        processed_without_pathboost,
        on="Dataset",
        how="inner"
    )

    print(f"Joined result shape: {joined_result.shape}")
    print(f"Total rows before grouping: {len(joined_result)}")

    # Group by Dataset and take maximum according to PathBoostAccuracy,
    # then by Weisfeiler-Lehman subtree kernel_accuracy
    print("\n" + "=" * 60)
    print("Grouping by Dataset and selecting maximum")
    print("=" * 60)

    # Check if PathBoostAccuracy column exists
    if "PathBoostAccuracy" in joined_result.columns:
        # Check for Weisfeiler-Lehman column
        wl_col = None
        if "Weisfeiler-Lehman subtree kernel_accuracy" in joined_result.columns:
            wl_col = "Weisfeiler-Lehman subtree kernel_accuracy"

        if wl_col:
            # Sort by PathBoostAccuracy (descending) then by WL accuracy (descending)
            sorted_result = joined_result.sort_values(
                by=["PathBoostAccuracy", wl_col],
                ascending=[False, False]
            )
            print(f"Sorting by: PathBoostAccuracy (primary), {wl_col} (secondary)")
        else:
            # Only sort by PathBoostAccuracy
            sorted_result = joined_result.sort_values(
                by=["PathBoostAccuracy"],
                ascending=False
            )
            print("Sorting by: PathBoostAccuracy only")
            print("Warning: Weisfeiler-Lehman subtree kernel_accuracy column not found")

        # Group by Dataset and take the first row (which is the maximum)
        final_result = sorted_result.groupby("Dataset", as_index=False).first()

        print(f"Final result shape after grouping: {final_result.shape}")
        print(f"Unique datasets: {final_result['Dataset'].nunique()}")
        print("\nTop 5 rows:")
        print(final_result.head())

        # Save result
        output_file = os.path.join(script_dir, "merged_result.csv")
        final_result.to_csv(output_file, index=False)
        print(f"\nResult saved to: {output_file}")
    else:
        print("Warning: PathBoostAccuracy column not found in joined result")
        print("Available columns:", joined_result.columns.tolist())

        # Save without grouping
        output_file = os.path.join(script_dir, "merged_result.csv")
        joined_result.to_csv(output_file, index=False)
        print(f"\nResult saved to: {output_file}")

elif processed_without_pathboost is not None:
    print("Only datasets without PathBoostAccuracy were processed")
    output_file = os.path.join(script_dir, "merged_result.csv")
    processed_without_pathboost.to_csv(output_file, index=False)
    print(f"Result saved to: {output_file}")

elif pathboost_cleaned is not None:
    print("Only PathBoost dataset was found")
    output_file = os.path.join(script_dir, "merged_result.csv")
    pathboost_cleaned.to_csv(output_file, index=False)
    print(f"Result saved to: {output_file}")

else:
    print("No valid datasets to process")