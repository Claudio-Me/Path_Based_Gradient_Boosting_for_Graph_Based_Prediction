import os
import numpy as np
import uuid
import networkx as nx
from sklearn.model_selection import GridSearchCV
from extended_path_boost import SequentialPathBoostClassifier

from tudataset.tud_benchmark.auxiliarymethods.datasets import get_dataset
from dataset_analysis import get_summary_table

from utils import (
    ensure_dir,
    get_unique_csv_path,
    append_row_to_csv,
    load_or_build_nx_graphs,
    evaluate_pathboost_with_timeout,
    evaluate_gnn_with_timeout,
)

datasets = [
    "MUTAG", "AIDS", #"BZR", "COX2_MD", "DHFR", "ER_MD",
    "MCF-7", "MCF-7H", "MOLT-4", "MOLT-4H", "Mutagenicity", "MUTAG", "NCI1", "NCI109",
    "NCI-H23", "NCI-H23H", "OVCAR-8", "OVCAR-8H", "P388", "P388H", "PC-3", "PC-3H",
    "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "SF-295", "SF-295H", "SN12C", "SN12CH",
    "SW-620", "SW-620H",
    "Tox21_AhR_training", "Tox21_AhR_testing", "Tox21_AhR_evaluation",
    "Tox21_AR_training", "Tox21_AR_testing", "Tox21_AR_evaluation",
    "Tox21_AR-LBD_training", "Tox21_AR-LBD_testing", "Tox21_AR-LBD_evaluation",
    "Tox21_ARE_training", "Tox21_ARE_testing", "Tox21_ARE_evaluation",
    "Tox21_aromatase_training", "Tox21_aromatase_testing", "Tox21_aromatase_evaluation",
    "Tox21_ATAD5_training", "Tox21_ATAD5_testing", "Tox21_ATAD5_evaluation",
    "Tox21_ER_training", "Tox21_ER_testing", "Tox21_ER_evaluation",
    "Tox21_ER-LBD_training", "Tox21_ER-LBD_testing", "Tox21_ER-LBD_evaluation",
    "Tox21_HSE_training", "Tox21_HSE_testing", "Tox21_HSE_evaluation",
    "Tox21_MMP_training", "Tox21_MMP_testing", "Tox21_MMP_evaluation",
    "Tox21_p53_training", "Tox21_p53_testing", "Tox21_p53_evaluation",
    "Tox21_PPAR-gamma_training", "Tox21_PPAR-gamma_testing", "Tox21_PPAR-gamma_evaluation",
    "UACC257", "UACC257H", "Yeast", "YeastH", "DD", "KKI", "OHSU", "Peking_1",
    "PROTEINS", "PROTEINS_full", "DBLP_v1", "TWITTER-Real-Graph-Partial", "SYNTHETIC", "DHFR_MD", "BZR_MD", "COX2",
]


def build_header() -> list[str]:
    return [
        "Dataset", "Graphs", "AvgNodes", "AvgEdges",
        "NodeFeatures", "EdgeFeatures", "TotalFeatures",
        "ClassPercentages", "CatAttrClasses",
        "PathBoostAccuracy", "PathBoostStd10", "PathBoostStd100",
        "GNNAccuracy", "GNNStd10", "GNNStd100",
        "KernelAccuracy"
    ]


def get_base_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'tudataset', 'tud_benchmark', 'datasets'
    )


# Global seed used for both PathBoost and GNN CV so folds are aligned
CV_SEED = 42


def process_dataset(dataset_name: str, base_dir: str, nx_graphs_dir: str, csv_path: str, timeout=36000) -> tuple:
    print(f"Processing {dataset_name}...")
    labels = get_dataset(dataset_name, regression=False)
    labels = np.array(labels).flatten()

    dataset_path = os.path.join(base_dir, dataset_name)
    nx_graphs = load_or_build_nx_graphs(dataset_name, dataset_path, nx_graphs_dir)


    # PathBoost with timeout -> returns dict of metric -> (mean, std10, std100)
    try:
        path_boost_results = evaluate_pathboost_with_timeout(
            nx_graphs, labels, dataset_name, timeout_sec=timeout, cv_seed=CV_SEED
        )
    except Exception as e:
        path_boost_results = {}
        print(f"PathBoost baseline failed for {dataset_name}: {e}")

    # GNN with timeout -> now returns dict of metric -> (mean, std10, std100)
    gnn_results = {}
    try:
        gnn_results = evaluate_gnn_with_timeout(
            dataset_name, timeout_sec=timeout, cv_seed=CV_SEED
        )
        if 'accuracy' in gnn_results:
            acc, s10, s100 = gnn_results['accuracy']
            print(f"GNN for {dataset_name}: acc={acc:.4f}, std10={s10:.4f}, std100={s100:.4f}")
        else:
            print(f"GNN for {dataset_name}: no accuracy reported.")
    except Exception as e:
        gnn_results = {}
        print(f"GNN baseline failed for {dataset_name}: {e}")

    # Append dataset info row with metrics
    info_table = get_summary_table([dataset_name], base_dir=base_dir, nx_graphs_dir=nx_graphs_dir)
    if info_table and len(info_table) > 0:
        row = info_table[0]
        # PathBoost metrics (unprefixed to keep backward compatibility)
        for metric, (mean, std10, std100) in path_boost_results.items():
            row[f'PB_{metric}'] = mean
            row[f'PB_{metric}_std10'] = std10
            row[f'PB_{metric}_std100'] = std100

        # GNN metrics (prefixed to avoid overwriting PathBoost columns)
        for metric, (mean, std10, std100) in gnn_results.items():
            row[f'GNN_{metric}'] = mean
            row[f'GNN_{metric}_std10'] = std10
            row[f'GNN_{metric}_std100'] = std100

        # placeholder if kernel not computed here
        if "CatAttr" in row:
            del row["CatAttr"]
        append_row_to_csv(csv_path, row)
        print(f"Appended row to summary table at {csv_path}")

    return path_boost_results, gnn_results


def main():
    best_dataset = None
    best_accuracy = -1.0
    results: dict[str, float] = {}
    gnn_results: dict[str, float] = {}

    base_dir = get_base_dir()
    nx_graphs_dir = "nx_graphs"
    ensure_dir(nx_graphs_dir)

    # unique CSV per run
    csv_path = get_unique_csv_path("combined_dataset_summary", ".csv")
    header = build_header()

    for dataset_name in datasets:
        pb_metrics, gnn_metrics = process_dataset(dataset_name=dataset_name, base_dir=base_dir, nx_graphs_dir=nx_graphs_dir,
                                                  csv_path=csv_path, timeout=10000)

    print("all done")


if __name__ == "__main__":
    main()
