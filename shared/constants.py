"""Shared constants across all evaluation scripts."""
import os

# Default timeout: 20 hours = 72000 seconds
DEFAULT_TIMEOUT = 72000

# Cross-validation seed for reproducibility
CV_SEED = 42

# Number of repetitions for CV
NUM_REPETITIONS = 10

# Number of folds
NUM_FOLDS = 10

# Base directory for datasets (relative to project root)
def get_base_dir():
    """Get the base directory for TU datasets."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'tudataset', 'tud_benchmark', 'datasets'
    )

# NetworkX graphs cache directory
NX_GRAPHS_DIR = "nx_graphs"

# Metrics tracked by each method
PATHBOOST_METRICS = ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'recall', 'roc_auc']
GNN_METRICS = ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'recall', 'roc_auc']
KERNEL_METRICS = ['accuracy']  # C++ kernels only return accuracy

# Kernel method names
KERNEL_METHODS = [
    "WL_subtree",
    "WL_subtree_linear",
    "Graphlet",
    "Graphlet_linear",
    "Shortest_path",
    "Shortest_path_linear",
    "WLOA",
]

# All available datasets (from tudataset)
ALL_DATASETS = [
    "AIDS", "BZR", "BZR_MD", "COLLAB", "COX2", "COX2_MD", "DBLP_v1", "DD",
    "DHFR", "DHFR_MD", "ENZYMES", "ER_MD", "IMDB-BINARY", "IMDB-MULTI", "KKI",
    "MCF-7", "MCF-7H", "MOLT-4", "MOLT-4H", "MUTAG", "Mutagenicity",
    "NCI-H23", "NCI-H23H", "NCI1", "NCI109", "OHSU", "OVCAR-8", "OVCAR-8H",
    "P388", "P388H", "PC-3", "PC-3H", "Peking_1", "PROTEINS", "PROTEINS_full",
    "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "REDDIT-BINARY", "REDDIT-MULTI-12K",
    "REDDIT-MULTI-5K", "SF-295", "SF-295H", "SN12C", "SN12CH", "SW-620", "SW-620H",
    "SYNTHETIC", "TWITTER-Real-Graph-Partial", "UACC257", "UACC257H", "Yeast", "YeastH",
    # Tox21 datasets
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
]
