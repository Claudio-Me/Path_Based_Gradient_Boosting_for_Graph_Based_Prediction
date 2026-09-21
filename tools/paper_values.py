"""The values as published in the manuscript.

Transcribed from the submitted PDF (Tables 2, 3 and 4) so that
``collect_paper_results.py`` can locate the run that produced each of them and
report anything that is no longer traceable. Accuracies and F1 scores are in
percent; "time limit" cells are recorded as None.
"""

# The 12 balanced datasets, in the order used in the paper.
# The paper abbreviates the Tox21 splits; TU names are spelled out.
DATASETS = [
    "AIDS",
    "BZR",
    "DHFR",
    "MUTAG",
    "PROTEINS_full",
    "PTC_FM",
    "SYNTHETIC",
    "Tox21_ARE_evaluation",
    "Tox21_ARE_testing",
    "Tox21_ARE_training",
    "Tox21_MMP_testing",
    "Tox21_MMP_training",
]

PAPER_LABEL = {
    "Tox21_ARE_evaluation": "Tox21_ARE_eval",
    "Tox21_ARE_testing": "Tox21_ARE_test",
    "Tox21_ARE_training": "Tox21_ARE_train",
    "Tox21_MMP_testing": "Tox21_MMP_test",
    "Tox21_MMP_training": "Tox21_MMP_train",
}

# Table 2: accuracy (mean, std) in percent. None = "Time limit" in the paper.
TABLE2 = {
    "AIDS":                 {"PB": (99.33, 0.09), "GNN": (96.66, 0.15), "WL": (96.42, 0.33), "GR": (99.04, 0.09), "SP": (99.02, 0.11)},
    "BZR":                  {"PB": (86.76, 0.55), "GNN": (82.51, 1.37), "WL": (86.90, 1.20), "GR": (82.30, 0.91), "SP": (82.30, 0.91)},
    "DHFR":                 {"PB": (78.19, 0.76), "GNN": (80.03, 1.08), "WL": (81.45, 1.22), "GR": (74.52, 0.72), "SP": (74.43, 0.90)},
    "MUTAG":                {"PB": (89.11, 1.08), "GNN": (82.83, 2.18), "WL": (76.35, 2.11), "GR": (85.32, 0.82), "SP": (85.32, 0.82)},
    "PROTEINS_full":        {"PB": (79.00, 0.51), "GNN": (71.50, 1.21), "WL": (73.06, 0.71), "GR": None,          "SP": None},
    "PTC_FM":               {"PB": (61.80, 0.95), "GNN": (59.55, 1.87), "WL": (62.84, 1.74), "GR": (56.64, 1.72), "SP": (56.64, 1.72)},
    "SYNTHETIC":            {"PB": (62.27, 1.67), "GNN": (48.53, 1.86), "WL": (45.53, 2.62), "GR": (45.73, 2.53), "SP": (45.73, 2.53)},
    "Tox21_ARE_evaluation": {"PB": (83.37, 0.18), "GNN": (81.95, 0.76), "WL": (81.27, 1.33), "GR": (82.24, 0.50), "SP": (82.54, 0.51)},
    "Tox21_ARE_testing":    {"PB": (79.40, 0.85), "GNN": (79.21, 1.89), "WL": (78.06, 2.13), "GR": (80.63, 1.24), "SP": (80.20, 1.26)},
    "Tox21_ARE_training":   {"PB": (84.79, 0.06), "GNN": (83.43, 1.32), "WL": None,          "GR": (84.74, 0.08), "SP": (84.74, 0.08)},
    "Tox21_MMP_testing":    {"PB": (85.84, 0.89), "GNN": (82.01, 2.88), "WL": (86.31, 2.11), "GR": (82.47, 1.53), "SP": (82.34, 1.20)},
    "Tox21_MMP_training":   {"PB": (85.95, 0.18), "GNN": None,          "WL": (91.14, 0.19), "GR": (84.96, 0.11), "SP": (84.99, 0.12)},
}

# Table 3: F1-macro (mean, std) in percent. Kernels cannot report it.
TABLE3 = {
    "AIDS":                 {"PB": (98.69, 0.12), "GNN": None},
    "BZR":                  {"PB": (70.81, 1.81), "GNN": (69.29, 2.32)},
    "DHFR":                 {"PB": (75.00, 5.00), "GNN": (78.60, 4.70)},
    "MUTAG":                {"PB": (83.86, 2.11), "GNN": (80.46, 3.29)},
    "PROTEINS_full":        {"PB": (69.14, 1.04), "GNN": None},
    "PTC_FM":               {"PB": (51.72, 1.91), "GNN": (53.28, 2.71)},
    "SYNTHETIC":            {"PB": (62.00, 19.00), "GNN": (32.00, 0.90)},
    "Tox21_ARE_evaluation": {"PB": (51.05, 1.99), "GNN": (52.69, 1.67)},
    "Tox21_ARE_testing":    {"PB": (54.39, 1.89), "GNN": (59.11, 3.48)},
    "Tox21_ARE_training":   {"PB": (50.16, 0.71), "GNN": None},
    "Tox21_MMP_testing":    {"PB": (62.14, 3.32), "GNN": (61.14, 4.69)},
    "Tox21_MMP_training":   {"PB": (60.83, 0.89), "GNN": None},
}

# Table 4: regression on alchemy_full, target index 0 (dipole moment).
# Values are absolute (not percent).
TABLE4 = {
    "Complete":   {"mae": (0.0174, 0.0000), "r2": (0.6397, 0.0003), "time_s": 2377.8},
    "Restricted": {"mae": (0.0184, 0.0000), "r2": (0.6027, 0.0000), "time_s": 946.7},
}

# Table 2 carries a footnote: where the standard (non-linear) kernel exceeded the
# 20-hour budget, the value reported is the one obtained with the linear kernel.
# The audit therefore accepts either variant for the kernel columns and records
# which one it found.
LINEAR_FALLBACK_KEYS = {"WL", "GR", "SP"}

# Metric name in the stored CSVs for each Table 2 column.
CSV_METRIC = {
    "PB": ("pathboost", "accuracy"),
    "GNN": ("gnn", "accuracy"),
    "WL": ("kernel", "accuracy_WL_subtree"),
    "GR": ("kernel", "accuracy_Graphlet"),
    "SP": ("kernel", "accuracy_Shortest_path"),
}

CSV_METRIC_LINEAR = {
    "WL": ("kernel", "accuracy_WL_subtree_linear"),
    "GR": ("kernel", "accuracy_Graphlet_linear"),
    "SP": ("kernel", "accuracy_Shortest_path_linear"),
}

METHOD_LABEL = {
    "PB": "PathBoost",
    "GNN": "GINE (GNN)",
    "WL": "Weisfeiler-Lehman subtree kernel",
    "GR": "Graphlet kernel",
    "SP": "Shortest-path kernel",
}
