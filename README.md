# Path-Based Gradient Boosting for Graph-Level Prediction

Code, data pointers and results for the paper

> **Path-Based Gradient Boosting for Graph-Level Prediction**
> Claudio Meggio, Johan Pensar, Riccardo De Bin
> Department of Mathematics, University of Oslo

The paper introduces **PathBoost**, a gradient tree boosting method for graph-level
classification and regression that adaptively learns path-based features, and compares it
against graph neural networks and graph kernels on twelve TUDatasets benchmark datasets.

This repository contains the **experiment harness**: dataset preparation, the evaluation
protocol, the competitor baselines, and the results reported in the paper. The PathBoost
algorithm itself lives in a separate package:

- **Method:** [`path_boost`](https://github.com/Claudio-Me/extended_path_boost) — also on
  [PyPI](https://pypi.org/project/path_boost/) (`pip install path_boost==2.1.0`)
- **Benchmark datasets:** [TUDatasets](https://www.graphlearning.io/), downloaded
  automatically (see [Data](#data))

---

## Table of contents

1. [Quick start](#quick-start)
2. [What this repository reproduces](#what-this-repository-reproduces)
3. [Installation](#installation)
4. [Building the graph kernels](#building-the-graph-kernels)
5. [Data](#data)
6. [Datasets used in the paper](#datasets-used-in-the-paper)
7. [Reproducing each table and figure](#reproducing-each-table-and-figure)
8. [Evaluation protocol](#evaluation-protocol)
9. [Hyperparameters](#hyperparameters)
10. [Computing environment](#computing-environment)
11. [Output format](#output-format)
12. [Published results](#published-results)
13. [Repository layout](#repository-layout)
14. [Running on a cluster](#running-on-a-cluster)
15. [Troubleshooting](#troubleshooting)
16. [License and citation](#license-and-citation)

---

## Quick start

From a clean clone, on a machine with Python 3.10 or newer:

```bash
git clone https://github.com/Claudio-Me/Path_Based_Gradient_Boosting_for_Graph_Based_Prediction.git
cd Path_Based_Gradient_Boosting_for_Graph_Based_Prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional, needed only for the graph-kernel baselines (see below)
./build_kernels.sh

# Download the datasets used in the paper
python download_datasets.py --paper

# Check the whole pipeline end to end on one small dataset
./reproduce_paper.sh smoke
```

`smoke` runs PathBoost, GINE and all seven kernel variants on MUTAG with a reduced
cross-validation and hyperparameter grid (`--quick`: 2 repetitions, 3 folds, one grid
point). It takes **about a minute** on a laptop CPU and confirms that the method, the GNN
baseline, the compiled kernels and the dataset loader all work.

It does **not** reproduce Table 2. Expect accuracies roughly in the 75-88% range; a run on
this machine gave PathBoost 83.2, GINE 75.3, WL 80.7, Graphlet 85.6, Shortest-path 80.6,
against the published 89.11 / 82.83 / 76.35 / 85.32 / 85.32 obtained with the full 10x10
cross-validation over the complete grid. For the published numbers see
[Reproducing each table and figure](#reproducing-each-table-and-figure).

## What this repository reproduces

| Paper artifact | Command | Output |
|---|---|---|
| Table 1 — dataset characteristics | `./reproduce_paper.sh table1` | `dataset_summary.csv` |
| Table 2 — accuracy, 5 methods | `./reproduce_paper.sh classification` | `PathBoost_results/`, `GNN_results/`, `Kernel_results/` |
| Table 3 — F1-macro, PathBoost vs GINE | same run (the `f1_macro` rows) | same CSVs |
| Table 4 — regression on alchemy_full | `./reproduce_paper.sh regression` | `Sequential_PathBoost_Regression_Performance_*.csv` |
| Figure 1 — learning curve on PROTEINS_full | `./reproduce_paper.sh figure1` | `subsample_dataset/PROTEINS_full_results.json` + `.png` |
| Table S2 — linear-kernel variants | part of `classification` (the `*_linear` kernels) | `Kernel_results/` |
| Published tables, without retraining | `./reproduce_paper.sh tables` | `paper_results/` |

Every target accepts `--dry-run` to print the commands it would execute:

```bash
./reproduce_paper.sh classification --dry-run
```

## Installation

**Requirements**

- Python 3.10 or newer (3.12 was used for the reported experiments)
- A C++11 compiler and `eigen3` — only for the graph-kernel baselines
- A CUDA-capable GPU — optional; the GINE baseline runs on CPU with `--device cpu`

**Python packages**

```bash
pip install -r requirements.txt
```

This installs `path_boost==2.1.0` (the method), PyTorch and PyTorch Geometric (the GINE
baseline), scikit-learn, NetworkX and the analysis dependencies.

To install PyTorch against a specific CUDA build, install it first following
[pytorch.org](https://pytorch.org/get-started/locally/) and then run the command above.

**Exact versions.** `requirements-lock.txt` pins every package version of the machine that
produced the published numbers:

```bash
pip install -r requirements-lock.txt
pip install path_boost==2.1.0
```

**A note on the PathBoost package name.** Releases up to 1.6 published the estimators under
the module name `extended_path_boost`; from 2.0.0 the module is called `path_boost`. The
class names never changed. The scripts import through `shared/pathboost_import.py`, which
accepts either, so an older local install keeps working.

## Building the graph kernels

The WL, Graphlet, Shortest-path and WLOA baselines are the original C++ implementations
from [TUDataset](https://github.com/chrsmrrs/tudataset), vendored under `tudataset/`. They
must be compiled once into a Python extension. **Skip this if you only want to run
PathBoost and the GNN.**

System dependencies:

```bash
sudo apt install g++ libeigen3-dev      # Debian / Ubuntu
brew install eigen                      # macOS
```

Then, with the virtual environment active:

```bash
./build_kernels.sh
```

The script locates the `eigen3` headers (via `pkg-config`, falling back to the usual
Homebrew and Linux locations), picks the right compiler flags for your platform, builds
`tudataset/tud_benchmark/kernel_baselines<suffix>.so` and verifies that it imports. If
eigen lives somewhere unusual, pass it explicitly:

```bash
EIGEN_FLAGS=-I/path/to/eigen3 ./build_kernels.sh
```

Equivalent manual command, if you prefer:

```bash
cd tudataset/tud_benchmark/kernel_baselines
g++ -O3 -shared -std=c++17 -fPIC $(pkg-config --cflags eigen3) $(python -m pybind11 --includes) \
    kernel_baselines.cpp src/*.cpp -o ../kernel_baselines$(python-config --extension-suffix)
```

On macOS replace `-fPIC` with `-undefined dynamic_lookup`. Note `-std=c++17`: eigen 5
requires at least C++14, whereas the upstream TUDataset instructions say C++11.

The compiled extension is platform- and Python-version-specific and is not committed;
build it once per environment.

## Data

All datasets come from the public [TUDatasets](https://www.graphlearning.io/) collection
and are **not** redistributed here. They are downloaded through PyTorch Geometric into
`tudataset/tud_benchmark/datasets/` on first use.

```bash
python download_datasets.py --paper         # the 12 classification datasets (~100 MB)
python download_datasets.py --regression    # alchemy_full (~200k graphs, several GB)
python download_datasets.py MUTAG BZR       # named datasets only
```

You can also skip this step: naming a dataset on any run script downloads it on demand.

On first use each dataset is converted to NetworkX graphs and cached as a pickle under
`nx_graphs/`. This speeds up later runs considerably but the cache is large — the full set
of 80+ datasets reaches several GB. It can be deleted at any time and will be rebuilt.

## Datasets used in the paper

Twelve balanced binary datasets: every node carries a categorical label, the response is
binary, and the minority class holds at least 15% of the samples.

| Paper (Table 1) | TUDatasets name | Graphs | Avg. nodes |
|---|---|---:|---:|
| AIDS | `AIDS` | 2000 | 15.69 |
| BZR | `BZR` | 405 | 35.75 |
| DHFR | `DHFR` | 756 | 42.43 |
| MUTAG | `MUTAG` | 188 | 17.93 |
| PROTEINS_full | `PROTEINS_full` | 1113 | 39.06 |
| PTC_FM | `PTC_FM` | 349 | 14.11 |
| SYNTHETIC | `SYNTHETIC` | 300 | 100.00 |
| Tox21_ARE_eval | `Tox21_ARE_evaluation` | 970 | 17.01 |
| Tox21_ARE_test | `Tox21_ARE_testing` | 234 | 21.99 |
| Tox21_ARE_train | `Tox21_ARE_training` | 5670 | 16.28 |
| Tox21_MMP_test | `Tox21_MMP_testing` | 238 | 21.68 |
| Tox21_MMP_train | `Tox21_MMP_training` | 5418 | 17.49 |

The paper abbreviates the Tox21 split names; use the TUDatasets spelling on the command
line. The list is also available programmatically as `shared.constants.PAPER_DATASETS`.

The regression experiment (Table 4) uses `alchemy_full`, target index 0 (dipole moment).

## Reproducing each table and figure

### Table 1 — dataset characteristics

```bash
python dataset_analysis.py
```

Writes `dataset_summary.csv` (graph counts, average nodes and edges, feature counts, class
balance, number of categorical attribute classes). Minutes.

### Tables 2 and 3 — classification

```bash
./reproduce_paper.sh classification -j 4 --device gpu
```

Or one method and one dataset at a time:

```bash
python run_pathboost.py MUTAG
python run_gnn.py MUTAG --device cpu
python run_kernel.py MUTAG
python run_pathboost.py AIDS BZR DHFR       # several datasets in one run
```

Table 2 reads the `accuracy` rows of the resulting CSVs, Table 3 the `f1_macro` rows
(kernels report accuracy only — a constraint of the C++ implementation).

**This is the expensive target.** The paper allowed a 20-hour budget per dataset per
method, and cells that exceeded it are printed as "Time limit". PathBoost on MUTAG, the
smallest dataset, is 10 repetitions × 10 folds × 12 grid points = 1200 model fits. Budget
days for the full sweep and use `-j` to run datasets in parallel.

Use `--quick` on any of the three scripts for a fast pipeline check with a reduced grid.

### Table 4 — regression

```bash
python run_pathboost_regression_alchemy.py alchemy_full
```

Runs both variants reported in the paper: **Complete** (all node attributes) and
**Restricted** (`categorical_only`, graph structure alone). Useful flags:

```bash
python run_pathboost_regression_alchemy.py alchemy_full --max-graphs 5000   # subsample
python run_pathboost_regression_alchemy.py alchemy_full --quick             # 2x2 CV
```

Note that `--max-graphs` changes the result: the published Table 4 uses the full dataset.

### Figure 1 — learning curve on PROTEINS_full

```bash
python subsample_dataset/run_evaluation.py --dataset PROTEINS_full --plot
```

Trains PathBoost and GIN on 10%, 20%, …, 100% of the training data over 4 train/test
splits, writing `subsample_dataset/PROTEINS_full_results.json` and a PNG. To redraw the
figure from the stored results without retraining:

```bash
python subsample_dataset/run_evaluation.py --dataset PROTEINS_full --plot-only
```

### Table S2 — linear kernel variants

`run_kernel.py` runs all seven kernel variants by default: `WL_subtree`,
`WL_subtree_linear`, `Graphlet`, `Graphlet_linear`, `Shortest_path`,
`Shortest_path_linear`, `WLOA`. Table 2 reports the non-linear ones; Table S2 reports the
linear ones, used in the paper wherever the non-linear kernel exceeded the time limit.

```bash
python run_kernel.py PROTEINS_full --kernels Graphlet_linear Shortest_path_linear
```

## Evaluation protocol

Identical across all methods, following the TUDatasets protocol:

```
for rep in 0..9:                       # 10 repetitions
    KFold(n_splits=10, shuffle=True, random_state=42 + rep)
    grid search over the hyperparameters of the method
    record the scores of the best configuration on each of the 10 folds

Mean        = average of the 10 per-repetition scores
Std_Top10   = std. dev. of those 10 per-repetition scores      <- reported in the paper
Std_All100  = std. dev. of all 100 individual fold scores
```

The base seed is `CV_SEED = 42` (`shared/constants.py`); repetition *i* uses seed `42 + i`,
so splits are identical across methods and reruns.

Metrics: accuracy, balanced accuracy, F1, F1-macro, recall and ROC-AUC for PathBoost and
GINE; accuracy only for the kernels. Regression reports MAE, MSE and R².

Datasets whose response is not binary are skipped, as are datasets with no categorical node
attribute (PathBoost needs one to anchor its paths — see
`utils.find_categorical_node_attributes`, which accepts an attribute with at most 200
distinct values).

## Hyperparameters

The grids of Table S1, as implemented.

**PathBoost** (`run_pathboost.py`, `QUICK_PARAM_GRID` is not this one):

| Parameter | Values |
|---|---|
| `learning_rate` | 0.1, 0.02 |
| `max_path_length` | 3, 5 |
| `n_iter` | 500, 1500, 2000 |
| base learner | `DecisionTreeRegressor(max_depth=4, max_leaf_nodes=10)` |
| selector | `DecisionTreeClassifier` |
| anchor labels | all values of the selected categorical node attribute |
| refit criterion | balanced accuracy |

**GINE** (`run_gnn.py`; GIN is used for datasets without edge features):

| Parameter | Values |
|---|---|
| layers | 1, 2, 3, 4, 5 |
| hidden units | 32, 64, 128 |
| learning rate | 0.01 |
| epochs | up to 200 |
| batch size | 64 |
| pooling / head | mean pooling, 2-layer MLP, dropout 0.5 |

**Graph kernels** (`run_kernel.py` and `tudataset/tud_benchmark/auxiliarymethods/`):

| Parameter | Values |
|---|---|
| WL / WLOA iterations | 1–5 |
| SVM `C` | 10³, 10², 10¹, 10⁰, 10⁻¹, 10⁻², 10⁻³ |
| classifier | `SVC(kernel="precomputed")`, or `LinearSVC` for the `*_linear` variants |
| gram matrices | normalised (`normalize_gram_matrix`) |

**PathBoost regression** (`run_pathboost_regression_alchemy.py`): same grid as the
classifier, squared-error loss, refit on negative MAE.

## Computing environment

The published experiments ran on a Linux server with two Intel Xeon Gold 6226R CPUs
(2.90 GHz, 16 cores each) and NVIDIA A10 GPUs, under Red Hat Enterprise Linux 8.10.
PathBoost and the graph kernels are CPU-only; GINE used a single GPU.

`requirements-lock.txt` records the exact package versions. Because the machine was shared,
the wall-clock times reported in the paper are indicative rather than benchmark-grade.

Results are not expected to be bit-identical across machines: the cross-validation splits
are seeded and deterministic, but GPU kernels, BLAS threading and library versions
introduce small numerical differences.

## Output format

One timestamped CSV per run, so nothing is ever overwritten:

```
PathBoost_results/Sequential_PathBoost_Performance_20260920_143022.csv
GNN_results/GNN_Performance_20260920_143022.csv
Kernel_results/Kernel_Performance_20260920_143022.csv
```

| Column | Meaning |
|---|---|
| `Dataset` | dataset name |
| `Metric` | `accuracy`, `f1_macro`, … (kernels: `accuracy_<kernel>`) |
| `Mean` | mean over the 10 repetitions |
| `Std_Top10` | std. dev. over the 10 repetitions — the ± reported in the paper |
| `Std_All100` | std. dev. over all 100 fold scores |
| `Training_Time_Mean`, `Training_Time_Std` | seconds per fit for the selected configuration |

A run that hit the time budget or crashed is recorded as `TIMEOUT` or `FAILED` in every
column, and the sweep moves on to the next dataset.

Note the scale: PathBoost and GNN store accuracies as fractions (`0.8911`), the C++ kernels
as percentages (`89.11`).

## Published results

`paper_results/` holds the numbers reported in the paper together with the runs they came
from, so the tables can be inspected without re-running anything. It is built — and
audited — by:

```bash
python tools/collect_paper_results.py            # report only
python tools/collect_paper_results.py --write    # rebuild paper_results/
```

The script searches the stored result CSVs for a run reproducing each published value to
the precision at which it was printed, and reports anything it cannot find rather than
substituting the closest available run. See `paper_results/README.md` for the current
status of that audit.

If some values are unaccounted for, `tools/run_missing.py` turns the report into the
minimal set of commands that would regenerate exactly those, and can run them:

```bash
python tools/run_missing.py                        # show the plan, run nothing
python tools/run_missing.py --execute -j 7         # run it
python tools/run_missing.py --extra-dir /synced    # re-check against synced results first
```

Note that this recomputes rather than recovers: the splits are seeded, but library
versions have moved on since the paper, so new numbers land near the published ones
rather than on them.

The analysis notebook `results_csv_files/analysis_csv_results.ipynb` turns the raw result
CSVs into the comparison tables, win/loss summaries and charts.

## Repository layout

```
.
├── run_pathboost.py                     # PathBoost classification (Tables 2, 3)
├── run_gnn.py                           # GIN / GINE baseline    (Tables 2, 3)
├── run_kernel.py                        # 7 graph kernel variants (Tables 2, S2)
├── run_pathboost_regression_alchemy.py  # regression             (Table 4)
├── subsample_dataset/                   # learning-curve experiment (Figure 1)
├── dataset_analysis.py                  # dataset characteristics   (Table 1)
├── download_datasets.py                 # fetch TU datasets
├── reproduce_paper.sh                   # one entry point for all of the above
├── build_kernels.sh                     # compile the C++ graph kernels
├── tools/collect_paper_results.py       # rebuild + audit paper_results/
├── tools/run_missing.py                 # re-run only the experiments the audit lacks
├── paper_results/                       # published numbers and their provenance
├── shared/                              # CLI, constants, CSV writer, timeout, imports
├── utils.py                             # dataset loading, caching, label preprocessing
├── tudataset/                           # vendored TUDataset benchmark code (C++ kernels)
├── results_csv_files/                   # analysis notebooks and aggregated tables
├── requirements.txt / requirements-lock.txt
└── *_results/                           # timestamped run outputs
```

## Running on a cluster

SLURM submission scripts for the UiO Fox cluster are included and serve as templates:

| Script | Purpose |
|---|---|
| `run_classification_fox.sh` | all three classifiers in one job |
| `run_regression_fox.sh` | one job per regression dataset |
| `run_gnn_regression_fox.sh` | GNN regression baseline |
| `download_datasets.sh` | dataset download as a batch job |

Locally, `run_parallel.sh` and `run_parallel_regression.sh` spread datasets over cores:

```bash
./run_parallel.sh pathboost -j 7
./run_parallel.sh gnn -j 7 --device cpu
./run_parallel.sh kernel -j 1          # the C++ kernels already use every core
```

## Troubleshooting

**`Could not import the PathBoost package`** — `pip install path_boost==2.1.0`.

**`Dataset 'X' not found` / `No datasets to process`** — running with no dataset argument
only uses datasets already downloaded. Name the dataset explicitly, or run
`python download_datasets.py --paper` first. `--no-download` restores the old skip
behaviour for batch jobs on a shared dataset directory.

**`ModuleNotFoundError: kernel_baselines`** — the C++ extension has not been compiled; run
`./build_kernels.sh`. A binary built on one platform or Python version will not load on
another.

**`Eigen/Sparse file not found`, or eigen errors about `enable_if_t`** — `./build_kernels.sh`
handles both (it passes the right `-I` and compiles with C++17). If you are invoking `g++`
by hand, see [Building the graph kernels](#building-the-graph-kernels).

**All kernels report `FAILED - Process terminated without result` and print
`!!! Unable to open file 1 !!!`** — the C++ code resolves dataset paths relative to the
working directory. `run_kernel.py` switches to `tudataset/tud_benchmark/` for the duration
of the computation, so this should no longer happen; if it does, check that the dataset
exists under `tudataset/tud_benchmark/datasets/<name>/<name>/raw/`.

**`CUDA not available`** — pass `--device cpu`, or install a CUDA build of PyTorch.

**Out of memory** — use `--device cpu` for the GNN, `--max-graphs N` for regression, and
note that the largest TU datasets (`naphthalene`, `toluene`) need subsampling. The
`nx_graphs/` cache can also be deleted to reclaim disk space.

**A dataset is skipped** — it is either not binary, or has no categorical node attribute
with 200 or fewer distinct values, which PathBoost requires. `salicylic_acid` is excluded
by default because of a PyTorch Geometric loader bug with empty node labels.

## License and citation

This repository is released under the MIT License (`LICENSE`). The vendored `tudataset/`
directory carries its own license and citation requirements.

```bibtex
@article{meggio2026pathboost,
  title  = {Path-Based Gradient Boosting for Graph-Level Prediction},
  author = {Meggio, Claudio and Pensar, Johan and De Bin, Riccardo},
  year   = {2026}
}
```

Please also cite the TUDataset benchmark:

```bibtex
@inproceedings{Morris+2020,
  title     = {TUDataset: A collection of benchmark datasets for learning with graphs},
  author    = {Morris, Christopher and Kriege, Nils M. and Bause, Franka and
               Kersting, Kristian and Mutzel, Petra and Neumann, Marion},
  booktitle = {ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020)},
  year      = {2020},
  url       = {www.graphlearning.io}
}
```
