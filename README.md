# Graph Classification Research Codebase

A research codebase for evaluating graph classification algorithms on TU datasets using three methods: **Sequential PathBoost**, **Graph Neural Networks (GNN)**, and **Graph Kernels**.

## Quick Start

```bash
# Run PathBoost on specific datasets
python run_pathboost.py MUTAG PTC_MR NCI1

# Run GNN with CPU (default is GPU)
python run_gnn.py MUTAG --device cpu

# Run all kernel methods
python run_kernel.py MUTAG AIDS
```

## Project Structure

```
Different_datasets/
├── run_pathboost.py          # Sequential PathBoost evaluation
├── run_gnn.py                # GNN (GIN/GINE) evaluation
├── run_kernel.py             # Kernel methods evaluation
├── shared/                   # Shared utilities
│   ├── cli.py                # Command-line interface
│   ├── csv_utils.py          # CSV output handling
│   ├── timeout.py            # Process-based timeout
│   ├── constants.py          # Configuration constants
│   └── logging_config.py     # Logging setup
├── utils.py                  # Data loading utilities
├── GNN_baseline.py           # GNN wrapper module
├── requirements.txt          # Python dependencies
├── tudataset/                # TU Benchmark dataset library
│   └── tud_benchmark/
│       ├── datasets/         # 80+ graph datasets
│       ├── auxiliarymethods/  # Evaluation functions
│       ├── gnn_baselines/    # GIN/GINE architectures
│       └── kernel_baselines/ # C++ kernel implementations
└── nx_graphs/                # Cached NetworkX graphs
```

## Installation

### Requirements
- Python 3.10+
- PyTorch (with CUDA for GPU support)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if using GPU)
# See: https://pytorch.org/get-started/locally/
```

## Usage

### Common Options

All three scripts share the same CLI interface:

| Option | Description | Default |
|--------|-------------|---------|
| `datasets` | Dataset names (positional) | All datasets |
| `--timeout` | Timeout per dataset (seconds) | 72000 (20 hours) |
| `--device` | cpu/gpu (GNN only) | gpu |
| `--verbose, -v` | Enable verbose output | False |

### Running PathBoost

```bash
# Single dataset
python run_pathboost.py MUTAG

# Multiple datasets with custom timeout
python run_pathboost.py MUTAG AIDS PTC_MR --timeout 36000

# All datasets (no arguments)
python run_pathboost.py
```

**Metrics computed:** accuracy, balanced_accuracy, f1, f1_macro, recall, roc_auc

### Running GNN

```bash
# With GPU (default)
python run_gnn.py MUTAG

# Force CPU
python run_gnn.py MUTAG --device cpu

# Verbose output
python run_gnn.py MUTAG -v
```

**Metrics computed:** accuracy, balanced_accuracy, f1, f1_macro, recall, roc_auc

### Running Kernel Methods

```bash
# Run all 7 kernel methods
python run_kernel.py MUTAG

# Multiple datasets
python run_kernel.py MUTAG AIDS NCI1
```

**Kernel methods:** WL subtree (dense/sparse), Graphlet (dense/sparse), Shortest-path (dense/sparse), WLOA

**Metrics computed:** accuracy only (C++ implementation)

## Output Format

### CSV Files

Results are saved with timestamps to prevent overwrites:

```
PathBoost_results/Sequential_PathBoost_Performance_20241227_143022.csv
GNN_results/GNN_Performance_20241227_143022.csv
Kernel_results/Kernel_Performance_20241227_143022.csv
```

### CSV Structure

| Column | Description |
|--------|-------------|
| Dataset | Dataset name (e.g., MUTAG) |
| Metric | Metric name (e.g., accuracy, f1_macro) |
| Mean | Average performance across repetitions |
| Std_Top10 | Std dev of best scores per repetition |
| Std_All100 | Std dev across all 100 fold evaluations |

**Example:**
```csv
Dataset,Metric,Mean,Std_Top10,Std_All100
MUTAG,accuracy,0.857143,0.012345,0.034567
MUTAG,f1_macro,0.849876,0.012567,0.033890
MUTAG,roc_auc,0.862345,0.010234,0.031234
AIDS,accuracy,TIMEOUT,TIMEOUT,TIMEOUT
```

### Understanding Std_Top10 vs Std_All100

The evaluation uses **10x10 fold cross-validation** (10 repetitions × 10 folds = 100 runs):

- **Std_Top10**: For each of the 10 repetitions, take the mean across folds → 10 values → compute std
- **Std_All100**: Standard deviation across all 100 individual fold scores

## Cross-Validation Methodology

```
For rep in 1..10:
    Shuffle dataset with seed = 42 + rep
    For fold in 1..10:
        Split: 80% train, 10% val, 10% test
        Train model with hyperparameter search
        Record test metrics

    Best_score[rep] = mean of fold scores for best hyperparams

Mean = average(Best_score)
Std_Top10 = std(Best_score)           # Stability across splits
Std_All100 = std(all 100 fold scores) # Total variance
```

## Timeout Handling

- Default timeout: **20 hours** (72000 seconds) per dataset
- Uses process-based isolation for hard timeout enforcement
- Failed/timed-out datasets are recorded as "FAILED" or "TIMEOUT" in CSV
- Execution continues to next dataset after timeout

## Available Datasets

The codebase includes 80+ TU Benchmark datasets:

**Molecular:** MUTAG, AIDS, NCI1, NCI109, PTC_*, Mutagenicity, Tox21_*, etc.

**Biological:** ENZYMES, PROTEINS, DD, etc.

**Social:** IMDB-BINARY, IMDB-MULTI, REDDIT-*, COLLAB, etc.

## Results Analysis

The `results_csv_files/` directory contains tools for analyzing and visualizing benchmark results.

### Analysis Notebook

`results_csv_files/analysis_csv_results.ipynb` provides:
- Comparison of PathBoost vs GNN vs Graph Kernels
- Win/loss summary tables
- Correlation analysis (dataset features vs performance)
- Scatter plots and bar charts

**Dependencies:**
```bash
pip install itables seaborn scipy
```

### Exporting to HTML

Export the notebook as an interactive HTML report (tables are sortable/searchable):

```bash
# Export with code cells hidden (report style)
jupyter nbconvert --to html --no-input results_csv_files/analysis_csv_results.ipynb

# Export with code visible
jupyter nbconvert --to html results_csv_files/analysis_csv_results.ipynb

# Custom output filename
jupyter nbconvert --to html --no-input --output report.html results_csv_files/analysis_csv_results.ipynb
```

The exported HTML includes interactive DataTables powered by `itables` - you can sort columns, search, and paginate results directly in the browser.

## Configuration

Key constants in `shared/constants.py`:

```python
DEFAULT_TIMEOUT = 72000  # 20 hours
CV_SEED = 42             # Random seed for reproducibility
NUM_REPETITIONS = 10     # CV repetitions
NUM_FOLDS = 10           # Folds per repetition
```

## Extending the Codebase

### Adding a New Algorithm

1. Create `run_newalgo.py` following the pattern of existing scripts
2. Import shared utilities:
   ```python
   from shared import (
       create_argument_parser,
       validate_datasets,
       ResultsCSVWriter,
       get_timestamped_path,
       run_with_timeout,
       setup_logging,
   )
   ```
3. Implement evaluation function returning `dict[metric -> (mean, std10, std100)]`

### Adding New Metrics

1. Update `shared/constants.py` with new metrics list
2. Modify evaluation function to compute and return new metrics

## Troubleshooting

### "CUDA not available" warning
- Install PyTorch with CUDA support
- Or use `--device cpu` flag

### Kernel methods failing
- The C++ kernel code may have path configuration issues
- Ensure datasets are properly downloaded in `tudataset/tud_benchmark/datasets/`

### Out of memory
- Reduce batch size in GNN evaluation
- Use `--device cpu` for large datasets
- Set shorter timeout to skip problematic datasets

## License

Research code - see individual dependencies for their licenses.
