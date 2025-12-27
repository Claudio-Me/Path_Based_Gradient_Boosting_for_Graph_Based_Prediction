"""
GNN baseline wrapper module.

Provides a high-level interface to run GNN (GIN/GINE) evaluations
on TU datasets with cross-validation.
"""
import os


def run_gnn_baseline(dataset_name, num_reps=10, use_gine=None, cv_seed: int | None = 42, device: str = 'auto'):
    """
    Runs GNN baseline with CV and hyperparameter exploration.

    Args:
        dataset_name: Name of the TU dataset
        num_reps: Number of CV repetitions (default: 10)
        use_gine: Whether to use GINE (edge-aware) instead of GIN.
                  If None, automatically determined based on dataset.
        cv_seed: Random seed for reproducibility (default: 42)
        device: Device to run on: 'cpu', 'gpu', 'cuda', or 'auto' (default).
                'auto' uses GPU if available, otherwise CPU.

    Returns:
        Dict mapping metric name to (mean, std_10, std_100) tuple.
    """
    import torch

    # Handle device selection via environment variable
    # This must be done BEFORE importing torch_geometric modules
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif device in ('gpu', 'cuda'):
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            print(f"WARNING: GPU requested but CUDA not available, using CPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # 'auto' - let torch decide based on CUDA availability

    import tudataset.tud_benchmark.auxiliarymethods.datasets as dp
    from tudataset.tud_benchmark.auxiliarymethods.gnn_evaluation import gnn_evaluation
    from tudataset.tud_benchmark.gnn_baselines.gnn_architectures import GIN, GINE

    # Load dataset
    dp.get_dataset(dataset_name)

    # Check for edge labels or continuous node attributes
    dataset_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tudataset", "tud_benchmark", "datasets", dataset_name, dataset_name, "raw"
    )
    has_edge_labels = os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_edge_labels.txt"))
    has_edge_attributes = os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_edge_attributes.txt"))

    if use_gine is None:
        use_gine_auto = has_edge_labels or has_edge_attributes
    else:
        use_gine_auto = use_gine

    GNNLayer = GINE if use_gine_auto else GIN

    res = gnn_evaluation(
        GNNLayer,
        dataset_name,
        [1, 2, 3, 4, 5],       # layers
        [32, 64, 128],         # hidden dimensions
        max_num_epochs=200,
        batch_size=64,
        start_lr=0.01,
        num_repetitions=num_reps,
        all_std=True,
        cv_seed=cv_seed
    )

    return res
