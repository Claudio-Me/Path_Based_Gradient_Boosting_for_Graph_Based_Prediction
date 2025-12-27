

def run_gnn_baseline(dataset_name, num_reps=10, use_gine=None, cv_seed: int | None = 42):
    """
    Runs GNN baseline with CV and hyperparameter exploration.
    Returns dict: metric -> (mean, std_10, std_100).
    """
    import tudataset.tud_benchmark.auxiliarymethods.datasets as dp
    from tudataset.tud_benchmark.auxiliarymethods.gnn_evaluation import gnn_evaluation
    from tudataset.tud_benchmark.gnn_baselines.gnn_architectures import GIN, GINE
    import os



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
        [1, 2, 3, 4, 5],
        [32, 64, 128],
        max_num_epochs=200,
        batch_size=64,
        start_lr=0.01,
        num_repetitions=num_reps,
        all_std=True,
        cv_seed=cv_seed
    )


    return res
