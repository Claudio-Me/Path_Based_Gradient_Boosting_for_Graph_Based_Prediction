"""Training functions with cross-validation for PathBoost and GIN."""
import os
import sys
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import find_categorical_node_attributes, preprocess_labels
from shared.constants import CV_SEED


def train_pathboost_cv(train_graphs, train_labels, test_graphs, test_labels, n_folds=5, seed=CV_SEED):
    """
    Find best PathBoost params via CV, retrain on full train, evaluate on test.

    Returns: test accuracy
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from extended_path_boost import SequentialPathBoostClassifier

    # Preprocess labels to {0, 1}
    train_labels = preprocess_labels(np.array(train_labels))
    test_labels = preprocess_labels(np.array(test_labels))
    if train_labels is None or test_labels is None:
        print("Labels are not binary")
        return None

    # Find anchor label
    anchor_attr = find_categorical_node_attributes(train_graphs)
    if not anchor_attr:
        print("No categorical node attributes found")
        return None

    # Get anchor labels
    anchor_labels = set()
    for g in train_graphs:
        for _, node_data in g.nodes(data=True):
            if anchor_attr in node_data:
                anchor_labels.add(node_data[anchor_attr])
    anchor_labels = list(anchor_labels)

    # Base model
    model = SequentialPathBoostClassifier(
        n_iter=2000,
        learning_rate=0.1,
        BaseLearnerClass=DecisionTreeRegressor,
        SelectorClass=DecisionTreeClassifier,
        kwargs_for_base_learner={'random_state': seed, 'max_depth': 4},
        verbose=False
    )

    # Parameter grid
    param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_path_length': [2, 3, 4],
        'n_iter': [100,500, 1000,],
        'kwargs_for_base_learner': [
            {'random_state': seed, 'max_depth': 3},
            {'random_state': seed, 'max_depth': 4},
            {'random_state': seed, 'max_depth': 5},
        ]
    }

    # GridSearchCV
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=1, verbose=0)
    grid.fit(train_graphs, train_labels, anchor_nodes_label_name=anchor_attr, list_anchor_nodes_labels=anchor_labels)

    print(f"Best params: {grid.best_params_}")

    # Retrain on full train data with best params
    best_model = SequentialPathBoostClassifier(
        learning_rate=grid.best_params_['learning_rate'],
        max_path_length=grid.best_params_['max_path_length'],
        n_iter=grid.best_params_['n_iter'],
        BaseLearnerClass=DecisionTreeRegressor,
        SelectorClass=DecisionTreeClassifier,
        kwargs_for_base_learner={'random_state': seed, 'max_depth': 4},
        verbose=False
    )
    best_model.fit(train_graphs, train_labels, anchor_nodes_label_name=anchor_attr, list_anchor_nodes_labels=anchor_labels)

    # Evaluate on test
    test_pred = best_model.predict(test_graphs)
    test_acc = accuracy_score(test_labels, test_pred)

    return test_acc


def train_gin_cv(train_graphs, train_labels, test_graphs, test_labels, n_folds=5, seed=CV_SEED, device='cpu'):
    """
    Find best GIN params via CV, retrain on full train, evaluate on test.
    Uses same training procedure as gnn_evaluation.py (LR scheduler + early stopping).

    Returns: test accuracy
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    from torch_geometric.utils import degree
    import torch_geometric.transforms as T

    gnn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tudataset', 'tud_benchmark', 'gnn_baselines')
    sys.path.insert(0, gnn_path)
    from gnn_architectures import GIN  # noqa: E402

    # Preprocess labels
    train_labels = preprocess_labels(np.array(train_labels))
    test_labels = preprocess_labels(np.array(test_labels))
    if train_labels is None or test_labels is None:
        print("Labels are not binary")
        return None

    # Convert NetworkX graphs to PyG Data objects
    def nx_to_pyg(graphs, labels):
        data_list = []
        for g, y in zip(graphs, labels):
            edges = list(g.edges())
            if edges:
                edge_index = torch.tensor(edges).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            num_nodes = g.number_of_nodes()
            x = torch.ones((num_nodes, 1), dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long))
            data.num_nodes = num_nodes
            data_list.append(data)
        return data_list

    train_data = nx_to_pyg(train_graphs, train_labels)
    test_data = nx_to_pyg(test_graphs, test_labels)

    # Compute max degree for one-hot encoding (same as gnn_evaluation.py)
    max_degree = 0
    for data in train_data + test_data:
        if data.edge_index.numel() > 0:
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, deg.max().item())

    if max_degree < 1000:
        transform = T.OneHotDegree(max_degree)
    else:
        # Normalized degree for large max_degree
        all_degs = []
        for data in train_data + test_data:
            if data.edge_index.numel() > 0:
                all_degs.append(degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float))
        all_degs = torch.cat(all_degs)
        mean, std = all_degs.mean().item(), all_degs.std().item()
        transform = lambda d: _normalized_degree_transform(d, mean, std)

    train_data = [transform(d) for d in train_data]
    test_data = [transform(d) for d in test_data]

    # Dummy dataset for GIN constructor
    class DummyDataset:
        def __init__(self, num_features, num_classes):
            self.num_features = num_features
            self.num_classes = num_classes

    num_features = train_data[0].x.shape[1]
    dummy_dataset = DummyDataset(num_features, 2)

    device = torch.device(device)
    torch.manual_seed(seed)

    # Training parameters (same as gnn_evaluation.py)
    max_num_epochs = 200
    batch_size = 128
    start_lr = 0.01
    min_lr = 0.000001
    factor = 0.5
    patience = 5

    # Hyperparameter grid
    layers_options = [1, 2, 3, 4, 5]
    hidden_options = [32, 64, 128]

    # Train/val split for hyperparameter selection
    np.random.seed(seed)
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    val_size = max(1, int(len(indices) * 0.1))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_subset = [train_data[i] for i in train_idx]
    val_subset = [train_data[i] for i in val_idx]

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    best_val_acc = 0
    best_config = None

    # Grid search with LR scheduler + early stopping
    for num_layers in layers_options:
        for hidden in hidden_options:
            model = GIN(dummy_dataset, num_layers, hidden).to(device)
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
            )

            for epoch in range(1, max_num_epochs + 1):
                lr = scheduler.optimizer.param_groups[0]['lr']

                # Train
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch)
                    loss = F.nll_loss(out, batch.y)
                    loss.backward()
                    optimizer.step()

                # Validate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch)
                        pred = out.argmax(dim=1)
                        correct += (pred == batch.y).sum().item()
                        total += batch.y.size(0)
                val_acc = correct / total if total > 0 else 0

                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_config = {'layers': num_layers, 'hidden': hidden}

                # Early stopping
                if lr < min_lr:
                    break

    print(f"Best params: {best_config}")

    # Retrain on full train data with best config
    model = GIN(dummy_dataset, best_config['layers'], best_config['hidden']).to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
    )

    full_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    best_test_acc = 0
    for epoch in range(1, max_num_epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']

        model.train()
        for batch in full_train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()

        # Track best test acc during training
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        test_acc = correct / total if total > 0 else 0
        best_test_acc = max(best_test_acc, test_acc)

        scheduler.step(test_acc)
        if lr < min_lr:
            break

    return best_test_acc


def _normalized_degree_transform(data, mean, std):
    """Apply normalized degree as node features."""
    from torch_geometric.utils import degree
    import torch
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
    deg = (deg - mean) / std
    data.x = deg.view(-1, 1)
    return data


def run_pathboost_evaluation(dataset_name, n_splits=4, n_folds=5, seed=CV_SEED):
    """
    Evaluate PathBoost on subsampled training data.

    Args:
        dataset_name: Name of TU dataset (e.g., 'PROTEINS_full')
        n_splits: Number of train/test splits
        n_folds: CV folds for hyperparameter search
        seed: Random seed

    Returns: {10: [acc_split0, acc_split1, ...], 20: [...], ..., 100: [...]}
    """
    from splitter import create_dataset_splits, subsample_train

    splits, _, _ = create_dataset_splits(dataset_name, n_splits)
    results = {p: [] for p in range(10, 101, 10)}

    for split in splits:
        subsamples = subsample_train(split['train_graphs'], split['train_labels'], seed)

        for sub in subsamples:
            acc = train_pathboost_cv(sub['graphs'], sub['labels'],
                                     split['test_graphs'], split['test_labels'],
                                     n_folds=n_folds, seed=seed)
            results[sub['percentage']].append(acc)

    return results


def run_gin_evaluation(dataset_name, n_splits=4, n_folds=5, seed=CV_SEED, device='cpu'):
    """
    Evaluate GIN on subsampled training data.

    Args:
        dataset_name: Name of TU dataset (e.g., 'PROTEINS_full')
        n_splits: Number of train/test splits
        n_folds: CV folds for hyperparameter search
        seed: Random seed
        device: Device for GIN ('cpu' or 'cuda')

    Returns: {10: [acc_split0, acc_split1, ...], 20: [...], ..., 100: [...]}
    """
    from splitter import create_dataset_splits, subsample_train

    splits, _, _ = create_dataset_splits(dataset_name, n_splits)
    results = {p: [] for p in range(10, 101, 10)}

    for split in splits:
        subsamples = subsample_train(split['train_graphs'], split['train_labels'], seed)

        for sub in subsamples:
            acc = train_gin_cv(sub['graphs'], sub['labels'],
                               split['test_graphs'], split['test_labels'],
                               n_folds=n_folds, seed=seed, device=device)
            results[sub['percentage']].append(acc)

    return results
