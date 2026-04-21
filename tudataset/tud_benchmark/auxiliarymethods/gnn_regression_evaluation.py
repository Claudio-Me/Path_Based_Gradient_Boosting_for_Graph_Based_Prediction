import os.path as osp
import time
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def _train(train_loader, model, optimizer, device):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y.float().view(-1))
        loss.backward()
        optimizer.step()


def _collect(loader, model, device):
    model.eval()
    ys = []
    yhat = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).detach().cpu().numpy().reshape(-1)
            ys.extend(data.y.detach().cpu().numpy().reshape(-1).tolist())
            yhat.extend(out.tolist())
    return np.asarray(ys), np.asarray(yhat)


def _metrics(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def gnn_regression_evaluation(
    gnn,
    ds_name,
    layers,
    hidden,
    targets,
    max_num_epochs=200,
    batch_size=64,
    start_lr=0.01,
    min_lr=1e-6,
    factor=0.5,
    patience=5,
    num_repetitions=10,
    n_folds=10,
    cv_seed=None,
    sample_indices=None,
):
    if cv_seed is not None:
        random.seed(cv_seed)
        np.random.seed(cv_seed)
        torch.manual_seed(cv_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cv_seed)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', ds_name)
    dataset = TUDataset(path, name=ds_name)

    targets = np.asarray(targets, dtype=float).reshape(-1)
    if len(dataset) != len(targets):
        raise ValueError(f"Target length mismatch for {ds_name}: {len(targets)} vs {len(dataset)} graphs")

    if sample_indices is None:
        dataset_indices = np.arange(len(dataset), dtype=int)
        selected_targets = targets
    else:
        dataset_indices = np.asarray(sample_indices, dtype=int)
        selected_targets = targets[dataset_indices]

    data_list = []
    for pos, dataset_idx in enumerate(dataset_indices.tolist()):
        data = dataset[int(dataset_idx)]
        data.y = torch.tensor([selected_targets[pos]], dtype=torch.float)
        data_list.append(data)

    if data_list[0].x is None:
        max_degree = 0
        degs = []
        for data in data_list:
            deg = degree(data.edge_index[0], dtype=torch.long)
            degs.append(deg)
            max_degree = max(max_degree, int(deg.max().item()))

        if max_degree < 1000:
            transform = T.OneHotDegree(max_degree)
            data_list = [transform(data) for data in data_list]
        else:
            deg_all = torch.cat(degs, dim=0).to(torch.float)
            mean = deg_all.mean().item()
            std = deg_all.std().item()
            if std == 0.0:
                std = 1.0
            transform = NormalizedDegree(mean, std)
            data_list = [transform(data) for data in data_list]

    edge_features = 0
    edge_attr0 = getattr(data_list[0], 'edge_attr', None)
    if edge_attr0 is not None:
        edge_features = 1 if edge_attr0.dim() == 1 else edge_attr0.size(1)

    dataset_shim = SimpleNamespace(
        num_features=data_list[0].x.size(1),
        num_edge_features=edge_features,
        num_classes=1,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_metrics_results = {
        'mae': {'best_scores': [], 'fold_scores': []},
        'mse': {'best_scores': [], 'fold_scores': []},
        'r2': {'best_scores': [], 'fold_scores': []},
    }
    config_times = []

    n_graphs = len(data_list)

    for rep in range(num_repetitions):
        seed_i = (cv_seed + rep) if cv_seed is not None else None
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_i)
        if seed_i is not None:
            torch.manual_seed(seed_i)
        rep_fold_metrics = {m: [] for m in all_metrics_results.keys()}

        for train_idx, test_idx in kf.split(range(n_graphs)):
            train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed_i)
            train_dataset = [data_list[i] for i in train_idx.tolist()]
            val_dataset = [data_list[i] for i in val_idx.tolist()]
            test_dataset = [data_list[i] for i in test_idx.tolist()]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            best_val_mae = float('inf')
            best_fold_metrics = None

            for l in layers:
                for h in hidden:
                    config_start = time.time()

                    model = gnn(dataset_shim, l, h).to(device)
                    model.reset_parameters()

                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=factor,
                        patience=patience,
                        min_lr=min_lr,
                    )

                    for _ in range(1, max_num_epochs + 1):
                        lr = scheduler.optimizer.param_groups[0]['lr']
                        _train(train_loader, model, optimizer, device)
                        y_val, pred_val = _collect(val_loader, model, device)
                        val_mae = float(np.mean(np.abs(y_val - pred_val)))
                        scheduler.step(val_mae)

                        if val_mae < best_val_mae:
                            best_val_mae = val_mae
                            y_test, pred_test = _collect(test_loader, model, device)
                            best_fold_metrics = _metrics(y_test, pred_test)

                        if lr < min_lr:
                            break

                    config_times.append(time.time() - config_start)

            if best_fold_metrics:
                for metric in rep_fold_metrics.keys():
                    rep_fold_metrics[metric].append(float(best_fold_metrics[metric]))

        for metric, fold_values in rep_fold_metrics.items():
            if fold_values:
                all_metrics_results[metric]['best_scores'].append(float(np.mean(fold_values)))
                all_metrics_results[metric]['fold_scores'].extend(fold_values)

    results_dict = {}
    for metric, buckets in all_metrics_results.items():
        best_scores = np.asarray(buckets['best_scores'], dtype=float)
        fold_scores = np.asarray(buckets['fold_scores'], dtype=float)
        if best_scores.size == 0:
            results_dict[metric] = (-1.0, -1.0, -1.0)
            continue

        results_dict[metric] = (
            float(best_scores.mean()),
            float(best_scores.std(ddof=1)) if best_scores.size > 1 else 0.0,
            float(fold_scores.std(ddof=1)) if fold_scores.size > 1 else 0.0,
        )

    if config_times:
        results_dict['_timing'] = (
            float(np.mean(config_times)),
            float(np.std(config_times, ddof=1)) if len(config_times) > 1 else 0.0,
        )
    else:
        results_dict['_timing'] = (None, None)

    return results_dict

