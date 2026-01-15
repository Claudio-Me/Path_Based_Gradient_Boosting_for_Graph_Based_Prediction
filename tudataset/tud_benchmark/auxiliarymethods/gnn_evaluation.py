import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import random
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def _collect_preds_and_probs(loader, model, device):
    # Collect true labels, predicted labels, probabilities.
    model.eval()
    ys, yhat, yprob = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)          # log-probabilities
            probs = out.exp().cpu().numpy()
            preds = probs.argmax(axis=1)
            ys.extend(data.y.cpu().numpy().tolist())
            yhat.extend(preds.tolist())
            yprob.append(probs)
    y_true = np.asarray(ys)
    y_pred = np.asarray(yhat)
    y_proba = np.vstack(yprob) if len(yprob) else None
    return y_true, y_pred, y_proba


def _compute_metrics(y_true, y_pred, y_proba):
    # Compute all requested metrics; return None if not computable.
    res = {}
    try:
        res['accuracy'] = float(accuracy_score(y_true, y_pred))
    except Exception:
        res['accuracy'] = None
    try:
        res['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        res['balanced_accuracy'] = None

    classes = np.unique(y_true)
    binary = len(classes) == 2
    avg_main = 'binary' if binary else 'weighted'

    try:
        res['f1'] = float(f1_score(y_true, y_pred, average=avg_main))
    except Exception:
        res['f1'] = None
    try:
        res['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
    except Exception:
        res['f1_macro'] = None
    try:
        res['recall'] = float(recall_score(y_true, y_pred, average=avg_main))
    except Exception:
        res['recall'] = None
    try:
        if y_proba is not None:
            if binary:
                pos_col = 1 if y_proba.shape[1] > 1 else 0
                res['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, pos_col]))
            else:
                res['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
        else:
            res['roc_auc'] = None
    except Exception:
        res['roc_auc'] = None
    return res


# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, ds_name, layers, hidden, max_num_epochs=200, batch_size=128, start_lr=0.01, min_lr = 0.000001, factor=0.5, patience=5,
                       num_repetitions=10, all_std=True, cv_seed=None):
    # Seed everything for reproducible CV and shuffles
    if cv_seed is not None:
        random.seed(cv_seed)
        np.random.seed(cv_seed)
        torch.manual_seed(cv_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cv_seed)

    # Load dataset and shuffle.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', ds_name)
    dataset = TUDataset(path, name=ds_name).shuffle()

    # One-hot degree if node labels are not available.
    # The following if clause is taken from  https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/datasets.py.
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare metric aggregation containers
    all_metrics_results = {
        'accuracy': {'best_scores': [], 'fold_scores': []},
        'balanced_accuracy': {'best_scores': [], 'fold_scores': []},
        'f1': {'best_scores': [], 'fold_scores': []},
        'f1_macro': {'best_scores': [], 'fold_scores': []},
        'recall': {'best_scores': [], 'fold_scores': []},
        'roc_auc': {'best_scores': [], 'fold_scores': []},
    }

    # Timing tracking for training time per hyperparameter config
    config_times = []

    for i in range(num_repetitions):
        seed_i = (cv_seed + i) if cv_seed is not None else None
        kf = KFold(n_splits=10, shuffle=True, random_state=seed_i)
        if seed_i is not None:
            torch.manual_seed(seed_i)
        dataset.shuffle()
        # Collect per-fold metric values for this repetition
        rep_fold_metrics = {m: [] for m in all_metrics_results.keys()}

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=seed_i)
            best_val_acc = 0.0
            best_fold_metrics = None

            # Split data.
            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            # Prepare batching.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # Collect val. and test acc. over all hyperparameter combinations.
            for l in layers:
                for h in hidden:
                    config_start = time.time()

                    # Setup model.
                    model = gnn(dataset, l, h).to(device)
                    model.reset_parameters()

                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=factor, patience=patience,
                                                                           min_lr=0.0000001)
                    for epoch in range(1, max_num_epochs + 1):
                        lr = scheduler.optimizer.param_groups[0]['lr']
                        train(train_loader, model, optimizer, device)
                        val_acc = test(val_loader, model, device)
                        scheduler.step(val_acc)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            y_true, y_pred, y_proba = _collect_preds_and_probs(test_loader, model, device)
                            best_fold_metrics = _compute_metrics(y_true, y_pred, y_proba)

                        # Break if learning rate is smaller 10**-6.
                        if lr < min_lr:
                            break

                    config_times.append(time.time() - config_start)

            # Store metrics for this fold
            if best_fold_metrics:
                for m in rep_fold_metrics.keys():
                    v = best_fold_metrics.get(m)
                    if v is not None:
                        rep_fold_metrics[m].append(float(v))

        # After folds: aggregate repetition means & extend fold scores
        for m, fold_vals in rep_fold_metrics.items():
            if fold_vals:
                all_metrics_results[m]['best_scores'].append(float(np.mean(fold_vals)))
                all_metrics_results[m]['fold_scores'].extend(fold_vals)

    # Build final dict: metric -> (mean, std_10, std_100)
    results_dict = {}
    for m, buckets in all_metrics_results.items():
        best_scores = np.array(buckets['best_scores'], dtype=float)
        fold_scores = np.array(buckets['fold_scores'], dtype=float)
        if best_scores.size == 0 or fold_scores.size == 0:
            results_dict[m] = (-1.0, -1.0, -1.0)
            continue
        mean_score = float(best_scores.mean())
        std_10 = float(best_scores.std(ddof=1)) if best_scores.size > 1 else 0.0
        std_100 = float(fold_scores.std(ddof=1)) if fold_scores.size > 1 else 0.0
        results_dict[m] = (mean_score, std_10, std_100)

    # Calculate timing statistics (time per config)
    if config_times:
        results_dict['_timing'] = (np.mean(config_times), np.std(config_times, ddof=1) if len(config_times) > 1 else 0.0)
    else:
        results_dict['_timing'] = (None, None)

    return results_dict
