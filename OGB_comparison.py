# main.py
# Deterministic end-to-end: load OGB (PyG), build deterministic DataLoaders,
# then create or load the cached NetworkX dataset.

from __future__ import annotations

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import average_precision_score
from extended_path_boost import SequentialPathBoostClassifier

from run_pathboost_all_datasets import find_categorical_node_attributes
from ogb_dataset.OGB_utils.to_networkx import (
    set_seed_everywhere,
    make_worker_init_fn,
    make_dataloader_generator,
    get_or_create_nx_dataset_from_loaders,
)

SEED = 44
DATASET_NAME = "ogbg-molhiv" # ogbg-molhiv  ogbg-molpcba
BASE_DIR = "./ogb_dataset"   # change to "./ogb_dataset" if you prefer


def main():
    # 1) Determinism/seeds — do this *before* constructing DataLoaders
    set_seed_everywhere(SEED, use_cuda_determinism=True)
    worker_init_fn = make_worker_init_fn(SEED)
    gen = make_dataloader_generator(SEED)

    # 2) Standard OGB PyG dataset + deterministic DataLoaders
    dataset = PygGraphPropPredDataset(name=DATASET_NAME, root="ogb_dataset/")
    split_idx = dataset.get_idx_split()

    # NOTE:
    #  - shuffle=True on train is now deterministic thanks to the fixed `generator=gen`.
    #  - if you keep num_workers=0, worker_init_fn is ignored (fine & deterministic).
    #  - set num_workers>0 for speed; determinism is preserved by worker_init_fn + generator.
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=32,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=gen,
        persistent_workers=False,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=32,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=gen,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=32,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=gen,
        persistent_workers=False,
    )

    # 3) Create or load the NetworkX dataset
    nx_data = get_or_create_nx_dataset_from_loaders(
        loaders={"train": train_loader, "valid": valid_loader, "test": test_loader},
        dataset_name=DATASET_NAME,
        base_dir=BASE_DIR,
        overwrite=False,
    )

    # Sanity check
    print(
        f"NetworkX graphs -> train={len(nx_data['train'])}, "
        f"valid={len(nx_data['valid'])}, test={len(nx_data['test'])}"
    )
    G0 = nx_data["train"][0]
    print("G0:", G0.number_of_nodes(), "nodes,", G0.number_of_edges(), "edges,", "y=", G0.graph.get("y"))

    # 4) Run PathBoost
    print("\nRunning PathBoost...")

    # Extract graphs and labels
    X_train = nx_data["train"]
    y_train = np.array([g.graph.get("y", 0) for g in X_train]).flatten()

    X_test = nx_data["test"]
    y_test_true = np.array([g.graph.get("y", 0) for g in X_test]).flatten()

    # Find the best categorical attribute from node features
    categorical_attr = find_categorical_node_attributes(X_train)
    if not categorical_attr:
        print("Could not find a suitable categorical attribute for PathBoost. Exiting.")
        return

    print(f"Using '{categorical_attr}' as anchor attribute for PathBoost.")

    # Get all possible anchor labels from the training set
    anchor_labels = []
    for graph in X_train:
        for _, node_data in graph.nodes(data=True):
            if categorical_attr in node_data:
                anchor_labels.append(node_data[categorical_attr])
    anchor_labels = sorted(list(set(anchor_labels)))

    # Configure and train PathBoost
    kwargs_for_base_learner = {
        'random_state': SEED,
        'splitter': 'best',
        'criterion': "squared_error",
        'max_depth': 2,
    }
    model = SequentialPathBoostClassifier(
        n_iter=100,
        learning_rate=0.1,
        BaseLearnerClass=DecisionTreeRegressor,
        SelectorClass=DecisionTreeClassifier,
        kwargs_for_base_learner=kwargs_for_base_learner,
        kwargs_for_selector={'random_state': SEED},
        verbose=True,
        use_tree_boost = False
    )

    model.fit(
        X_train, y_train,
        anchor_nodes_label_name=categorical_attr,
        list_anchor_nodes_labels=anchor_labels
    )

    # 5) Evaluate with OGB Evaluator
    y_pred = model.predict(X = X_test, class_probability= True)  # Get probabilities for the positive class

    evaluator = Evaluator(name=DATASET_NAME)
    input_dict = {"y_true": y_test_true.reshape(-1, 1), "y_pred": y_pred.reshape(-1, 1)}
    result_dict = evaluator.eval(input_dict)

    # Calculate average precision score
    avg_precision = average_precision_score(y_test_true, y_pred)

    print(f"\nPathBoost Evaluation on {DATASET_NAME}:")
    for metric, value in result_dict.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  average_precision: {avg_precision:.4f}")


if __name__ == "__main__":
    main()




#Evaluation
"""
evaluator = Evaluator(name = "ogbg-molhiv")
# You can learn the input and output format specification of the evaluator as follows.
# print(evaluator.expected_input_format) 
# print(evaluator.expected_output_format) 
input_dict = {"y_true": y_true, "y_pred": y_pred}
result_dict = evaluator.eval(input_dict) # E.g., {"rocauc": 0.7321}
"""