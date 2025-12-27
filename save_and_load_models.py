import os
import pickle

def save_pathboost_model(model, dataset_name, verbose =False):
    """
    Save a PathBoost model to the path_boost_models directory with a name based on the dataset.
    """
    dir_path = "path_boost_models"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, f"{dataset_name}_path_boost_model.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    if verbose:
        print(f"Model saved to {file_path}")

def load_pathboost_model(dataset_name):
    """
    Load a PathBoost model from the path_boost_models directory based on the dataset name.
    """
    file_path = os.path.join("path_boost_models", f"{dataset_name}_path_boost_model.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No model found for dataset {dataset_name} at {file_path}")
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model

