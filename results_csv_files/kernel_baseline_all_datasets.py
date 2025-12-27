import sys
import os

# Add the parent directory of tudataset to sys.path
sys.path.append(os.path.abspath("/mn/sarpanitu/ansatte-u6/claudm/PycharmProjects/Different_datasets/Different_datasets"))



import tudataset.tud_benchmark.auxiliarymethods.datasets as dp
import tudataset.tud_benchmark.auxiliarymethods.auxiliary_methods as aux
from tudataset.tud_benchmark.auxiliarymethods.kernel_evaluation import kernel_svm_evaluation, linear_svm_evaluation
import tudataset.tud_benchmark.kernel_baselines as kb
from multiprocessing import Process, Queue
import uuid
from datetime import datetime

import csv
import signal


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException

def _run_in_process(q, func, *args, **kwargs):
    try:
        res = func(*args, **kwargs)
        q.put(("OK", res))
    except Exception as e:
        # send a simple string to avoid pickling complex exceptions
        q.put(("ERR", f"{type(e).__name__}: {e}"))

def run_with_timeout(func, timeout_sec, *args, **kwargs):
    """
    Run `func(*args, **kwargs)` in a separate process. Return (result, False)
    on success, or ((-1,-1,-1), True) on timeout or error.
    """
    q = Queue()
    p = Process(target=_run_in_process, args=(q, func) + args, kwargs=kwargs)
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join()
        return (-1, -1, -1), True

    # process finished
    if q.empty():
        # no result posted -> treat as error
        return (-1, -1, -1), True

    status, payload = q.get()
    if status == "OK":
        return payload, False
    else:
        # payload is an error string
        return (-1, -1, -1), True

"""
def run_with_timeout(func, timeout_sec, *args, **kwargs):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout_sec))
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result, False
    except TimeoutException:
        signal.alarm(0)
        return (-1, -1, -1), True
    except Exception:
        # to clear the time allarm in case of other exceptions
        signal.alarm(0)
        return (-1,-1,-1), True
"""

def _wl_dense_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    # time check: compute gram matrices and feature vectors for 5 iterations
    wl_gram_matrix_collection = []
    for iterations in range(1, 6):
        print(f"Computing WL iteration {iterations}")
        wl_gram_matrix = kb.compute_wl_1_dense(dataset_name, int(iterations), use_labels, has_edge_labels)
        wl_gram_matrix = aux.normalize_gram_matrix(wl_gram_matrix)
        wl_gram_matrix_collection.append(wl_gram_matrix)

    wl_accuracy, wl_std_10, wl_std_100 = kernel_svm_evaluation(
        wl_gram_matrix_collection, dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )
    return wl_accuracy, wl_std_10, wl_std_100
    # end time check


def _wl_sparse_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    # Time check: compute feature vectors for 5 iterations
    wl_feature_vector_collection = []
    for iterations in range(1, 6):
        print(f"Computing WL iteration {iterations}")
        wl_feature_vectors = kb.compute_wl_1_sparse(dataset_name, iterations, use_labels, has_edge_labels)
        wl_feature_vectors = aux.normalize_feature_vector(wl_feature_vectors)
        wl_feature_vector_collection.append(wl_feature_vectors)

    wl_linear_accuracy, wl_linear_std_10, wl_linear_std_100 = linear_svm_evaluation(
        wl_feature_vector_collection, dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )
    return wl_linear_accuracy, wl_linear_std_10, wl_linear_std_100
    # end time check


def _graphlet_dense_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    # time check: compute gram matrix
    gr_gram_matrix = kb.compute_graphlet_dense(dataset_name, use_labels, has_edge_labels)
    gr_accuracy, gr_std_10, gr_std_100 = kernel_svm_evaluation(
        [gr_gram_matrix], dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )
    return gr_accuracy, gr_std_10, gr_std_100
    # end time check


def _graphlet_sparse_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    # Time check: compute feature vectors
    gr_feature_vectors = kb.compute_graphlet_sparse(dataset_name, use_labels, has_edge_labels)
    gr_linear_accuracy, gr_linear_std_10, gr_linear_std_100 = linear_svm_evaluation(
        [gr_feature_vectors], dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )
    return gr_linear_accuracy, gr_linear_std_10, gr_linear_std_100
    # end time check


def _sp_dense_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    # time check: compute gram matrix and feature vectors
    sp_gram_matrix = kb.compute_graphlet_dense(dataset_name, use_labels, has_edge_labels)
    sp_accuracy, sp_std_10, sp_std_100 = kernel_svm_evaluation(
        [sp_gram_matrix], dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )
    return sp_accuracy, sp_std_10, sp_std_100
    # end time check


def _sp_sparse_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    # Time check: compute feature vectors
    sp_feature_vectors = kb.compute_graphlet_sparse(dataset_name, use_labels, has_edge_labels)
    sp_linear_accuracy, sp_linear_std_10, sp_linear_std_100 = linear_svm_evaluation(
        [sp_feature_vectors], dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )
    return sp_linear_accuracy, sp_linear_std_10, sp_linear_std_100
    # end time check


def all_kernels_baseline(dataset_name, num_reps=10, use_labels=True, snippet_timeout_sec=36000):
    dp.get_dataset(dataset_name)
    # Check for edge labels or continuous node attributes
    dataset_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tudataset", "tud_benchmark", "datasets", dataset_name, dataset_name, "raw"
    )
    has_edge_labels = os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_edge_labels.txt"))
    has_edge_labels = has_edge_labels or os.path.exists(
        os.path.join(dataset_dir, f"{dataset_name}_edge_attributes.txt"))

    results = dict()

    # WL dense
    wl_dense_res, timed_out = run_with_timeout(
        _wl_dense_eval, snippet_timeout_sec, dataset_name, num_reps, use_labels, has_edge_labels
    )
    if timed_out or wl_dense_res is None:
        print(f"Timeout in WL dense for {dataset_name} (>{snippet_timeout_sec/3600:.2f}h).")
        wl_dense_res = (-1, -1, -1)
    results["Weisfeiler-Lehman subtree kernel"] = wl_dense_res

    # WL sparse linear
    wl_sparse_res, timed_out = run_with_timeout(
        _wl_sparse_eval, snippet_timeout_sec, dataset_name, num_reps, use_labels, has_edge_labels
    )
    if timed_out or wl_sparse_res is None:
        print(f"Timeout in WL sparse for {dataset_name} (>{snippet_timeout_sec/3600:.2f}h).")
        wl_sparse_res = (-1, -1, -1)
    results["Weisfeiler-Lehman subtree kernel linear"] = wl_sparse_res

    print(f"WL kernel done for {dataset_name}")

    # Graphlet dense
    gr_dense_res, timed_out = run_with_timeout(
        _graphlet_dense_eval, snippet_timeout_sec, dataset_name, num_reps, use_labels, has_edge_labels
    )
    if timed_out or gr_dense_res is None:
        print(f"Timeout in Graphlet dense for {dataset_name} (>{snippet_timeout_sec/3600:.2f}h).")
        gr_dense_res = (-1, -1, -1)
    results["Graphlet kernel"] = gr_dense_res


    # Graphlet sparse linear
    gr_sparse_res, timed_out = run_with_timeout(
        _graphlet_sparse_eval, snippet_timeout_sec, dataset_name, num_reps, use_labels, has_edge_labels
    )
    if timed_out or gr_sparse_res is None:
        print(f"Timeout in Graphlet sparse for {dataset_name} (>{snippet_timeout_sec/3600:.2f}h).")
        gr_sparse_res = (-1, -1, -1)
    results["Graphlet kernel linear"] = gr_sparse_res
    print(f"Graphlet kernel done for {dataset_name}")

    # Shortest-path dense
    sp_dense_res, timed_out = run_with_timeout(
        _sp_dense_eval, snippet_timeout_sec, dataset_name, num_reps, use_labels, has_edge_labels
    )
    if timed_out or sp_dense_res is None:
        print(f"Timeout in Shortest-path dense for {dataset_name} (>{snippet_timeout_sec/3600:.2f}h).")
        sp_dense_res = (-1, -1, -1)
    results["Shortest-path kernel"] = sp_dense_res

    # Shortest-path sparse linear
    sp_sparse_res, timed_out = run_with_timeout(
        _sp_sparse_eval, snippet_timeout_sec, dataset_name, num_reps, use_labels, has_edge_labels
    )
    if timed_out or sp_sparse_res is None:
        print(f"Timeout in Shortest-path sparse for {dataset_name} (>{snippet_timeout_sec/3600:.2f}h).")
        sp_sparse_res = (-1, -1, -1)
    results["Shortest-path kernel linear"] = sp_sparse_res
    print(f"Shortest-path kernel done for {dataset_name}")

    # Weisfeiler-Lehman optimal assignment kernel
    """
    # we preserve this block without explicit time markers, as no #time check comments exist here in the original code
    wlpa_gram_matrix_collection = []
    for i in range(1, 6):
        print(f"Computing WLOA iteration {i}")
        wlpa_gram_matrix = kb.compute_wloa_dense(dataset_name, use_labels, has_edge_labels)
        wlpa_gram_matrix = aux.normalize_gram_matrix(wlpa_gram_matrix)
        wlpa_gram_matrix_collection.append(wlpa_gram_matrix)

    wlpa_accuracy, wlpa_std_10, wlpa_std_100 = kernel_svm_evaluation(
        wlpa_gram_matrix_collection, dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )
    """
    # Weisfeiler-Lehman optimal assignment kernel
    wlpa_res, timed_out = run_with_timeout(
        _wlpa_dense_eval, snippet_timeout_sec, dataset_name, num_reps, use_labels, has_edge_labels
    )
    if timed_out or wlpa_res is None:
        print(f"Timeout in WLOA for {dataset_name} (>{snippet_timeout_sec / 3600:.2f}h).")
        wlpa_res = (-1, -1, -1)
    results["Weisfeiler-Lehman optimal assignment kernel"] = wlpa_res
    print(f"WLOA kernel done for {dataset_name}")

    return results

def _wlpa_dense_eval(dataset_name, num_reps, use_labels, has_edge_labels):
    # Time check: compute gram matrices for 5 iterations
    wlpa_matrices = []
    for iteration in range(1, 6):
        print(f"Computing WLOA iteration {iteration}")
        gram = kb.compute_wloa_dense(dataset_name, use_labels, has_edge_labels)
        gram = aux.normalize_gram_matrix(gram)
        wlpa_matrices.append(gram)

    wlpa_acc, wlpa_std_10, wlpa_std_100 = kernel_svm_evaluation(
        wlpa_matrices, dp.get_dataset(dataset_name), num_repetitions=num_reps, all_std=True
    )
    return wlpa_acc, wlpa_std_10, wlpa_std_100

def save_results_to_csv(csv_path, dataset_name, results):
    file_exists = os.path.exists(csv_path)
    # Flatten results for CSV
    row = {"Dataset": dataset_name}
    for k, v in results.items():
        # v is a tuple (accuracy, std_10, std_100)
        row[f"{k}_accuracy"] = v[0]
        row[f"{k}_std_10"] = v[1]
        row[f"{k}_std_100"] = v[2]
    # Write header if file does not exist
    with open(csv_path, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    datasets =   [
    #"MUTAG", "AIDS", "BZR", "COX2_MD", "DHFR", "ER_MD",
    #"MCF-7", "MCF-7H", "MOLT-4", "MOLT-4H", "Mutagenicity", "MUTAG", "NCI1", "NCI109",
    #"NCI-H23", "NCI-H23H", "OVCAR-8", "OVCAR-8H", "P388", "P388H", "PC-3", "PC-3H",
    #"PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "SF-295", "SF-295H", "SN12C", "SN12CH",
    #"SW-620", "SW-620H",
    #"Tox21_AhR_training", "Tox21_AhR_testing", "Tox21_AhR_evaluation",
    #"Tox21_AR_training", "Tox21_AR_testing", "Tox21_AR_evaluation",
    #"Tox21_AR-LBD_training", "Tox21_AR-LBD_testing", "Tox21_AR-LBD_evaluation",
    #"Tox21_ARE_training", "Tox21_ARE_testing", "Tox21_ARE_evaluation",
    #"Tox21_aromatase_training", "Tox21_aromatase_testing", "Tox21_aromatase_evaluation",
    #"Tox21_ATAD5_training", "Tox21_ATAD5_testing", "Tox21_ATAD5_evaluation",
    #"Tox21_ER_training", "Tox21_ER_testing", "Tox21_ER_evaluation",
    #"Tox21_ER-LBD_training", "Tox21_ER-LBD_testing", "Tox21_ER-LBD_evaluation",
    #"Tox21_HSE_training", "Tox21_HSE_testing", "Tox21_HSE_evaluation",
    #"Tox21_MMP_training", "Tox21_MMP_testing", "Tox21_MMP_evaluation",
    #"Tox21_p53_training", "Tox21_p53_testing", "Tox21_p53_evaluation",
    #"Tox21_PPAR-gamma_training", "Tox21_PPAR-gamma_testing", "Tox21_PPAR-gamma_evaluation",
    #"UACC257", "UACC257H", "Yeast", "YeastH", "DD", "KKI", "OHSU", "Peking_1",
    #"PROTEINS", "PROTEINS_full", "DBLP_v1", "TWITTER-Real-Graph-Partial", "SYNTHETIC", "DHFR_MD",  "BZR_MD", "COX2",
]

    # create a unique csv filename per run using timestamp + short uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    csv_path = f"kernel_baseline_results_{timestamp}_{unique_id}.csv"

    for dataset in datasets:
        try:
            print(f"Running all kernels baseline for dataset {dataset}")
            results = all_kernels_baseline(dataset, snippet_timeout_sec=36000)
            save_results_to_csv(csv_path, dataset, results)
            print(f"Results for {dataset} saved to {csv_path}")
        except Exception as e:
            print(f"Failed for {dataset}: {e}")


