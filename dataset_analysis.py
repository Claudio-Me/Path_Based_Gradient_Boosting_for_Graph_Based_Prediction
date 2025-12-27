import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def find_categorical_node_attributes(graphs, max_unique_values=200):
    from collections import defaultdict
    from collections.abc import Hashable
    attr_values = defaultdict(list)
    for graph in graphs:
        for _, node_data in graph.nodes(data=True):
            for attr, value in node_data.items():
                attr_values[attr].append(value)
    max_classes = 0
    selected_attr = None
    for attr, values in attr_values.items():
        if not all(isinstance(v, Hashable) for v in values):
            continue
        unique_vals = set(values)
        if (all(isinstance(v, (str, int)) for v in unique_vals)
                and 1 < len(unique_vals) <= max_unique_values):
            if len(unique_vals) > max_classes:
                max_classes = len(unique_vals)
                selected_attr = attr
    return selected_attr, max_classes

def dataset_prescreening(labels, dataset_name=None, show_plot=True, save_plot=False, plot_dir="plots", nx_graphs=None):
    """
    Analyze class distribution, plot histogram, and provide graph statistics.
    Also print percentage for each class and average graph size if graphs are provided.
    Additionally, report number of observations, node features, edge features,
    and number of classes of the selected categorical attribute.
    """
    labels = np.array(labels).flatten()
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    # Clean up output: convert keys and values to int
    clean_dist = {int(k): int(v) for k, v in class_dist.items()}
    total = sum(clean_dist.values())
    percentages = {k: (v / total * 100) for k, v in clean_dist.items()}
    print(f"Class distribution for {dataset_name}: {clean_dist}")
    print(f"Class percentages for {dataset_name}: " +
          ", ".join([f"{int(k)}: {percentages[k]:.2f}%" for k in sorted(percentages.keys())]))

    # Always plot using the actual class values
    plot_classes = list(clean_dist.keys())
    plot_counts = list(clean_dist.values())

    if save_plot or show_plot:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.figure()
        plt.bar(plot_classes, plot_counts)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(f"Class Distribution: {dataset_name}")
        plt.xticks(plot_classes)  # Set x-axis ticks to actual class values
        if save_plot:
            plt.savefig(f"{plot_dir}/{dataset_name}_class_distribution.png")
        if show_plot:
            plt.show()
        plt.close()

    categorical_attr = None
    num_categorical_classes = None

    # Graph statistics if graphs are provided
    if nx_graphs is not None and len(nx_graphs) > 0:
        num_nodes = [g.number_of_nodes() for g in nx_graphs]
        num_edges = [g.number_of_edges() for g in nx_graphs]
        avg_nodes = np.mean(num_nodes)
        avg_edges = np.mean(num_edges)
        print(f"Number of observations (graphs): {len(nx_graphs)}")
        print(f"Average number of nodes per graph: {avg_nodes:.2f}")
        print(f"Average number of edges per graph: {avg_edges:.2f}")
        print(f"Min nodes: {np.min(num_nodes)}, Max nodes: {np.max(num_nodes)}")
        print(f"Min edges: {np.min(num_edges)}, Max edges: {np.max(num_edges)}")

        # Node features
        node_feature_counts = []
        for g in nx_graphs:
            if g.number_of_nodes() == 0:
                continue
            # Take the first node with features
            for _, node_data in g.nodes(data=True):
                if node_data:
                    # Count features that are not empty
                    node_feature_counts.append(len(node_data))
                    break
        avg_node_features = np.mean(node_feature_counts) if node_feature_counts else 0
        print(f"Average number of node features: {avg_node_features:.2f}")

        # Edge features
        edge_feature_counts = []
        for g in nx_graphs:
            if g.number_of_edges() == 0:
                continue
            for _, _, edge_data in g.edges(data=True):
                if edge_data:
                    edge_feature_counts.append(len(edge_data))
                    break
        avg_edge_features = np.mean(edge_feature_counts) if edge_feature_counts else 0
        print(f"Average number of edge features: {avg_edge_features:.2f}")

        # Categorical attribute analysis
        categorical_attr, num_categorical_classes = find_categorical_node_attributes(nx_graphs)
        if categorical_attr:
            print(f"Selected categorical node attribute: {categorical_attr} ({num_categorical_classes} classes)")
        else:
            print("No suitable categorical node attribute found.")

    return {
        "class_distribution": clean_dist,
        "percentages": percentages,
        "num_graphs": len(nx_graphs) if nx_graphs is not None else None,
        "avg_nodes": avg_nodes if nx_graphs is not None else None,
        "avg_edges": avg_edges if nx_graphs is not None else None,
        "avg_node_features": avg_node_features if nx_graphs is not None else None,
        "avg_edge_features": avg_edge_features if nx_graphs is not None else None,
        "categorical_attribute": categorical_attr,
        "num_categorical_classes": num_categorical_classes
    }

def get_summary_table(datasets, base_dir=None, nx_graphs_dir="nx_graphs"):
    """
    Returns the summary table of key statistics for each dataset as a list of dicts.
    """
    from tudataset.tud_benchmark.auxiliarymethods.datasets import get_dataset

    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'tudataset', 'tud_benchmark', 'datasets'
        )

    summary = []
    for dataset_name in datasets:
        print(f"\n--- Analyzing {dataset_name} ---")
        try:
            labels = get_dataset(dataset_name, regression=False)
            nx_graphs_file = os.path.join(nx_graphs_dir, dataset_name + "_nx_graphs.pkl")
            nx_graphs = None
            if os.path.exists(nx_graphs_file):
                with open(nx_graphs_file, "rb") as f:
                    nx_graphs = pickle.load(f)
            stats = dataset_prescreening(labels, dataset_name=dataset_name, show_plot=False, save_plot=False, nx_graphs=nx_graphs)
            node_feat = stats.get('avg_node_features', 0)
            edge_feat = stats.get('avg_edge_features', 0)
            sum_feat = node_feat + edge_feat if node_feat is not None and edge_feat is not None else "-"
            percentages = stats.get("percentages", {})
            class_percent_str = ", ".join([f"{int(k)}: {percentages[k]:.2f}%" for k in sorted(percentages.keys())]) if percentages else "-"
            summary.append({
                "Dataset": dataset_name,
                "Graphs": stats.get("num_graphs", "-"),
                "AvgNodes": f"{stats.get('avg_nodes', 0):.2f}" if stats.get("avg_nodes") is not None else "-",
                "AvgEdges": f"{stats.get('avg_edges', 0):.2f}" if stats.get("avg_edges") is not None else "-",
                "NodeFeatures": int(round(node_feat)) if node_feat is not None else "-",
                "EdgeFeatures": int(round(edge_feat)) if edge_feat is not None else "-",
                "TotalFeatures": int(round(sum_feat)) if isinstance(sum_feat, (float, int)) else "-",
                "ClassPercentages": class_percent_str,
                "CatAttr": stats.get("categorical_attribute", "-"),
                "CatAttrClasses": stats.get("num_categorical_classes", "-")
            })
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
            summary.append({
                "Dataset": dataset_name,
                "Graphs": "-",
                "AvgNodes": "-",
                "AvgEdges": "-",
                "NodeFeatures": "-",
                "EdgeFeatures": "-",
                "TotalFeatures": "-",
                "ClassPercentages": "-",
                "CatAttr": "-",
                "CatAttrClasses": "-"
            })
    return summary

def analyze_all_datasets(datasets, base_dir=None, nx_graphs_dir="nx_graphs", csv_path="dataset_summary.csv"):
    """
    Run dataset_prescreening for all datasets in the list.
    Loads nx_graphs from nx_graphs_dir if available.
    Prints a summary table of key statistics for each dataset.
    Saves the summary table as a CSV file.
    Adds percentage of observations for each class.
    Removes 'Classes' column and ensures NodeFeatures, EdgeFeatures, TotalFeatures are integers.
    """
    from tudataset.tud_benchmark.auxiliarymethods.datasets import get_dataset
    import csv

    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'tudataset', 'tud_benchmark', 'datasets'
        )

    summary = []
    for dataset_name in datasets:
        print(f"\n--- Analyzing {dataset_name} ---")
        try:
            labels = get_dataset(dataset_name, regression=False)
            nx_graphs_file = os.path.join(nx_graphs_dir, dataset_name + "_nx_graphs.pkl")
            nx_graphs = None
            if os.path.exists(nx_graphs_file):
                with open(nx_graphs_file, "rb") as f:
                    nx_graphs = pickle.load(f)
            stats = dataset_prescreening(labels, dataset_name=dataset_name, show_plot=False, save_plot=False, nx_graphs=nx_graphs)
            node_feat = stats.get('avg_node_features', 0)
            edge_feat = stats.get('avg_edge_features', 0)
            sum_feat = node_feat + edge_feat if node_feat is not None and edge_feat is not None else "-"
            # Prepare class percentages string
            percentages = stats.get("percentages", {})
            class_percent_str = ", ".join([f"{int(k)}: {percentages[k]:.2f}%" for k in sorted(percentages.keys())]) if percentages else "-"
            summary.append({
                "Dataset": dataset_name,
                "Graphs": stats.get("num_graphs", "-"),
                "AvgNodes": f"{stats.get('avg_nodes', 0):.2f}" if stats.get("avg_nodes") is not None else "-",
                "AvgEdges": f"{stats.get('avg_edges', 0):.2f}" if stats.get("avg_edges") is not None else "-",
                "NodeFeatures": int(round(node_feat)) if node_feat is not None else "-",
                "EdgeFeatures": int(round(edge_feat)) if edge_feat is not None else "-",
                "TotalFeatures": int(round(sum_feat)) if isinstance(sum_feat, (float, int)) else "-",
                "ClassPercentages": class_percent_str,
                "CatAttr": stats.get("categorical_attribute", "-"),
                "CatAttrClasses": stats.get("num_categorical_classes", "-")
            })
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
            summary.append({
                "Dataset": dataset_name,
                "Graphs": "-",
                "AvgNodes": "-",
                "AvgEdges": "-",
                "NodeFeatures": "-",
                "EdgeFeatures": "-",
                "TotalFeatures": "-",
                "ClassPercentages": "-",
                "CatAttr": "-",
                "CatAttrClasses": "-"
            })

    # Print summary table
    print("\nSummary Table:")
    header = [
        "Dataset", "Graphs", "AvgNodes", "AvgEdges", "NodeFeatures", "EdgeFeatures", "TotalFeatures",
        "ClassPercentages", "CatAttr", "CatAttrClasses"
    ]
    print("{:<25} {:>7} {:>9} {:>9} {:>13} {:>13} {:>13} {:>25} {:>15} {:>16}".format(*header))
    print("-" * 150)
    for row in summary:
        print("{:<25} {:>7} {:>9} {:>9} {:>13} {:>13} {:>13} {:>25} {:>15} {:>16}".format(
            row["Dataset"], row["Graphs"], row["AvgNodes"], row["AvgEdges"],
            row["NodeFeatures"], row["EdgeFeatures"], row["TotalFeatures"],
            row["ClassPercentages"], str(row["CatAttr"]), str(row["CatAttrClasses"])
        ))

    # Save summary table as CSV
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print(f"\nSummary table saved to {csv_path}")

if __name__ == "__main__":
    datasets = [
        "AIDS", "BZR", "COX2", "COX2_MD", "DHFR", "ER_MD",
        "MCF-7", "MCF-7H", "MOLT-4", "MOLT-4H", "Mutagenicity", "MUTAG", "NCI1", "NCI109",
        "NCI-H23", "NCI-H23H", "OVCAR-8", "OVCAR-8H", "P388", "P388H", "PC-3", "PC-3H",
        "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "SF-295", "SF-295H", "SN12C", "SN12CH",
        "SW-620", "SW-620H",
        "Tox21_AhR_training", "Tox21_AhR_testing", "Tox21_AhR_evaluation",
        "Tox21_AR_training", "Tox21_AR_testing", "Tox21_AR_evaluation",
        "Tox21_AR-LBD_training", "Tox21_AR-LBD_testing", "Tox21_AR-LBD_evaluation",
        "Tox21_ARE_training", "Tox21_ARE_testing", "Tox21_ARE_evaluation",
        "Tox21_aromatase_training", "Tox21_aromatase_testing", "Tox21_aromatase_evaluation",
        "Tox21_ATAD5_training", "Tox21_ATAD5_testing", "Tox21_ATAD5_evaluation",
        "Tox21_ER_training", "Tox21_ER_testing", "Tox21_ER_evaluation",
        "Tox21_ER-LBD_training", "Tox21_ER-LBD_testing", "Tox21_ER-LBD_evaluation",
        "Tox21_HSE_training", "Tox21_HSE_testing", "Tox21_HSE_evaluation",
        "Tox21_MMP_training", "Tox21_MMP_testing", "Tox21_MMP_evaluation",
        "Tox21_p53_training", "Tox21_p53_testing", "Tox21_p53_evaluation",
        "Tox21_PPAR-gamma_training", "Tox21_PPAR-gamma_testing", "Tox21_PPAR-gamma_evaluation",
        "UACC257", "UACC257H", "Yeast", "YeastH", "DD", "KKI", "OHSU", "Peking_1",
        "PROTEINS", "PROTEINS_full", "DBLP_v1", "TWITTER-Real-Graph-Partial", "SYNTHETIC", "DHFR_MD", "BZR_MD"
    ]
    analyze_all_datasets(datasets)
