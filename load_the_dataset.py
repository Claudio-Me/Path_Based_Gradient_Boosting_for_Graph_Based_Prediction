from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# Load the MUTAG dataset
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

print(f"Number of graphs in MUTAG: {len(dataset)}")


# Convert the first graph to a NetworkX graph
data = dataset[0]
G = to_networkx(data, to_undirected=True)

# Print basic info
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
