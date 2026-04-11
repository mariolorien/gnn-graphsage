import torch
import networkx as nx
from torch_geometric.utils import from_networkx, degree


def create_graph_dataset():
    # Creates scale-free network (similar to social networks)
    G = nx.barabasi_albert_graph(
        n=300,   # number of nodes
        m=4      # edges added per new node
    )

    # Convert to PyTorch Geometric format
    data = from_networkx(G)

    # Create simple structural node features
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    deg_norm = deg / deg.max()
    deg_log = torch.log(deg + 1)

    # Stack features into a [num_nodes, 3] tensor
    data.x = torch.stack([deg, deg_norm, deg_log], dim=1)

    # ---------------------------------------------------
    # Create labels for node classification
    #
    # Class meaning:
    # 0 = A = low degree
    # 1 = B = medium degree
    # 2 = C = high degree
    #
    # You can later rename these to something more descriptive
    # like "low", "medium", "high" if you prefer.
    # ---------------------------------------------------
    data.y = torch.zeros(data.num_nodes, dtype=torch.long)  #data.y contains the labels. 

    data.y[deg >= 5] = 1
    data.y[deg >= 10] = 2

    # Optional human-readable label mapping
    label_names = {
        0: "A",
        1: "B",
        2: "C"
    }

    print("Dataset Created")
    print("--------------")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Node feature shape: {data.x.shape}")
    print(f"Label shape: {data.y.shape}")
    print(f"Number of classes: {len(label_names)}")
    print(f"First 10 labels (numeric): {data.y[:10].tolist()}")
    print("First 10 labels (named):")
    for i in range(10):
        print(f"Node {i}: class {data.y[i].item()} = {label_names[data.y[i].item()]}")

    return data


if __name__ == "__main__":
    data = create_graph_dataset()

    print("\nFinal PyG data object:")
    print(data)

    print("\nFirst 5 node feature rows:")
    print(data.x[:5])

    print("\nFirst 5 labels:")
    print(data.y[:5])