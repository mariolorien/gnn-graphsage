import random
import networkx as nx
import torch
from torch_geometric.data import Data


"""
DTDGSnapshotCreator.py

This script creates a Discrete-Time Dynamic Graph (DTDG) as a sequence
of PyTorch Geometric Data snapshots.

Main idea:
1) Start with an initial Barabási-Albert graph
2) At each time step:
   - update node features
   - add a new node using preferential attachment
   - optionally add an extra edge
   - optionally remove an edge
3) Save the graph as a PyG Data snapshot

Important:
Because the number of nodes grows over time, each snapshot is padded
to the same size (max_nodes), so that later we can stack snapshot
embeddings across time inside the DTDG model.

Each snapshot contains:
- x           : padded node feature matrix
- edge_index  : graph connectivity
- y           : padded node labels
- active_mask : True for real nodes, False for padded nodes
- time        : snapshot index
"""


def choose_preferential_targets(G, m, rng):
    """
    Select m existing nodes with probability proportional to degree.

    This mimics preferential attachment:
    nodes with higher degree are more likely to receive new links.
    """

    existing_nodes = list(G.nodes())
    degrees = [G.degree(node) + 1 for node in existing_nodes]
    """
    +1 avoids zero probability for isolated nodes.
    """

    chosen = set()

    while len(chosen) < min(m, len(existing_nodes)):
        target = rng.choices(existing_nodes, weights=degrees, k=1)[0]
        chosen.add(target)

    return list(chosen)


def graph_to_pyg_data(G, features_dict, time_step, max_nodes):
    """
    Convert a NetworkX graph into a padded PyTorch Geometric Data object.

    Parameters
    ----------
    G : networkx.Graph
        Current graph snapshot

    features_dict : dict
        Dictionary mapping node_id -> feature tensor

    time_step : int
        Snapshot index

    max_nodes : int
        Maximum number of nodes across all snapshots
        used for padding

    Returns
    -------
    data : torch_geometric.data.Data
        A padded PyG snapshot with:
        - x
        - edge_index
        - y
        - active_mask
        - time
    """

    num_nodes = G.number_of_nodes()

    # ------------------------------------------------------------
    # NODE FEATURES
    # ------------------------------------------------------------
    x_real = torch.stack(
        [features_dict[node] for node in range(num_nodes)],
        dim=0
    )

    feature_dim = x_real.size(1)

    """
    Pad node features so every snapshot has shape:
    [max_nodes, feature_dim]
    """
    x = torch.zeros((max_nodes, feature_dim), dtype=x_real.dtype)
    x[:num_nodes] = x_real

    # ------------------------------------------------------------
    # EDGE INDEX
    # ------------------------------------------------------------
    edges = list(G.edges())

    if len(edges) > 0:
        """
        Since the graph is undirected, add both directions.
        PyG stores edges explicitly in directed format.
        """
        undirected_edges = edges + [(v, u) for (u, v) in edges]
        edge_index = torch.tensor(
            undirected_edges,
            dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # ------------------------------------------------------------
    # NODE LABELS
    # ------------------------------------------------------------
    """
    Example toy labels for node classification.

    Label rule:
    y = 1 if node degree >= average degree of the current graph
    y = 0 otherwise
    """
    degrees = torch.tensor(
        [G.degree(node) for node in range(num_nodes)],
        dtype=torch.float
    )

    avg_degree = degrees.mean()
    y_real = (degrees >= avg_degree).long()

    """
    Pad labels to shape [max_nodes]
    """
    y = torch.zeros(max_nodes, dtype=torch.long)
    y[:num_nodes] = y_real

    # ------------------------------------------------------------
    # ACTIVE MASK
    # ------------------------------------------------------------
    """
    active_mask tells us which nodes are real at this time step.

    True  = node really exists in this snapshot
    False = padded placeholder node
    """
    active_mask = torch.zeros(max_nodes, dtype=torch.bool)
    active_mask[:num_nodes] = True

    # ------------------------------------------------------------
    # BUILD PYG DATA OBJECT
    # ------------------------------------------------------------
    data = Data(x=x, edge_index=edge_index, y=y)
    data.time = torch.tensor([time_step], dtype=torch.long)
    data.active_mask = active_mask

    return data


def create_dynamic_ba_snapshots(
    initial_num_nodes=8,
    m=2,
    num_snapshots=6,
    feature_dim=4,
    feature_noise_std=0.05,
    p_extra_edge=0.40,
    p_remove_edge=0.20,
    seed=42,
):
    """
    Create a sequence of padded discrete-time graph snapshots.

    Parameters
    ----------
    initial_num_nodes : int
        Number of nodes in the first BA graph

    m : int
        Number of edges each newly arriving node creates

    num_snapshots : int
        Total number of snapshots to generate

    feature_dim : int
        Number of features per node

    feature_noise_std : float
        Standard deviation of Gaussian noise added to node features
        at each time step

    p_extra_edge : float
        Probability of adding one extra random edge at each step

    p_remove_edge : float
        Probability of removing one existing edge at each step

    seed : int
        Random seed for reproducibility

    Returns
    -------
    snapshots : list
        List of padded PyG Data snapshots

    max_nodes : int
        Maximum padded node count across all snapshots
    """

    rng = random.Random(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------
    # INITIAL GRAPH
    # ------------------------------------------------------------
    """
    First snapshot starts as a Barabási-Albert graph.
    """
    G = nx.barabasi_albert_graph(
        n=initial_num_nodes,
        m=m,
        seed=seed
    )

    # ------------------------------------------------------------
    # INITIAL NODE FEATURES
    # ------------------------------------------------------------
    features_dict = {
        node: torch.randn(feature_dim)
        for node in G.nodes()
    }

    snapshots = []

    """
    We add exactly one new node after each snapshot,
    except after the final snapshot.

    So the maximum number of nodes is:
    initial_num_nodes + (num_snapshots - 1)
    """
    max_nodes = initial_num_nodes + (num_snapshots - 1)

    # ------------------------------------------------------------
    # BUILD SNAPSHOTS OVER TIME
    # ------------------------------------------------------------
    for t in range(num_snapshots):
        """
        Save current graph as snapshot G_t.
        """
        snapshot = graph_to_pyg_data(
            G=G,
            features_dict=features_dict,
            time_step=t,
            max_nodes=max_nodes
        )
        snapshots.append(snapshot)

        """
        Stop after saving the final snapshot.
        """
        if t == num_snapshots - 1:
            break

        # --------------------------------------------------------
        # STEP 1: UPDATE NODE FEATURES
        # --------------------------------------------------------
        """
        Small Gaussian noise simulates features changing over time.
        """
        for node in G.nodes():
            features_dict[node] = (
                features_dict[node]
                + feature_noise_std * torch.randn(feature_dim)
            )

        # --------------------------------------------------------
        # STEP 2: ADD A NEW NODE
        # --------------------------------------------------------
        """
        New node joins and connects to m existing nodes
        using preferential attachment.
        """
        targets = choose_preferential_targets(G, m, rng)

        new_node = G.number_of_nodes()
        G.add_node(new_node)
        features_dict[new_node] = torch.randn(feature_dim)

        for target in targets:
            G.add_edge(new_node, target)

        # --------------------------------------------------------
        # STEP 3: OPTIONAL EXTRA EDGE
        # --------------------------------------------------------
        """
        Simulates new relationships forming between older nodes.
        """
        if rng.random() < p_extra_edge:
            possible_non_edges = list(nx.non_edges(G))
            if possible_non_edges:
                u, v = rng.choice(possible_non_edges)
                G.add_edge(u, v)

        # --------------------------------------------------------
        # STEP 4: OPTIONAL EDGE REMOVAL
        # --------------------------------------------------------
        """
        Simulates old relationships disappearing.
        """
        if rng.random() < p_remove_edge and G.number_of_edges() > m:
            edge_to_remove = rng.choice(list(G.edges()))
            G.remove_edge(*edge_to_remove)

    return snapshots, max_nodes


if __name__ == "__main__":
    snapshots, max_nodes = create_dynamic_ba_snapshots(
        initial_num_nodes=8,
        m=2,
        num_snapshots=6,
        feature_dim=5,
        seed=123,
    )

    print("Number of snapshots:", len(snapshots))
    print("Maximum padded node count:", max_nodes)
    print()

    for i, snapshot in enumerate(snapshots):
        print(f"Snapshot {i}")
        print(snapshot)
        print("time:", snapshot.time.item())
        print("x shape:", snapshot.x.shape)
        print("edge_index shape:", snapshot.edge_index.shape)
        print("y shape:", snapshot.y.shape)
        print("active nodes:", snapshot.active_mask.sum().item())
        print("-" * 50)