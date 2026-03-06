import torch
import random
import math

def make_station_graph(
    n_nodes: int = 100,
    k_nearest: int = 4,
    seed: int = 42,
    self_loops: bool = True,
):
    random.seed(seed)
    torch.manual_seed(seed)

    # -------------------------
    # 1) Node "positions" (for edges)
    # -------------------------
    # Think of these as station coordinates on a map (0..1 range).
    pos = torch.rand(n_nodes, 2)  # [N, 2]

    # -------------------------
    # 2) Node features (your 2 features)
    # -------------------------
    # busy: 1 or 2
    busy = torch.randint(low=1, high=3, size=(n_nodes, 1))  # {1,2}
    # tube: 0 or 1
    tube = torch.randint(low=0, high=2, size=(n_nodes, 1))  # {0,1}

    x = torch.cat([busy, tube], dim=1).float()  # [N, 2]

    # -------------------------
    # 3) Build edges: connect each node to its k nearest neighbours
    # -------------------------
    # Pairwise distances
    # dist[i, j] = Euclidean distance between station i and j
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)   # [N, N, 2]
    dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-12)  # [N, N]

    edges = set()
    for i in range(n_nodes):
        # sort by distance, skip itself at index 0
        nn = torch.argsort(dist[i])[1 : 1 + k_nearest].tolist()
        for j in nn:
            edges.add((i, j))
            edges.add((j, i))  # undirected stored as two directed edges

    if self_loops:
        for i in range(n_nodes):
            edges.add((i, i))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()  # [2, E]

    return x, edge_index, pos

# ---- Generate dataset ----
x, edge_index, pos = make_station_graph(n_nodes=100, k_nearest=4, seed=123)

print("x shape:", x.shape)                 # [100, 2]
print("edge_index shape:", edge_index.shape)  # [2, E]
print("First 5 node feature rows:\n", x[:5])
print("First 10 edges (columns):\n", edge_index[:, :10])