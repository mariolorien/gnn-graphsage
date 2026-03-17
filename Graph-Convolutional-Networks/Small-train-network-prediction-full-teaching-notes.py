import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Node features: [busy_level, has_tube]
# Indices: 0=A, 1=B, 2=C, 3=D, 4=E
x = torch.tensor([
    [2, 1],  # A
    [2, 0],  # B
    [1, 0],  # C
    [1, 1],  # D
    [2, 0],  # E
], dtype=torch.float)

# Edge list (directed)
edge_index = torch.tensor([
    [0, 4,  0, 1,  0, 3,  1, 2,  0, 1, 2, 3, 4],
    [4, 0,  1, 0,  3, 0,  2, 1,  0, 1, 2, 3, 4],
], dtype=torch.long)

print("x shape:", x.shape)
print("x:\n", x)
print("\nedge_index shape:", edge_index.shape)
print("edge_index:\n", edge_index)

"""
We define a graph using:

(1) x → node features
(2) edge_index → connectivity

For link prediction, we do NOT predict node labels.
Instead, we predict whether a connection (edge) exists
between two nodes.

So we must construct a dataset of node-pairs:

- Positive edges → edges that exist (label = 1)
- Negative edges → edges that do NOT exist (label = 0)

This converts the graph into a supervised binary classification problem.
"""

# --- Positive edges ---

src, dst = edge_index

mask_no_loops = src != dst

"""
We remove self-loops (edges like i → i), because they do not provide
useful supervision for link prediction.

Example:

src = [0, 1, 2, 3]
dst = [0, 2, 2, 3]

mask_no_loops =
[False, True, False, False]

So we keep only edges where src != dst.
"""

pos_edge_index = edge_index[:, mask_no_loops]

print("Positive edges:", pos_edge_index)

"""
pos_edge_index contains all real edges in the graph.

These will be labelled as 1.
"""

# --- Negative edges ---

neg_edge_index = torch.tensor([
    [2, 4, 3, 4],
    [4, 2, 4, 3],
], dtype=torch.long)

"""
Negative edges are node pairs that are NOT connected.

Example:
(2,4), (4,2), (3,4), (4,3)

These will be labelled as 0.
"""

# --- Labels ---

num_pos = pos_edge_index.size(1)
num_neg = neg_edge_index.size(1)

y_pos = torch.ones(num_pos)
y_neg = torch.zeros(num_neg)

y = torch.cat([y_pos, y_neg])

"""
We create binary labels:

1 → edge exists
0 → edge does not exist

Example:

y_pos = [1,1,1,1,...]
y_neg = [0,0,0,0]

y = [1,1,1,...,0,0,0,...]

So each edge pair has a corresponding label.
"""

# --- Combine edges ---

edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

print("edge_label_index:", edge_label_index)
print("y:", y)

"""
edge_label_index defines the node-pairs we want to evaluate.

So the dataset becomes:

x → node features
edge_index → graph structure
edge_label_index → node pairs to classify
y → labels (0 or 1)
"""

# --- Encoder (GNN) ---

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        z = self.conv2(h, edge_index)
        return z

"""
The encoder transforms node features into embeddings.

Input:
x → [num_nodes, num_features]

Output:
z → [num_nodes, embedding_dim]

Each node now has a learned representation that includes:
- its own features
- information from its neighbours
"""

encoder = GNNEncoder(x.size(1), 8, 4)

Z = encoder(x, edge_index)

print("Z:", Z)

"""
Z contains node embeddings.

Example:

Z[0] → embedding of node 0
Z[1] → embedding of node 1

These embeddings are what we use to predict links.
"""

# --- Decoder (dot product) ---

src_nodes = edge_label_index[0]
dst_nodes = edge_label_index[1]

z_src = Z[src_nodes]
z_dst = Z[dst_nodes]

scores = (z_src * z_dst).sum(dim=1)

"""
We compute similarity between node pairs.

Step 1: element-wise multiplication
Step 2: sum → dot product

Example:

z_i = [1,2]
z_j = [3,4]

z_i * z_j = [3,8]
sum = 11

This gives a score for how similar the nodes are.
"""

probs = torch.sigmoid(scores)

"""
Sigmoid converts scores into probabilities:

large positive → ~1 (likely edge)
large negative → ~0 (unlikely edge)
"""

# --- Loss ---

loss = F.binary_cross_entropy(probs, y)

"""
Binary Cross Entropy compares:

predicted probabilities vs true labels

Example:

prob = 0.9, label = 1 → low loss
prob = 0.2, label = 1 → high loss

So the model learns to:
- assign high probability to real edges
- assign low probability to non-edges
"""

print("Loss:", loss.item())

# --- Training loop ---

optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

for epoch in range(1, 201):
    encoder.train()
    optimizer.zero_grad()

    Z = encoder(x, edge_index)

    z_src = Z[src_nodes]
    z_dst = Z[dst_nodes]

    scores = (z_src * z_dst).sum(dim=1)
    probs = torch.sigmoid(scores)

    loss = F.binary_cross_entropy(probs, y)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        pos_mean = probs[:num_pos].mean().item()
        neg_mean = probs[num_pos:].mean().item()
        print(f"epoch {epoch} loss={loss.item():.4f} pos={pos_mean:.3f} neg={neg_mean:.3f}")

"""
During training:

- encoder learns node embeddings
- embeddings make connected nodes similar
- embeddings make non-connected nodes dissimilar

So over time:
positive edges → high probability
negative edges → low probability
"""

# --- Prediction ---

def predict_link_prob(i, j):
    encoder.eval()
    with torch.no_grad():
        Z = encoder(x, edge_index)
        score = (Z[i] * Z[j]).sum()
        prob = torch.sigmoid(score)
    return float(prob)

"""
We can now predict new links.

Given two nodes (i, j):

1. get embeddings z_i, z_j
2. compute dot product
3. apply sigmoid

Output:
probability that an edge exists
"""

candidates = [(0,2), (1,4), (2,3), (3,4), (2,4)]

print("\nPredictions:")
for (i, j) in candidates:
    print(f"{i}->{j}: {predict_link_prob(i,j):.4f}")