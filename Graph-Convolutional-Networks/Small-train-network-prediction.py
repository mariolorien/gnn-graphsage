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
    [1, 1],  # D  <-- if this "2" is a mistake, change to 0/1 only
    [2, 0],  # E
], dtype=torch.float)

# Edge list (directed): add both directions for undirected edges
edge_index = torch.tensor([
    # sources
    [0, 4,  0, 1,  0, 3,  1, 2,  0, 1, 2, 3, 4],
    # targets
    [4, 0,  1, 0,  3, 0,  2, 1,  0, 1, 2, 3, 4],
], dtype=torch.long)


print("x shape:", x.shape)
print("x:\n", x)
print("\nedge_index shape:", edge_index.shape)
print("edge_index:\n", edge_index)

"""
We have now created the network by defining the labels, the connectivity and the features. 
Because our objective is to make link prediction (or maybe new nodes) we need to define 
positive and negative edges for link prediction. 
Positive edges are the ones that actually exist (excluding self loops)
Negative edges are node pairs that do no have an edge. 

"""
# --- Step 3a: Positive edges (true links) for supervision ---

src, dst = edge_index

"""
We can also write this as: 
src = edge_index[0]
dst = edge_index[1]

Remember that edge_index is our tensor with 2 rows. 
"""

# Mask out self-loops (i -> i)
mask_no_loops = src != dst

"""
Here we create a boolean condition 
if src[k] != dst[k] ->True 
if src[k] == dst[k] -> False: this edge is a self loop lik3 3->3 
"""

# These are the positive training examples (label = 1)
pos_edge_index = edge_index[:, mask_no_loops]

"""
The is tensor indexing: 
: , means take all the rows both source and destination
mask_no_loops: means take only the columns where the mask is True

"""
print("Positive edges (pos_edge_index):")
print(pos_edge_index)
print("Number of positive directed edges:", pos_edge_index.size(1))


# --- Step 3b: Negative edges (non-links) for supervision ---

# Manually chosen non-edges (directed, include both directions)
# We have chosen (2,4) and (3,4), but we treat this undirected edges as directed we have four instead: 
# [(2,4),(4,2), (3,4), (4,3)]
neg_edge_index = torch.tensor([
    [2, 4, 3, 4],   # sources
    [4, 2, 4, 3],   # targets
], dtype=torch.long)

print("\nNegative edges (neg_edge_index):")
print(neg_edge_index)
print("Number of negative directed edges:", neg_edge_index.size(1))

# --- Step 3c: Labels ---

num_pos = pos_edge_index.size(1)
num_neg = neg_edge_index.size(1)
"""
For edge_index:
.size(0) tells us the format (should be 2: source and destination)
.size(1) tells us how many edges (the useful one)

In our example num_pos will be 8 (8 positive) and num_neg would be (4)
"""
y_pos = torch.ones(num_pos)
y_neg = torch.zeros(num_neg)

y = torch.cat([y_pos, y_neg])
"""
These create tensors filled with 1s or 0s.

torch.ones(num_pos) creates a vector like:

[1,1,1,1,1,1,1,1]
[1,1,1,1,1,1,1,1]

length = num_pos

torch.zeros(num_neg) creates:

[0,0,0,0]
[0,0,0,0]

length = num_neg

All positive node-pairs get label 1, all negative node-pairs get label 0.

"""

print("\nLabels y:")
print("y shape:", y.shape)
print(y)


# --- Step 4: Combine edges to score (positives + negatives) ---

edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

print("\nedge_label_index shape:", edge_label_index.shape)
print(edge_label_index)

# Sanity check: edge_label_index columns should match y length
print("Number of edge pairs:", edge_label_index.size(1))
print("Number of labels:", y.size(0))


"""
What we have so far

We have turned our network into:

x: raw node features (5 nodes x 2 features)
edge_index: who connects to who
edge_label_index: the node-pairs we want to judge
y: the yes/no labels for those pairs

That is the dataset.

Why that it is not enough by itself

It is not that “two numbers is always bad” — sometimes 2 features can be enough.

The real issue is this:
For link prediction, the label depends on the relationship between nodes and their position in the network, not just their raw features.
Our raw x only tells me “busy” and “tube”, but it does not directly encode:

That A sits in the middle connected to B, D, E
that C is only connected to B
that D is connected to A
that E is connected to A
…and those structural patterns matter a lot for predicting missing links.

What embeddings really are?

It is a learned transformation that creates a new set of numbers.

"""

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # IMPORTANT: we already included self-loops in edge_index,
        # so we set add_self_loops=False to avoid double self-loops.
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        z = self.conv2(h, edge_index)
        return z  # node embeddings

encoder = GNNEncoder(in_channels=x.size(1), hidden_channels=8, out_channels=4)

Z = encoder(x, edge_index)

print("\nNode embeddings Z:")
print("Z shape:", Z.shape)   # expect [5, 4]:  5 nodes 4 number embedding each 
print(Z)
print("These values are not meaningful yet because the encoder weights are still random. Training will shape them.")

# --- Step 7: Dot-product decoder (score each edge pair) ---

src_nodes = edge_label_index[0]  # [12]
dst_nodes = edge_label_index[1]  # [12]

z_src = Z[src_nodes]             # [12, 4]
z_dst = Z[dst_nodes]             # [12, 4]

scores = (z_src * z_dst).sum(dim=1)   # [12]
"""
This is the dot product per edge.

z_src * z_dst

This is element-wise multiplication.
both are [12, 4]
result is [12, 4]

For each edge k, it multiplies the 4 numbers.
Why do we need the dot product? 
Because in link prediction we need a way to turn two node embeddings 
into one number that says “how likely is there a link between them?”. 
The dot product is the simplest, standard way to do that.
We can also use cosine similarity, bilinear and MLP decoder. 
"""
probs  = torch.sigmoid(scores)        # [12]
"""
sigmoid maps any real number to 
[0,1]
[0,1]:

big positive score → probability near 1
big negative score → probability near 0

score near 0 → probability near 0.5
So now probs is shape [12].

"""
print("\nEdge scores and probabilities:")
for k in range(edge_label_index.size(1)):
    i = int(src_nodes[k])
    j = int(dst_nodes[k])
    print(f"edge ({i}->{j})  score={scores[k]: .3f}  prob={probs[k]:.3f}  label={int(y[k].item())}")


# --- Step 8: Loss (how wrong are our probabilities?) ---


"""
For binary link prediction we use binary cross-entropy. Intuition:

if label = 1, we want prob close to 1
if label = 0, we want prob close to 0

The loss is a single number that is smaller when we are right

"""
loss = F.binary_cross_entropy(probs, y)

print("\nLoss:", loss.item())

# --- Step 9: One training step ---

"""
Example of what it would look liek with JUST one training step

optimizer = torch.optim.Adam(encoder.parameters(), lr=0.00)

optimizer.zero_grad()
loss.backward()
optimizer.step()

# Recompute after one update (forward pass again)
Z = encoder(x, edge_index)

src_nodes = edge_label_index[0]
dst_nodes = edge_label_index[1]
scores = (Z[src_nodes] * Z[dst_nodes]).sum(dim=1)
probs = torch.sigmoid(scores)

loss2 = F.binary_cross_entropy(probs, y)
print("Loss after 1 step:", loss2.item())

"""

# --- Step 10: Training loop (stable) ---

optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

for epoch in range(1, 201):
    encoder.train()
    optimizer.zero_grad()

    Z = encoder(x, edge_index)
    src_nodes = edge_label_index[0]
    dst_nodes = edge_label_index[1]
    scores = (Z[src_nodes] * Z[dst_nodes]).sum(dim=1)
    probs = torch.sigmoid(scores)

    loss = F.binary_cross_entropy(probs, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        # quick separation check: mean prob for positives vs negatives
        pos_mean = probs[:num_pos].mean().item()
        neg_mean = probs[num_pos:].mean().item()
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}  pos_mean={pos_mean:.3f}  neg_mean={neg_mean:.3f}")



#after training our model, we predict 

def predict_link_prob(i, j):
    encoder.eval()
    with torch.no_grad():
        Z = encoder(x, edge_index)
        score = (Z[i] * Z[j]).sum()          # dot product
        prob = torch.sigmoid(score)
    return float(prob)

candidates = [(0,2), (1,4), (2,3), (3,4), (2,4)]
print("\nUnseen candidate links:")
for (i, j) in candidates:
    print(f"({i}->{j}) prob = {predict_link_prob(i,j):.4f}")

# =============================================================================
# Conclusion (what we built and what we observed)
# =============================================================================
#
# 1) We started from a hand-drawn toy rail network and converted it into the
#    standard tensors used by Graph Neural Networks:
#       - x: node feature matrix  [N, F]  (busy level + tube flag per station)
#       - edge_index: graph connectivity [2, E] (directed edges + self-loops)
#
# 2) Because our task was link prediction, we converted the graph into a
#    supervised learning dataset of node-pairs:
#       - pos_edge_index: edges that exist in the graph (labels = 1)
#       - neg_edge_index: edges that do NOT exist (labels = 0)
#       - edge_label_index: concatenation of positives and negatives
#       - y: the corresponding binary labels
#
# 3) We built a simple CGNN-style pipeline:
#       - Encoder: a 2-layer GCN that maps (x, edge_index) -> Z
#                  where Z is a learned embedding vector for each node.
#       - Decoder: dot-product scoring s_ij = z_i^T z_j for each candidate pair,
#                  followed by sigmoid to get a probability in [0, 1].
#
# 4) We trained the encoder parameters using binary cross-entropy loss between
#    predicted probabilities and y. With enough epochs, the model achieved
#    near-perfect separation on the training pairs (loss -> ~0), meaning it
#    could "memorise" which training pairs were labelled as links vs non-links.
#
#
# =============================================================================
# Limitations (why results on "new links" must be interpreted carefully)
# =============================================================================
#
# A) Tiny dataset / overfitting:
#    This graph has only 5 nodes and very few labelled edge pairs. With such a
#    small dataset, a GNN can easily overfit (memorise the training labels),
#    so very low training loss does NOT guarantee meaningful generalisation.
#
# B) Negative sampling matters:
#    Link prediction requires both positives (existing links) and negatives
#    (non-links). If we provide too few or unrepresentative negative examples,
#    the model will not learn a reliable boundary between "link" and "no link"
#    and may assign high probabilities to many unseen non-edges.
#
# C) We did not perform a proper evaluation split:
#    In a standard link prediction experiment, we hide some true edges
#    (train/validation/test split), train on the remaining edges + negatives,
#    and then test whether the model recovers the hidden edges better than it
#    scores true non-edges. We stopped before implementing that hold-out test.
#
# D) Decoder simplicity:
#    Dot-product is a simple and common baseline decoder, but it assumes a
#    relatively simple notion of compatibility between node embeddings. More
#    expressive decoders (e.g., bilinear or MLP) can capture richer patterns.
#
# E) Toy features and realism:
#    Our node features were minimal (2 features). Real transport networks would
#    require richer features (location, line constraints, demand flows, travel
#    times) and careful definition of what a "missing link" means in reality.
#
# Next logical extension (not implemented here):
#   - Hold out one real edge as a "missing link" test case.
#   - Train on the remaining graph with robust negative sampling.
#   - Evaluate whether the held-out true edge is ranked above non-edges.
# =============================================================================