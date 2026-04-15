"""
GRAPH AUTOENCODER (GAE) WITH CORA
=================================

This script shows a complete first example of a Graph Autoencoder (GAE)
using the Cora citation network.

Main goal:
----------
Learn node embeddings that can reconstruct graph connections.

In other words, the model tries to answer:

    "Given the learned node embeddings, can we predict which pairs of nodes
     should have an edge between them?"

This is the standard link-prediction setup for Graph Autoencoders.


WHY THIS IS CALLED AN AUTOENCODER
---------------------------------
A Graph Autoencoder has two main parts:

1) Encoder
   Takes the graph as input and produces node embeddings.

2) Decoder
   Uses those embeddings to reconstruct graph structure,
   usually by predicting whether an edge exists.

So the full logic is:

    graph --> encoder --> node embeddings --> decoder --> reconstructed edges


WHY THIS IS DIFFERENT FROM NORMAL GCN NODE CLASSIFICATION
---------------------------------------------------------
In a normal GCN node-classification script, the output is usually:

    one class prediction per node

In a Graph Autoencoder, the output is usually:

    one link probability per pair of nodes

So the task is no longer "what class is this node?",
but rather:

    "should these two nodes be connected?"


DATASET USED
------------
We use the Cora citation network from PyTorch Geometric.

Cora is convenient because:
- it is small
- it downloads automatically
- it is a standard beginner dataset
- it works well for teaching


IMPORTANT FIX USED IN THIS VERSION
----------------------------------
We use:

    split_labels=True

inside RandomLinkSplit.

This is important because it creates:
- pos_edge_label_index
- neg_edge_label_index

which match the variables used later in the script.

Without split_labels=True, PyG may instead create:
- edge_label_index
- edge_label

and then the older-style pos/neg attributes would not exist.


WHAT THIS SCRIPT DOES
---------------------
1) Load Cora
2) Split edges into train / validation / test for link prediction
3) Build a GCN encoder
4) Wrap it inside a GAE model
5) Train the model to reconstruct positive edges
6) Evaluate with AUC and Average Precision


METRICS
-------
We use:

- AUC (Area Under the ROC Curve)
- AP  (Average Precision)

Higher is better for both.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE


# ------------------------------------------------------------------
# PART 1: LOAD AND SPLIT THE DATA
# ------------------------------------------------------------------

"""
For link prediction, we do not keep the graph exactly as-is.

Instead, we split edges into:

- training positive edges
- validation positive / negative edges
- test positive / negative edges

The model trains on the training graph, then we test whether it can
predict missing links on validation and test sets.

RandomLinkSplit is a standard PyG transform for this.

Important:
split_labels=True creates:
- pos_edge_label_index
- neg_edge_label_index

This makes the code below consistent with the variables we want to use.
"""
dataset = Planetoid(
    root="data/Planetoid",
    name="Cora",
    transform=RandomLinkSplit(
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False,
        num_val=0.05,
        num_test=0.10,
    ),
)

"""
The transformed dataset returns three graph objects:
- train_data
- val_data
- test_data
"""
train_data, val_data, test_data = dataset[0]

print("Train data:")
print(train_data)
print()

print("Validation data:")
print(val_data)
print()

print("Test data:")
print(test_data)
print()

print("Node feature matrix shape:", train_data.x.shape)
print("Training graph edge_index shape:", train_data.edge_index.shape)
print("Training positive edge label index shape:", train_data.pos_edge_label_index.shape)
print("Validation positive edge label index shape:", val_data.pos_edge_label_index.shape)
print("Validation negative edge label index shape:", val_data.neg_edge_label_index.shape)
print("Test positive edge label index shape:", test_data.pos_edge_label_index.shape)
print("Test negative edge label index shape:", test_data.neg_edge_label_index.shape)
print()


# ------------------------------------------------------------------
# PART 2: DEFINE THE ENCODER
# ------------------------------------------------------------------

"""
The encoder is the part that turns raw node features into node embeddings.

Here we use a simple 2-layer GCN encoder.

Important idea:
---------------
GAE itself is not the graph convolution architecture.
GAE is the overall encoder-decoder framework.

Inside that framework, we still need an encoder architecture.

Here:
- GAE = overall architecture
- GCN = encoder inside the GAE
"""


class GCNEncoder(torch.nn.Module):
    """
    A simple GCN encoder for Graph Autoencoder.

    Input:
    - x
    - edge_index

    Output:
    - latent node embeddings z
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        First graph convolution:
        raw node features --> hidden features
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        """
        Second graph convolution:
        hidden features --> latent embeddings z
        """
        x = self.conv2(x, edge_index)

        return x


# ------------------------------------------------------------------
# PART 3: BUILD THE GAE MODEL
# ------------------------------------------------------------------

"""
GAE wraps the encoder and provides:
- encode(...)
- recon_loss(...)
- test(...)

By default, GAE uses an inner-product decoder.

That means:
-----------
If two node embeddings are similar and point in compatible directions,
their inner product will be high, and the model will tend to predict
that an edge should exist between them.

So the decoder is effectively learning:

    "Are these two node embeddings compatible enough to form an edge?"
"""
in_channels = dataset.num_features
hidden_channels = 32
out_channels = 16

encoder = GCNEncoder(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
)

model = GAE(encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Model:")
print(model)
print()


# ------------------------------------------------------------------
# PART 4: TRAINING FUNCTION
# ------------------------------------------------------------------

def train():
    """
    Train the Graph Autoencoder.

    Main steps:
    -----------
    1) Encode nodes into latent embeddings z
    2) Reconstruct training positive edges
    3) Compute reconstruction loss
    4) Backpropagate and update parameters
    """
    model.train()
    optimizer.zero_grad()

    """
    Encode the training graph.
    z shape = [num_nodes, out_channels]
    """
    z = model.encode(train_data.x, train_data.edge_index)

    """
    Reconstruction loss:
    --------------------
    The model is encouraged to assign high probability to real edges.

    Since add_negative_train_samples=False in RandomLinkSplit,
    we do not rely on train_data.neg_edge_label_index.

    If neg_edge_index is omitted here, PyG samples negative edges internally.
    So the model learns to separate:
    - real edges
    - non-edges
    """
    loss = model.recon_loss(z, train_data.pos_edge_label_index)

    loss.backward()
    optimizer.step()

    return loss.item()


# ------------------------------------------------------------------
# PART 5: EVALUATION FUNCTION
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(data_split, split_name="Validation"):
    """
    Evaluate the model on validation or test edges.

    We first compute node embeddings z using the training graph.
    Then we test whether those embeddings can discriminate:

    - positive edges
    - negative edges

    Metrics returned:
    - AUC
    - AP
    """
    model.eval()

    """
    Important:
    ----------
    We still encode using the training graph structure.
    This avoids leaking validation/test edges into message passing.
    """
    z = model.encode(train_data.x, train_data.edge_index)

    auc, ap = model.test(
        z,
        data_split.pos_edge_label_index,
        data_split.neg_edge_label_index,
    )

    print(f"{split_name} AUC: {auc:.4f} | {split_name} AP: {ap:.4f}")

    return auc, ap


# ------------------------------------------------------------------
# PART 6: TRAIN THE MODEL
# ------------------------------------------------------------------

print("Training Graph Autoencoder...\n")

for epoch in range(1, 201):
    loss = train()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
        evaluate(val_data, split_name="Validation")
        print()

print("Final evaluation:")
val_auc, val_ap = evaluate(val_data, split_name="Validation")
test_auc, test_ap = evaluate(test_data, split_name="Test")


# ------------------------------------------------------------------
# PART 7: INSPECT THE LEARNED EMBEDDINGS
# ------------------------------------------------------------------

@torch.no_grad()
def inspect_embeddings():
    """
    Show the shape of the final learned node embeddings.

    These embeddings are the compressed latent representation learned
    by the encoder.
    """
    model.eval()
    z = model.encode(train_data.x, train_data.edge_index)

    print()
    print("Final latent embedding matrix z shape:", z.shape)
    print("First 5 node embeddings:")
    print(z[:5])

    return z


z = inspect_embeddings()


# ------------------------------------------------------------------
# PART 8: SIMPLE MANUAL EDGE PROBABILITY EXAMPLE
# ------------------------------------------------------------------

@torch.no_grad()
def edge_probability(node_u, node_v):
    """
    Compute the decoder probability for one node pair.

    Since GAE uses an inner-product decoder by default,
    this gives a probability that an edge exists between node_u and node_v.
    """
    model.eval()
    z = model.encode(train_data.x, train_data.edge_index)

    """
    model.decoder(...) gives a score for the edge pair.
    Applying sigmoid turns that into a probability-like value.
    """
    edge_index = torch.tensor([[node_u], [node_v]], dtype=torch.long)
    prob = model.decoder(z, edge_index).sigmoid().item()

    print(f"Predicted edge probability between node {node_u} and node {node_v}: {prob:.4f}")
    return prob


print()
edge_probability(0, 1)
edge_probability(0, 100)


"""
FINAL LEARNING INTERPRETATION
=============================

What did the model learn?

- The encoder learned a latent embedding for each node.
- The decoder learned to use those embeddings to reconstruct graph edges.

So if two nodes receive embeddings that are highly compatible,
the model will predict a high probability of an edge between them.

This is why Graph Autoencoders are commonly used for:

- link prediction
- graph reconstruction
- unsupervised node embedding learning

The most important conceptual point is this:

GCN:
    tells us HOW to compute neighbourhood-based node representations

GAE:
    tells us WHAT the full model is trying to do with those representations,
    namely reconstruct graph structure
"""

"""
The Graph Autoencoder ran successfully. The GCN encoder learned a
16-dimensional latent embedding for each of the 2708 nodes, and the
inner-product decoder used those embeddings to predict whether edges
should exist between node pairs. The final test AUC of 0.8950 and
test AP of 0.9027 indicate good link-prediction performance.
"""


"""
To reuse a trained PyTorch model on new compatible data, the standard
approach is to save its learned weights with torch.save(...state_dict()).

For example:

    torch.save(model.encoder.state_dict(), "vgae_encoder_weights.pth")

Later, recreate the same encoder architecture and load the weights with:

    encoder.load_state_dict(torch.load("vgae_encoder_weights.pth"))
    encoder.eval()

The loaded encoder can then be applied to new graph data, provided the
new data has the same input feature size and the same architecture is used.
"""

torch.save(model.encoder.state_dict(), "vgae_encoder_weights.pth")