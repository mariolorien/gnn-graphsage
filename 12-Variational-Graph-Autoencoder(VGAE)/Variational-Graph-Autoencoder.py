"""
VARIATIONAL GRAPH AUTOENCODER (VGAE) WITH CORA
==============================================

This script shows a complete first example of a Variational Graph Autoencoder
(VGAE) using the Cora citation network.

Main goal:
----------
Learn probabilistic node embeddings that can reconstruct graph connections.

In other words, the model tries to answer:

    "Given a probabilistic latent representation of each node,
     can we predict which pairs of nodes should have an edge between them?"

This is the standard link-prediction setup for Variational Graph Autoencoders.


WHY THIS IS CALLED A VARIATIONAL AUTOENCODER
--------------------------------------------
A VGAE has two main parts:

1) Variational encoder
   Takes the graph as input and produces:
   - mu      (mean of the latent distribution)
   - logstd  (log standard deviation of the latent distribution)

2) Decoder
   Uses sampled latent embeddings z to reconstruct graph structure,
   usually by predicting whether an edge exists.

So the full logic is:

    graph --> variational encoder --> distribution over z --> sampled z --> decoder --> reconstructed edges


HOW THIS DIFFERS FROM A NORMAL GAE
----------------------------------
In a standard Graph Autoencoder (GAE):
- the encoder produces one fixed embedding per node

In a Variational Graph Autoencoder (VGAE):
- the encoder produces a distribution for each node
- the model samples from that distribution
- a KL-divergence regularization term is added to the loss

So:

GAE:
    deterministic latent embeddings

VGAE:
    probabilistic latent embeddings


WHY THIS IS USEFUL
------------------
The variational setup encourages a smoother and more regularized latent space.

This can help with:
- uncertainty-aware latent representations
- better regularization
- generative modelling ideas
- robust link prediction


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

This creates:
- pos_edge_label_index
- neg_edge_label_index

which match the variable names used later in the script.


WHAT THIS SCRIPT DOES
---------------------
1) Load Cora
2) Split edges into train / validation / test for link prediction
3) Build a variational GCN encoder
4) Wrap it inside a VGAE model
5) Train the model using:
   - reconstruction loss
   - KL loss
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
from torch_geometric.nn.models import VGAE


# ------------------------------------------------------------------
# PART 1: LOAD AND SPLIT THE DATA
# ------------------------------------------------------------------

"""
For link prediction, we split edges into:

- training positive edges
- validation positive / negative edges
- test positive / negative edges

RandomLinkSplit is a standard PyG transform for this.

Important:
split_labels=True creates:
- pos_edge_label_index
- neg_edge_label_index
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
# PART 2: DEFINE THE VARIATIONAL ENCODER
# ------------------------------------------------------------------

"""
The encoder now produces two outputs instead of one:

- mu
- logstd

These describe the latent Gaussian distribution for each node.

Important idea:
---------------
VGAE is the overall variational encoder-decoder framework.

Inside that framework, we still need a graph encoder architecture.

Here:
- VGAE = overall probabilistic autoencoder framework
- GCN  = encoder style used inside it
"""


class VariationalGCNEncoder(torch.nn.Module):
    """
    A variational GCN encoder for VGAE.

    Input:
    - x
    - edge_index

    Output:
    - mu
    - logstd

    Each node gets:
    - a latent mean vector
    - a latent log-standard-deviation vector
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        First graph convolution:
        raw node features --> hidden features
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        """
        Two parallel graph convolutions:
        hidden features --> mu
        hidden features --> logstd
        """
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)

        return mu, logstd


# ------------------------------------------------------------------
# PART 3: BUILD THE VGAE MODEL
# ------------------------------------------------------------------

"""
VGAE wraps the variational encoder and provides:
- encode(...)
- recon_loss(...)
- kl_loss(...)
- test(...)

By default, VGAE also uses an inner-product decoder.

Important difference from GAE:
------------------------------
model.encode(...) now internally samples a latent embedding z
from the node-wise Gaussian distributions described by mu and logstd.
"""
in_channels = dataset.num_features
hidden_channels = 32
out_channels = 16

encoder = VariationalGCNEncoder(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
)

model = VGAE(encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Model:")
print(model)
print()


# ------------------------------------------------------------------
# PART 4: TRAINING FUNCTION
# ------------------------------------------------------------------

def train():
    """
    Train the Variational Graph Autoencoder.

    Main steps:
    -----------
    1) Encode nodes into latent embeddings z
    2) Reconstruct training positive edges
    3) Add KL regularization
    4) Backpropagate and update parameters

    Total loss:
    -----------
    recon_loss + kl_loss / num_nodes
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
    Encourage the model to assign high probability to real edges.
    Negative edges are sampled internally if omitted.
    """
    recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)

    """
    KL loss:
    --------
    Regularizes the latent distribution so that it does not drift too far
    from a standard normal prior.

    Dividing by num_nodes is a common stabilization choice.
    """
    kl = model.kl_loss() / train_data.num_nodes

    loss = recon_loss + kl

    loss.backward()
    optimizer.step()

    return loss.item(), recon_loss.item(), kl.item()


# ------------------------------------------------------------------
# PART 5: EVALUATION FUNCTION
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(data_split, split_name="Validation"):
    """
    Evaluate the model on validation or test edges.

    We compute node embeddings z using the training graph,
    then test whether those embeddings can discriminate:

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

print("Training Variational Graph Autoencoder...\n")

for epoch in range(1, 201):
    loss, recon_loss, kl = train()

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Total Loss: {loss:.4f} | "
            f"Recon: {recon_loss:.4f} | "
            f"KL: {kl:.4f}"
        )
        evaluate(val_data, split_name="Validation")
        print()

print("Final evaluation:")
val_auc, val_ap = evaluate(val_data, split_name="Validation")
test_auc, test_ap = evaluate(test_data, split_name="Test")


# ------------------------------------------------------------------
# PART 7: INSPECT THE LEARNED LATENT EMBEDDINGS
# ------------------------------------------------------------------

@torch.no_grad()
def inspect_embeddings():
    """
    Show the shape of the final sampled latent node embeddings.

    These embeddings come from the variational latent space.
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
# PART 8: INSPECT MU AND LOGSTD DIRECTLY
# ------------------------------------------------------------------

@torch.no_grad()
def inspect_distribution_parameters():
    """
    Inspect the encoder's mean and log-standard-deviation outputs directly.

    This helps us see the parameters of the latent distributions.
    """
    model.eval()
    mu, logstd = model.encoder(train_data.x, train_data.edge_index)

    print()
    print("Mu shape:", mu.shape)
    print("Logstd shape:", logstd.shape)
    print()
    print("First 5 mu vectors:")
    print(mu[:5])
    print()
    print("First 5 logstd vectors:")
    print(logstd[:5])

    return mu, logstd


mu, logstd = inspect_distribution_parameters()


# ------------------------------------------------------------------
# PART 9: SIMPLE MANUAL EDGE PROBABILITY EXAMPLE
# ------------------------------------------------------------------

@torch.no_grad()
def edge_probability(node_u, node_v):
    """
    Compute the decoder probability for one node pair.

    Since VGAE uses an inner-product decoder by default,
    this gives a probability that an edge exists between node_u and node_v.
    """
    model.eval()
    z = model.encode(train_data.x, train_data.edge_index)

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

- The variational encoder learned a latent distribution for each node.
- The decoder learned to use sampled latent embeddings to reconstruct graph edges.

So if two nodes receive sampled embeddings that are highly compatible,
the model will predict a high probability of an edge between them.

This is why Variational Graph Autoencoders are commonly used for:

- link prediction
- graph reconstruction
- probabilistic node embedding learning
- uncertainty-aware latent representations

The most important conceptual point is this:

GCN:
    tells us HOW to compute neighbourhood-based node representations

VGAE:
    tells us WHAT the full probabilistic model is trying to do with those
    representations, namely reconstruct graph structure from a latent distribution
"""

"""
Inspect the learned latent distribution for one specific node.

In VGAE, the actual learned distribution for node i is:

    q(z_i | x, A) = Normal(mu_i, sigma_i^2)

where:
- mu_i     = learned mean vector for node i
- logstd_i = learned log standard deviation vector for node i
- sigma_i  = exp(logstd_i)

The sampled latent embedding z_i is drawn from that distribution.
"""

@torch.no_grad()
def inspect_node_distribution(node_idx=0):
    model.eval()

    """
    Get the learned distribution parameters directly from the encoder.
    """
    mu, logstd = model.encoder(train_data.x, train_data.edge_index)

    """
    Convert log standard deviation into standard deviation.
    """
    std = logstd.exp()

    """
    Sampled latent embedding currently produced by model.encode(...).
    """
    z = model.encode(train_data.x, train_data.edge_index)

    print(f"\nNode {node_idx} latent distribution:")
    print("-" * 50)

    print("mu vector:")
    print(mu[node_idx])

    print("\nlogstd vector:")
    print(logstd[node_idx])

    print("\nstd vector = exp(logstd):")
    print(std[node_idx])

    print("\nSampled z vector:")
    print(z[node_idx])

    return mu[node_idx], logstd[node_idx], std[node_idx], z[node_idx]


inspect_node_distribution(node_idx=2)


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