"""
EXPLAINABILITY METHODS FOR GRAPH NEURAL NETWORKS
===============================================

These methods are used to understand why a GNN made a prediction.

Typical questions they try to answer are:
- Why was this node classified as class A?
- Which neighbouring nodes mattered most?
- Which edges were most important?
- Which node features influenced the decision?
- Which subgraph drove the prediction?

This is important because GNNs can otherwise behave like black boxes.



1) GNNExplainer
----------------
GNNExplainer is a graph-specific explainability method designed especially
for Graph Neural Networks.

Main idea:
For one specific prediction, GNNExplainer tries to identify a small and
important part of the graph that is sufficient to preserve the model's
decision.

In practice, it tries to learn masks over:
- edges
- node features

The goal is to keep only the most relevant structure and feature information.

So, for example, if a GNN predicts that a node belongs to class 1,
GNNExplainer asks:

    "What is the smallest subgraph and smallest set of features
     that still make the model predict class 1?"

This makes GNNExplainer mainly an instance-level explainer:
it usually explains one node, one graph, or one prediction at a time.

In simple words:
GNNExplainer tries to show the important local subgraph and features
behind a single prediction.



2) PGExplainer
---------------
PGExplainer is also a graph-specific explainability method,
but it works differently from GNNExplainer.

Main idea:
Instead of solving a fresh explanation optimization problem separately
for every single prediction, PGExplainer trains a separate explainer model.

That explainer learns how to generate explanations across many instances.

This means:
- GNNExplainer = optimize a new explanation per case
- PGExplainer = train an explanation model that can be reused

This is useful when many predictions need to be explained,
because it can be more scalable than explaining every example from scratch.

Like GNNExplainer, PGExplainer usually focuses on:
- important edges
- important neighbourhood structure
- important subgraphs

In simple words:
PGExplainer learns how to produce explanations for GNN predictions
in a more general and reusable way.



3) SHAP for Graphs
-------------------
SHAP originally comes from general explainable AI, not specifically from GNNs.

SHAP is based on Shapley values from game theory.

Main idea:
Measure how much each input component contributes to the final prediction.

In standard tabular machine learning, SHAP often tells us:
- how much feature 1 contributed
- how much feature 2 contributed
- etc.

For graphs, this becomes more complicated because the important components
may include:
- node features
- edges
- neighbouring nodes
- subgraphs

So SHAP-style explanations for graphs try to estimate contribution scores
for graph components.

Important note:
SHAP was not originally built for relational graph structure,
so graph-based SHAP methods are usually adaptations rather than one single
standard graph-native method.

In simple words:
SHAP for graphs tries to assign importance values to the parts of the graph
that influenced the prediction.



4) LIME for Graphs
-------------------
LIME also comes from general explainable AI, not specifically from GNNs.

LIME stands for:
Local Interpretable Model-agnostic Explanations

Main idea:
For one specific prediction, perturb the input locally,
observe how the prediction changes,
and fit a simpler interpretable model around that local region.

For graph data, this may mean perturbing:
- node features
- edges
- neighbouring nodes
- local subgraph structure

Then LIME tries to infer which parts were important for that local decision.

Important note:
Like SHAP, LIME was not originally created for graphs,
so graph-LIME methods are also adaptations.

In simple words:
LIME for graphs tries to build a simple local explanation around one
specific graph prediction by testing small changes to the input.



SUMMARY OF THE DIFFERENCES
===========================

GNNExplainer:
- graph-specific
- explains one prediction at a time
- finds an important subgraph and/or important features

PGExplainer:
- graph-specific
- learns an explainer model across many examples
- can generate explanations more efficiently for multiple predictions

SHAP for graphs:
- based on Shapley-value logic
- estimates contribution scores
- often adapted to node features, edges, or graph components

LIME for graphs:
- local approximation method
- perturbs the input around one instance
- fits a simpler interpretable explanation locally



IMPORTANT DISTINCTION
======================

These names refer first to explanation METHODS or ALGORITHMS,
not just code files.

However, many of them also have real implementations in libraries,
for example in graph ML frameworks or research repositories.

So it is correct to say:

- they are explainability methods
- and they often also exist as implementations



VERY SHORT MEMORY AID
======================

GNNExplainer  -> find the important subgraph/features for one prediction
PGExplainer   -> learn how to generate graph explanations across examples
SHAP          -> assign contribution scores
LIME          -> build a simple local approximation-based explanation
"""
"""
GNNExplainer and PGExplainer are best suited to static graph models.

They can also be applied, with caution, to discrete-time dynamic graphs
when these are represented as a sequence of static snapshots, especially
if the explanation focuses on one snapshot at a time.

However, for continuous-time dynamic graph models based on event streams,
timestamps, and memory updates, vanilla GNNExplainer and PGExplainer are
usually not directly appropriate without additional adaptation.
"""

"""
GCN + GNNExplainer
==================

This script does three things:

1) Loads the Cora citation network
2) Trains a small GCN for node classification
3) Uses GNNExplainer to explain one node prediction

Why this is a good first explainability example:
- It is static, not dynamic
- It uses a standard GCN
- GNNExplainer is naturally suited to this setting
- PyG's official explainability docs show this workflow through
  the Explainer interface for node classification

Key idea of GNNExplainer:
It tries to identify the important local graph structure and/or
important node features that are sufficient to preserve the model's
prediction for a specific node.

Note:
This script assumes your model returns log probabilities because the
Explainer configuration below uses:
    return_type='log_probs'
"""
import os 
print(f"Current Working Directory: {os.getcwd()}")

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib
matplotlib.use("Agg")


# ------------------------------------------------------------
# PART 1: LOAD DATA
# ------------------------------------------------------------

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]

print("Dataset:", dataset)
print("Data object:", data)
print("Number of node features:", dataset.num_node_features)
print("Number of classes:", dataset.num_classes)
print()


# ------------------------------------------------------------
# PART 2: DEFINE A SMALL GCN
# ------------------------------------------------------------

class GCN(torch.nn.Module):
    """
    Simple 2-layer GCN for node classification.

    Input:
    - x
    - edge_index

    Output:
    - log probabilities for each node
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        First graph convolution:
        raw node features -> hidden node features
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        """
        Second graph convolution:
        hidden node features -> class logits
        """
        x = self.conv2(x, edge_index)

        """
        Return log probabilities because the Explainer configuration
        below will be set to return_type='log_probs'.
        """
        return F.log_softmax(x, dim=1)


model = GCN(
    in_channels=dataset.num_node_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# ------------------------------------------------------------
# PART 3: TRAINING
# ------------------------------------------------------------

def train():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    """
    Supervised loss only on training nodes.
    """
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test():
    model.eval()

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []

    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = int(mask.sum())
        acc = correct / total if total > 0 else 0.0
        accs.append(acc)

    return accs


print("Training GCN...\n")

for epoch in range(1, 201):
    loss = train()

    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss:.4f} | "
            f"Train: {train_acc:.4f} | "
            f"Val: {val_acc:.4f} | "
            f"Test: {test_acc:.4f}"
        )

print()

train_acc, val_acc, test_acc = test()
print("Final Train Accuracy:", train_acc)
print("Final Val Accuracy:", val_acc)
print("Final Test Accuracy:", test_acc)
print()


# ------------------------------------------------------------
# PART 4: PICK ONE NODE TO EXPLAIN
# ------------------------------------------------------------

model.eval()
log_probs = model(data.x, data.edge_index)
pred = log_probs.argmax(dim=1)

"""
Choose one test node to explain.
"""
test_node_indices = data.test_mask.nonzero(as_tuple=False).view(-1)
node_idx = int(test_node_indices[0])

print("Node chosen for explanation:", node_idx)
print("Predicted class:", int(pred[node_idx]))
print("True class:", int(data.y[node_idx]))
print()


# ------------------------------------------------------------
# PART 5: GNNExplainer THROUGH PYG'S EXPLAINER INTERFACE
# ------------------------------------------------------------

"""
PyG's explainability interface lets us wrap the model together with
an explanation algorithm.

Here:
- explanation_type='model' means we explain the model's prediction
- node_mask_type='attributes' means we explain feature importance
- edge_mask_type='object' means we explain edge importance
- task_level='node' means this is node classification
- return_type='log_probs' matches the model output
"""

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification",
        task_level="node",
        return_type="log_probs",
    ),
)

"""
Generate explanation for one node.
"""
explanation = explainer(
    data.x,
    data.edge_index,
    index=node_idx,
)

print("Explanation object:")
print(explanation)
print()

print("Node mask shape:", explanation.node_mask.shape)
print("Edge mask shape:", explanation.edge_mask.shape)
print()

print("First 10 edge importance values:")
print(explanation.edge_mask[:10])
print()

"""
For attribute explanations, node_mask typically has one importance
value per feature (or a feature-related mask depending on configuration).
"""
print("Node feature importance mask:")
print(explanation.node_mask)
print()


# ------------------------------------------------------------
# PART 6: OPTIONAL VISUALIZATION
# ------------------------------------------------------------

"""
These helpers are part of the PyG explanation workflow.

Depending on your environment, graph plotting may open a figure window
or require a working matplotlib backend.
"""

try:
    explanation.visualize_feature_importance(top_k=10, path="feature_importance.png")
    print("Saved feature importance plot to feature_importance.png")
except Exception as e:
    print("Could not save feature importance plot:", e)

try:
    explanation.visualize_graph(path="explanation_graph.pdf")
    print("Saved explanation graph to explanation_graph.pdf")
except Exception as e:
    print("Could not save explanation graph:", e)