"""
GCN + PGExplainer
=================

This script does four things:

1) Loads the Cora citation network
2) Trains a small GCN for node classification
3) Trains PGExplainer on node-level predictions
4) Generates an edge-level explanation for one chosen node

Important notes:
- PGExplainer must be trained separately before it can generate explanations.
- PGExplainer explains edge/subgraph importance, not node-feature importance.
- Therefore, we do NOT pass node_mask_type here.

This script follows the current PyTorch Geometric explainability style,
but minor API differences may appear across PyG versions.
"""

import matplotlib
matplotlib.use("Agg")

import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, PGExplainer


# ------------------------------------------------------------
# PART 1: LOAD DATA
# ------------------------------------------------------------

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]

print("Dataset:", dataset)
print("Data object:", data)
print("Number of node features:", dataset.num_node_features)
print("Number of classes:", dataset.num_classes)
print("Current working directory:", os.getcwd())
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
        Return log probabilities.
        """
        return F.log_softmax(x, dim=1)


model = GCN(
    in_channels=dataset.num_node_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# ------------------------------------------------------------
# PART 3: TRAIN THE GCN
# ------------------------------------------------------------

def train_model():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_model():
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
    loss = train_model()

    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test_model()
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss:.4f} | "
            f"Train: {train_acc:.4f} | "
            f"Val: {val_acc:.4f} | "
            f"Test: {test_acc:.4f}"
        )

print()

train_acc, val_acc, test_acc = test_model()
print("Final Train Accuracy:", train_acc)
print("Final Val Accuracy:", val_acc)
print("Final Test Accuracy:", test_acc)
print()


# ------------------------------------------------------------
# PART 4: FIX A TARGET TO EXPLAIN
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

"""
For PGExplainer training, we need a target.

Here we use the model's own predicted classes as the target to explain.
That makes the explainer learn to explain the trained model's behaviour.

If you wanted instead to explain the underlying ground-truth phenomenon,
you could use data.y as the target.
"""
target = pred


# ------------------------------------------------------------
# PART 5: BUILD THE EXPLAINER WRAPPER
# ------------------------------------------------------------

"""
Important:
PGExplainer does not provide node-feature masks in the same way as
GNNExplainer, so we only request an edge mask here.
"""

explainer = Explainer(
    model=model,
    algorithm=PGExplainer(epochs=30, lr=0.003),
    explanation_type="phenomenon",
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification",
        task_level="node",
        return_type="log_probs",
    ),
      
    threshold_config=dict(
        threshold_type="topk",
        value=15,
    ),
)


# ------------------------------------------------------------
# PART 6: TRAIN PGEXPLAINER
# ------------------------------------------------------------

"""
PGExplainer is a parametric explainer.
That means it has its own trainable neural network and must be trained
before explanations can be generated.

We train it on a subset of training nodes.
"""

train_node_indices = data.train_mask.nonzero(as_tuple=False).view(-1)

print("Training PGExplainer...\n")

for epoch in range(30):
    total_loss = 0.0

    for idx in train_node_indices[:100]:
        loss = explainer.algorithm.train(
            epoch=epoch,
            model=model,
            x=data.x,
            edge_index=data.edge_index,
            target=target,
            index=int(idx),
        )
        total_loss += float(loss)

    avg_loss = total_loss / min(100, train_node_indices.numel())

    print(f"PGExplainer Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f}")

print()


# ------------------------------------------------------------
# PART 7: GENERATE EXPLANATION FOR ONE NODE
# ------------------------------------------------------------

explanation = explainer(
    data.x,
    data.edge_index,
    target=target,
    index=node_idx,
)

print("Explanation object:")
print(explanation)
print()

print("Edge mask shape:", explanation.edge_mask.shape)
print("Edge mask min:", explanation.edge_mask.min().item())
print("Edge mask max:", explanation.edge_mask.max().item())
print()

print("First 20 edge importance values:")
print(explanation.edge_mask[:20])
print()


# ------------------------------------------------------------
# PART 8: SAVE THE EXPLANATION GRAPH
# ------------------------------------------------------------

"""
PGExplainer mainly gives an edge-level explanation,
so the most relevant visualization is the explanation graph.
"""

try:
    explanation.visualize_graph(path="pgexplanation_graph.pdf")
    print("Saved explanation graph to pgexplanation_graph.pdf")
except Exception as e:
    print("Could not save explanation graph:", e)