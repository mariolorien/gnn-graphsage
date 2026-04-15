import torch
import torch.nn.functional as F

from torch_geometric.datasets import PPI
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader

"""
The original GraphSAGE model uses node features and graph connectivity 
(neighbourhood structure), but it does not incorporate edge features.

In this PPI dataset:
Nodes = proteins
Edges = interactions between proteins
Node features = biological properties of each protein
Node labels = biological functions of each protein

Each node has 50 numerical attributes (features).

Example:

Protein(node)   Feature1   Feature2   ...   Feature50
P1              0.2        0.5               1.1
P2              0.4        1.2               1.2

The model predicts 121 possible biological functions.

Protein     Function1   Function2   ...   Function121
P1              1           0               0
P2              0           1               1

This is multi-label classification: a protein can belong to multiple classes.

Edges do NOT have features here. They only define the structure of the graph.
GraphSAGE uses edges to determine which neighbours influence a node.
"""

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

train_dataset = PPI(root="./data/PPI", split="train")
val_dataset = PPI(root="./data/PPI", split="val")
test_dataset = PPI(root="./data/PPI", split="test")

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of validation graphs: {len(val_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")

data = train_dataset[0]
print(data)

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(train_dataset, batch_size=2)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

print(f"Number of node features: {train_dataset.num_node_features}")
print(f"Number of classes: {train_dataset.num_classes}")

# --------------------------------------------------
# GraphSAGE Model
# --------------------------------------------------

class GraphSAGE(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)

        return x


# --------------------------------------------------
# Instantiate model
# --------------------------------------------------

model_node_cls = GraphSAGE(
    in_channels=train_dataset.num_node_features,
    hidden_channels=128,
    out_channels=train_dataset.num_classes,
)


"""
50 node features
121 output classes

Here we choose: 
hidden size = 128

So the layers become:

Layer 1
50  → 128

Layer 2
128 → 121
"""
# --------------------------------------------------
# Device (GPU or CPU)
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_node_cls = model_node_cls.to(device)

# --------------------------------------------------
# Loss function and optimizer
# --------------------------------------------------

optimizer_node_cls = torch.optim.Adam(
    model_node_cls.parameters(), lr=0.005, weight_decay=5e-4
)

"""
Weight decay prevents the weights from becoming too large. 
5e-4 is very commonly usde in GNN papers 

learning rate here is set at 0.005, which means how "large"
each weight update step is. 
ADAM is our optimizer - the model that we use to update our weights during training- 
"""
criterion_node_cls = torch.nn.BCEWithLogitsLoss()

# --------------------------------------------------
# Training function
# --------------------------------------------------

def training_node_cls():

    model_node_cls.train()

    total_loss = 0

    for data in train_loader:

        data = data.to(device)

        optimizer_node_cls.zero_grad()

        out = model_node_cls(data.x, data.edge_index)

        loss = criterion_node_cls(out, data.y.float())

        loss.backward()

        optimizer_node_cls.step()

        total_loss += loss.item()

    return total_loss / sum([d.num_nodes for d in train_dataset])


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

from sklearn.metrics import f1_score


def evaluate_node_cls(loader):

    model_node_cls.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            out = model_node_cls(data.x, data.edge_index)

            preds = (out > 0).float()

            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())

    preds_tensor = torch.cat(all_preds, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    f1 = f1_score(labels_tensor.numpy(), preds_tensor.numpy(), average="micro")

    return f1


# --------------------------------------------------
# Training loop
# --------------------------------------------------

epochs_node_cls = 50

print("\n--- Training GraphSAGE for Node Classification ---")

for epoch in range(1, epochs_node_cls + 1):

    loss = training_node_cls()

    val_f1 = evaluate_node_cls(val_loader)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}")

# --------------------------------------------------
# Test evaluation
# --------------------------------------------------

test_f1 = evaluate_node_cls(test_loader)

print(f"Test F1 Score: {test_f1:.4f}")

"""
We compute F1 score because this si multilabel classfication (accuracy would be missleading)
F1 score balances precison and recall F1 = 2(precision.recall)/(precision + recall) 
"""

