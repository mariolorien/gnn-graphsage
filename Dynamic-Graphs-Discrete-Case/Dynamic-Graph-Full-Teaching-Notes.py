
"""
A dynamic graph is a graph whose structure (nodes and edges)
and/or attributes change over time.

These changes can include:

1) Node arrivals and departures:
   New users joining a platform, or existing users leaving.

2) Edge additions and deletions:
   New friendships forming, or old connections disappearing.

3) Feature updates:
   A user's profile changing, or a product's price fluctuating.

4) Edge attribute changes:
   The strength of a relationship increasing or decreasing.

Temporal information is the key distinguishing characteristic of dynamic graphs.

Events happen at specific timestamps, and the order in which they occur matters.

Ignoring the temporal dimension in dynamic graphs can lead to:

- Stale information:
  Models trained on static snapshots can quickly become outdated.

- Loss of context:
  The sequence of events often provides critical information for prediction.

- Suboptimal performance:
  Models cannot capture evolving patterns or effectively predict future changes.
"""
"""
Dynamic graphs can be broadly categorised according to how their changes are represented:

1) Discrete-Time Dynamic Graphs (DTDG):
   The graph is observed at discrete time steps, forming a sequence of
   static graph snapshots.

   Examples:
   - Daily snapshots of a social network
   - Hourly states of a traffic network

2) Continuous-Time Dynamic Graphs (CTDG), also called Temporal Event Graphs:
   Changes occur asynchronously as individual events at specific timestamps.

   Examples:
   - An edge appearing or disappearing
   - A message being sent between two users
   - A node feature being updated at a precise time
"""

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import random


"""
DTDG EXAMPLE: Discrete-Time Dynamic Graph with GCN + GRU

Idea:
- We have a sequence of graph snapshots: G_0, G_1, ..., G_T
- Each snapshot is processed with a GCN
- The node embeddings over time are passed into a GRU
- The final GRU output is used for node classification

The GNU (the convolutional network) learns from the graph structure at one snapshot, 
the GRU, (Gated Recurrent Unit) learns from the sequence over time (i.e., looks at how each node's 
embedding changes across snapshots)

The GRU does the temporal learning across time steps.
"""

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear
from torch_geometric.nn import GCNConv

from DTDGSnapshotCreator import create_dynamic_ba_snapshots


"""
DTDG MODEL SCRIPT: GCN + GRU

This script does not create the dynamic graph data itself.
Instead, it imports the snapshot generator from DTDGSnapshotCreator.py

Pipeline:
1) Load a sequence of discrete-time graph snapshots
2) Apply a GCN to each snapshot
3) Feed the node embeddings across time into a GRU
4) Use the final temporal embedding for node classification
"""


# ------------------------------------------------------------------
# PART 1: SNAPSHOT GCN ENCODER
# ------------------------------------------------------------------

class SnapshotGCN(torch.nn.Module):
    """
    A small GCN encoder applied independently to each graph snapshot.

    Input:
    - x
    - edge_index

    Output:
    - node embeddings for that snapshot
    
    For small, medium size we use normal GCNs -like the one below-, 
    for large and very large graph size, we would use GraphSAGE
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        First graph convolution:
        input node features -> hidden node features
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        """
        Second graph convolution:
        hidden node features -> final snapshot embeddings
        """
        x = self.conv2(x, edge_index)

        return x


# ------------------------------------------------------------------
# PART 2: DTDG MODEL
# ------------------------------------------------------------------

class DTDGModel(torch.nn.Module):
    """
    Discrete-Time Dynamic Graph model.

    For each snapshot:
    - apply a GCN to obtain node embeddings

    Across snapshots:
    - collect each node's embeddings over time
    - process that temporal sequence with a GRU

    Final step:
    - classify each node using its final GRU representation
    """

    def __init__(self, in_channels, gcn_hidden, gcn_out, gru_hidden, num_classes):
        super().__init__()

        self.snapshot_gcn = SnapshotGCN(in_channels=in_channels, hidden_channels=gcn_hidden, out_channels=gcn_out)

        """
        The Line above: 
        
        DTDGModel needs to create an instance of SnapshotGCN because that is the encoder it uses to process each snapshot.
        This means: 
        DTDGModel is saying: “I need a graph encoder inside me”
        in this script, that encoder is SnapshotGCN
        so it instantiates it and stores it as self.snapshot_gcn

        Then later, during forward(), DTDGModel uses that instantiated object to encode each snapshot.
        """
        self.gru = GRU(input_size=gcn_out,hidden_size=gru_hidden,batch_first=True)

        """
        Final classifier:
        temporal embedding -> class logits
        """
        self.classifier = Linear(gru_hidden, num_classes)

    def forward(self, snapshots):
        """
        snapshots:
        list of PyG Data objects representing G_0, G_1, ..., G_T

        Goal:
        1) Encode each snapshot with the GCN
        2) Stack those embeddings across time
        3) Feed them into the GRU
        4) Predict node labels from the final temporal state
        """

        per_snapshot_embeddings = []

        for snapshot in snapshots:
            z_t = self.snapshot_gcn(snapshot.x, snapshot.edge_index)
            per_snapshot_embeddings.append(z_t)

        """
        After stacking:
        shape = [num_snapshots, num_nodes, gcn_out]
        """
        z = torch.stack(per_snapshot_embeddings, dim=0)

        """
        GRU expects:
        [batch_size, sequence_length, feature_dim]

        Here:
        - batch_size = num_nodes
        - sequence_length = num_snapshots
        - feature_dim = gcn_out
        """
        z = z.transpose(0, 1)

        """
        gru_out shape:
        [num_nodes, num_snapshots, gru_hidden]
        """
        gru_out, h_n = self.gru(z)

        """
        Keep the final GRU output for each node:
        shape = [num_nodes, gru_hidden]
        """
        final_node_embeddings = gru_out[:, -1, :]

        logits = self.classifier(final_node_embeddings)

        return logits


# ------------------------------------------------------------------
# PART 3: TRAINING FUNCTION
# ------------------------------------------------------------------

def train(model, snapshots, optimizer):
    """
    Train the model using the labels from the final snapshot.

    This is a standard temporal setting:
    earlier snapshots provide history,
    final snapshot provides the target labels.
    """
    model.train()
    optimizer.zero_grad()

    logits = model(snapshots)

    final_snapshot = snapshots[-1]
    mask = final_snapshot.active_mask
    y = final_snapshot.y

    """
    Only compute the loss on nodes that truly exist
    in the final snapshot.
    """
    loss = F.cross_entropy(logits[mask], y[mask])

    loss.backward()
    optimizer.step()

    return loss.item()


# ------------------------------------------------------------------
# PART 4: EVALUATION FUNCTION
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, snapshots):
    """
    Evaluate node classification accuracy on the final snapshot.
    """
    model.eval()

    logits = model(snapshots)

    final_snapshot = snapshots[-1]
    mask = final_snapshot.active_mask
    y = final_snapshot.y

    preds = logits.argmax(dim=1)

    correct = (preds[mask] == y[mask]).sum().item()
    total = mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0

    return accuracy, preds


# ------------------------------------------------------------------
# PART 5: MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    """
    Import and create the dynamic graph snapshots from DTDGSnapshotCreator.py
    """
    snapshots, max_nodes = create_dynamic_ba_snapshots(
        initial_num_nodes=6,
        m=2,
        num_snapshots=6,
        feature_dim=5,
        seed=123
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
        print("active nodes:", snapshot.active_mask.sum().item())
        print("-" * 50)

    """
    Infer input feature size directly from the first snapshot.
    This avoids hardcoding it.
    """
    in_channels = snapshots[0].x.size(1)

    model = DTDGModel(
        in_channels=in_channels,
        gcn_hidden=8,
        gcn_out=8,
        gru_hidden=8,
        num_classes=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nTraining...\n")

    for epoch in range(1, 101):
        loss = train(model, snapshots, optimizer)

        if epoch % 10 == 0:
            acc, _ = evaluate(model, snapshots)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    final_acc, final_preds = evaluate(model, snapshots)

    print("\nFinal accuracy:", final_acc)
    print("Predicted labels for all padded nodes:")
    print(final_preds)

    print("\nTrue labels at final snapshot:")
    print(snapshots[-1].y)

    print("\nActive mask at final snapshot:")
    print(snapshots[-1].active_mask)