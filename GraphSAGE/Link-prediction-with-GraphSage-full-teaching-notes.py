import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import KarateClub
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score

# ---- Link Prediction with GraphSAGE -----




from Create_BAModel_Network import create_graph_dataset

data_lp = create_graph_dataset()
"""
This creates a Barabási-Albert style network for a more realistic
link prediction experiment.
"""

print(data_lp)
"""
Print the synthetic graph to inspect number of nodes, edges, and features.
"""


class LinkPredictionGraphSAGE(torch.nn.Module):
    """
    GraphSAGE model used as an encoder for link prediction.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        """
        First GraphSAGE layer:
        input features -> hidden embeddings
    
        Second GraphSAGE layer:
        hidden embeddings -> final node embeddings
        """

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) # First neighbour aggregation step.
        x = F.relu(x) #        Apply non-linearity.
        x = F.dropout(x, p=0.05, training=self.training) # Apply dropout during training.
        x = self.conv2(x, edge_index) #Second GraphSAGE layer produces final embeddings.
       

        return x #Return one embedding vector per node.
          

    def decoder(self, z, edge_label_index):
        """
        Decode candidate edges using node embeddings.

        z:
        node embedding matrix

        edge_label_index:
        pairs of nodes to score as possible edges
        """

        row, col = edge_label_index #row = source nodes col = destination nodes
        

        return (z[row] * z[col]).sum(dim=1)
        """
        Compute one dot-product score per candidate edge.
         (z[row] * z[col]) = tensor([
                                    [1*5, 2*6],
                                    [3*7, 4*8]
                                   ])
        =
        tensor([
                  [5, 12],
                  [21, 32]
              ])
              
        Then sum across each row: 
        
        (z[row] * z[col]).sum(dim=1) = tensor([
                                              5 + 12,
                                              21 + 32
                                             ])
        =
        tensor([17, 53])
        So:
        edge 1 gets score 17
        edge 2 gets score 53
        """


model_lp = LinkPredictionGraphSAGE(in_channels=data_lp.num_node_features, hidden_channels=64, out_channels=64)

"""
Create the GraphSAGE link prediction model.

Input size  = number of node features
Hidden size = chosen by us
Output size = final embedding dimension
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
Use GPU if available, otherwise CPU.
"""

model_lp = model_lp.to(device)
"""
Move the model to the selected device.
"""


transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0
)
"""
Split edges into training, validation, and test sets.

num_val=0.1:
10% of edges used for validation

num_test=0.1:
10% of edges used for testing

is_undirected=True:
the graph is treated as undirected

add_negative_train_samples=False:
negative training edges will be generated manually during training

neg_sampling_ratio=1.0:
validation and test sets use equal numbers of positive and negative edges
"""

train_data_lp, val_data_lp, test_data_lp = transform(data_lp)
"""
Apply the split.

Each split contains:
- edge_index for message passing
- edge_label_index for edges to predict
- edge_label for true edge labels
"""


train_edge_label_index = train_data_lp.edge_label_index.to(device)
"""
this contains the list of node pairs begin treated as training examples 
"""

train_edge_label = train_data_lp.edge_label.to(device)
"""
Example: 

If:

train_data_lp.edge_label_index = tensor([
    [0, 1, 2],
    [2, 3, 4]
])

then candidate edges are:

(0,2)
(1,3)
(2,4)

and if:

train_data_lp.edge_label = tensor([1, 0, 1])

that means:

(0,2) is a real edge
(1,3) is a fake edge
(2,4) is a real edge

That is what the two lines above are extracting.

"""


train_data_lp = train_data_lp.to(device)
val_data_lp = val_data_lp.to(device)
test_data_lp = test_data_lp.to(device)
"""
Move all data splits to the selected device.
"""


optimizer_lp = torch.optim.Adam(model_lp.parameters(), lr=0.01)
"""
Adam optimiser for updating model parameters.
"""

criterion_lp = torch.nn.BCEWithLogitsLoss()
"""
Loss function for binary link prediction.
It expects raw edge scores (logits), so no sigmoid is applied before loss.
"""


def train_lp():
    """
    One full training step for link prediction.
    """

    model_lp.train() #   Enable training mode.
    optimizer_lp.zero_grad() # Clear old gradients.
   
    z = model_lp(train_data_lp.x, train_data_lp.edge_index) #  Compute node embeddings using the training graph.

    pos_edge_index = train_data_lp.edge_label_index
    """
    These are all positive training edges.
    These are real edges used as positive examples.
    How do we knwo they are ALL positives? 
    
    Because of the way RandomLinkSplit was configured.

    In our code:

    add_negative_train_samples=False

    So the training split does not come with negative training edges added into train_data_lp.edge_label_index.
    """

    neg_edge_index = negative_sampling(
        edge_index=train_data_lp.edge_index,
        num_nodes=train_data_lp.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    """
    Generate negative edges using the method negative_sampling. 
    These are node pairs that are not real edges and will be used
    as negative examples.
    """

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1) #Combine positive and negative candidate edges into one tensor.
  

    edge_label = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ]).to(device)
    """
    Create binary labels for the candidate edges.
    1 = real edge
    0 = fake edge
    """

    out = model_lp.decoder(z, edge_label_index) # Compute one raw score for each candidate edge.
    loss = criterion_lp(out, edge_label) #Compare predicted edge scores with true edge labels.
    loss.backward() #Compute gradients.
    optimizer_lp.step() # Update the model weights.
  
    return loss.item() #Return the training loss for this step.
    


def evaluate_lp(data_split):
    """
    Evaluate the model on a validation or test split.

    Returns:
    AUC score
    """

    model_lp.eval() #Switch to evaluation mode.
   

    with torch.no_grad(): #No gradients are needed during evaluation.
        z = model_lp(data_split.x, data_split.edge_index)  # Compute node embeddings for this split.
        edge_label_index = data_split.edge_label_index # Candidate edges to score.
        edge_label = data_split.edge_label #  True labels for the candidate edges.
        out = model_lp.decoder(z, edge_label_index) # Compute raw edge scores.
        out = torch.sigmoid(out) #Convert raw scores into probabilities between 0 and 1.

        auc = roc_auc_score(edge_label.cpu().numpy(),out.cpu().numpy())
        """
        Compute the AUC score.

        AUC measures how well the model separates positive edges from
        negative edges across different thresholds.
        """

    return auc


epochs_lp = 500
"""
Train for 500 epochs.
"""

print("\n----- Training GraphSAGE for Link Prediction -----")

for epoch in range(1, epochs_lp + 1):

    loss = train_lp()
    val_auc = evaluate_lp(val_data_lp) # Evaluate on the validation split

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")


test_auc = evaluate_lp(test_data_lp)

print(f"Test AUC: {test_auc:.4f}")
