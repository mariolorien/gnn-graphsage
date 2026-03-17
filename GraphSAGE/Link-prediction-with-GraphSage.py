import torch 
import torch.nn.functional as F 
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling 
from torch_geometric.datasets import KarateClub

#----Link Prediction Implementation-----

dataset_lp = KarateClub()
data_lp = dataset_lp[0]

print(data_lp)
"""
Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
This means 34 nodes - 156 edges stored as directed pair and 34 node features
"""

"""
The problem with KarateClub is that is often too small for link prediction experiments. 
A better approach is to generate a synthetic social network using a model like Barabasi-Albert model, 
wich produces realisitc networks with bus. 

The script in our folder (create_BAModel_Network) does this for us. 
"""

from Create_BAModel_Network import create_graph_dataset 

data_lp = create_graph_dataset()

print(data_lp)

"""
Dataset Created
--------------
Nodes: 300
Edges: 2368
Node feature dimesnion: torch.Size([300, 16])
Data(edge_index=[2, 2368], num_nodes=300, x=[300, 16])
"""

# Define the GraphSAGE Encoder (to get node embeddings) 

class LinkPredictionGraphSAGE(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LinkPredictionGraphSAGE, self).__init__()
        
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x 
    
    # Here we define a simple decoder for link prediction (dot product)
    
    def decoder(self, z, edge_label_index):
        """
        z are the node embeddings 
        edge_label_index are the indices of edges to predict
        """
        
        row, col = edge_label_index 
        
        return (z[row] * z[col]).sum(dim=1)
    

#Now we instantiate the encoder model. 
    
model_lp = LinkPredictionGraphSAGE(
    in_channels = data_lp.num_node_features,
    hidden_channels = 64, #Embedding dimensions
    out_channels = 64 # Output embedding dimensions
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_lp = model_lp.to(device)


#Prepare data for Link Prediction 
"""
For link prediction, we need positive examples (existing links) and negative examples (non-exisitng links)
We will split exisitng edges into training and test sets, and generate negative samples
We split edges into training and test sets (80/20 split)
"""

from torch_geometric.transforms import RandomLinkSplit 

transform = RandomLinkSplit(
    num_val  = 0.1, # 10% for validation
    num_test = 0.1, # 10% for test 
    is_undirected = True, #Our data set is undirected
    add_negative_train_samples = False, # We will manually add negative samples for flexibility
    neg_sampling_ratio = 1.0 # Ratio of negative to positive samples in val/test
)

"""
The transformation returns 3 data objects: train_data, val_data, and test_data. 
Each contains: 
1) edge_index : edges used for message passing 
2) edge_label_index : edges for prediction 
3) edge_label: labels for edges (1 = real edge, 0 = fake edge)
"""

train_data_lp, val_data_lp, test_data_lp = transform(data_lp)


"""
We combine now positive and negative edges for training labels 
For training we only need positive edges for message passing and then we will generate 
negative samples on the fly or use a fixed set for loss calculation. 
For simplicity, we will use the edge_label_index and edge_label returned by the transform
"""

train_edge_label_index = train_data_lp.edge_label_index.to(device)
train_edge_label = train_data_lp.edge_label.to(device)


"""
NOTE ON PYTORCH GEOMETRIC VERSION DIFFERENCES

Older tutorials and books often refer to the attributes:

    pos_edge_label_index
    neg_edge_label_index

However, newer versions of PyTorch Geometric changed the behaviour of the
RandomLinkSplit transformation.

The transformation now returns:

    edge_label_index
    edge_label

where:

    edge_label_index → contains BOTH positive and negative edges
    edge_label       → contains the labels (1 = real edge, 0 = fake edge)

Therefore, we no longer need to manually concatenate positive and negative edges
using torch.cat(). The transform already produces a dataset suitable for
link prediction training and evaluation.
"""


"""
NOTE ON NEGATIVE SAMPLES DURING TRAINING

We set:

    add_negative_train_samples = False

This means that RandomLinkSplit will NOT automatically generate negative
samples for the training set.

Instead, negative edges are generated dynamically during training using:

    negative_sampling(...)

This approach is generally preferred because it avoids the model memorising
a fixed set of negative edges.
"""


#Move all data to the device 

train_data_lp = train_data_lp.to(device)
val_data_lp = val_data_lp.to(device)
test_data_lp = test_data_lp.to(device)

#Define loss function and Optimizer

optimizer_lp = torch.optim.Adam(model_lp.parameters(), lr=0.01)
criterion_lp = torch.nn.BCEWithLogitsLoss() # Binary Cross Entropy with Logits for link Prediction. 


# Training Loop for Link Prediction 

def train_lp():

    model_lp.train()
    optimizer_lp.zero_grad()

    # node embeddings
    z = model_lp(train_data_lp.x, train_data_lp.edge_index)

    # positive edges
    pos_edge_index = train_data_lp.edge_label_index

    # generate negative edges
    neg_edge_index = negative_sampling(
        edge_index=train_data_lp.edge_index,
        num_nodes=train_data_lp.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )

    # combine edges
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    # create labels
    edge_label = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ]).to(device)

    # predictions
    out = model_lp.decoder(z, edge_label_index)

    # compute loss
    loss = criterion_lp(out, edge_label)

    loss.backward()
    optimizer_lp.step()

    return loss.item()

#Evaluation for Link Prediction (using AUC)

from sklearn.metrics import roc_auc_score 

def evaluate_lp(data_split):
    
    model_lp.eval()
    
    with torch.no_grad():
        
        z = model_lp(data_split.x, data_split.edge_index)
        
        edge_label_index = data_split.edge_label_index
        edge_label = data_split.edge_label

        out = model_lp.decoder(z, edge_label_index)

        out = torch.sigmoid(out)

        auc = roc_auc_score(
            edge_label.cpu().numpy(),
            out.cpu().numpy()
        )

    return auc


# Training and evaluation Loop 

epochs_lp = 2000

print("\n-----Training GraphSAGE for Link Prediction---")

for epoch in range(1, epochs_lp + 1):

    loss = train_lp()
    val_auc = evaluate_lp(val_data_lp)
    
    if epoch % 10 == 0: 
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")
        

test_auc = evaluate_lp(test_data_lp)

print(f"Test AUC: {test_auc:.4f}")