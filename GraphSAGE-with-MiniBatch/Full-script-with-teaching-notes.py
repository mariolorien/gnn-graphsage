"""
Mini-batch GraphSAGE for node classification using PyTorch Geometric.

This script demonstrates how to train a GraphSAGE model on a large graph
using mini-batches instead of loading the full graph into memory at once.

Why do we need this?
For small graphs, we can often pass the whole graph into the GNN in one go.
But for large graphs, this becomes too expensive in memory and computation.
To solve this, PyTorch Geometric provides NeighborLoader, which samples only
the target nodes and their local neighbourhoods for each batch.

In this example, we:
1. Create a synthetic large graph
2. Define a 2-layer GraphSAGE model
3. Build NeighborLoaders for train, validation, and test
4. Train using mini-batches
5. Evaluate using mini-batches
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

# --------------------------------------------------
# 0. Device
# --------------------------------------------------

"""
We use GPU if available, otherwise CPU.

The model and each mini-batch must be moved to the same device.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------
# 1. Create a dummy large graph
# --------------------------------------------------

"""
In a real project, this data would usually come from:
- a dataset file,
- a graph database,
- a social network,
- a citation network,
- a recommender system graph,
- or another real-world source.

Here we create a synthetic graph just to understand the mechanics of
mini-batch training with NeighborLoader.
"""

# Example graph size
num_nodes_large = 10_000
num_edges_large = 50_000
num_features_large = 32
num_classes_large = 10

# Random node features
x_large = torch.randn(num_nodes_large, num_features_large)

# Random edges
"""
edge_index has shape [2, num_edges].

The first row contains source nodes.
The second row contains destination nodes.

Each column represents one edge:
[source_node, destination_node]

The line below could have been written also as: 

low_index = 0
high_index = num_nodes_large
edge_shape = (2, num_edges_large)

edge_index_large = torch.randint(
    low=low_index,
    high=high_index,
    size=edge_shape,
    dtype=torch.long
)

the num_edges_large tell us how many nedge to generate, in this case 50,000 or: 
Create a tensor with shape (2, 50000), and fill it with random integers from 0 to 9999.
"""
edge_index_large = torch.randint(0, num_nodes_large, (2, num_edges_large), dtype=torch.long)


# Random labels for node classification
"""
Each node gets a class label from 0 to num_classes_large - 1.

This is a semi-supervised setting:
all nodes may have labels stored in y,
but only some nodes are used for training,
some for validation,
and some for testing.
"""
y_large = torch.randint(0, num_classes_large, (num_nodes_large,),dtype=torch.long)

"""
The alternative version is: 

low_class = 0
high_class = num_classes_large
label_shape = (num_nodes_large,)

y_large = torch.randint(
    low=low_class,
    high=high_class,
    size=label_shape,
    dtype=torch.long
)

"""
# Create train / validation / test masks
train_mask_large = torch.zeros(num_nodes_large, dtype=torch.bool)
val_mask_large = torch.zeros(num_nodes_large, dtype=torch.bool)
test_mask_large = torch.zeros(num_nodes_large, dtype=torch.bool)
"""
In the above three line what we do is creating tensors filled with 
False; exactly 10,000 each. 
#How? 

torch.zeros creates all zeros...but because of the dtype=torch.bool,
we change them to False. 

"""
train_mask_large[:int(num_nodes_large * 0.6)] = True
val_mask_large[int(num_nodes_large * 0.6):int(num_nodes_large * 0.8)] = True
test_mask_large[int(num_nodes_large * 0.8):] = True

"""
Now how do we do the split? 
Line 1 says first 60% are for training, hence change them to True (6,000)
Line 2 says from 60% to 80% - 2,000 - use them for validation changing them to True (2,000)
Line 3 says from 80% to the end use them for testing also, changing them to True (2,000)

"""
# Build PyG Data object
data_large = Data(
    x=x_large,
    edge_index=edge_index_large,
    y=y_large,
    train_mask=train_mask_large,
    val_mask=val_mask_large,
    test_mask=test_mask_large
)
"""
Why we are creating this very large object and why not keep it all the variables separated? 
In theory, we could but, PyTorch Geometric is built to work with a single graph data object, usually called Data.
That makes it much easier to pass the graph around.

Instead of handling 6 separate things all the time, we can just use:  data_large

Is like saying: 
"“Here is my graph.
Here are its node features, its edges, its labels, and the masks 
telling me which nodes are for training, validation, and testing.”

What does the Object looks like? 

x:               torch.Size([10000, 32])   #10,000 nodes x 32 features per node
edge_index:      torch.Size([2, 50000])    # 2 rows and 50,000 columns, where each column is one edge
y:               torch.Size([10000])       # 1 label per each of the 10,000 nodes
train_mask:      torch.Size([10000])       # 10,000 values, 6000 True and 4000 False
val_mask:        torch.Size([10000])       # 10,000 values, 2000 True and 8,000 False
test_mask:       torch.Size([10000])       # 10,000 values, 2000 True and 8,000 False 

"""
print(f"Dummy Large Graph: {data_large}")
print(f"Number of training nodes: {int(data_large.train_mask.sum())}")
print(f"Number of validation nodes: {int(data_large.val_mask.sum())}")
print(f"Number of test nodes: {int(data_large.test_mask.sum())}")


# --------------------------------------------------
# 2. Define the GraphSAGE model
# --------------------------------------------------

"""
This is a 2-layer GraphSAGE model.

Layer 1:
- takes the original node features
- aggregates neighbourhood information
- produces hidden embeddings

Layer 2:
- takes those hidden embeddings
- aggregates again
- produces output logits for each class

The forward method returns raw logits.
We do NOT apply softmax here because CrossEntropyLoss expects raw logits.
"""

class LargeGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LargeGraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First GraphSAGE layer
        x = self.conv1(x, edge_index)

        # Non-linearity
        x = F.relu(x)

        # Dropout only during training
        x = F.dropout(x, p=0.5, training=self.training)

        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)

        return x


model_large_graph = LargeGraphSAGE(
    in_channels=num_features_large,
    hidden_channels=64,
    out_channels=num_classes_large
).to(device)


# --------------------------------------------------
# 3. Create NeighborLoaders
# --------------------------------------------------

"""
NeighborLoader is the key part of mini-batch GNN training.

Instead of loading the full graph into memory for each update,
it samples only a local subgraph around the target nodes.

Important parameters:

num_neighbors = [10, 5]
This means:
- at the first GNN layer, sample up to 10 neighbours per target node
- at the second GNN layer, sample up to 5 neighbours per node

Since this is a 2-layer GraphSAGE model, we provide two numbers.

batch_size = 128
This is the number of target nodes in each mini-batch.

input_nodes = data_large.train_mask
This tells the loader which nodes should serve as target nodes for batching.
Only nodes marked True in the training mask are used as supervised training targets.

shuffle = True for training
This randomises the order of training target nodes.

shuffle = False for validation and test
This keeps evaluation deterministic and stable.

num_workers = 0
This means data loading happens in the main process.
It is simple and reliable for demonstration.
In production, increasing num_workers can speed up loading.


The full training mask contains 6,000 possible target nodes overall,
but each mini-batch uses only 128 of them as target nodes at a time.
"""

train_loader_large = NeighborLoader(
    data_large,
    num_neighbors=[10, 5], # sample 10 neighbours of the target node (128 nodes)on 1st layer and then 5 on the 2nd 
    batch_size=128,
    input_nodes=data_large.train_mask, #6,000 available
    shuffle=True,
    num_workers=0, # helper. PyTorch can create background helper process called workers to load 
)

val_loader_large = NeighborLoader(
    data_large,
    num_neighbors=[10, 5],
    batch_size=128,
    input_nodes=data_large.val_mask,
    shuffle=False, #shuffle is good for learning, not on validation or testing 
    num_workers=0,
)

test_loader_large = NeighborLoader(
    data_large,
    num_neighbors=[10, 5],
    batch_size=128,
    input_nodes=data_large.test_mask,
    shuffle=False,
    num_workers=0,
)

print(f"\nNumber of batches in training loader: {len(train_loader_large)}")

# Example of one sampled batch
for batch_data in train_loader_large:
    print(f"Batch nodes: {batch_data.num_nodes}, Batch edges: {batch_data.num_edges}")
    print(f"Batch x shape: {batch_data.x.shape}, Batch y shape: {batch_data.y.shape}")
    print(f"Batch size (target nodes only): {batch_data.batch_size}")
    break


# --------------------------------------------------
# 4. Define optimizer and loss
# --------------------------------------------------

# Adam optimizer updates the model parameters using gradients
optimizer_large_graph = torch.optim.Adam(
    model_large_graph.parameters(),
    lr=0.005,
    weight_decay=5e-4
)

# CrossEntropyLoss is standard for multi-class classification
criterion_large_graph = torch.nn.CrossEntropyLoss()


# --------------------------------------------------
# 5. Training loop
# --------------------------------------------------

"""
How mini-batch training works here:

Each batch returned by NeighborLoader is a sampled subgraph.
It includes:
- the target nodes we want to learn from in this batch
- extra neighbour nodes needed to support message passing

Very important:
NeighborLoader places the target nodes at the beginning of the batch.

So:
- batch_data.x contains features for all sampled nodes in the subgraph
- batch_data.edge_index contains the edges for that sampled subgraph
- batch_data.y contains labels for all sampled nodes
- batch_data.batch_size tells us how many nodes at the beginning are
  the actual target nodes for this batch

That is why the loss is computed only on:
out[:batch_data.batch_size]
and
batch_data.y[:batch_data.batch_size]

We ignore the rest because those extra nodes are only there to provide
neighbourhood context for message passing.
"""

def train_large_graph():
    model_large_graph.train()
    total_loss = 0

    for batch_idx, batch_data in enumerate(train_loader_large):
        batch_data = batch_data.to(device)
        """
        The above data can be written as: 
        
        loader_with_index = enumerate(train_loader_large)
        
        for item in loader_with_index:
            batch_idx = item[0]
            batch_data = item[1]
            batch_data = batch_data.to(device)
                
        What has happened? 
        
        enumerate(train_loader_large) produces pairs like:

                (0, first_batch)
                (1, second_batch)
                (2, third_batch)
        
        batch_data would look like this: 
        
        Data(x=[842, 32], edge_index=[2, 3910], y=[842], batch_size=128)
        Which means, 
        This sampled batch has 842 total nodes, each node has 32 features. 
        Out of these 842 total nodes, 128 are the target nodes, the rest are 
        sampled neighbourgs needed for messaging passing. 
        
        The sampled subgraph has 3910 edges stored in the usual edge_index format 
        1 label per sampled node( 842 in total)
        
        The first 128 nodes in this batch are the target node. 
        """
        optimizer_large_graph.zero_grad()

        # Forward pass on the sampled subgraph
        out = model_large_graph(batch_data.x, batch_data.edge_index)

        # Compute loss only for the target nodes
        loss = criterion_large_graph(out[:batch_data.batch_size], batch_data.y[:batch_data.batch_size])

        loss.backward()
        optimizer_large_graph.step()

        # Weight the batch loss by number of target nodes
        total_loss += loss.item() * batch_data.batch_size

    # Average loss over all training target nodes
    return total_loss / int(data_large.train_mask.sum())


# --------------------------------------------------
# 6. Evaluation loop
# --------------------------------------------------

"""
Evaluation is similar to training, but:

- model_large_graph.eval() switches the model to evaluation mode
- dropout is disabled
- we do not calculate gradients
- predictions are made only for the target nodes in each batch

Again, we only measure accuracy on the first batch_size nodes,
because those are the actual validation or test targets.
"""

def evaluate_large_graph(loader):
    model_large_graph.eval()

    correct_predictions = 0
    total_nodes = 0

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)

            out = model_large_graph(batch_data.x, batch_data.edge_index)

            # Predictions only for target nodes
            pred = out[:batch_data.batch_size].argmax(dim=1)

            correct_predictions += (
                pred == batch_data.y[:batch_data.batch_size]
            ).sum().item()

            total_nodes += batch_data.batch_size

    return correct_predictions / total_nodes


# --------------------------------------------------
# 7. Train and validate
# --------------------------------------------------

"""
We train for a few epochs just for demonstration.

At each epoch:
- train on mini-batches from train_loader_large
- evaluate on validation mini-batches from val_loader_large
"""

epochs_large_graph = 10

print("\n--- Training GraphSAGE for Node Classification on Large Graph with NeighborLoader ---")

for epoch in range(1, epochs_large_graph + 1):
    loss = train_large_graph()
    val_acc = evaluate_large_graph(val_loader_large)

    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")


# --------------------------------------------------
# 8. Final test evaluation
# --------------------------------------------------

test_acc_large = evaluate_large_graph(test_loader_large)
print(f"\nTest Accuracy on Large Graph: {test_acc_large:.4f}")


# --------------------------------------------------
# 9. Final notes
# --------------------------------------------------

"""
Key ideas to remember:

1. Full-batch training
   - uses the whole graph at once
   - fine for small graphs
   - often impossible for very large graphs

2. Mini-batch training with NeighborLoader
   - samples small subgraphs around target nodes
   - greatly reduces memory use
   - makes large-graph training practical

3. Why extra nodes appear in each batch
   - the target nodes need neighbours for message passing
   - so the sampled batch contains both:
       a) target nodes
       b) supporting neighbour nodes

4. Why loss uses only the first batch_size nodes
   - NeighborLoader puts the target nodes first
   - only those are supervised targets for the batch
   - the other sampled nodes are only context

5. Meaning of num_neighbors = [10, 5]
   - for a 2-layer GNN:
       first layer samples up to 10 neighbours
       second layer samples up to 5 neighbours

6. Scalability benefit
   - lets GNNs work on graphs much larger than GPU memory would allow
   - this is one reason GraphSAGE is popular for large real-world graphs

7. Inductive capability
   - GraphSAGE learns how to aggregate local neighbourhood information
   - this helps it generalise to unseen nodes better than some older methods
"""