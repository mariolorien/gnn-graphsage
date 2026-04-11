"""
Foundational GNNs (GCN, GraphSAGE, GAT) operate on homgeneous graphs, where all the nodes are of the same type 
and all edges represent a single type of relationship.
The problem is that many real world systems are more complex. Data often involves entities of different categories
interacing through varios kinds or relationships, and these evolve over time.

Consider a typical e-commerce platform: 

* NODES: Users, Products, Categories, Brands 
* EDGES: 
         - User buys Product 
         - User views Product 
         - Product belongs to Category 
         - Product produced by Brand 
         - User friends with User 

Applying standard GNNs to heterogeneous graphs directly presents several challenges: 
1) Feature incompatibility: Nodes of different types often have different feature dimensions 
2) Semantic differences: Differen edges represent distinct relationships. A simple aggregation across all edges 
would mix semantically different information. 
3) Aggregation acroos types: How should information "flow" from a "Product" node to a "User" node via a "buys" edge, 
compared to a "views" edge? 
4) Meta Path Information: Complex relationships can be described by "meta-paths" (eg. User buys Product bought by User). 
Capturing these orders is powerful but challenging

"""
"""
R-GCNs are an extension of GCNs to handle multiple relation types in heterogeneous graphs. 
The core idea is to use type-specific weight matrices for each relation. 

A normal GCN says something like:

“Take neighbour information, average/aggregate it, transform it, and update the node.”

An R-GCN says:

“Group neighbours by relation type first, use a different transformation for each relation, then combine them.”

So instead of one shared weight matrix for all neighbours, R-GCN uses one weight matrix per relation type.
"""

"""
Suppose node i is a person, and the graph has two relation types:
  friend 
  works_with

Suppose the neighbours are:
  Alice via friend
  Bob via friend
  Carol via works_with
 
Then the update is roughly:

new embedding of i= W_friend(Alice+Bob)+W_works_with(Carol)+W_0(current embedding of i)

Then apply activation.

The important thing is that Alice and Bob are processed differently from Carol, because the relation type is different.

"""

"""
MAIN CHALLENGE

If there are many relation types, then having one full weight matrix per relation becomes expensive.
Why? Because if we have: 100 relation types and each relation has its own matrix, then the number of parameters can explode.

That is why the original R-GCN paper introduced parameter sharing tricks, like BASIS DECOMPOSTION AND BLOCK DIAGONAL DECOMPOSTION
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear  # HGTConv for heterogeneous graphs


"""
1. Create a conceptual HeteroData object

We build a tiny heterogeneous graph for a movie recommendation scenario.

Node types:
- user: 
- movie: 
- genre

Edge types:
- user --rates--> movie
- movie --has_genre--> genre

This is just a conceptual example to show how heterogeneous graphs
are represented and how a heterogeneous GNN can operate on them.
"""

data = HeteroData()

"""
WHAT IS HETERODATA? 

HeteroData is PyG's container for a heterogeneous graph.

That means the graph can have:

    * different node types
    * different edge types
    *different feature sizes for each node type

Unlike a normal homogeneous graph, we do not have just one:
    x
    edge_index

Instead we usually have:

    one x per node type
    one edge_index per relation type
"""


# Define node features for 'user', 'movie', and 'genre'

# Users: 3 nodes, 16 features each (e.g. initial user embeddings)
data['user'].x = torch.randn(3, 16)

"""
WHAT DATA['USER].X WOULD ACTUALLY LOOK LIKE? 

data['user'].x =
tensor([
    [ 0.42, -1.10,  0.33,  1.27, -0.88,  0.51,  0.09, -0.45,
      1.12, -0.67,  0.28,  0.73, -1.31,  0.04,  0.95, -0.22],

    [-0.18,  0.77, -1.42,  0.36,  0.11, -0.59,  1.49,  0.24,
     -0.73,  0.88, -0.05, -1.20,  0.63,  0.17, -0.91,  0.40],

    [ 1.03, -0.26,  0.58, -0.94,  0.67,  1.15, -0.33,  0.72,
     -1.08,  0.19,  0.44, -0.61,  0.29, -1.14,  0.53,  0.81]
])

Same for the next two lines
"""
# Movies: 4 nodes, 32 features each (e.g. initial movie embeddings)
data['movie'].x = torch.randn(4, 32)

# Genres: 2 nodes, 8 features each (e.g. initial genre embeddings)
data['genre'].x = torch.randn(2, 8)


# Define edge indices for different relation types

# User rates movie (user_id, movie_id)
# User rates movie
data['user', 'rates', 'movie'].edge_index = torch.tensor([
    [0, 1, 2, 0],
    [0, 1, 2, 3],
], dtype=torch.long)

"""
In here:

data['user', 'rates', 'movie'] means:

source node type = 'user'
edge / relation type = 'rates'

target node type = 'movie'

user 0 ---rates---> movie 0
user 1 ---rates---> movie 1
user 2 ---rates---> movie 2
user 0 ---rates---> movie 3

This labelling 'rates' is not banal. For example different labels like 

('user', 'rates', 'movie')
('user', 'likes', 'movie')
('user', 'follows', 'user') may each get different learned transformations.

So 'rates' is not just decoration — it is a key identifier for the relation type.
In a plain HeteroData object, at the storage level, it is mainly a label naming that edge set.
It helps organise the graph.

In the model itself, it becomes more important, because layers like HGTConv use the full metadata:

data.metadata()

which includes relation types like 'rates'.

That allows the model to know:

    * which message-passing channels exist
    * which edge set belongs to which relation
    * potentially which parameters to use for each relation type

Summarising: 

'rates' does not carry numeric edge values by itself.
It gives the edge set its semantic identity as a relation type, 
and heterogeneous GNNs can use that identity to process different relations differently.
"""

# Reverse: movie rated by user
data['movie', 'rev_rates', 'user'].edge_index = torch.tensor([
    [0, 1, 2, 3],
    [0, 1, 2, 0],
], dtype=torch.long)

# Movie has genre
data['movie', 'has_genre', 'genre'].edge_index = torch.tensor([
    [0, 1, 2, 3],
    [0, 0, 1, 1],
], dtype=torch.long)

# Reverse: genre belongs to movie
data['genre', 'rev_has_genre', 'movie'].edge_index = torch.tensor([
    [0, 0, 1, 1],
    [0, 1, 2, 3],
], dtype=torch.long)

"""
Why we need the reverse? 
You need the reverse edges because message passing follows edge direction, 
and without reverse edges some node types never receive messages and cannot be updated.

The reverse edge is mainly a computational device for message passing.

It says:
“Let information travel back along this connection too.”

So the semantic meaning of the reverse is often just:
“this node is connected back to that one for learning purposes”
It is not necessarily a real-world statement
"""


"""
The graph can support different learning tasks.

Because the relation ('user', 'rates', 'movie') represents user-item
interactions, the most natural real-world task is often link prediction:
predict whether a user should be connected to a movie.

However, link prediction needs extra machinery such as positive/negative
edge sampling and an edge-scoring step.

To keep the teaching example simpler, we instead assign labels to user
nodes and treat the problem as node classification on users.
"""

data['user'].y = torch.tensor([0, 1, 0], dtype=torch.long)  # Example labels for users
"""
Because we have 3 nodes to label in the line above 
we pretened that we know the real classes: 
It is like saying:

“Pretend we know user 0 belongs to class 0”
“Pretend we know user 1 belongs to class 1”
“Pretend we know user 2 belongs to class 0”

This is what the line above does. 
"""

print(data)
print(f"User features shape: {data['user'].x.shape}")
print(f"Movie features shape: {data['movie'].x.shape}")
print(f"User-rates-movie edges: {data['user', 'rates', 'movie'].edge_index.shape}")
print(f"Movie-has_genre-genre edges: {data['movie', 'has_genre', 'genre'].edge_index.shape}")


# Move data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)


"""
2. Define the Heterogeneous GNN model using HGTConv

HGTConv requires:
- metadata about node types
- metadata about edge types

A key issue in heterogeneous graphs is that different node types can
have different feature dimensions:
- user: 16
- movie: 32
- genre: 8

So before applying HGTConv, we first project all node types into a
common hidden dimension using separate linear layers.
"""


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads):
        super().__init__()

        # HGTConv requires a dictionary of node types and their input dimensions
        # and a list of all edge types (metadata)

        """
        We need initial linear layers to project different node types
        into a common embedding space before the HGTConv layer.

        This is crucial because different node types have different
        input dimensions.
        """
        self.user_lin = Linear(data['user'].x.shape[1], hidden_channels)
        self.movie_lin = Linear(data['movie'].x.shape[1], hidden_channels)
        self.genre_lin = Linear(data['genre'].x.shape[1], hidden_channels)

        # HGTConv takes metadata about node and edge types
        # We pass the initial projected features to HGTConv
        self.conv1 = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
        self.conv2 = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)  # Another HGT layer

        # Output layer for 'user' nodes (e.g. for user classification)
        self.user_out = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Apply initial linear transformations to project node features into common space
        x_dict['user'] = self.user_lin(x_dict['user'])
        x_dict['movie'] = self.movie_lin(x_dict['movie'])
        x_dict['genre'] = self.genre_lin(x_dict['genre'])

        # Pass through HGTConv layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply ReLU to all node types

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # For node classification on 'user' nodes, apply final linear layer
        return self.user_out(x_dict['user'])


# Instantiate the model
model_hetero = HeteroGNN(
    hidden_channels=64,
    out_channels=data['user'].y.max().item() + 1 if hasattr(data['user'], 'y') else 2,
    num_heads=2
)

model_hetero = model_hetero.to(device)

print(f"\nModel architecture (HeteroGNN):\n{model_hetero}")


"""
3. Define loss function and optimizer
"""

optimizer_hetero = torch.optim.Adam(model_hetero.parameters(), lr=0.01, weight_decay=5e-4)
criterion_hetero = torch.nn.CrossEntropyLoss()


"""
4. Training loop for heterogeneous GNN

The forward pass is slightly different from homogeneous GNNs because
we pass:
- a dictionary of node features
- a dictionary of edge indices

The model outputs logits only for 'user' nodes because this example
assumes a user-node classification task.
"""


def train_hetero():
    model_hetero.train()
    optimizer_hetero.zero_grad()

    # Forward pass: pass the dictionary of node features and edge indices
    out = model_hetero(data.x_dict, data.edge_index_dict)

    # Calculate loss only on 'user' nodes (assuming user classification)
    # For a real dataset, you'd usually have a train mask for 'user' nodes
    loss = criterion_hetero(out, data['user'].y)  # Assuming all users are training nodes for simplicity

    loss.backward()
    optimizer_hetero.step()

    return loss.item()


# Training loop
epochs_hetero = 50  # Reduced for demonstration

print("\n--- Training Heterogeneous GNN ---")

for epoch in range(1, epochs_hetero + 1):
    loss = train_hetero()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")


"""
5. Evaluation (conceptual, since we do not have train/val/test masks here)

In a real scenario, you would define an evaluate_hetero() function
similar to previous examples, but using proper masks for the relevant
node type.

Since this toy example trains on all user nodes, there is no meaningful
test accuracy to report.
"""

print("\nEvaluation for Heterogeneous GNN would typically involve specific metrics for each task type")
print("(classification accuracy, movie recommendation metrics, etc.).")
print("For this simple example, we only trained on all user nodes (no split),")
print("so a test accuracy cannot be meaningfully computed without proper data masks.")