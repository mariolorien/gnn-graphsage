import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


"""
We start by creating our network based on our small train network graph (see .png on the foloder)

We create our list of nodes

The tensor x has shape [num_nodes, num_features].

In this case:
- num_nodes = 5
- num_features = 2

Each row represents a node, and each column represents a feature.

So each node is described by a 2-dimensional feature vector:
[busy_level, has_tube]
"""
# Node features: [busy_level, has_tube]
# Indices: 0=A, 1=B, 2=C, 3=D, 4=E
x = torch.tensor([
    [2, 1],  # A
    [2, 0],  # B
    [1, 0],  # C
    [1, 1],  # D
    [2, 0],  # E
], dtype=torch.float)

# Edge list (directed)
edge_index = torch.tensor([
    [0, 4,  0, 1,  0, 3,  1, 2,  0, 1, 2, 3, 4],
    [4, 0,  1, 0,  3, 0,  2, 1,  0, 1, 2, 3, 4],
], dtype=torch.long)

print("x shape:", x.shape)
print("x:\n", x)
print("\nedge_index shape:", edge_index.shape)
print("edge_index:\n", edge_index)

"""
We define a graph using:

(1) x → node features
(2) edge_index → connectivity

For link prediction, we do NOT predict node labels.
Instead, we predict whether a connection (edge) exists
between two nodes.

So we must construct a dataset of node-pairs:

- Positive edges → edges that exist (label = 1)
- Negative edges → edges that do NOT exist (label = 0)

This converts the graph into a supervised binary classification problem.
"""

# --- Positive edges ---

src, dst = edge_index

"""
We go from: 
(Example)
[
  [0, 1, 2],
  [1, 2, 2]
]

To: 
src = [0, 1, 2]
dst = [1, 2, 2]
"""


mask_no_loops = src != dst

"""
We remove self-loops (edges like i → i), because they do not provide
useful supervision for link prediction.

Example:

src = [0, 1, 2, 3]
dst = [0, 2, 2, 3]

mask_no_loops =
[False, True, False, False]

So we keep only edges where src != dst.
"""

pos_edge_index = edge_index[:, mask_no_loops]

"""
What do we do here? our previous line created a list 
of boolean (True/False) where source was not the same as the destination. 
Something like this: 

[False, True, False, False]

Now we need the actual list. What this line does is getting the whole thing
source and destination and applying mask_no_loops

"""
"""

edge_index[:, mask_no_loops] uses tensor indexing.

The syntax tensor[rows, columns] allows us to select specific parts
of the tensor.

Here:
- ':' means select all rows
- mask_no_loops is a boolean tensor that selects specific columns

So we keep all rows, but only the columns (edges) where the mask is True.

If we used edge_index[:], it would return the entire tensor with no filtering.

This type of indexing works not only with boolean masks,
but also with integer indices.

For example:
- Boolean indexing: select elements based on a condition (our example)
- Integer indexing: select elements by position (Example: edge_index[:, [1, 3]])

This mechanism is general and can be used on any PyTorch tensor,
including node features, predictions, and labels.
"""

print("Positive edges:", pos_edge_index)

"""
pos_edge_index contains all real edges in the graph.

These will be labelled as 1.
"""

# --- Negative edges ---

neg_edge_index = torch.tensor([
    [2, 4, 3, 4],
    [4, 2, 4, 3],
], dtype=torch.long)

"""
neg_edge_index defines negative examples for link prediction.

These are pairs of nodes that are NOT connected in the graph,
and therefore should be labelled as 0.

Example:
(2 → 4), (4 → 2), (3 → 4), (4 → 3)

These edges must not appear in edge_index, otherwise we would
create contradictory supervision.

We include both directions because edges are represented as directed
pairs, even if the underlying graph is conceptually undirected.

Negative edges are essential so that the model learns to distinguish
between existing and non-existing connections.
"""

# --- Labels ---

num_pos = pos_edge_index.size(1)
num_neg = neg_edge_index.size(1)

"""
Why do we use .size(1) in here if this is "just" a count -i.e, a single digit number-? 
Because these are tensors, not just counts, and, in here they are 2D tensors: 

For example, 

pos_edge_index.shape = [2, 8]

Where in the dimension 0 we have the size = 2 (source + destination)
and in dimension 1 we have the number of edges: in this example 8 

We do we count nodes? 

In link prediction, negative edges are created by selecting pairs of nodes
that are not connected in the graph.

So nodes define all possible pairs, while edges represent the pairs that
actually exist.

num_pos and num_neg do not define which edges exist or not;
they simply count how many positive and negative examples we have.

These counts are used to create the label vector y and to ensure that
each edge pair has a corresponding label.

"""
y_pos = torch.ones(num_pos)
y_neg = torch.zeros(num_neg)

y = torch.cat([y_pos, y_neg])

"""
We are not replacing edge_index with 1s and 0s.

Instead, we create a separate dataset for link prediction:

- edge_label_index contains node pairs (both real edges and non-edges)
- y contains the corresponding labels:
    1 for existing edges
    0 for non-existing edges

The order is important because each edge pair in edge_label_index
must match its corresponding label in y.

So position k in edge_label_index corresponds to position k in y.

node i	node j	label
A	     B	     1
C	     D	     1
B	     E	     0
D	     E	     0

y is a 1D tensor containing the labels for each edge pair.

In the next example:
- The first num_pos entries are 1 (existing edges)
- The next num_neg entries are 0 (non-existing edges)

Example:
y = tensor([1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.])

Each position in y corresponds to the same position in edge_label_index.
"""

# --- Combine edges ---

edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

print("edge_label_index:", edge_label_index)
print("y:", y)

"""
edge_label_index defines the node-pairs we want to evaluate.

So the dataset becomes:

x → node features
edge_index → graph structure
edge_label_index → node pairs to classify
y → labels (0 or 1)
"""

# --- Encoder (GNN) ---

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        z = self.conv2(h, edge_index)
        return z

"""
In node classification, the GNN directly outputs final predictions
(logits) for each node, and the loss is computed on those outputs.

In link prediction, the GNN is used ONLY as an encoder to generate
node embeddings.

The actual prediction is made by a separate decoder, which takes two
node embeddings and computes a score (e.g., dot product), followed by
a sigmoid to obtain a probability.

The loss is then computed by comparing these probabilities with the
true labels (1 for existing edges, 0 for non-existing edges).

Therefore, the key difference is that in link prediction the prediction
step happens outside the GNN, while in node classification it happens
inside the GNN.

"""

encoder = GNNEncoder(x.size(1), 8, 4)
"""
The line above does the following. 

the x.size(1) extracts the number of features (2,3,6, whatever is the value)
8 is the hidden-channels 
4 is the out channels. 

Say for the sake of the argument aht the features are 6, then 

then any node would have 6 input features, 
after the first layer it will 8 hidden values, 
after the second layer it might end up with 4 final embedding values 
"""
Z = encoder(x, edge_index)

"""
Here in Z, we save the resulting embeddings. 
At his point each node as its own features and information about its neighbours 

Z.shape = [100, 4] This would be one row per node (100 nodes) x 4 learned numbders for each node. 

tensor([
  [0.2, 1.1, -0.4, 0.7],   # node 0
  [0.9, 0.3,  0.2, 1.5],   # node 1
  [1.2, 0.8, -0.1, 0.4],   # node 2
])
"""
print("Z:", Z)

"""
Z contains node embeddings.

Example:

Z[0] → embedding of node 0
Z[1] → embedding of node 1

These embeddings are what we use to predict links.
"""

#=====Decoder Par=========

"""
Why do we need a decoder?

Because embeddings by themselves are just vectors of numbers.

For example, after the encoder we may have:

Z = [
  [0.2, 1.1, -0.4, 0.7],   # node 0
  [0.9, 0.3,  0.2, 1.5],   # node 1
  [1.2, 0.8, -0.1, 0.4],   # node 2
]

That is useful, but it does not yet directly say:
if the edge exists or does not exist or give 
a probability like say = 0.84

We still need something that reads two node embeddings and turns them into an edge score.

That is the decoder's job.

"""

src_nodes = edge_label_index[0]
dst_nodes = edge_label_index[1]

"""
Here we go from: 

edge_label_index = tensor([
    [0, 1, 2, 3],
    [1, 2, 0, 4]
])

src_nodes = tensor([0, 1, 2, 3])
dst_nodes = tensor([1, 2, 0, 4])

"""
z_src = Z[src_nodes]
z_dst = Z[dst_nodes]

"""
What is going on in here? 
Suppose for the sake of the argument that we want to score: 


    (0, 1), (2, 4), (3, 0)

Then edge_label_index would be:

    edge_label_index = tensor([
        [0, 2, 3],   # source nodes
        [1, 4, 0]    # destination nodes
    ])

So:

    src_nodes = edge_label_index[0]
    dst_nodes = edge_label_index[1]
--------------------------------------------------------
Step 1: extract the source and destination embeddings
--------------------------------------------------------

    z_src = Z[src_nodes]
    z_dst = Z[dst_nodes]

This uses advanced indexing.

So:

    z_src = Z[[0, 2, 3]]

becomes:

    z_src = tensor([
        [1, 2],   # embedding of node 0
        [3, 4],   # embedding of node 2
        [5, 6]    # embedding of node 3
    ])

and:

    z_dst = Z[[1, 4, 0]]

becomes:

    z_dst = tensor([
        [7, 8],   # embedding of node 1
        [9,10],   # embedding of node 4
        [1, 2]    # embedding of node 0
    ])

Important:
the rows are matched by position, not all-against-all.

So the model compares:
- row 0 of z_src with row 0 of z_dst  -> edge (0,1)
- row 1 of z_src with row 1 of z_dst  -> edge (2,4)
- row 2 of z_src with row 2 of z_dst  -> edge (3,0)

--------------------------------------------------------
Step 2: elementwise multiplication
--------------------------------------------------------

    z_src * z_dst

This multiplies the rows pairwise, element by element:

    tensor([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    *
    tensor([
        [7, 8],
        [9,10],
        [1, 2]
    ])
    =
    tensor([
        [1*7,  2*8 ],
        [3*9,  4*10],
        [5*1,  6*2 ]
    ])
    =
    tensor([
        [ 7, 16],
        [27, 40],
        [ 5, 12]
    ])

--------------------------------------------------------
Step 3: sum across each row
--------------------------------------------------------

    scores = (z_src * z_dst).sum(dim=1)

Here, dim=1 means:
sum across the columns of each row.

So:

    tensor([
        [ 7, 16],
        [27, 40],
        [ 5, 12]
    ]).sum(dim=1)

becomes:

    tensor([
        7 + 16,
        27 + 40,
        5 + 12
    ])
    =
    tensor([23, 67, 17])

These are the link scores:

- score for edge (0,1) = 23
- score for edge (2,4) = 67
- score for edge (3,0) = 17

--------------------------------------------------------
What this operation is doing mathematically
--------------------------------------------------------

This is a dot product between each source embedding and its
corresponding destination embedding.

For each candidate edge (i, j), the score is:

    score(i, j) = z_i · z_j

That means:
multiply matching embedding components and then add them.

Example for edge (0,1):

    z_0 = [1, 2]
    z_1 = [7, 8]

    score = 1*7 + 2*8 = 7 + 16 = 23

--------------------------------------------------------
Why this is useful for link prediction
--------------------------------------------------------

The encoder gives one embedding per node, but link prediction is about pairs of nodes.

So after obtaining Z, we need a way to compare:
- source node embedding
- destination node embedding

This decoder uses a simple dot product:
higher score -> embeddings are more compatible
lower score  -> embeddings are less compatible

So the code:

    z_src = Z[src_nodes]
    z_dst = Z[dst_nodes]
    scores = (z_src * z_dst).sum(dim=1)

means:

"Take the embedding of each source node and the embedding of the
matching destination node, compare them with a dot product, and
return one score per candidate edge."

"""

scores = (z_src * z_dst).sum(dim=1)

"""
We compute similarity between node pairs.

Step 1: element-wise multiplication
Step 2: sum → dot product

Example:

z_i = [1,2]
z_j = [3,4]

z_i * z_j = [3,8]
sum = 11

This gives a score for how similar the nodes are.
"""

probs = torch.sigmoid(scores)


"""
Convert raw link scores into probabilities
------------------------------------------

After computing:

    scores = (z_src * z_dst).sum(dim=1)

we have one raw score for each candidate edge.

Example:

    scores = tensor([23, 67, 17])

These are not probabilities yet.
They are just raw numbers produced by the dot product decoder.

The problem is that raw scores:
- can be very large
- can be negative
- are not limited to the range 0 to 1

For link prediction, we usually want something we can interpret as:

    "How likely is this edge to exist?"

That is why we apply:

    probs = torch.sigmoid(scores)

The sigmoid function takes any real number and squashes it into
the interval (0, 1).

Mathematically:

    sigmoid(x) = 1 / (1 + e^(-x))

So:

- very large positive scores become values close to 1
- very large negative scores become values close to 0
- a score of 0 becomes 0.5

This is useful because:
- probability near 1 -> model thinks edge is likely
- probability near 0 -> model thinks edge is unlikely
- probability around 0.5 -> model is uncertain

-])


The decoder gives a compatibility score, but not a probability.

For example:
- score = 8.3
- score = -1.7

These values tell us something about how compatible the embeddings are,
but they are harder to interpret directly.

By applying sigmoid, we turn them into something easier to use:

- 8.3   -> near 1.0
- -1.7  -> near 0.15

Now we can say:
- "This edge is likely"
- "This edge is unlikely"

--------------------------------------------------------
Connection to training
--------------------------------------------------------

In link prediction, the true labels are usually:

    1 = edge exists
    0 = edge does not exist

So probabilities from sigmoid are convenient because they can be compared
to these binary labels.

Example:
- predicted probability = 0.92, true label = 1
- predicted probability = 0.08, true label = 0

This is why sigmoid is commonly used before a binary classification loss
or when inspecting predictions.


"""

# =====Loss====

loss = F.binary_cross_entropy(probs, y)

"""
Binary Cross Entropy compares:

predicted probabilities vs true labels

Example:


Suppose:

    probs = tensor([0.90, 0.20, 0.75, 0.10])
    y     = tensor([1.,   0.,   1.,   0. ])

Now calculate each edge's loss.

1) First edge:
   true label = 1
   predicted probability = 0.90

   loss_1 = -log(0.90)
          ≈ 0.105

2) Second edge:
   true label = 0
   predicted probability = 0.20

   loss_2 = -log(1 - 0.20)
          = -log(0.80)
          ≈ 0.223

3) Third edge:
   true label = 1
   predicted probability = 0.75

   loss_3 = -log(0.75)
          ≈ 0.288

4) Fourth edge:
   true label = 0
   predicted probability = 0.10

   loss_4 = -log(1 - 0.10)
          = -log(0.90)
          ≈ 0.105

So the per-edge losses are approximately:

    [0.105, 0.223, 0.288, 0.105]

By default, F.binary_cross_entropy takes the mean, so:

    total_loss ≈ (0.105 + 0.223 + 0.288 + 0.105) / 4
               ≈ 0.180

So:

    loss = 0.180
    
--------------------------------------------------------
What this means
--------------------------------------------------------

If the loss is small:
- the predicted probabilities match the true labels well

If the loss is large:
- the predictions are poor

So the goal of training is to make this loss smaller over time.

--------------------------------------------------------
Why probs and y must have matching shapes
--------------------------------------------------------

Each predicted probability must be matched with the correct true label.

So if:

    probs.shape = [num_candidate_edges]

then:

    y.shape = [num_candidate_edges]

must match exactly.

Example:

    probs = tensor([0.90, 0.20, 0.75, 0.10])
    y     = tensor([1.,   0.,   1.,   0. ])

Position by position:
- 0.90 is compared with 1
- 0.20 is compared with 0
- 0.75 is compared with 1
- 0.10 is compared with 0

--------------------------------------------------------
In one sentence
--------------------------------------------------------

    loss = F.binary_cross_entropy(probs, y)

measures how far the predicted edge probabilities are from the true
edge labels, and gives the model a single number to minimize during
training.
"""


print("Loss:", loss.item())

# --- Training loop ---

optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

for epoch in range(1, 201):
    encoder.train()
    optimizer.zero_grad()

    Z = encoder(x, edge_index)

    z_src = Z[src_nodes]
    z_dst = Z[dst_nodes]

    scores = (z_src * z_dst).sum(dim=1)
    probs = torch.sigmoid(scores)

    loss = F.binary_cross_entropy(probs, y)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        pos_mean = probs[:num_pos].mean().item()
        neg_mean = probs[num_pos:].mean().item()
        print(f"epoch {epoch} loss={loss.item():.4f} pos={pos_mean:.3f} neg={neg_mean:.3f}")

"""

Training flow for link prediction
---------------------------------

1. Loop through the training process for many epochs:
       for epoch in range(1, 201)

2. Put the encoder in training mode:
       encoder.train()

3. Clear old gradients from the previous epoch:
       optimizer.zero_grad()

4. Pass node features and graph structure through the encoder:
       Z = encoder(x, edge_index)

5. Extract the embeddings for the source and destination nodes
   of each candidate edge:
       z_src = Z[src_nodes]
       z_dst = Z[dst_nodes]

6. Compute one raw score for each candidate edge using the
   dot product of source and destination embeddings:
       scores = (z_src * z_dst).sum(dim=1)

7. Convert raw scores into probabilities between 0 and 1:
       probs = torch.sigmoid(scores)

8. Compare predicted probabilities with the true labels:
       loss = F.binary_cross_entropy(probs, y)

9. Backpropagate the loss to compute gradients:
       loss.backward()

10. Update the encoder parameters using the optimizer:
        optimizer.step()

11. Every 20 epochs, print:
    - current epoch
    - current loss
    - average predicted probability for positive edges
    - average predicted probability for negative edges

        pos_mean = probs[:num_pos].mean().item()
        neg_mean = probs[num_pos:].mean().item()

Overall flow:
raw node features + graph structure
-> encoder
-> node embeddings Z
-> extract source/destination embeddings
-> compute edge scores
-> convert to probabilities
-> compute loss
-> backpropagation
-> update weights

"""

#====Prediction====


"""
During training, the model learned:

* what node embeddings should look like
* how connected pairs tend to score
* how non-connected pairs tend to score

Then at prediction time, we ask:

“For this pair of nodes, what probability does the model assign to an edge?”


"""

def predict_link_prob(i, j):
    encoder.eval() # We put the model in evaluation mode. 
    with torch.no_grad():  # We ask not to store gradients: we are predicting, not training
        Z = encoder(x, edge_index) # recompute the node embeddings for the whole graph
        score = (Z[i] * Z[j]).sum() # Gives one raw score for each pair
        prob = torch.sigmoid(score) #Converts the raw score into a value between 0 and 1 (our probability)
    return float(prob)

"""
GNN for link prediction usually does not directly output:

“Here are the top 5 new edges in the graph.”

Instead, it usually works like this:
 * build node embeddings
 * choose node pairs to inspect
 * score those pairs 
  * rank or threshold the results

So the model is more like a scoring machine than a fully automatic suggestion machine
"""

candidates = [(0,2), (1,4), (2,3), (3,4), (2,4)]

""""
Why do we suggest candidates? 
Is the model suggesting or not?

In this script, we provide the candidates.
The model only says how likely each one is.

So this code is doing:

This is normal.
Because in many graphs, the number of possible pairs is huge.

If you have N nodes, possible node pairs are roughly N^(2) or N(N-1)/2 if undirected. 

So with many nodes, checking every pair can be expensive.

That is why many pipelines do this in two stages:
1) Generate promising candidate pairs
2) Let the model score them

"""
print("\nPredictions:")
for (i, j) in candidates:
    print(f"{i}->{j}: {predict_link_prob(i,j):.4f}")