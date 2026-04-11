import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.data import Data  # For type hinting


"""
Load the Cora dataset.

This is the same dataset-loading process as in the GCN example.
The graph data structure is consistent in PyTorch Geometric,
so the only main change later will be the model architecture.
"""
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]

print(f"Dataset: {dataset.name}")
print(f"Graph object: {data}")
print(f"Number of training nodes: {data.train_mask.sum()}")


# Move data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)


"""
Define the GAT model architecture.

What are attention heads? 
Attention heads are multiple separate attention mechanisms running in parallel inside the same GAT layer.

A good way to picture it is the following: 

A node looks at its neighbours and asks:

“Which of these neighbours should I pay more attention to?”

With 1 attention head, there is just one way of answering that question.

With 8 attention heads, there are 8 different sets of attention weights, 
so the node gets 8 different opinions about which neighbours matter most.
This also means that the layer will compute attention 8 times in parallel, 
and each head produces its own output features

PyG usually concatenates those outputs together

Here we replace GCNConv with GATConv.

Key ideas:
- GATConv applies attention over neighbours.
- heads specifies how many attention heads to use.
- dropout is applied to attention coefficients and features for regularisation.
- when heads > 1, the outputs of the heads are concatenated,
  so the output size becomes hidden_channels * heads.
- the final layer usually uses heads=1 so that the output dimension
  matches the number of classes.
- ELU is commonly used as the activation function in GATs.
"""


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()

        # GATConv implements the Graph Attention Network layer
        # heads: number of attention heads
        # dropout: dropout probability applied to the attention coefficients
        self.conv1 = GATConv(in_channels,hidden_channels,heads=heads,dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels,heads=1, dropout=0.6)
        """
        For the second layer, input features are multiplied by heads
        because the outputs of multiple heads are concatenated.

        The last layer typically uses heads=1 to produce the final
        output dimension.
        "
        Why we multiply heads * hidden_channels -i.e imbeddings-?
        
        Suppose one node goes through a GAT layer with:

        hidden_channels = 2
        heads = 3

        Then:

        Head 1 might output: [0.4, 1.2]
        Head 2 might output: [2.1, -0.3]
        Head 3 might output: [0.7, 0.9]

        After concatenation, the node's new representation becomes:

        [0.4, 1.2, 2.1, -0.3, 0.7, 0.9]
        
        That vector has length:

        2 * 3 = 6
        """

    def forward(self, x, edge_index):
        # First GAT layer
        x = F.dropout(x, p=0.6, training=self.training)  
        # Dropout on input features to randomly hide some features during training
        # we also force the model not to depend too much on specific inputs and make it more robust 
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # ELU is often used with GATs

        # Second GAT layer
        x = F.dropout(x, p=0.6, training=self.training)  # Dropout on hidden features 
        x = self.conv2(x, edge_index)

        return x

        """
        How many dropouts we have? Are all the same? 
        In training, we have 4 dropout events in total in this 2-layer model:

        1) Feature dropout before conv1
        2) Attention dropout inside conv1
        3) Feature dropout before conv2
        4) Attention dropout inside conv2
        
        In short we have "2" types of dropout: one hides what information we have 
        and the other hides who we listen to.        
        
        """

# Instantiate the model
# For single-head attention (heads=1)
model_gat_single_head = GAT(
    in_channels=dataset.num_node_features,
    hidden_channels=8,  # Common hidden dimension for GAT
    out_channels=dataset.num_classes,
    heads=1
)

print(f"Model architecture (Single-Head GAT):\n{model_gat_single_head}")


"""
Instantiate the model with multi-head attention (e.g. 8 heads).

The output of the first layer will be:
hidden_channels * heads = 8 * 8 = 64
"""
model_gat_multi_head = GAT(
    in_channels=dataset.num_node_features,
    hidden_channels=8,
    out_channels=dataset.num_classes,
    heads=8  # Using 8 attention heads
)

print(f"\nModel architecture (Multi-Head GAT):\n{model_gat_multi_head}")


# Move model to device
model_gat_single_head = model_gat_single_head.to(device)
model_gat_multi_head = model_gat_multi_head.to(device)


"""
GATConv explanation:

- GATConv:
    This layer implements the attention mechanism.

- heads:
    Specifies the number of attention heads.

- dropout:
    This dropout is applied to the normalized attention coefficients,
    which helps regularize the attention mechanism itself.

- hidden_channels * heads:
    When heads > 1, PyG's GATConv concatenates the outputs of
    individual attention heads. Therefore, the input channels for
    the next layer must be hidden_channels * heads.

- Last layer heads=1:
    The final layer typically uses heads=1 to produce a single
    output feature vector per node, matching the out_channels
    (number of classes).

- F.elu(x):
    The Exponential Linear Unit (ELU) activation function is often
    used with GATs, as suggested in the original paper, due to its
    smooth negative part which can help with gradient flow.

- Dropout on features:
    Dropout is applied to the input features before the first GATConv
    and after the ELU activation in the hidden layer. This is a common
    practice to prevent overfitting.
"""


"""
Step 3: Define loss function and optimizer.

These remain standard for node classification,
just as in the GCN example.
"""

# For single-head GAT
optimizer_single_head = torch.optim.Adam(
    model_gat_single_head.parameters(),
    lr=0.005,
    weight_decay=5e-4
)

criterion = torch.nn.CrossEntropyLoss()

# For multi-head GAT
optimizer_multi_head = torch.optim.Adam(
    model_gat_multi_head.parameters(),
    lr=0.005,
    weight_decay=5e-4
)


"""
Step 4: Training loop for GAT.

The training loop is structurally identical to the GCN version.
Only the model itself has changed.
"""
def train_gat(model, optimizer, data_obj):
    model.train()
    optimizer.zero_grad()

    out = model.forward(data_obj.x, data_obj.edge_index)
    """
    here we apply our forward () method to our model. 
    What goes in?

    data_obj.x = node feature matrix
    data_obj.edge_index = graph connections

    For Cora (that we are now using):

    data_obj.x.shape is roughly [2708, 1433]
    data_obj.edge_index.shape is roughly [2, num_edges]

    What comes out?

    [number_of_nodes, number_of_classes]
    [2708, 7]
    """
    
    loss = criterion(out[data_obj.train_mask], data_obj.y[data_obj.train_mask])
    """
    Here we say: only look at the training nodes, compare the model's predicitons 
    with the true labels, and compute the loss. 
 
    Tiny fake example

    Suppose we have 5 nodes and 3 classes.

    Model output for all nodes:
    out =
    [
    [2.0, 0.5, -1.0],   # node 0
    [0.2, 1.8, 0.1],    # node 1
    [1.1, 0.4, 0.3],    # node 2
    [0.0, 0.2, 2.5],    # node 3
    [1.7, 0.1, 0.2]     # node 4
    ]

    True labels:
    y = [0, 1, 0, 2, 1]

    Suppose only nodes 0, 2, and 4 are training nodes:
    train_mask = [True, False, True, False, True]

    Then:

    out[train_mask]

    becomes:

    [
    [2.0, 0.5, -1.0],   # node 0
    [1.1, 0.4, 0.3],    # node 2
    [1.7, 0.1, 0.2]     # node 4
    ]

    And:

    y[train_mask]

    becomes:

    [0, 0, 1]

    Then CrossEntropyLoss compares those predictions against those true labels.
    
    The largest logit gives the predicted class, but training uses the full logit 
    vector so the loss can measure not only whether the prediction was right, 
    but also how confident or uncertain it was.
    
    What it counts is the postion of the largest value in our vector and then 
    CrossEntropyLoss takes it and the true class labels and compute the loss. 
    
    When we have more than one label to classify -Multilabel classification- we use 
    BCEWithLogitsLoss
    because each class is treated more like an independent yes/no decision.
    
    """ 
    
    """
    IMPORTANT 
    For node classification, we often use either CrossEntropyLoss
    (for single-class classification) or BCEWithLogitsLoss
    (for multilabel classification).

    In both cases, the loss is computed from raw logits and true labels,
    not from final predicted classes.

    Predictions such as argmax or sigmoid-thresholded outputs are mainly
    used for evaluation and interpretation, rather than for the loss itself.
    """
    
    loss.backward()
    optimizer.step()

    return loss.item()


# Training for Single-Head GAT
epochs = 200

print("\n--- Training Single-Head GAT ---")
for epoch in range(1, epochs + 1):
    loss = train_gat(model_gat_single_head, optimizer_single_head, data)

    if epoch % 50 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")


# Training for Multi-Head GAT (using the same data, but a different model and optimizer)
print("\n--- Training Multi-Head GAT ---")
for epoch in range(1, epochs + 1):
    loss = train_gat(model_gat_multi_head, optimizer_multi_head, data)

    if epoch % 50 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")


"""
Step 5: Evaluation for GAT.

The evaluation function remains the same because it only depends
on the model outputs and the data masks.
"""
def evaluate_gat(model, data_obj):
    model.eval()

    with torch.no_grad():
        out = model(data_obj.x, data_obj.edge_index)
        pred = out.argmax(dim=1)
        """
        We do not use softmax there because argmax of 
        the raw logits gives the same predicted class 
        as argmax after softmax, so softmax would be unnecessary extra work
        
        argmax(logits) == argmax(softmax(logits))
        """
        correct = (pred[data_obj.test_mask] == data_obj.y[data_obj.test_mask]).sum()
        acc = int(correct) / int(data_obj.test_mask.sum())

    return acc


# Evaluate Single-Head GAT
test_acc_single_head = evaluate_gat(model_gat_single_head, data)
print(f"\nSingle-Head GAT Test Accuracy: {test_acc_single_head:.4f}")

# Evaluate Multi-Head GAT
test_acc_multi_head = evaluate_gat(model_gat_multi_head, data)
print(f"Multi-Head GAT Test Accuracy: {test_acc_multi_head:.4f}")