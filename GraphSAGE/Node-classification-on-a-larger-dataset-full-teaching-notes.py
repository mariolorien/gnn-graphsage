import torch
import torch.nn.functional as F

from torch_geometric.datasets import PPI
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score


train_dataset = PPI(root="./data/PPI", split="train")
"""
Introduction to the PPI dataset
-------------------------------

The PPI dataset stands for Protein-Protein Interaction dataset.

It is a node classification dataset in which:

- each node represents a protein
- each edge represents an interaction between two proteins
- each node has a feature vector describing biological properties
- each node also has one or more labels to predict

Important:
this dataset is not a single giant graph.
Instead, it is a collection of separate graphs.

In the PyTorch Geometric version, the dataset is split into:
- 20 training graphs
- 2 validation graphs
- 2 test graphs

Each node has:
- 50 input features
- 121 possible labels (proteins can have more than one lable each )

x = tensor([
    [0.2, 1.1, 0.0, ..., 0.7],   # node 0 -> 50 features
    [1.5, 0.3, 0.8, ..., 0.1],   # node 1 -> 50 features
    [0.4, 0.9, 1.2, ..., 0.6],   # node 2 -> 50 features
    [1.0, 0.0, 0.5, ..., 1.3],   # node 3 -> 50 features
])


Because a protein can belong to more than one label at the same time,
this is a multi-label node classification task.

Why this dataset is useful:
it is commonly used to test inductive learning, where the model learns
patterns on some graphs and is then evaluated on completely unseen graphs.

So the task is:

given the node features and the interaction structure of each graph,
predict the correct labels for each protein node.
"""

val_dataset = PPI(root="./data/PPI", split="val")
"""
Load the validation split.

This is used during training to check how well the model is generalising
to unseen graphs without touching the test set.
"""

test_dataset = PPI(root="./data/PPI", split="test")
"""
Load the test split.

This is used only at the end for the final evaluation.
"""

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of validation graphs: {len(val_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")

data = train_dataset[0]
"""
Take the first graph from the training dataset.

This lets us inspect one sample graph and see what kind of object
the dataset returns.
"""

print(data)


train_loader = DataLoader(train_dataset, batch_size=2)
"""
Create a DataLoader for the training graphs.

batch_size=2 means two graphs are grouped together at a time.
PyG merges them into one larger disconnected graph batch.

In simple words, this means: 

take 2 graphs at a time: make 1 batch out of them

The total number of batches depends on how many graphs are in train_dataset.

So if:

len(train_dataset) = 10

then with:

batch_size=2

we get:

batch 1 = graphs 0 and 1
batch 2 = graphs 2 and 3
batch 3 = graphs 4 and 5
batch 4 = graphs 6 and 7
batch 5 = graphs 8 and 9

5 batches of 2.

In our case for PPI specifically, trainins set has 20 graphs, hence 
we get 10 batches per epoch, meaning that on one epoch it goes 
theough all 20 training graphs once. 
"""

val_loader = DataLoader(val_dataset, batch_size=2)
"""
Validation DataLoader.
Used when computing validation performance after training epochs.
"""

test_loader = DataLoader(test_dataset, batch_size=2)
"""
Test DataLoader.
Used for the final model evaluation.
"""
"""
What DataLoader does? 

For graph datasets, PyG's DataLoader does something special:
when it takes 2 graphs, it combines them into one big disconnected graph batch.

That means:

nodes from graph A stay connected only to graph A
nodes from graph B stay connected only to graph B
no fake edges are added between the two graphs

So the model can process both graphs in one forward pass.

"""
print(f"Number of node features: {train_dataset.num_node_features}")
"""
This tells us how many input features each node has.
In this dataset, each protein is described by a vector of numerical features.
"""

print(f"Number of classes: {train_dataset.num_classes}")
"""
This tells us how many output labels/classes there are.

Because this is multi-label classification, a node can belong to more
than one class at the same time.
"""


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for node classification.

    It has:
    - one GraphSAGE convolution from input to hidden space
    - ReLU activation
    - dropout during training
    - one GraphSAGE convolution from hidden space to output space

    Output:
    one logit per class for each node
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        """
        Initialise the neural network module and define the layers.
        """

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        """
        First GraphSAGE layer.

        Input:
        one feature vector per node of size in_channels
        Output:
        one hidden representation per node of size hidden_channels
        """

        self.conv2 = SAGEConv(hidden_channels, out_channels)
        """
        Second GraphSAGE layer.

        Input:
        hidden node representations

        Output:
        one output vector per node of size out_channels
        """
        
        """
        What is the difference between using GCNConv and using  SAGEConv? 
        
        GCN: “blend me and my neighbours together first, using normalization”
        GraphSAGE: “make one summary of my neighbours, keep my own features separate, then combine the two”
  
        CHECK THE MARKDWON FILE  GNCCONV VS GRAPHSAGE EXPLANING THE DIFFERENCES BETWEEN THE BOTH OF THEM 
        
        """

    def forward(self, x, edge_index):
        """
        Forward pass of the model.

        x:
        node feature matrix of shape [num_nodes, num_node_features]

        edge_index:
        graph connectivity in COO format with shape [2, num_edges]
        """

        x = self.conv1(x, edge_index)
        """
        Apply the first GraphSAGE convolution.

        Each node updates its representation by combining:
        - its own features
        - information aggregated from its neighbours
        """

        x = F.relu(x)
        """
        Apply ReLU non-linearity.

        This helps the model learn non-linear patterns instead of behaving
        like a purely linear transformation.
        """

        x = F.dropout(x, p=0.5, training=self.training)
        """
        Randomly drop some values during training.

        This helps reduce overfitting.
        The dropout is active only when self.training is True.
        """

        x = self.conv2(x, edge_index)
        """
        Apply the second GraphSAGE convolution.

        This produces the final output logits for each node.
        """

        return x
        """
        Return the output logits.

        In this node classification task, the output shape is:

            [num_nodes, num_classes]

        Each row corresponds to one node.
        Each column corresponds to one class.
        """


model_node_cls = GraphSAGE(
    in_channels=train_dataset.num_node_features,
    hidden_channels=128,
    out_channels=train_dataset.num_classes,
)
"""
Instantiate the GraphSAGE model.

in_channels:
number of input features per node

hidden_channels:
size of the hidden representation chosen by us

out_channels:
number of classes to predict per node

So the flow is:

input node features
-> GraphSAGE hidden layer
-> GraphSAGE output layer
-> one logit per class for each node
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Choose GPU if available, otherwise CPU.
"""

model_node_cls = model_node_cls.to(device)
"""
Move the model parameters to the selected device.
"""


optimizer_node_cls = torch.optim.Adam(
    model_node_cls.parameters(), lr=0.005, weight_decay=5e-4
)
"""
Create the optimiser.

Adam updates the model weights during training.

lr=0.005:
the learning rate, which controls the step size of parameter updates

weight_decay=5e-4:
L2 regularisation, used to discourage overly large weights
"""

criterion_node_cls = torch.nn.BCEWithLogitsLoss()
"""
Loss function for multi-label classification.
WE USE CROSSENTROPY LOSS WHEN THERE IS ONLY ONE CORRECT ANSWER (CAT, DOG, BIRD)
WE USE BCEWITHLOGITSLOSS WHEN EACH EXMAPLE CAN BELONG TO MULITPLE CLASSES AT ONCE. 


In our example: 
Each node has 121 possible labels
Each label is a separate yes/no question

For one node, the model might output something like:

out = [2.1, -1.4, 0.7, -3.0, ...]

That means:
label 1 → probably yes
label 2 → probably no
label 3 → maybe yes
label 4 → strongly no



Why BCEWithLogitsLoss?
Because each class is treated independently:
for each node, each class can be either present or absent.

Important:
this loss expects raw logits, so we do NOT apply sigmoid inside the model.
The sigmoid is handled internally by BCEWithLogitsLoss.

Basically BCE is sigmoid + Binary Cross Entropy 

Sigmoid squashes the raw logit between 0 and 1, then this gets interpreted 
as a probability, 
then the binary cross-entropy checks for each label separately how close 
that probability is to the true answer 0 or 1. 

So for one node with 121 labels:

the model outputs 121 logits
sigmoid converts them into 121 probabilities
BCE compares those 121 probabilities with the 121 true labels
"""


def training_node_cls():
    """
    Run one full training epoch over the training DataLoader.

    Returns:
    average loss across the training dataset
    """

    model_node_cls.train()
    """
    Put the model in training mode.

    This matters because dropout behaves differently in training mode.
    """

    total_loss = 0
    """
    Variable used to accumulate the loss over all batches.
    """

    for data in train_loader:
        """
        Loop over batches of training graphs.
        """

        data = data.to(device)
        """
        Move the batch data to the same device as the model.
        """

        optimizer_node_cls.zero_grad()
        """
        Clear old gradients from the previous batch.
        """

        out = model_node_cls(data.x, data.edge_index)
        """
        Forward pass.

        Input:
        - data.x = node features
        - data.edge_index = graph connectivity

        Output:
        raw logits for every node and every class
        """

        loss = criterion_node_cls(out, data.y.float())
        """
        Compute the batch loss.

        data.y contains the true labels.
        We convert to float because BCEWithLogitsLoss expects float targets.
        """

        loss.backward()
        """
        Backpropagation.

        Compute gradients of the loss with respect to all model parameters.
        """

        optimizer_node_cls.step()
        """
        Update the model parameters using the computed gradients.
        """

        total_loss += loss.item()
        """
        Add the current batch loss to the running total.
        """

    return total_loss / sum([d.num_nodes for d in train_dataset])
    """
    Return a normalised average loss.

    Here the total loss is divided by the total number of nodes in the
    training dataset.
    """


def evaluate_node_cls(loader):
    """
    Evaluate the model on a given DataLoader.

    Returns:
    micro F1 score
    """

    model_node_cls.eval()
    """
    Put the model in evaluation mode.

    This disables training-specific behaviour such as dropout.
    """

    all_preds = []
    """
    Store predicted labels from all batches.
    """

    all_labels = []
    """
    Store true labels from all batches.
    """

    with torch.no_grad():
        """
        Disable gradient tracking because we are only evaluating.
        """

        for data in loader:
            """
            Loop over the batches from the provided loader.
            """

            data = data.to(device)
            """
            Move the batch to the correct device.
            """

            out = model_node_cls(data.x, data.edge_index)
            """
            Compute raw output logits for the nodes in this batch.
            """

            preds = (out > 0).float() # if the logit is > 0, predict 1; otherweise predict 0
            #the .float() converts True -> 1.0 , False ->0 
            """
            Convert logits into binary predictions.

            Why threshold at 0?
            Because BCEWithLogitsLoss uses logits.

            A logit > 0 corresponds to a sigmoid probability > 0.5
            A logit < 0 corresponds to a sigmoid probability < 0.5
            
            So this line is equivalent to saying:

            predict label 1 if probability > 0.5

            without explicitly calling sigmoid.
            """

            all_preds.append(preds.cpu())
            """
            Move predictions to CPU and store them.
            """

            all_labels.append(data.y.cpu())
            """
            Move true labels to CPU and store them.
            """

    preds_tensor = torch.cat(all_preds, dim=0)
    """
    Concatenate all batch predictions into one tensor.
    """

    labels_tensor = torch.cat(all_labels, dim=0)
    """
    Concatenate all batch labels into one tensor.
    """

    f1 = f1_score(labels_tensor.numpy(), preds_tensor.numpy(), average="micro")
    """
    Compute the micro F1 score.

    Micro F1 is commonly used in multi-label classification because it
    aggregates contributions from all labels across all nodes.
    """

    return f1
    """
    Return the evaluation score.
    """
    """
    What is Micro F1

    """

epochs_node_cls = 50
"""
    Number of training epochs.
    Micro F1 does one global count first, instead of computing a separate F1 for each label and averaging later.

    So micro F1 says:

    treat every node-label decision as one big pool of binary predictions

    Micro F1 measures performance by looking at all label decisions together before computing precision and recall.

    In our PPI case, every node has 121 yes/no labels. So the model makes a huge number of binary decisions:

    node 1, label 1
    node 1, label 2
    ...
    node 2, label 1
    and so on

    How micro F1 works?

    First, across all nodes and all labels, it counts:

    TP = predicted 1 and true label is 1 (True Positive)
    FP = predicted 1 but true label is 0 (False Positive)
    FN = predicted 0 but true label is 1 (False Negative)

    Then it computes:

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)

    Then:

    F1 = 2 * precision * recall / (precision + recall)

    Suppose across all nodes and labels we get:

    TP = 80
    FP = 20
    FN = 10

    Then:

    precision = 80 / (80 + 20) = 0.80
    recall    = 80 / (80 + 10) = 0.889

    So:

    F1 ≈ 2 * 0.80 * 0.889 / (0.80 + 0.889) ≈ 0.842
"""

print("\n--- Training GraphSAGE for Node Classification ---")

for epoch in range(1, epochs_node_cls + 1):
    """
    Main training loop.

    For each epoch:
    - train the model on the training set
    - evaluate on the validation set
    """

    loss = training_node_cls()
    """
    Run one training epoch and get the average loss.
    """

    val_f1 = evaluate_node_cls(val_loader)
    """
    Evaluate the current model on the validation data.
    """

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}")
        """
        Every 10 epochs, print:
        - epoch number
        - training loss
        - validation F1 score
        """


test_f1 = evaluate_node_cls(test_loader)
"""
Evaluate the trained model on the test set.
"""

print(f"Test F1 Score: {test_f1:.4f}")
"""
Print the final test F1 score.
"""