import torch 
import networkx as nx 
from torch_geometric.utils import from_networkx, degree
from Create_BAModel_Network import create_graph_dataset 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

data = create_graph_dataset()


#===General Purpose Data Splitting for Node Classification on a Single Graph==

num_nodes = data.num_nodes
perm = torch.randperm(num_nodes)

train_size = int(0.6 * num_nodes)
val_size = int(0.2 * num_nodes)

data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

data.train_mask[perm[:train_size]] = True
data.val_mask[perm[train_size:train_size + val_size]] = True
data.test_mask[perm[train_size + val_size:]] = True


#====Isolate node connectivity list and node features====
edge_index = data.edge_index
x = data.x

#====Inspect the data====

print(data)
print(data)
print("data.y:", data.y)

if data.y is not None:
    num_classes = int(data.y.max().item()) + 1
    print("num_classes:", num_classes)
else:
    print("No labels found. data.y is None.")
print(f"Edge index type is: {type(edge_index)}, shape is: {edge_index.shape} and the first 10 edges are: {edge_index[:, :10]}")
print(f"Node Features type is: {type(x)}, shape is: {x.shape} and the first 10 features are: {x[:10]}")


#====We start by creating the classes====

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    """
    For GRAPHSage we would use something very similar: 

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GraphSAGE, self).__init__()
            
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)

    GCN is often used for simpler or smaller transductive settings, 
    while GraphSAGE is better suited to larger or inductive settings where 
    the model must scale and generalize to unseen nodes or graphs.

    """        
    def forward(self, x, edge_index):
        #First Convolution Layer 
        x = self.conv1(x,edge_index)
        """
        What this line does? It takes through our list, learns how the nodes are connected 
        -using edge_index- and the node features. 
        
        Say that we have 6 hidden channels and 300 nodes and our nodes had 3 features each one:
        degree, normalised degree and log degree,  then after this first convolution 
        we will have a matrix that goes from [300, 3] to [300,6]
        # 
        # [31.0000, 0.4627, 3.4657] -> [ 0.84, -1.12, 0.37, 2.05, -0.49, 1.31]
        # [13.0000, 0.1940, 2.6391] -> [-0.27,  0.91, 1.44, 0.52,  0.18, -0.63]
        # [67.0000, 1.0000, 4.2195] -> [ 1.76, -0.34, 0.88, 2.91, -1.15, 0.47]
        # [15.0000, 0.2239, 2.7726] -> [ 0.13,  0.54, 1.02, 0.74, -0.21, 0.09]
        # [43.0000, 0.6418, 3.7842] -> [ 1.12, -0.67, 0.45, 1.98, -0.72, 0.88]
        
        This means that we have 6 learned features 
        In practice we use more than 6, for example, 16. 
        """
        x = F.relu(x) # Apply ReLu activation function 
        
        """
        ReLU is applied element by element: positive values stay the same and negative values become 0
        [ 0.84, -1.12, 0.37, 2.05, -0.49, 1.31] -> [0.84, 0.00, 0.37, 2.05, 0.00, 1.31]
        [-0.27,  0.91, 1.44, 0.52,  0.18, -0.63] -> [0.00, 0.91, 1.44, 0.52, 0.18, 0.00]
        [ 1.76, -0.34, 0.88, 2.91, -1.15, 0.47] -> [1.76, 0.00, 0.88, 2.91, 0.00, 0.47]
        [0.13,  0.54, 1.02, 0.74, -0.21, 0.09] -> [0.13, 0.54, 1.02, 0.74, 0.00, 0.09]
        [1.12, -0.67, 0.45, 1.98, -0.72, 0.88] -> [1.12, 0.00, 0.45, 1.98, 0.00, 0.88]
        """
        
        x = F.dropout(x, p=0.5, training=self.training) #Apply dropout for regularisation

        """
        Here the convolution drops each value -makes it zero- half of the time (0.5)
        This is to avoid depending too much on the value and force the model to learn 
        a more distributed representation -but values of 0.3 and 0.2 can be used- 
        
        The values that remain are scaled up based on 1/(1-p).
        In our case is: 
        
        1/1-0.5 = 1/0.5 = 2 
        
        For example, 
        
        [0.84, 0.00, 0.37, 2.05, 0.00, 1.31] -> 
        
        -> drop out aplies and just 0.84 and 2.05 remains 
        
        [0.84, 0.00, 0.00, 2.05, 0.00, 0.00] ->
        
        We multiply by 2 
          
        [1.68, 0.00, 0.00, 4.10, 0.00, 0.00] 
        
        
        """
        #Second Convolution Layer

        x = self.conv2(x,edge_index)
        """
        We convolute and aggregate again because the second GNN layer uses the first-layer 
        embeddings as new node features, allowing each node to gather 
        information from a wider part of the graph, not just its immediate neighbours
        
        Now in our example we have defined the hidden channels as 6. 
        Now we need to define how many output channels and this number should match 
        what we are trying to predict. 
        For example, if we are doing node classification and we have 3 possible labels 
        then our output channels must be 3. 
       
        This is why when we instantiate our model, we set the out-channels equal 
        to the number of classes and the in-channels equals to the number of features
        
        Finally, the weight matrix is already part of our model through GCNConv;
        we do not need to define it separately.
        
        IMPORTANT: THE SAME WEIGHT IS SHARED ACROSS ALL NODES. 
        
        So the weights belong to the layer, not the individual nodes. 
        If a layer maps: 
        
        in_channels = 3 
        hidden_channels = 6
        
        then the weight matrix shaped roughly like 3 x 6 
        or 
        any other equivalent set of weight that tells the model how to turn 
        a 3-dimensional node feature vector into a 6-dimensional embedding.
        
        If each node had its own weights, the model
        would not generalize well and would become enormous.
        
        IN BOTH STANDARD NEURAL NETWORKS AND GNNs, THE WEIGHTS ARE SHARED 
        ACROSS ALL INPUTS TO THE LAYER; THE DIFFERENCE IS THAT GNNS APPLY 
        THE SHARED TRANSFORMATION TO NODES WHILE ALSO INCORPORATING 
        GRAPH NEIGHBOURHOOD INFORMATION.

        """
        return x 

num_classes = int(data.y.max().item()) + 1
"""
We need this line above to count how many classes we have in our model 
The line above extracts this info for us and allow us to use it when 
instantiating the model. 
"""   
model = GCN(in_channels=data.num_node_features, hidden_channels=16, out_channels=num_classes)

print(f"Model architecture: \n{model}")

#===Define Optimizer and Loss function===
""" 
We define the optimizer and loss function outside the training method 
because they are part of the training setup, and in particular 
the optimizer must preserve its internal state across epochs 
instead of being recreated every time.

For example, ADAM what it does is 
"""   
optimizer = torch.optim.Adam (model.parameters(), 
                             lr=0.01, 
                             weight_decay=5e-4)

"""
What are .parameters()? 

Here WE DO NOT REFER to the architecture of the model -like the number of channels-
but to all the LEARNABLE TENSORS inside the model. 

ADAM will use the actual weight and biases stored inside the model. 
In PyTorch, optimizers are constructed from an iterable of Parameter objects
and model.parameters() provide exactly that. 

We can inspect them by running: 
 
for name, p in model.named_parameters():
    print(name, p.shape)


The following table shows the evolution of the shape, after each passage:

input x                [300, 3]
after conv1            [300, 16]
after ReLU             [300, 16]
after dropout          [300, 16]
after conv2            [300, 3]


"""
#===Inspecting our model and its parameters===
for name, p in model.named_parameters():
    print(name, p.shape)
    
criterion = torch.nn.CrossEntropyLoss()

#===The Training Loop===
def train(): 
    model.train() # Set the model to training mode (enables dropout, etc)
    """
    Why do we need to tell PyTorch that we are training now? 
    This matters because some layers behave differently during training and testing. 
    For example, dropout is active during trainig and off during evaluation. 
    
    """
    optimizer.zero_grad() #Clear gradients from previous step 

    # Forward Pass: compute the output logits 

    out = model(data.x, data.edge_index)
    """
    This line above is of extreme importance, and wee need to know 
    what is hapenning here. 
    
    What we are taking from our graph:

    (1) The node feature matrix data.x of shape [300, 3]
    (2) The graph connectivity data.edge_index

    and passing both into the GCN.

    So the model does not look only at each node’s own 3 features in isolation. 
    It uses edge_index to know who is connected to whom.


    At the beginning, each node has a simple feature vector like:

    [degree, normalized_degree, log_degree]

    So initially it is just a graph plus basic node features.

    Then the first GCN layer does two things together:

    (1) It aggregates information from neighbours
    (2) It applies a learned transformation to that aggregated information

    So after the first layer, each node no longer represents only its own original features. 
    Its new representation has been influenced by its local neighbourhood.

    Then:

    (1) ReLU changes the values nonlinearly
    (2) Dropout randomly removes some activations during training

    The second GCN layer again aggregates and transforms

    So by the end, each node still corresponds to one row, but that row 
    is no longer a simple hand-made feature vector. 
    It is now a learned representation shaped by:

    (1) The original features
    (2) The graph structure
    (3) The model’s weights
    (4) The nonlinear transformations

    The final row contains 3 raw logits.

    For example, one node might end with:

    [1.8, -0.4, 0.9]

    These are not labels yet, and not probabilities yet.

    They are just raw class scores.

    The model is basically saying:

    class 0 score = 1.8
    class 1 score = -0.4
    class 2 score = 0.9

    The highest one is class 0, so that would be the predicted class.

    Then what do we do with those logits?

    We compare them to the true label.
        
    """
      
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    """
    This line above saves, inside the loss variable, the nodes 
    for training and pair each of them with its respective boolean value. 
    
    Recall that we have defined what our criterion was above : CrossEntropy Loss
    
    CrossEntropy means: 
    
    
    CrossEntropyLoss compares the raw logits predicted by the model
    with the true labels of the training nodes.

    It checks whether the correct class receives a high score:
    if it does, the loss is small; if it does not, the loss is larger.

    The losses across all selected training nodes are then combined
    -usually averaged- into a single number, which is stored in loss.

    Example:

    Suppose after applying data.train_mask we keep 3 training nodes.

    Predicted logits:
    [
        [2.0, 0.5, -1.0],   # node 1
        [0.1, 1.8, 0.3],    # node 2
        [1.2, 0.4, 2.1]     # node 3
    ]

    True labels:
    [0, 1, 2]

    This means:
    - for node 1, the correct class is 0
    - for node 2, the correct class is 1
    - for node 3, the correct class is 2

    In all three cases, the correct class has the highest score,
    so the loss for each node will be relatively small.

    If instead a node had logits like:
    [2.5, 0.1, -0.3]

    but its true label was:
    2

    then the model would be assigning the highest score to class 0
    instead of the correct class 2, so the loss for that node would be large.

    CrossEntropyLoss computes this error for each selected node and
    then combines them -usually by averaging- into one single number.
    """
    loss.backward() #Compute gradients 
    optimizer.step() # Update model parameters 
    
    """
    loss.backward()  is a method of the loss tensor returned by PyTorch.

    It computes the gradients of the loss with respect to all 
    the learnable parameters that were involved in producing that loss.

    optimizer.step() is a method of the optimizer object, here Adam.

    What it does is use the gradients that were just computed
    and update the model parameters accordingly.

    - loss.backward() computes the gradients
    - optimizer.step() uses those gradients to update the weights

    """
    return loss.item()

# Mode model and data to GPU if available 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Check whether the model is actually on the GPU or not

print("Model device:", next(model.parameters()).device)
data = data.to(device)

#Training loop 

epochs = 1000# Number of training epochs 

for epoch in range (1, epochs +1):
    loss = train()
    if epoch % 20 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:4f}")

def evaluate(): 
    model.eval() # Set the model to evaluation mode 
    """
    During evaluation, we run the model on the entire graph 
    (all nodes and edges), but we compute performance only 
    on the subset of nodes selected by the evaluation mask.
    """
    with torch.no_grad(): #Disable gradinet calculations for efficiency 
        out = model(data.x, data.edge_index)
        #Get predicted class by taking argmax of logits 
        pred = out.argmax(dim=1)
        """
        pred = out.argmax(dim=1) converts the model output (logits)
        into predicted class labels.

        After the forward pass, out has shape [num_nodes, num_classes].
        Each row corresponds to a node, and each value in that row is the
        score (logit) for a class.

        argmax(dim=1) means:
        - look across each row (i.e. across classes for each node)
        - find the index of the largest value
        - return that index as the predicted class

        So the result is a 1D tensor of shape [num_nodes],
        where each element is the predicted class label for a node.

        Example:

        out =
        [
            [2.1, 0.3, -1.2],   # node 0
            [0.5, 1.7, 0.2],    # node 1
            [-0.4, 0.1, 2.3]    # node 2
        ]

        For each node, we select the index of the largest value:

        node 0 → max is 2.1 → index 0  
        node 1 → max is 1.7 → index 1  
        node 2 → max is 2.3 → index 2  

        So:

        pred = [0, 1, 2]

        This means:
        - node 0 is predicted as class 0
        - node 1 is predicted as class 1
        - node 2 is predicted as class 2

        We do not need to convert logits into probabilities,
        because the class with the highest logit is the same
        as the class with the highest probability.
        """
        #Print the resuls
        
        test_nodes = data.test_mask.nonzero(as_tuple=True)[0]

        """
        The model predicts outputs for all nodes,
        but we use the mask to select only the test nodes
        when computing accuracy or inspecting results.
        
        test_nodes = data.test_mask.nonzero(as_tuple=True)[0] extracts 
        the indices  of the nodes that belong to the test set.

        data.test_mask is a boolean tensor of shape [num_nodes], where:
        - True means the node is part of the test set
        - False means it is not

        Example:

        data.test_mask =
        [False, True, False, True, True]

        This means:
        - node 1, 3, and 4 are test nodes

        .nonzero(as_tuple=True) finds the indices where the value is True.

        It returns a tuple (because tensors can be multi-dimensional),
        so we take [0] to extract the actual indices.

        So:

        data.test_mask.nonzero(as_tuple=True) → (tensor([1, 3, 4]),)

        Then:

        test_nodes = tensor([1, 3, 4])

        This gives us a list of node indices that we can loop over.


        """
        for node in test_nodes[:50]:   # first 50 test nodes
            print(
                 f"Node {node.item()} | "
                 f"Predicted: {pred[node].item()} | "
                 f"True label: {data.y[node].item()}"
                )
        # Calculate accuracy only on text nodes 

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        """
        
        computes how many test nodes were classified correctly.

        Step by step:

        1. data.test_mask selects only the test nodes.
        So:
        pred[data.test_mask] → predicted labels for test nodes
        data.y[data.test_mask] → true labels for test nodes

        2. We compare them element-wise:
        pred[...] == data.y[...]

        This produces a boolean tensor:
        True  → correct prediction
        False → incorrect prediction

        3. .sum() counts how many True values there are.
        In PyTorch, True = 1 and False = 0.

        Example:

        pred =        [0, 1, 2, 1, 0]
        data.y =      [0, 2, 2, 1, 1]
        test_mask =   [F, T, T, F, T]

        Step 1: select test nodes
        pred[test_mask]   → [1, 2, 0]
        data.y[test_mask] → [2, 2, 1]

        Step 2: compare
        [1 == 2, 2 == 2, 0 == 1] → [False, True, False]

        Step 3: sum
        False = 0, True = 1

        correct = 1

        So only 1 test node was classified correctly.

        This value is later divided by the total number of test nodes
        to compute the test accuracy.
        """
        acc = int(correct)/int(data.test_mask.sum())

        return acc
    
#Evaluate the model after training 

test_acc = evaluate()

print(f"Test Accuracy: {test_acc:4f}")