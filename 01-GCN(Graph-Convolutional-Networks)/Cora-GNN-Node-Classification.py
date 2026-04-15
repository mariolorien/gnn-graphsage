"""
Implementing a Graph Convolutional Network (GCN) from scratch provides a deep understanding 
of its mechanics. We will use PyTorch Geometric (PyG). 
Our task will be node classification on a classic graph dataset.

Project: Node Classification on the Cora Dataset

The Cora Dataset is a widely used benchmark for graph based machine learning. 
It consists of scientific papers - nodes - and citation links between them -edges-. 
Each paper has a bag-of-words feature vector and a single topic label. 
Our goal is to classify the topic of each paper using GCN.
"""
import torch 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load the Cora dataset 

dataset = Planetoid(root='./data/Cora', name='Cora')
"""
The 'root='./data/Cora', specifies where to store the dataset
Teh 'name = 'Cora', selects the dataset. 

"""
data = dataset[0] # The Planetoi dataset contains only one graph 
print(f"Dataset: {dataset.name}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features (per node): {dataset.num_node_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Graph Object: {data}")

print(f"-----------We Inspect the loaded object------------")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edeges: {data.num_edges}")
print(f"Node feature matrix shape: {data.x.shape}") # number of nodes, number of nodes features
print(f"Edge index shape: {data.edge_index.shape}") # 2, number of edges
print(f"Node labels shape: {data.y.shape}") # number of nodes

print(f"-----------We check masks for training, validation, and testing")
print(f"Number of training nodes: {data.train_mask.sum()}")
print(f"Number of validation nodes: {data.val_mask.sum()}")
print(f"Number of test nodes: {data.test_mask.sum()}")      

class GCN(torch.nn.Module):  
    """
    Our parent class is torch.nn.module and our child class is our own GCN class.
    In PyTorch, almost all neural networks are built as a subclasses of torch.nn.Module    
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
    
        self.conv1=GCNConv(in_channels, hidden_channels) #First GCN Layer 
        self.conv2=GCNConv(hidden_channels, out_channels) #Second GCn Layer 
    

    def forward(self, x, edge_index):
        #First Convolution Layer 
        x = self.conv1(x,edge_index)
        x = F.relu(x) # Apply ReLu activation function 
        x = F.dropout(x, p=0.5, training=self.training) #Apply dropout for regularisation

        #Second Convolution Layer

        x = self.conv2(x,edge_index)
        """
        Output Layer: No activation here if using CrossEntropy Loss 
        CrossEntropy Loss expects raw logits for classification
        """
        return x 
    
    #---------Now we instantiate the model----------------
    #   input_features = dataset.num_node_features
    #   hidden_dim = 16 (a common choice, it can be tuned)
    #   output_classes = dataset.num_classes

model = GCN(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes)

print(f"Model architecture: \n{model}")

#Define the optimizer 

"""
For multi-class classification, CrossEntropyLoss is standard. Adam is a popular optimizer
"""

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

"""
lr= learning rate, controlls the step size druing optimization
weight_decay: L2 regularization (prevents overfitting by penalizing large weights)
CrossEntropy: Computes the cross entropy loss between the models ouput logits and the true labels
"""
#Define the loss function 

criterion = torch.nn.CrossEntropyLoss()


# The training Loop 

def train(): 
    model.train() # Set the model to training mode (enables dropout, etc)
    optimizer.zero_grad() #Clear gradients from previous step 

    # Forward Pass: compute the output logits 

    out = model(data.x, data.edge_index)

    # Calculate loss only on training nodes 
    #data.train_mask is a boolean tensor indicating training nodes 

    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward() #Compute gradients 

    optimizer.step() # Update model parameters 

    return loss.item()

# Mode model and data to GPU if available 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Check whether the model is actually on the GPU or not

print("Model device:", next(model.parameters()).device)
data = data.to(device)

#Training loop 

epochs = 200 # Number of training epochs 

for epoch in range (1, epochs +1):
    loss = train()
    if epoch % 20 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:4f}")



#Evaluation 

def evaluate(): 
    model.eval() # Set the model to evaluation mode 

    with torch.no_grad(): #Disable gradinet calculations for efficiency 
        out = model(data.x, data.edge_index)
    #Get predicted class by taking argmax of logits 
        pred = out.argmax(dim=1)

        #Print the resuls
        
        test_nodes = data.test_mask.nonzero(as_tuple=True)[0]

        for node in test_nodes[:50]:   # first 50 test nodes
            print(
                 f"Node {node.item()} | "
                 f"Predicted: {pred[node].item()} | "
                 f"True label: {data.y[node].item()}"
                )
        # Calculate accuracy only on text nodes 

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()

        acc = int(correct)/int(data.test_mask.sum())

        return acc
    
#Evaluate the model after training 

test_acc = evaluate()

print(f"Test Accuracy: {test_acc:4f}")