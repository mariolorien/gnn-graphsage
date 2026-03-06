# GCN Notes: Parent Class, `super()`, and the Meaning of the Channels

## What I have so far

So far, we have done two main things:

1. **Loaded the Cora dataset**
   - papers = nodes
   - citations = edges
   - words = node features
   - topics = labels

This means `data` is now my graph.

2. **Started defining a GCN model as a Python class**

```python
class GCN(torch.nn.Module):
```

This means:
- `GCN` is my own class
- it **inherits** from `torch.nn.Module`

## What is the parent class?

The parent class is:

```python
torch.nn.Module
```

So:
- `GCN` = child class
- `torch.nn.Module` = parent class

## Why inherit from `torch.nn.Module`?

Because in PyTorch, neural network models are usually built as subclasses of `torch.nn.Module`.

This parent class gives useful functionality such as:
- storing layers
- storing learnable parameters
- working with optimizers
- switching between training and evaluation mode
- moving the model to CPU or GPU

So by inheriting from it, `GCN` becomes a proper PyTorch model.

## What does `super().__init__()` do?

```python
super().__init__()
```

This runs the constructor of the parent class.

In simple terms, it means:
- first initialize the `torch.nn.Module` part
- then initialize the custom GCN layers

For this PyTorch example, **yes, I do need it**.

If I do not call the parent constructor, PyTorch may not correctly register the layers and parameters of my model.

## In general, do child classes always need to call the parent constructor?

**No, not always.**

General rule:
- if the parent class does important setup, call it
- if the parent class does nothing special, it may not matter

In frameworks like PyTorch, it is usually necessary.

## What is `__init__`?

```python
def __init__(self, in_channels, hidden_channels, out_channels):
```

This is the constructor of the class.

It runs when I create an object, for example:

```python
model = GCN(in_channels=1433, hidden_channels=16, out_channels=7)
```

## What is `self`?

`self` refers to the specific object being created.

It lets the object store its own layers and data.

Example:

```python
self.conv1 = GCNConv(in_channels, hidden_channels)
self.conv2 = GCNConv(hidden_channels, out_channels)
```

This means the model object has two graph convolution layers: `conv1` and `conv2`.

## What do `in_channels`, `hidden_channels`, and `out_channels` mean?

They describe the size of the feature vectors as they move through the network.

### `in_channels`

The number of input features per node.

For Cora:

```python
in_channels = dataset.num_node_features
```

This is usually **1433**.

So each paper starts with a vector of length 1433.

### `hidden_channels`

The size of the hidden representation produced by the first layer.

Example:

```python
hidden_channels = 16
```

This means the first layer turns each node from 1433 features into 16 learned features.

### `out_channels`

The size of the final output per node.

For node classification, this is usually the number of classes.

For Cora:

```python
out_channels = dataset.num_classes
```

This is usually **7**.

So the final layer produces 7 output values per node, one for each topic class.

## What do the two layers do?

```python
self.conv1 = GCNConv(in_channels, hidden_channels)
self.conv2 = GCNConv(hidden_channels, out_channels)
```

This means:

- first layer: `1433 -> 16`
- second layer: `16 -> 7`

So:
- each paper starts as a bag-of-words vector
- the first layer creates a hidden learned representation
- the second layer produces class scores


## Short final reminder

- parent class: `torch.nn.Module`
- `super().__init__()` initializes the parent part
- `in_channels` = input feature size
- `hidden_channels` = hidden representation size
- `out_channels` = output size / number of classes

## Understanding the `GCN` class and the `forward()` method

### What do you have so far?

You have done two main things.

First, you loaded the Cora dataset:

- papers = nodes
- citations = edges
- words = node features
- topics = labels

So now `data` is your graph.

Second, you started defining a GCN model as a Python class:

```python
class GCN(torch.nn.Module):
```

This means:

- you are creating your own neural network class called `GCN`
- and it inherits from `torch.nn.Module`

---

## What is the parent class?

The parent class is:

```python
torch.nn.Module
```

So in:

```python
class GCN(torch.nn.Module):
```

- `GCN` = your child class
- `torch.nn.Module` = the parent class

---

## Why do we inherit from `torch.nn.Module`?

Because in PyTorch, almost all neural networks are built as subclasses of `torch.nn.Module`.

That parent class gives you useful neural-network behaviour, such as:

- storing layers
- storing learnable parameters
- moving the model to CPU/GPU
- switching between training and evaluation mode
- making the model compatible with optimizers

So by inheriting from it, your `GCN` becomes a proper PyTorch model.

---

## What does this line do?

```python
super().__init__()
```

This says:

- run the constructor of the parent class too

In other words:

- first initialize the base `torch.nn.Module` part
- then initialize your own custom layers

Very intuitively:

> Before building my special GCN, make sure the normal PyTorch neural network machinery is set up.

So yes, this is connected to the parent class.

---

## Do you need `super().__init__()` here?

Yes, in this case you do.

Why? Because `GCN` inherits from `torch.nn.Module`, and `torch.nn.Module` has its own initialization logic. That setup is important because PyTorch needs to properly register:

- submodules
- layers
- parameters
- buffers
- internal model state

If you do not call the parent constructor, PyTorch may not correctly treat your class as a proper model.

So for a PyTorch model, this is not just a style choice - it is part of making the model work correctly.

---

## In general, do we always need to initialize the parent class?

No, not always.

The general rule is:

- if the parent class has an `__init__()` method that does important work, then you should call it
- if the parent class does nothing special, then it may not matter
- in many real object-oriented designs, it is good practice to call it when inheritance is meaningful

So the answer is not "always", but very often yes, especially in frameworks like:

- PyTorch
- Tkinter widgets
- many library base classes

Think of inheritance like this:

- the parent class builds the foundation
- the child class adds the specialized part

If you skip the parent constructor, you may be building the specialized part on top of a foundation that was never created.

---

## What is `__init__`?

This method:

```python
def __init__(self, in_channels, hidden_channels, out_channels):
```

is the constructor of your class.

It runs when you create an object from the class.

For example, later you might do:

```python
model = GCN(in_channels=1433, hidden_channels=16, out_channels=7)
```

When that happens, Python calls `__init__` and uses those values to build the model.

---

## What is `self`?

`self` means:

- the specific object being created

It lets the object store its own data and layers.

So when you write:

```python
self.conv1 = GCNConv(in_channels, hidden_channels)
```

you are saying:

- this model object has a first convolution layer called `conv1`

and when you write:

```python
self.conv2 = GCNConv(hidden_channels, out_channels)
```

you are saying:

- this model object also has a second convolution layer called `conv2`

---

## What are `in_channels`, `hidden_channels`, and `out_channels`?

These describe the sizes of the feature vectors as they move through the network.

Think of them as the number of values each node representation has at each stage.

### `in_channels`

This is the number of input features per node.

In Cora, each paper has a bag-of-words vector.

So:

```python
in_channels = dataset.num_node_features
```

In Cora, this is usually 1433.

That means each node starts with a vector of length 1433.

So `in_channels` means:

- how many features come in at the start

### `hidden_channels`

This is the size of the hidden representation created by the first layer.

For example:

```python
hidden_channels = 16
```

means:

- the first GCN layer will transform each node from 1433 features
- into a new representation of 16 features

This is like a compressed learned summary.

So `hidden_channels` means:

- how many features the network uses internally

### `out_channels`

This is the size of the final output per node.

For node classification, this is usually the number of classes.

In Cora, there are usually 7 classes, so:

```python
out_channels = dataset.num_classes
```

So `out_channels = 7`.

This means the final layer produces 7 numbers per node, one for each possible topic.

So `out_channels` means:

- how many outputs we want at the end

---

## So what do your two layers do?

You wrote:

```python
self.conv1 = GCNConv(in_channels, hidden_channels)
self.conv2 = GCNConv(hidden_channels, out_channels)
```

This means:

### First layer

It takes node features from `in_channels` to `hidden_channels`.

For Cora, roughly:

```text
1433 -> 16
```

So each paper starts with 1433 word features and gets transformed into 16 learned features.

### Second layer

It takes those hidden features from `hidden_channels` to `out_channels`.

For Cora, roughly:

```text
16 -> 7
```

So each node ends up with 7 output values, corresponding to the 7 topic classes.

---

## Very intuitive picture

### Input

Each paper begins as a large word-based vector:

- word 1 present?
- word 2 present?
- word 3 present?
- etc.

### Hidden layer

The model learns a more useful internal summary:

- maybe this looks mathematical
- maybe this looks like neural networks
- maybe this seems related to reinforcement learning

Not literally these exact meanings, but something like that.

### Output layer

The model turns that summary into class scores:

- topic A score
- topic B score
- topic C score
- etc.

---

## Why two layers?

Because one layer lets each node gather information from its immediate neighbours.

Two layers let each node gather information from:

- its neighbours
- and neighbours of neighbours

So two GCN layers allow information to travel farther in the graph.

That is one reason two-layer GCNs are very common.

---

## What my class means in plain English

Your class currently says:

> I want to build a graph neural network model called `GCN`. It is based on PyTorch's neural network class. It has two graph convolution layers: the first transforms the input node features into hidden features, and the second transforms hidden features into output class scores.


## The `forward()` method

```python
def forward(self, x, edge_index):
    # First Convolution Layer 
    x = self.conv1(x, edge_index)
    x = F.relu(x)  # Apply ReLU activation function 
    x = F.dropout(x, p=0.5, training=self.training)  # Apply dropout for regularisation

    # Second Convolution Layer
    x = self.conv2(x, edge_index)

    """
    Output Layer: No activation here if using CrossEntropy Loss 
    CrossEntropy Loss expects raw logits for classification
    """
    return x
```

### What is `forward()`?

The `forward()` method defines how the data moves through the network.

Earlier, in `__init__()`, we created the layers.  
Now, in `forward()`, we specify in which order those layers are applied to the input data.

In other words:

- `__init__()` = builds the model
- `forward()` = explains how the model uses the data

When we later do something like:

```python
out = model(data.x, data.edge_index)
```

PyTorch will use this `forward()` method to compute the output.

---

## What are `x` and `edge_index`?

The method takes two inputs:

- `x`: the node feature matrix
- `edge_index`: the graph connectivity information

### `x`

This is the matrix containing the features of all nodes.

For Cora:

- each row = one paper
- each column = one word feature

So `x` contains the bag-of-words representation of the papers.

### `edge_index`

This tells the GCN which nodes are connected to which other nodes.

It is the graph structure: in Cora, it tells us which papers cite which other papers.

So the model needs both:

- `x` to know the content of each node
- `edge_index` to know how nodes are linked

---

## First convolution layer

```python
x = self.conv1(x, edge_index)
```

This applies the first graph convolution layer.

It takes:

- the current node features `x`
- the graph connections `edge_index`

and produces new node representations.

This is where message passing begins: each node updates its features using information from its neighbours.

The shape changes from:

- input features per node (`in_channels`)

to:

- hidden features per node (`hidden_channels`)

So in the Cora example, roughly:

- from 1433 features
- to 16 hidden features

---

## ReLU activation

```python
x = F.relu(x)
```

After the first convolution, we apply the ReLU activation function.

ReLU stands for Rectified Linear Unit and works like this:

- if the value is positive, keep it
- if the value is negative, replace it with 0

Mathematically:

```python
ReLU(z) = max(0, z)
```

Why do we use it?

Because without an activation function, the network would just be a sequence of linear transformations, which would limit what it can learn.

So ReLU introduces non-linearity, allowing the model to learn more complex patterns.

---

## Dropout

```python
x = F.dropout(x, p=0.5, training=self.training)
```

This applies dropout, which is a regularisation technique.

What it does:

- during training, it randomly sets some values in `x` to zero
- this prevents the model from relying too much on specific neurons
- this helps reduce overfitting

Here:

- `p=0.5` means 50% dropout probability

The argument:

```python
training=self.training
```

means:

- apply dropout during training
- do not apply it during evaluation/testing

This is important because dropout is only used while the model is learning.

---

## Second convolution layer

```python
x = self.conv2(x, edge_index)
```

This applies the second graph convolution layer.

It takes the hidden node representations produced by the first layer and transforms them into the final output dimension.

So the shape changes from:

- hidden features per node (`hidden_channels`)

to:

- output features per node (`out_channels`)

In Cora, roughly:

- from 16 hidden features
- to 7 output values

These 7 output values correspond to the 7 topic classes.

---

## Why is there no activation after the second layer?

There is no activation function after `self.conv2(...)` because for this classification setup we usually use `CrossEntropyLoss`.

`CrossEntropyLoss` expects raw scores, also called logits.

Logits are just the unnormalised outputs of the model before applying softmax.

So:

- do not apply `softmax`
- do not apply `ReLU`
- just return the raw output

That is why the code ends with:

```python
return x
```

---

## What does the method return?

The method returns the final output `x`.

This output contains one row per node and one score per class.

So each node gets a vector like:

- score for class 1
- score for class 2
- score for class 3
- etc.

The class with the highest score is the predicted class.

---

## Instantiating the model

```python
model = GCN(
    in_channels=dataset.num_node_features,
    hidden_channels=16,
    out_channels=dataset.num_classes
)
```

This creates an instance of the `GCN` class.

We are telling the model:

- `in_channels = dataset.num_node_features`  
  number of input features per node  
  for Cora, this is usually 1433

- `hidden_channels = 16`  
  the size of the hidden layer  
  this is a design choice and can be tuned

- `out_channels = dataset.num_classes`  
  number of output classes  
  for Cora, this is usually 7

So this model has:

- input: 1433 features per node
- hidden layer: 16 features per node
- output: 7 class scores per node

---

## Printing the model

```python
print(f"Model architecture: \n{model}")
```

This prints a summary of the model architecture.

It helps us check that the model has been built correctly and shows the layers it contains.

For example, it should show that the model includes:

- one first `GCNConv` layer
- one second `GCNConv` layer

This is a useful quick check before training the model.

---

## Final intuitive summary

The `forward()` method says:

1. take the node features and graph connections
2. apply the first graph convolution
3. apply ReLU
4. apply dropout
5. apply the second graph convolution
6. return the raw class scores

So in plain English:

> the model first learns a hidden representation of each node from its own features and its neighbours, then it transforms that representation into class scores for node classification


## Optimizer, loss function, training loop, and epochs

### Define the optimizer

```python
# Define the optimizer

"""
For multi-class classification, CrossEntropyLoss is standard. Adam is a popular optimizer.
"""

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
```

### What is the Adam optimizer?

`Adam` is an optimization algorithm used to update the model's parameters during training.

In simple terms, after the model makes a prediction and we compute the error, Adam decides **how to change the weights** so that the model performs better next time.

It is popular because:

- it usually works well in practice
- it adapts the learning step for different parameters
- it often converges faster than basic gradient descent

So you can think of Adam as the rule that tells the model **how to learn from its mistakes**.

### What do `lr` and `weight_decay` mean?

```python
lr=0.01
weight_decay=5e-4
```

- `lr` stands for **learning rate**. It controls the step size during optimization.
  - if it is too large, learning can become unstable
  - if it is too small, learning can be very slow

- `weight_decay` is a form of **L2 regularization**.
  It penalizes very large weights, which helps reduce overfitting.

So:

- learning rate controls **how big each update is**
- weight decay helps keep the model **from becoming too extreme or overfitting the training data**

---

## Define the loss function

```python
# Define the loss function
criterion = torch.nn.CrossEntropyLoss()
```

### What is the criterion?

The `criterion` is the **loss function**.

A loss function measures **how wrong the model's predictions are**.

For this node classification task, we use `CrossEntropyLoss`, which is the standard choice for **multi-class classification**.

It compares:

- the model's raw output scores (logits)
- the true class labels

and returns a number representing the error.

If the predictions are poor, the loss is larger. If the predictions improve, the loss gets smaller.

So the criterion is the function that tells the model:

> how bad was this prediction?

---

## The training loop

```python
def train():
    model.train()  # Set the model to training mode (enables dropout, etc.)
    optimizer.zero_grad()  # Clear gradients from previous step

    # Forward pass: compute the output logits
    out = model(data.x, data.edge_index)

    # Calculate loss only on training nodes
    # data.train_mask is a boolean tensor indicating training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()  # Compute gradients
    optimizer.step()  # Update model parameters

    return loss.item()
```

### What does `model.train()` do?

```python
model.train()
```

This puts the model into **training mode**.

This matters because some layers behave differently during training and testing.
For example:

- `dropout` is active during training
- `dropout` is turned off during evaluation

So this line tells PyTorch:

> we are training now, so use training behaviour

---

### Why do we need `optimizer.zero_grad()`?

```python
optimizer.zero_grad()
```

PyTorch accumulates gradients by default.
That means if you do not clear them, the gradients from the previous step will be added to the new ones.

Usually, during standard training, we want each update to be based only on the **current forward and backward pass**.

So we clear old gradients before computing new ones.

In simple terms:

- old gradients belong to the previous step
- we do not want them to interfere with the current step

So `zero_grad()` means:

> start this training step with a clean gradient buffer

---

### Forward pass

```python
out = model(data.x, data.edge_index)
```

This runs the data through the model.

The model takes:

- `data.x` = node feature matrix
- `data.edge_index` = graph connectivity

and returns `out`, which contains the output logits for each node.

These logits are the raw class scores produced by the model.

---

### Why do we calculate the loss only on training nodes?

```python
loss = criterion(out[data.train_mask], data.y[data.train_mask])
```

In the Cora dataset, not all nodes are used for training.
The dataset is split into:

- training nodes
- validation nodes
- test nodes

`data.train_mask` is a boolean mask that selects only the training nodes.

So this line means:

- take the outputs for training nodes only
- take the true labels for training nodes only
- compute the loss using those nodes only

This is important because we do not want to train on validation or test data.

---

### What does `loss.backward()` do?

```python
loss.backward()
```

This computes the gradients of the loss with respect to the model parameters.

In simple terms, it answers the question:

> how should each weight change to reduce the loss?

This is the backpropagation step.

---

### What does `optimizer.step()` do?

```python
optimizer.step()
```

After gradients have been computed, `optimizer.step()` updates the model parameters.

So this is the actual learning step.

In very simple language:

- `loss.backward()` computes what direction to move in
- `optimizer.step()` actually moves the weights

---

### Why do we return `loss.item()`?

```python
return loss.item()
```

`loss` is a PyTorch tensor.

`loss.item()` extracts the numerical value as a normal Python number.

This is useful for printing and tracking training progress.

---

## Move model and data to GPU if available

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

### What does this do?

This checks whether a GPU is available.

- if yes, use `cuda`
- if not, use `cpu`

Then both the model and the data are moved to the same device.

This is necessary because PyTorch expects the model and the data to be on the same device.

---

## Training loop over epochs

```python
epochs = 200  # Number of training epochs

for epoch in range(1, epochs + 1):
    loss = train()
    if epoch % 20 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
```

### What is an epoch?

An **epoch** is one complete training pass through the training data.

In this example, each epoch means:

- run one forward pass
- compute the loss on the training nodes
- run backpropagation
- update the model parameters

So if you train for 200 epochs, the model repeats this learning process 200 times.

Why do we need many epochs?

Because the model usually does not learn everything in one step. It improves gradually over many updates.

So an epoch is basically:

> one round of learning

---

### Why print every 20 epochs?

```python
if epoch % 20 == 0:
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
```

This prints the training progress every 20 epochs.

It helps us monitor whether the loss is going down.
If the loss decreases over time, that usually means the model is learning.

---


## Final intuitive summary

This part of the code does the following:

1. chooses how the model will learn (`Adam` optimizer)
2. chooses how error will be measured (`CrossEntropyLoss`)
3. defines one training step in `train()`
4. clears old gradients before each step
5. runs the model forward
6. computes the loss on training nodes only
7. backpropagates the error
8. updates the model weights
9. repeats this process for many epochs

So in plain English:

> the model makes predictions, measures how wrong they are, computes how to improve, updates itself, and repeats this many times until it learns better node classifications

# GCN Evaluation: Testing the Model After Training

## Evaluation code

```python
# Evaluation

def evaluate():
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculations for efficiency
        out = model(data.x, data.edge_index)

        # Get predicted class by taking argmax of logits
        pred = out.argmax(dim=1)

        # Calculate accuracy only on test nodes
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()

        acc = int(correct) / int(data.test_mask.sum())

        return acc

# Evaluate the model after training
test_acc = evaluate()

print(f"Test Accuracy: {test_acc:.4f}")
```

---

## What is this part doing?

This section checks how well the trained model performs on the **test set**.

During training, the model learned from the training nodes.  
Now we want to measure how accurately it predicts the labels of nodes it did **not** train on.

That is the purpose of the `evaluate()` function.

---

## `model.eval()`

```python
model.eval()
```

This sets the model to **evaluation mode**.

This is important because some layers behave differently during training and testing.

For example:

- **dropout** is active during training
- **dropout** is turned off during evaluation

So `model.eval()` tells PyTorch:

> we are no longer training; we are now testing the model

---

## Why do we use `torch.no_grad()`?

```python
with torch.no_grad():
```

This tells PyTorch **not** to calculate gradients.

Why?

Because during evaluation we are **not updating the model**.  
We only want predictions.

Disabling gradients makes evaluation:

- faster
- lighter in memory
- more efficient

So the idea is:

- during training → compute gradients
- during evaluation → no gradients needed

---

## Forward pass during evaluation

```python
out = model(data.x, data.edge_index)
```

This runs the whole graph through the model again.

The output `out` contains the **raw class scores** (logits) for each node.

So for every node, the model produces a vector of scores, one score per class.

---

## Predicted class

```python
pred = out.argmax(dim=1)
```

This chooses the class with the highest score for each node.

### Why `argmax(dim=1)`?

Because:

- each row corresponds to one node
- each column corresponds to one class score

So `argmax(dim=1)` means:

- look across the class scores of each node
- pick the index of the largest one

That index is the predicted class.

---

## Why only use `data.test_mask`?

```python
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
```

We only want to evaluate on the **test nodes**.

`data.test_mask` is a boolean mask that selects only those nodes reserved for testing.

This is important because:

- training nodes were used to fit the model
- test nodes were not seen during training
- test accuracy gives a fairer idea of generalization

So this line compares:

- predicted labels on test nodes
- true labels on test nodes

and counts how many predictions are correct.

---

## Accuracy calculation

```python
acc = int(correct) / int(data.test_mask.sum())
```

This computes the test accuracy.

It is:

- number of correct predictions
- divided by
- total number of test nodes

So if the model predicts 800 test nodes correctly out of 1000, then:

```text
accuracy = 800 / 1000 = 0.8
```

which means **80% accuracy**.

---

## Returning the accuracy

```python
return acc
```

The function returns the final accuracy value.

Then we call:

```python
test_acc = evaluate()
```

and print it with:

```python
print(f"Test Accuracy: {test_acc:.4f}")
```

The `:.4f` part means:

- show the number with 4 decimal places

So an accuracy of `0.812345` would print as:

```text
Test Accuracy: 0.8123
```

---

## Training mode vs evaluation mode

It is useful to remember the difference:

### Training mode
```python
model.train()
```

Used when learning from data.

- dropout is active
- gradients are computed
- parameters are updated

### Evaluation mode
```python
model.eval()
```

Used when testing the model.

- dropout is turned off
- gradients are not needed
- parameters are not updated

---

## Very simple intuition

Training asks:

> how can the model improve its weights?

Evaluation asks:

> now that the weights are learned, how well does the model perform?

So evaluation is the model's **exam**, after training was the **study phase**.

---

## Final summary

This evaluation code does the following:

1. switches the model to evaluation mode
2. disables gradients
3. runs the graph through the model
4. picks the most likely class for each node
5. compares predictions with true labels on the test nodes only
6. computes and returns the test accuracy

In plain English:

> after training, the model predicts the class of each test node, and we measure the proportion of correct predictions