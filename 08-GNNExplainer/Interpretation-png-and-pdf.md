# Interpretation of the GNNExplainer Outputs

## 1. Feature importance plot

The feature-importance plot shows which **input features of the explained node** were most influential for the model’s prediction.

In this example, the most important features are:

- **495**
- **507**
- **1263**
- **1177**

These are followed by features such as **1151**, **99**, **140**, **1381**, **1262**, and **774**, which also contribute, but less strongly.

### How to interpret this

This means that, for the specific node being explained, the GNN relied more heavily on these feature dimensions than on the others when producing its prediction.

Because this example uses the **Cora** dataset, these feature numbers are usually not directly human-readable concepts. They are typically just **column indices in the node feature matrix**. In Cora, node features are often represented as a bag-of-words vector, so each number refers to one particular input feature column.

The values shown on the bars should be interpreted as **relative importance scores**, not probabilities. A larger value means the explainer judged that feature to be more influential for the prediction of this particular node.

### Important caution

This is a **local explanation**, not a global one.

So it does **not** mean that feature 495 is the most important feature in the whole dataset or for the model overall. It only means that, for this specific node and this specific prediction, that feature was especially important.

---

## 2. Graph explanation PDF

The graph explanation shows the **subgraph structure** that GNNExplainer considered most relevant for the prediction.

In other words, it is trying to answer:

> Which nearby nodes and edges were most important for this node being classified the way it was?

### How to interpret this

The PDF represents the **local neighbourhood explanation** around the target node.

This means the model’s prediction was not based only on the target node’s own features, but also on information coming from its surrounding graph structure.

The important point is that this explanation concerns the **structural side** of the decision:

- which neighbouring nodes matter
- which edges matter
- which local connectivity pattern supports the prediction

In this case, the explanation graph still looks quite dense, which suggests that the model may be relying on a **fairly broad local neighbourhood** rather than only one or two very specific neighbours.

### Practical reading of the graph

The graph should not be read as if every node ID in the picture is equally important.

Instead, it should be interpreted more generally as:

- the prediction depends on a local citation/network neighbourhood
- several surrounding nodes and edges appear relevant
- the explanation is structural rather than purely feature-based

So the graph explanation tells us **where in the graph** the important evidence is located.

---

## 3. Combined interpretation

Together, the two outputs give two complementary explanations:

### Feature importance plot
Shows **which node features** mattered most.

### Graph explanation
Shows **which local graph structure** mattered most.

This is exactly the main purpose of GNNExplainer:

- to identify the important **features**
- and the important **subgraph**

behind a specific prediction.

---

## 4. Final summary

In this example:

- the **bar chart** suggests that a small set of feature dimensions played the strongest role in the node’s classification
- the **graph explanation** suggests that the model also relied on a non-trivial local neighbourhood structure

So the model’s decision appears to be based on **both**:

1. strong signals from a handful of node features, and  
2. evidence coming from the surrounding graph connectivity.

This is a typical GNN behaviour: the prediction is not driven only by the node itself, but also by information aggregated from its neighbours.