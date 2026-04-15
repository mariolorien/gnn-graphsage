## GCNConv vs GraphSAGE (SAGEConv)

Both **GCNConv** and **SAGEConv** are graph convolution layers used in GNNs, and both update a node by using information from its neighbours. The main difference is in **how the node combines its own features with the neighbours' features**.

### Important clarification

There is not really a big difference between **"neighbour information"** and **"neighbour features"** in this context. In practice, both layers are mostly passing and combining **feature vectors** or **hidden representations** from neighbouring nodes.

So the real difference is **not the word information vs features**.

The real difference is:

- **GCNConv** mixes the node and its neighbours together first, then applies a transformation.
- **SAGEConv** first aggregates the neighbours, keeps the node's own features separate, and then combines the two parts.

---

## 1. GCNConv formula

\[
x'_i = \Theta^T \sum_{j \in \mathcal{N}(i)\cup\{i\}} \frac{e_{j,i}}{\sqrt{\hat d_j \hat d_i}} x_j
\]

### What this means

For node \(i\):

- take the features of all its neighbours
- also include the node itself through the self-loop term \(\{i\}\)
- normalise each contribution using the node degrees
- add everything together
- apply a learned transformation

### Intuition

GCN says:

> "Mix me and my neighbours into one normalised average-like combination, then transform the result."

So in GCN, the node's own features and the neighbours' features are blended together early.

---

## 2. GraphSAGE / SAGEConv formula

\[
x'_i = W_1 x_i + W_2 \cdot \mathrm{mean}_{j \in \mathcal{N}(i)} x_j
\]

### What this means

For node \(i\):

- keep the node's own features \(x_i\) as one separate part
- compute a summary of the neighbours, usually their mean
- apply one learned weight matrix to the node itself
- apply another learned weight matrix to the neighbour summary
- combine the two

### Intuition

GraphSAGE says:

> "First summarise my neighbours, keep my own features separate, then combine both parts."

So in GraphSAGE, the node's own features are treated more explicitly as separate from the aggregated neighbours.

---

## Main conceptual difference

### GCNConv

\[
\text{new node} = \text{transform}\big(\text{normalised sum of self + neighbours}\big)
\]

### GraphSAGE

\[
\text{new node} = \text{transform(self)} + \text{transform(neighbour summary)}
\]

---

## Short comparison

| Layer | Self node included how? | Neighbours included how? | Main idea |
|---|---|---|---|
| **GCNConv** | Included directly inside the same sum as neighbours | Combined in one normalised aggregation | Mix self and neighbours first, then transform |
| **SAGEConv** | Kept as a separate term | Aggregated first, often by mean | Keep self separate, summarise neighbours, then combine |

---

## Simple summary

Both layers use neighbour features.

The main difference is:

- **GCNConv**: self and neighbours are merged together into one normalised combination
- **SAGEConv**: neighbours are aggregated separately, then combined with the node's own features

So the distinction is not really **information vs features**, but rather **how self and neighbours are mathematically combined**.