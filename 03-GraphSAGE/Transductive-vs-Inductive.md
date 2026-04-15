# Transductive vs Inductive Learning in Graph Neural Networks

Graph Neural Networks (GNNs) can operate under two main learning paradigms:

- **Transductive learning**
- **Inductive learning**

Understanding the difference between these two settings is important because many real-world graphs are **large, dynamic, and constantly evolving**.

---

# Transductive Learning

In **transductive learning**, the **entire graph is available during training**.

The model has access to:

- all nodes
- all edges
- all node features

during the training phase.

The model learns **embeddings for all nodes within this fixed graph structure**.

Formally, the model learns a mapping:

\[
f: V \rightarrow \mathbb{R}^d
\]

where:

- \(V\) is the set of nodes in the graph
- \(d\) is the embedding dimension

Each node receives a vector representation (embedding) based on its features and its position in the graph.

### Key characteristics

- The graph is **fixed**
- All nodes are known in advance
- Predictions are made only for nodes that already exist in the graph

### Example

A classic example is **citation networks**, such as:

- **Cora**
- **Citeseer**
- **PubMed**

Each node represents a paper, and edges represent citations.

During training, the model sees the **entire citation network** and learns embeddings for all papers.

Even though only some nodes are labelled, the full graph structure is available.

### Limitation

The model **cannot easily generalise to new nodes or new graphs**.

If new nodes appear, the model usually must be **retrained**.

---

# Inductive Learning

**Inductive learning** refers to the ability of a model to **generalise to unseen data**.

In the context of GNNs, this means the model can generate embeddings and make predictions for:

1. **Nodes that were not present during training**
2. **Entirely new graphs that the model has never encountered before**

Instead of learning a **fixed embedding for each node**, the model learns a **function** that computes node embeddings from:

- node features
- neighbourhood structure

Formally:

\[
h_v = f(x_v, N(v))
\]

where:

- \(x_v\) = feature vector of node \(v\)
- \(N(v)\) = neighbourhood of node \(v\)
- \(h_v\) = embedding of node \(v\)

This means the model learns **how to compute embeddings**, not just **what embeddings should be**.

### Example

Consider a **social network**:

- new users join every day
- new connections are constantly created

A model trained in an **inductive way** can generate embeddings for new users without retraining the entire network.

---

# Why Inductive Learning Matters

Many real-world graphs are:

- **large**
- **dynamic**
- **constantly changing**

Examples include:

- social networks
- recommendation systems
- biological interaction networks
- financial transaction networks

In these systems, retraining the entire graph model every time new nodes appear is impractical.

Inductive GNNs solve this problem.

---

# GraphSAGE: An Inductive Graph Neural Network

**GraphSAGE (Graph Sample and Aggregate)** was specifically designed to address the **inductive learning challenge**.

Instead of learning node embeddings directly, GraphSAGE learns a **function that generates embeddings** by aggregating information from a node's local neighbourhood.

This makes it possible to compute embeddings for:

- unseen nodes
- evolving graphs
- entirely new graphs

Because the learned function depends only on **features and local structure**, the model can generalise beyond the training graph.

This makes GraphSAGE highly scalable and suitable for **real-world AI systems with dynamic graph data**.

---

# GraphSAGE Embedding Generation

The process of generating a node embedding in GraphSAGE involves two main steps, repeated for each layer of the network:

1. **Neighbourhood Sampling**
2. **Information Aggregation**

---

# 1. Neighbourhood Sampling

In very large graphs, a node might have:

- thousands
- millions

of neighbours.

Aggregating information from all neighbours would be computationally expensive.

GraphSAGE solves this problem by **sampling a fixed number of neighbours** for each node at each layer.

For example:

- Sample **25 neighbours at layer 1**
- Sample **10 neighbours at layer 2**

This creates a manageable computation tree.

### Example

Suppose node **A** has 1,000 neighbours.

Instead of using all of them, GraphSAGE might sample:


Layer 1: sample 25 neighbours
Layer 2: sample 10 neighbours for each of those


This dramatically reduces computational cost.

---

# 2. Information Aggregation

After sampling neighbours, their feature vectors are **aggregated**.

GraphSAGE proposes several **learnable aggregation functions** that combine neighbour information in different ways.

The aggregated information is then combined with the node's own features to produce a new embedding.

---

# Common Aggregation Functions in GraphSAGE

GraphSAGE introduced several types of aggregators.

---

## 1. Mean Aggregator

The **mean aggregator** computes the element-wise mean of the sampled neighbour features.

\[
h_{N(v)} = \frac{1}{|N(v)|} \sum_{u \in N(v)} h_u
\]

Example:

Neighbour feature vectors:


[2,4], [4,6], [6,8]


Mean aggregation:


[4,6]


This is the **simplest and most commonly used aggregator**.

---

## 2. LSTM Aggregator

The **LSTM aggregator** uses a **Long Short-Term Memory (LSTM)** neural network to combine neighbour features.

LSTMs are normally used for **sequential data** such as text or time series.

However, graphs do not have an inherent order.

To address this, GraphSAGE:

- randomly **permutes the order of neighbours**
- processes them sequentially with the LSTM

This introduces a learnable aggregation mechanism while maintaining approximate permutation invariance.

---

## 3. Pooling Aggregators (Max or Mean Pooling)

Pooling aggregators apply a **neural transformation to each neighbour independently**, then aggregate the results.

Step-by-step process:

1. Each neighbour feature vector is passed through a **fully connected neural layer**

\[
z_u = \sigma(W h_u + b)
\]

2. The transformed features are aggregated using **pooling**

Examples:

### Max Pooling

Select the maximum value element-wise.

Example:


Transformed features:
[1,7], [5,4], [3,9]

Max pooling:
[5,9]


### Mean Pooling

Compute the average of the transformed features.

This approach allows the model to **learn richer representations** before aggregation.

---

# GraphSAGE Layer Update

After aggregation, the node updates its representation:

\[
h_v^{k+1} = \sigma \left( W \cdot [h_v^k \parallel h_{N(v)}^k] \right)
\]

where:

- \(h_v^k\) = current node embedding
- \(h_{N(v)}^k\) = aggregated neighbour embedding
- \(\parallel\) = concatenation
- \(W\) = learnable weight matrix
- \(\sigma\) = activation function

This update is repeated for multiple layers to allow information to propagate across the graph.

---

# Why GraphSAGE is Important

GraphSAGE introduced several key ideas that made GNNs scalable:

- **Neighbourhood sampling** reduces computation
- **Inductive embedding generation** allows generalisation
- **Flexible aggregation functions** allow expressive representations

Because of these properties, GraphSAGE is widely used in:

- recommendation systems
- fraud detection
- social network analysis
- biological networks
- knowledge graphs

---

# Summary

GraphSAGE enables **inductive graph learning** by learning a function that generates node embeddings based on:

- node features
- sampled neighbourhoods
- aggregation functions

This allows the model to produce embeddings for **new nodes and new graphs**, making it suitable for large-scale and dynamic graph applications.