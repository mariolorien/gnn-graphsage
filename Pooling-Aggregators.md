# Pooling Aggregators

Pooling aggregators are functions used in neural networks to combine a set of values or vectors into a single representative value or vector. They are widely used in machine learning models that must process variable-sized inputs, such as **Graph Neural Networks (GNNs)**, **Convolutional Neural Networks (CNNs)**, and **set-based learning models**.

The main purpose of pooling is to compress information from multiple inputs into a fixed-size representation that can be processed by later layers of a neural network.

---

# Why Pooling is Needed

In many machine learning problems, the number of inputs is not fixed.

Examples:# Pooling Aggregators

Pooling aggregators are functions used in neural networks to combine a set of values or vectors into a single representative value or vector. They are widely used in machine learning models that must process variable-sized inputs, such as **Graph Neural Networks (GNNs)**, **Convolutional Neural Networks (CNNs)**, and **set-based learning models**.

The main purpose of pooling is to compress information from multiple inputs into a fixed-size representation that can be processed by later layers of a neural network.

---

# Why Pooling is Needed

In many machine learning problems, the number of inputs is not fixed.

Examples:

- A node in a graph may have a different number of neighbours.
- A region of an image may contain many pixels.
- A dataset may contain sets with variable numbers of elements.

However, neural networks typically require inputs of fixed size. Pooling solves this problem by aggregating many inputs into one vector.

---

# Basic Idea

Suppose we have feature vectors from several elements:

\[
h_1, h_2, h_3, ..., h_n
\]

A pooling function combines them into a single vector:

\[
h_{agg} = POOL(h_1, h_2, ..., h_n)
\]

The **POOL** function determines how the information is summarised.

---

# Common Pooling Aggregators

## 1. Mean Pooling

Mean pooling computes the average of all vectors.

\[
h_{agg} = \frac{1}{n} \sum_{i=1}^{n} h_i
\]

Example:


[2,4], [4,6], [6,8] → [4,6]


Mean pooling captures the **average behaviour of the inputs** and is one of the most commonly used aggregators in graph neural networks.

---

## 2. Sum Pooling

Sum pooling adds all vectors together.

\[
h_{agg} = \sum_{i=1}^{n} h_i
\]

Example:


[2,4] + [4,6] + [6,8] = [12,18]


Sum pooling preserves the **total magnitude of the information** and is used in models such as the **Graph Isomorphism Network (GIN)**.

---

## 3. Max Pooling

Max pooling selects the **maximum value element-wise** across all vectors.

Example:


[2,7], [5,4], [3,9] → [5,9]


Max pooling captures the **strongest signal** among the inputs.

---

## 4. Min Pooling

Min pooling selects the smallest value element-wise across vectors.  
This is less commonly used but follows the same aggregation idea.

---

# Pooling in Graph Neural Networks

In Graph Neural Networks, pooling is used when a node aggregates information from its neighbours during **message passing**.

For a node \(v\) with neighbours \(N(v)\):

\[
h_v^{(k+1)} =
\sigma
\left(
W \cdot
POOL\left(h_u^{(k)} \mid u \in N(v)\right)
\right)
\]

Where:

- \(h_v^{(k)}\) = representation of node \(v\) at layer \(k\)
- \(W\) = learnable weight matrix
- \(\sigma\) = activation function
- \(N(v)\) = set of neighbours of node \(v\)

The pooling operation combines the neighbour features before they are transformed by the neural network layer.

---

# Permutation Invariance

Pooling aggregators must be **permutation invariant**.

This means the result should **not depend on the order of the inputs**.

Example:


POOL(B, C, D) = POOL(D, B, C)


Operations such as **mean**, **sum**, and **max** satisfy this property.

This requirement is essential because **graphs have no natural ordering of neighbours**.

---

# Graph-Level Pooling

Pooling can also be used to create a representation of an **entire graph**.

If a graph has node embeddings:

\[
h_1, h_2, ..., h_n
\]

A graph embedding can be computed as:

\[
h_G = POOL(h_1, h_2, ..., h_n)
\]

This representation can then be used for tasks such as:

- graph classification
- molecule property prediction
- network analysis

---

# Learnable Pooling Methods

More advanced neural networks use **learnable pooling mechanisms** that allow the model to determine which nodes or features are most important.

Examples include:

- Attention pooling
- Hierarchical pooling
- DiffPool
- TopK pooling

These methods extend simple pooling operations by introducing **trainable parameters** that adapt during learning.

---

# Summary

Pooling aggregators are functions that combine multiple input vectors into a single vector in a **permutation-invariant way**.  

They enable neural networks to process **variable-sized inputs** and are a fundamental component of architectures such as **Graph Neural Networks**.

- A node in a graph may have a different number of neighbours.
- A region of an image may contain many pixels.
- A dataset may contain sets with variable numbers of elements.

However, neural networks typically require inputs of fixed size. Pooling solves this problem by aggregating many inputs into one vector.

---

# Basic Idea

Suppose we have feature vectors from several elements:

\[
h_1, h_2, h_3, ..., h_n
\]

A pooling function combines them into a single vector:

\[
h_{agg} = POOL(h_1, h_2, ..., h_n)
\]

The **POOL** function determines how the information is summarised.

---

# Common Pooling Aggregators

## 1. Mean Pooling

Mean pooling computes the average of all vectors.

\[
h_{agg} = \frac{1}{n} \sum_{i=1}^{n} h_i
\]

Example:


[2,4], [4,6], [6,8] → [4,6]


Mean pooling captures the **average behaviour of the inputs** and is one of the most commonly used aggregators in graph neural networks.

---

## 2. Sum Pooling

Sum pooling adds all vectors together.

\[
h_{agg} = \sum_{i=1}^{n} h_i
\]

Example:


[2,4] + [4,6] + [6,8] = [12,18]


Sum pooling preserves the **total magnitude of the information** and is used in models such as the **Graph Isomorphism Network (GIN)**.

---

## 3. Max Pooling

Max pooling selects the **maximum value element-wise** across all vectors.

Example:


[2,7], [5,4], [3,9] → [5,9]


Max pooling captures the **strongest signal** among the inputs.

---

## 4. Min Pooling

Min pooling selects the smallest value element-wise across vectors.  
This is less commonly used but follows the same aggregation idea.

---

# Pooling in Graph Neural Networks

In Graph Neural Networks, pooling is used when a node aggregates information from its neighbours during **message passing**.

For a node \(v\) with neighbours \(N(v)\):

\[
h_v^{(k+1)} =
\sigma
\left(
W \cdot
POOL\left(h_u^{(k)} \mid u \in N(v)\right)
\right)
\]

Where:

- \(h_v^{(k)}\) = representation of node \(v\) at layer \(k\)
- \(W\) = learnable weight matrix
- \(\sigma\) = activation function
- \(N(v)\) = set of neighbours of node \(v\)

The pooling operation combines the neighbour features before they are transformed by the neural network layer.

---

# Permutation Invariance

Pooling aggregators must be **permutation invariant**.

This means the result should **not depend on the order of the inputs**.

Example:


POOL(B, C, D) = POOL(D, B, C)


Operations such as **mean**, **sum**, and **max** satisfy this property.

This requirement is essential because **graphs have no natural ordering of neighbours**.

---

# Graph-Level Pooling

Pooling can also be used to create a representation of an **entire graph**.

If a graph has node embeddings:

\[
h_1, h_2, ..., h_n
\]

A graph embedding can be computed as:

\[
h_G = POOL(h_1, h_2, ..., h_n)
\]

This representation can then be used for tasks such as:

- graph classification
- molecule property prediction
- network analysis

---

# Learnable Pooling Methods

More advanced neural networks use **learnable pooling mechanisms** that allow the model to determine which nodes or features are most important.

Examples include:

- Attention pooling
- Hierarchical pooling
- DiffPool
- TopK pooling

These methods extend simple pooling operations by introducing **trainable parameters** that adapt during learning.

---

# Summary

Pooling aggregators are functions that combine multiple input vectors into a single vector in a **permutation-invariant way**.  

They enable neural networks to process **variable-sized inputs** and are a fundamental component of architectures such as **Graph Neural Networks**.