## Node Classification vs Link Prediction in GNNs

| Aspect | Node Classification | Link Prediction |
|---|---|---|
| **Main goal** | Predict a **label for each node** | Predict whether an **edge exists between two nodes** |
| **What is the model learning about?** | Individual nodes | Pairs of nodes |
| **Typical output of `forward()`** | Node logits / class scores | Node embeddings |
| **Do we need a decoder?** | **No**. The output of `forward()` is already the prediction target | **Yes**. We need a decoder to turn two node embeddings into an edge score |
| **Typical prediction target** | `y` for nodes | `edge_label` for candidate edges |
| **What does the model compare against during training?** | Predicted node logits vs true node labels | Predicted edge scores vs true edge labels (`1` real edge, `0` fake edge) |
| **Typical loss** | `BCEWithLogitsLoss()` for multi-label node classification (like PPI) | `BCEWithLogitsLoss()` for binary edge classification |
| **Need sigmoid before loss?** | **No**, if using `BCEWithLogitsLoss` | **No**, if using `BCEWithLogitsLoss` |
| **Need sigmoid for evaluation / interpretation?** | Sometimes, if you want probabilities | Usually yes, to convert raw edge scores into probabilities |
| **Typical evaluation metric** | `micro F1` | `AUC` / `ROC-AUC` |
| **Prediction line** | `preds = (out > 0).float()` | `prob = torch.sigmoid(score)` or `out = torch.sigmoid(out)` |
| **What does one row in output represent?** | One node | One candidate edge score after decoding |
| **What is being batched?** | Usually graphs or nodes depending on dataset | Candidate edges are scored after node embeddings are computed |
| **Need candidate pairs?** | **No** | **Yes** |
| **Why candidate pairs are needed?** | Because the task is on nodes directly | Because the model must be told which node pairs to score |
| **Need negative sampling?** | Usually no | Usually yes, to create fake non-edges for training |
| **Need `RandomLinkSplit`?** | Usually no | Often yes, to split edges into train/val/test |
| **What is split into train/val/test?** | Nodes or graphs, depending on dataset | **Edges**, not nodes |
| **Example dataset structure** | PPI already comes with train / val / test graphs | A single graph can be split into train / val / test edges using `RandomLinkSplit` |

---

## Encoder differences by layer type

| Part | Node Classification with GCN | Node Classification with GraphSAGE | Link Prediction with GCN | Link Prediction with GraphSAGE |
|---|---|---|---|---|
| **Typical layer used** | `GCNConv` | `SAGEConv` | `GCNConv` | `SAGEConv` |
| **Parent class** | `torch.nn.Module` | `torch.nn.Module` | `torch.nn.Module` | `torch.nn.Module` |
| **Encoding meaning** | Create node embeddings / hidden representations | Create node embeddings / hidden representations | Create node embeddings / hidden representations | Create node embeddings / hidden representations |
| **Forward pass output** | Node logits or node representations | Node logits or node representations | Node embeddings | Node embeddings |
| **Core idea** | Aggregate self + neighbours with normalization | Aggregate neighbours, keep self separate, then combine | Same encoder idea, but used before decoding edges | Same encoder idea, but used before decoding edges |

---

## Practical code differences

| Practical item | Node Classification | Link Prediction |
|---|---|---|
| **Typical model call** | `out = model(data.x, data.edge_index)` | `z = model(data.x, data.edge_index)` |
| **Meaning of model output** | `out` = logits for each node and class | `z` = embedding for each node |
| **Need edge pairs to score?** | No | Yes: `edge_label_index` |
| **Typical unpacking of candidate edges** | Not needed | `row, col = edge_label_index` |
| **Decoder example** | Not needed | `(z[row] * z[col]).sum(dim=1)` |
| **True target tensor** | `data.y` | `edge_label` |
| **Thresholding for predictions** | `(out > 0).float()` if using logits | `torch.sigmoid(score)` then threshold if needed |
| **Negative examples** | Not needed in the same way | Created with `negative_sampling(...)` or via split transform |
| **Train/val/test split helper** | Usually dataset already provides it | `RandomLinkSplit(...)` is common |
| **Special PyG transform needed?** | Usually no | Often yes: `RandomLinkSplit` |

---

## Code examples by task and layer

### 1. Node Classification with GCN

```python
from torch_geometric.nn import GCNConv

class GCNNodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x