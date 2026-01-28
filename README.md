# GNN Bootcamp

This repository contains hands-on implementations of **Graph Neural Networks (GNNs)** using **PyTorch Geometric (PyG)**.
It is designed for learning, experimentation, and teaching core GNN concepts through well-known benchmark datasets.

The project focuses on two complementary tasks:
- **Node-level classification** on citation networks
- **Graph-level classification** on molecular graphs

---

## 📁 Contents

- `GCN_Cora.py` / `GCN_Cora.ipynb`  
  Graph Convolutional Network (GCN) and Graph Attention Network (GAT) models for node classification on the Cora dataset.

- `GNN_MUTAG.py` / `GNN_MUTAG.ipynb`  
  Graph Isomorphism Network (GIN) model for graph-level classification on the MUTAG dataset.

---

## 📌 GCN_Cora: Node Classification on Citation Networks

The **Cora** dataset is a standard benchmark in graph machine learning.
Nodes represent academic papers, edges represent citation links, and each node has a bag-of-words feature vector.

### Models Used

#### Graph Convolutional Network (GCN)

```python
self.conv1 = GCNConv(dataset.num_features, hidden_channels)
self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
```

#### Graph Attention Network (GAT)

```python
self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
self.conv2 = GATConv(heads * hidden_channels, dataset.num_classes, heads)
```

---

## 📌 GNN_MUTAG: Graph-Level Classification on Molecular Data

The **MUTAG** dataset consists of small molecular graphs.
Each graph represents a chemical compound, where nodes are atoms and edges are chemical bonds.

### Model Used: Graph Isomorphism Network (GIN)

```python
pyg_nn.GINConv(
    nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )
)
```

---

## 🔍 Model Comparison: MUTAG vs Cora

| Aspect | MUTAG (GIN) | Cora (GCN / GAT) |
|------|------------|----------------|
| Task Type | Graph-level | Node-level |
| Input | Many small graphs | One large graph |
| Complexity | Higher | Lower |
| Focus | Structural expressiveness | Efficient message passing |

---

## 🚀 Running the Code

```bash
pip install torch torch-geometric
python GCN_Cora.py
python GNN_MUTAG.py
```

---

## 📜 References

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*
- Veličković et al., *Graph Attention Networks*
- Xu et al., *How Powerful Are Graph Neural Networks?*
