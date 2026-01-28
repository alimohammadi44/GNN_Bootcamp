# GNN Bootcamp

This repository contains hands-on implementations of **Graph Neural Networks (GNNs)** using **PyTorch Geometric (PyG)**.
It is designed for learning, experimentation, and teaching core GNN concepts through well-known benchmark datasets.

# GNN_Bootcamp
Hands-on bootcamp materials for Graph Neural Networks (GNNs), including tutorials, exercises, and projects.

## Data
Datasets are downloaded and processed automatically using PyTorch Geometric / OGB.
Generated files are not tracked in the repository.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/alimohammadi44/GNN_Bootcamp/blob/main/GCN_Cora.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/alimohammadi44/GNN_Bootcamp/blob/main/GNN_MUTAG.ipynb)




The project focuses on two complementary tasks:
- **Node-level classification** on citation networks
- **Graph-level classification** on molecular graphs

---

## 📁 Contents

- `GCN_Cora.py` / `GCN_Cora.ipynb`  
  Graph Convolutional Network (GCN) and Graph Attention Network (GAT) models for node classification on the Cora dataset.
=======
🔍 Model Comparison: MUTAG vs Cora

Although both experiments use Graph Neural Networks, the nature of the tasks and data is fundamentally different, which motivates different model choices.

🧪 MUTAG: Graph-Level Classification with GIN

For the MUTAG dataset, the task is graph-level classification:
each input graph represents an entire molecule, and the model must predict a single label for the whole graph.

To handle this, the following model is used:

pyg_nn.GINConv(
    nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )
)

Why GIN?

Graph Isomorphism Network (GIN) is designed to be maximally expressive

It has been shown to be as powerful as the Weisfeiler–Lehman (WL) graph isomorphism test

This is crucial for molecular graphs, where small structural differences can change chemical properties

Why MUTAG is more complex

Each sample is a separate graph

Graph sizes vary

The model must:

Learn node representations

Aggregate them using global pooling

Produce a single graph embedding

Pros of GIN (MUTAG)

✅ Very expressive
✅ Strong at capturing subtle structural differences
✅ Well-suited for molecular data

Cons of GIN

❌ More parameters
❌ Higher risk of overfitting on small datasets
❌ Computationally heavier

📚 Cora: Node-Level Classification with GCN and GAT

For Cora, the task is node classification on a single large graph.
Each node (paper) must be classified, not the entire graph.

Two architectures are explored.

🟦 GCN on Cora
self.conv1 = GCNConv(dataset.num_features, hidden_channels)
self.conv2 = GCNConv(hidden_channels, dataset.num_classes)


Graph Convolutional Networks (GCN) perform neighborhood averaging with normalized adjacency.

Pros

Simple and efficient

Strong baseline for citation networks

Easy to train

Cons

Uniformly weights neighbors

Limited expressive power

Can oversmooth with deeper layers

🟨 GAT on Cora
self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
self.conv2 = GATConv(heads * hidden_channels, dataset.num_classes, heads)


Graph Attention Networks (GAT) introduce attention weights to learn which neighbors matter more.

Pros

Learns adaptive neighbor importance

More expressive than GCN

Often improves performance on heterophilic graphs

Cons

More computationally expensive

More hyperparameters (heads, attention)

Slower on large graphs

🧠 Key Differences (Summary)
Aspect	MUTAG (GIN)	Cora (GCN / GAT)
Task	Graph-level classification	Node-level classification
Input	Many small graphs	One large graph
Model focus	Structural expressiveness	Efficient message passing
Architecture	MLP-based aggregation	Linear + neighborhood aggregation
Complexity	Higher	Lower
Risk	Overfitting	Oversmoothing (GCN)
🎯 Takeaway

MUTAG requires a more expressive model because it must distinguish between entire graphs with subtle structural differences → GIN is ideal

Cora focuses on scalable node classification, where simpler architectures like GCN and GAT are sufficient and more efficient

Using multiple models on Cora highlights the trade-off between simplicity (GCN) and expressiveness (GAT)

Together, these experiments demonstrate how task type and data structure drive GNN architecture design.


>>>>>>> ce795850bb6ffc7d8c10ff003a7202093ae47c87

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
