# GNN Bootcamp

This repository contains hands-on implementations of **Graph Neural Networks (GNNs)** using **PyTorch Geometric (PyG)**.
It is designed for learning, experimentation, and teaching core GNN concepts through well-known benchmark datasets.

## Data
Datasets are downloaded and processed automatically using PyTorch Geometric / OGB.
Generated files are not tracked in the repository.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/alimohammadi44/GNN_Bootcamp/blob/main/GCN_Cora.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/alimohammadi44/GNN_Bootcamp/blob/main/GNN_MUTAG.ipynb)

# 🧠 GNN Bootcamp — From Nodes to Graphs

This repository is a **hands-on Graph Neural Networks (GNN) bootcamp** that demonstrates how **task type and dataset structure determine the choice of GNN architecture**.

The bootcamp focuses on two complementary problems in graph machine learning:
- **Graph-level classification** on molecular graphs (MUTAG)
- **Node-level classification** on citation networks (Cora)

Although both tasks rely on message passing, their **objectives, data characteristics, and modeling requirements are fundamentally different**, motivating different GNN designs.

---

## 📊 Dataset Nature

### MUTAG
MUTAG consists of **many small molecular graphs**, where:
- Each graph represents a chemical compound
- Nodes correspond to atoms and edges to chemical bonds
- Each graph has a single label indicating mutagenicity

The task is **graph-level classification**: the model must predict **one label per entire graph**.  
Small structural changes can significantly affect chemical properties, making **structural expressiveness critical**.

---

### Cora
Cora is a **single large citation graph**, where:
- Nodes represent academic papers
- Edges represent citation links
- Node features are sparse bag-of-words vectors
- Each node belongs to a research topic

The task is **node-level classification**: predict the label of each node using its features and neighborhood.  
Because the graph is large and fixed, **scalability and efficient message passing** are essential.

---

# 🧪 Graph-Level Classification on MUTAG — GIN

### Model
```python
pyg_nn.GINConv(
    nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )
)
```

Why GIN?

The Graph Isomorphism Network (GIN) is designed to be maximally expressive among message-passing GNNs.
It has been shown to be as powerful as the Weisfeiler–Lehman (WL) graph isomorphism test, meaning it can distinguish graph structures that simpler GNNs cannot.

This property is especially important for molecular graphs, where small changes in structure (e.g., functional groups or bond patterns) can significantly affect chemical behavior.

For MUTAG, the model must:

Learn informative node embeddings

Aggregate them via global pooling

Produce a single embedding representing the entire molecule

Pros and Cons of GIN (MUTAG)

## Pros

✅ Very strong structural expressiveness

✅ Captures subtle graph differences

✅ Well-suited for molecular and chemical data

## Cons

❌ More parameters than simpler GNNs

❌ Higher risk of overfitting on small datasets

❌ Computationally heavier

# 📚 Node-Level Classification on Cora — GCN and GAT
## Graph Convolutional Network (GCN)
```python
self.conv1 = GCNConv(dataset.num_features, hidden_channels)
self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
```

GCN performs normalized neighborhood averaging, treating all neighbors equally.

Pros

✅ Simple and efficient

✅ Strong baseline for citation networks

✅ Easy to train

Cons

❌ Uniform neighbor weighting

❌ Limited expressive power

❌ Oversmoothing with deeper layers

## Graph Attention Network (GAT)
```python
self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
self.conv2 = GATConv(heads * hidden_channels, dataset.num_classes, heads)
```

GAT introduces attention mechanisms that learn which neighbors are more important.

## Pros

✅ Learns adaptive neighbor importance

✅ More expressive than GCN

✅ Often improves performance on heterophilic graphs

## Cons

❌ More computationally expensive

❌ Additional hyperparameters (attention heads)

❌ Slower on large graphs

## 🔍 MUTAG vs Cora — Key Differences
Aspect	MUTAG (GIN)	Cora (GCN / GAT)
Task type	Graph-level	Node-level
Input	Many small graphs	One large graph
Dataset nature	Molecular structures	Citation network
Model focus	Structural expressiveness	Efficient message passing
Main risk	Overfitting	Oversmoothing (GCN)
## 🧠 Key Takeaways

Graph-level classification requires highly expressive models → GIN

Node-level classification benefits from scalable architectures → GCN / GAT

GAT trades efficiency for expressiveness compared to GCN

GNN architecture should be chosen based on task and dataset characteristics, not popularity

🚀 Running the Code
```python
pip install torch torch-geometric
python GCN_Cora.py
python GNN_MUTAG.py
```

# 📜 References

- Kipf & Welling — Semi-Supervised Classification with Graph Convolutional Networks (https://arxiv.org/abs/1609.02907)

- Veličković et al. — Graph Attention Networks (https://arxiv.org/abs/1710.10903)

- Xu et al. — How Powerful Are Graph Neural Networks?
