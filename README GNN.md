# Blockchain Fraud Detection using GNN and XGBoost

This project investigates fraud detection in blockchain transaction networks using:

Graph Neural Networks (GCN, GAT, GraphSAGE)

XGBoost (feature-based baseline)

Temporal evaluation protocol

Threshold tuning

Weighted loss for class imbalance

The goal is to compare graph-based learning vs classical machine learning under realistic conditions.

# 1️⃣ Graph Neural Networks (GNN)

Blockchain transactions naturally form a graph:

Nodes → Transactions

Edges → Flow of cryptocurrency

Node features → Engineered behavioral statistics

Traditional ML models treat transactions independently.
GNNs leverage both:

Node features

Graph structure (who transacts with whom)

This allows the model to capture fraud propagation patterns.

# 🔵 GCN (Graph Convolutional Network)

GCN aggregates information from neighboring nodes using a normalized adjacency matrix.

Idea:
Each node updates its representation by averaging its neighbors.

Strengths:

Simple

Stable

Efficient

Weaknesses:

Assumes all neighbors are equally important

Can oversmooth with depth

# 🟣 GAT (Graph Attention Network)

GAT introduces attention mechanisms to learn which neighbors are more important.

Idea:
Each neighbor gets a learnable weight (attention score).

Strengths:

More expressive than GCN

Learns importance dynamically

Weaknesses:

More parameters

Sensitive to hyperparameters

Can overfit

# 🟢 GraphSAGE

GraphSAGE learns aggregation functions to generate node embeddings.

Idea:
Instead of fixed averaging, it learns how to combine neighbor features.

Strengths:

Works well in inductive settings

Robust for unseen nodes

Performs well in temporal splits

Weaknesses:

Slightly more complex than GCN

# 2️⃣ Dataset: Elliptic Bitcoin Dataset

This project uses the Elliptic Bitcoin Transaction Dataset, a real-world blockchain dataset widely used in fraud detection research.

Dataset Characteristics

~200,000 transactions

166 engineered node features

49 temporal steps

Labels:

1 → Illicit (Fraud)

0 → Licit

Unknown (removed)

Graph Structure

Nodes: Bitcoin transactions

Edges: Money flow between transactions

Time step available per transaction

This dataset is highly imbalanced (~10% fraud).

3️⃣ Why Temporal Split Is Important

A random train/test split causes data leakage.

In blockchain:

Transactions are time-ordered.

Fraud patterns evolve over time.

If we randomly split:

Future transactions appear in training.

The model indirectly learns future information.

Results become overly optimistic.

Correct Approach

We use a temporal split:

Train on earlier time steps (e.g., ≤ 34)

Test on later time steps (> 34)

This simulates real-world deployment:

Train on past → Detect fraud in the future

Temporal split significantly reduces inflated performance.

4️⃣ Why Softmax Threshold Is Important

Neural networks output probabilities using softmax.

Default classification uses:

argmax → equivalent to threshold = 0.5


However, fraud detection is imbalanced.

Using threshold = 0.5 often:

Over-predicts fraud

Or under-detects fraud

By tuning threshold (e.g., 0.6):

Precision and recall can be balanced

F1-score improves

Model becomes operationally useful

Threshold tuning controls the trade-off between:

False positives

False negatives

This is critical in fraud detection systems.

5️⃣ Why Weighted Loss Is Important

Fraud detection datasets are imbalanced.

In Elliptic:

~90% licit

~10% fraud

Without weighting:

Model learns to predict majority class

Fraud recall becomes very low

Weighted CrossEntropy loss penalizes fraud misclassification more heavily.

Balanced weight formula:

𝑤
𝑐
=
𝑁
2
⋅
𝑛
𝑐
w
c
	​

=
2⋅n
c
	​

N
	​


Where:

𝑁
N = total training samples

𝑛
𝑐
n
c
	​

 = samples in class 
𝑐
c

This improves fraud recall significantly.

6️⃣ Experimental Results

Temporal split + weighted loss + threshold tuning.

Ranking by Fraud F1-score
Model	Fraud F1	Fraud Recall
XGBoost	0.7462	0.7396
GraphSAGE	0.4898	0.7128
GCN	0.4624	0.6297
GAT	0.3676	0.7378
Interpretation
🔹 XGBoost

Strongest overall.

Engineered features already capture strong fraud signals.

Performs best under temporal split.

🔹 GraphSAGE

Best performing GNN.

Handles inductive setting well.

Graph structure provides moderate benefit.

🔹 GAT

Very high recall.

Low precision → lower F1.

Sensitive to threshold tuning.

🔹 GCN

Stable baseline.

Simpler aggregation.

Moderate performance.

Key Insights

Random split inflates performance.

Temporal split gives realistic evaluation.

Feature-based models are very strong on Elliptic.

Graph structure provides additional but limited gain.

Threshold tuning significantly impacts neural models.

Weighted loss is essential for minority class detection.

Conclusion

This study demonstrates that:

Proper evaluation protocol matters more than model complexity.

Classical ML (XGBoost) can outperform GNNs when strong engineered features exist.

Graph models require careful calibration in temporal settings.

Fraud detection systems must balance recall and precision via threshold tuning.