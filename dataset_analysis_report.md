# Elliptic Bitcoin Dataset Analysis Report

## Overview

This report presents an analysis of the Elliptic Bitcoin dataset, which contains information about Bitcoin transactions and their connections. The dataset is used for fraud detection in blockchain networks, with the goal of identifying illicit transactions.

## Dataset Statistics

- **Number of nodes (transactions)**: 203,769
- **Number of edges (connections)**: 234,355
- **Number of features per transaction**: 166
- **Number of time steps**: 49

## Class Distribution

The dataset contains three classes of transactions:

- **Illicit transactions**: 4,545 (2.23% of total)
- **Licit transactions**: 42,019 (20.62% of total)
- **Unknown transactions**: 157,205 (77.15% of total)

This shows a significant class imbalance, with illicit transactions being a small minority. This imbalance presents a challenge for fraud detection models, as they need to be able to identify rare events effectively.

## Graph Structure

The analysis of the graph structure reveals:

- **Nodes with edges**: 203,769 (100.00%)
- **Isolated nodes**: 0 (0.00%)

Among edges connecting nodes with known classes:
- **Illicit to illicit connections**: 998 (2.72%)
- **Licit to licit connections**: 33,930 (92.64%)
- **Illicit to licit connections**: 1,696 (4.63%)

This distribution shows that licit transactions tend to connect with other licit transactions, while illicit transactions have a higher proportion of connections to licit transactions compared to their overall representation in the dataset.

## Temporal Distribution

The dataset spans 49 time steps, with varying numbers of transactions per time step. The temporal aspect of the data is crucial for understanding how transaction patterns evolve over time and for developing temporal graph neural networks that can capture these dynamics.

## Feature Analysis

The dataset includes 166 features per transaction. The feature statistics show varying distributions across different features, with some having higher variance than others. These features capture various aspects of the transactions and are essential for the fraud detection task.

## Implications for Fraud Detection

Based on this analysis, several considerations for developing fraud detection models emerge:

1. **Class Imbalance**: Models need to handle the significant imbalance between illicit and licit transactions. Techniques such as weighted loss functions, oversampling, or specialized metrics (e.g., precision-recall AUC instead of ROC AUC) should be employed.

2. **Graph Structure**: The connections between transactions provide valuable information. Graph Neural Networks (GNNs) can leverage this structure to improve detection performance.

3. **Temporal Dynamics**: The temporal nature of the data suggests that models incorporating time information, such as Temporal Graph Neural Networks (TGNs), may perform better than static models.

4. **Feature Engineering**: The large number of features provides rich information, but feature selection or dimensionality reduction might be beneficial to focus on the most discriminative features.

## Conclusion

The Elliptic Bitcoin dataset presents a challenging but realistic fraud detection scenario. The class imbalance, graph structure, and temporal aspects all contribute to the complexity of the task. Graph Neural Networks, particularly those that can incorporate temporal information, are promising approaches for this problem.

The full implementation of GNN-based fraud detection models, including Graph Attention Networks (GAT) and Temporal Graph Neural Networks (TGN), is available in the project repository.
