# GNN Fraud Detection Model Comparison Report

## Overview

This report compares the performance of two Graph Neural Network (GNN) models for fraud detection in blockchain transactions:

1. **Graph Attention Network (GAT)**: A static graph neural network that uses attention mechanisms to weigh neighbor nodes differently.
2. **Temporal Graph Neural Network (TGN)**: A dynamic graph neural network that incorporates temporal information to capture the evolution of the graph over time.

## Dataset

The models were trained and evaluated on the Elliptic Bitcoin dataset, which contains:
- Bitcoin transactions represented as nodes
- Transaction flows represented as directed edges
- Node features including transaction characteristics
- Binary labels indicating illicit (1) or licit (0) transactions

## Model Performance

### Key Metrics Comparison

| Metric | GAT | TGN |
|--------|-----|-----|
| Accuracy | 0.7668 | 0.8386 |
| Precision | 0.2946 | 0.3764 |
| Recall | 0.9963 | 0.9952 |
| F1 Score | 0.4548 | 0.5462 |
| AUC | 0.9849 | 0.9882 |
| AUPRC | 0.8939 | 0.9059 |

### Analysis

- **TGN outperforms GAT** across most metrics, demonstrating the value of incorporating temporal information in fraud detection.
- **High Recall for Both Models**: Both models achieve excellent recall (>99.5%), indicating they successfully identify almost all fraudulent transactions.
- **Precision is Low**: Both models have relatively low precision, suggesting many false positives, though TGN performs better.
- **Strong AUC and AUPRC**: The high AUC and AUPRC values indicate that both models have good discriminative power, with TGN slightly outperforming GAT.

## Visualizations

The following visualizations are available in the results directory:
- ROC curves comparing both models
- Precision-Recall curves comparing both models
- Confusion matrices for each model
- Training history plots showing loss and metrics over epochs

## Conclusions

1. **TGN is Superior**: The Temporal Graph Neural Network demonstrates better overall performance, particularly in terms of accuracy, precision, and F1 score.
2. **Trade-off Consideration**: Both models prioritize recall over precision, which is appropriate for fraud detection where missing fraudulent transactions is more costly than investigating legitimate ones.
3. **Future Improvements**: 
   - Further tuning of model hyperparameters could improve precision
   - Incorporating additional features or graph structures might enhance performance
   - Ensemble methods combining multiple models could provide more robust predictions

## Next Steps

1. **Model Deployment**: Implement the TGN model in a production environment for real-time fraud detection
2. **Monitoring System**: Develop a system to track model performance over time and detect concept drift
3. **Continuous Learning**: Implement a feedback loop to incorporate new labeled data for model retraining
4. **Explainability**: Develop methods to explain model predictions to help investigators understand why transactions are flagged as fraudulent
