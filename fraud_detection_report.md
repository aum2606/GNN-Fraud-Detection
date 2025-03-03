# Fraud Detection in Blockchain Networks: Results Report

## Project Overview

This report presents the results of our fraud detection system for blockchain networks, specifically focusing on the Elliptic Bitcoin dataset. The project aims to identify illicit transactions in a blockchain network using machine learning techniques.

## Dataset Analysis

The Elliptic Bitcoin dataset contains:
- **203,769 nodes (transactions)**
- **234,355 edges (connections)**
- **166 features per transaction**
- **49 time steps**

The dataset exhibits significant class imbalance:
- **Illicit transactions**: 4,545 (2.23%)
- **Licit transactions**: 42,019 (20.62%)
- **Unknown transactions**: 157,205 (77.15%)

## Model Performance

We trained a Random Forest classifier with class balancing to address the imbalance in the dataset. The model was evaluated on a temporal test set to simulate real-world deployment conditions.

### Validation Set Results
- **Accuracy**: 97.54%
- **Precision**: 46.83%
- **Recall**: 86.52%
- **F1 Score**: 60.77%
- **AUC-ROC**: 97.12%

### Test Set Results
- **Accuracy**: 97.72%
- **Precision**: 23.37%
- **Recall**: 46.57%
- **F1 Score**: 31.12%
- **AUC-ROC**: 87.08%

### Confusion Matrix Analysis
- **True Positives**: 190 (correctly identified illicit transactions)
- **False Positives**: 623 (licit transactions incorrectly flagged as illicit)
- **True Negatives**: 35,793 (correctly identified licit transactions)
- **False Negatives**: 218 (illicit transactions missed)

## Key Insights

1. **High Accuracy but Lower Precision**: The model achieves high overall accuracy (97.72%) but relatively low precision (23.37%) on the test set. This indicates that while the model is good at classifying transactions overall, it tends to generate a significant number of false positives when identifying illicit transactions.

2. **Recall Drop from Validation to Test**: The recall drops significantly from the validation set (86.52%) to the test set (46.57%). This suggests that the model's ability to identify illicit transactions deteriorates over time, possibly due to evolving patterns in fraudulent behavior.

3. **Class Imbalance Challenge**: The significant class imbalance (only 2.23% of transactions are illicit) makes the detection task challenging. Despite using class weights to balance the training, the model still struggles with precision.

4. **Temporal Effects**: The performance drop between validation and test sets suggests that temporal dynamics play a crucial role in fraud detection. Future models should better account for evolving patterns over time.

## Recommendations

1. **Implement Graph Neural Networks**: The current model doesn't fully leverage the graph structure of the transactions. Implementing Graph Neural Networks (GNNs) like Graph Attention Networks (GAT) or Temporal Graph Neural Networks (TGN) could improve performance by capturing relationships between transactions.

2. **Feature Engineering**: Analyze the top features identified by the Random Forest model and create additional derived features that might help distinguish between illicit and licit transactions.

3. **Anomaly Detection Approach**: Consider complementing the classification approach with anomaly detection techniques, which might be more effective at identifying novel fraud patterns.

4. **Ensemble Methods**: Combine multiple models with different strengths to improve overall performance, especially for capturing different aspects of fraudulent behavior.

5. **Active Learning**: Implement an active learning framework where human analysts review uncertain predictions, providing feedback to improve the model over time.

## Conclusion

The Random Forest model provides a solid baseline for fraud detection in blockchain networks, achieving high accuracy and reasonable AUC-ROC scores. However, the precision and recall metrics indicate room for improvement, particularly in reducing false positives while maintaining the ability to detect illicit transactions.

The next steps involve implementing more sophisticated models that can better leverage the graph structure and temporal dynamics of the blockchain network, as well as refining the feature engineering process to better capture indicators of fraudulent activity.

## Visualizations

The following visualizations have been generated to help understand the model's performance:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance Analysis

These visualizations can be found in the `results` directory.
