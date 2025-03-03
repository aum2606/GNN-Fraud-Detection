# GNN Fraud Detection Visualization Summary

This document provides an overview of the visualizations generated for the Graph Neural Network (GNN) fraud detection project.

## Graph Structure Visualizations

### 1. Full Graph Visualization
- **Filename**: `full_graph_visualization.png`
- **Description**: A visualization of the entire Bitcoin transaction graph (sampled to 500 nodes for clarity).
- **Color Scheme**: Nodes are colored based on their class (red for fraudulent, blue for legitimate).
- **Purpose**: Provides an overview of the transaction network structure.

### 2. Fraudulent Subgraph Visualization
- **Filename**: `fraudulent_subgraph_visualization.png`
- **Description**: A visualization of a subgraph containing fraudulent transactions and their neighbors.
- **Color Scheme**: Nodes are colored based on their class (red for fraudulent, blue for legitimate).
- **Purpose**: Highlights patterns and clusters of fraudulent activity within the network.

### 3. Class Graph Visualization
- **Filename**: `class_graph_visualization.png`
- **Description**: A visualization of the transaction graph with nodes colored by their class.
- **Color Scheme**: Viridis colormap (fraudulent transactions in warmer colors).
- **Purpose**: Shows the distribution of fraudulent transactions across the network.

## Model Prediction Visualizations

### 1. Prediction Visualization
- **Filename**: `prediction_visualization.png`
- **Description**: A visualization of the test set with nodes colored by their predicted probability of being fraudulent.
- **Color Scheme**: RdYlBu_r colormap (red for high fraud probability, blue for low).
- **Purpose**: Illustrates how the model's predictions align with the graph structure.

### 2. Transaction Analysis Visualizations
- **Filenames**: `transaction_XXXXX_analysis.png` (where XXXXX is the transaction index)
- **Description**: Detailed visualizations of specific transactions and their local neighborhoods.
- **Color Scheme**: Nodes are colored based on their predicted fraud probability.
- **Purpose**: Provides insights into how the model makes decisions for specific transactions by analyzing their local graph context.

## Performance Metric Visualizations

These visualizations are generated during model training and evaluation:

### 1. Training History
- **Description**: Line plots showing the training and validation loss/accuracy over epochs.
- **Purpose**: Monitors model convergence and potential overfitting.

### 2. ROC Curves
- **Description**: Receiver Operating Characteristic curves showing the trade-off between true positive rate and false positive rate.
- **Purpose**: Evaluates model performance across different classification thresholds.

### 3. Precision-Recall Curves
- **Description**: Curves showing the trade-off between precision and recall.
- **Purpose**: Particularly useful for imbalanced datasets like ours where fraudulent transactions are rare.

### 4. Confusion Matrices
- **Description**: Visual representation of the model's predictions versus actual labels.
- **Purpose**: Shows the distribution of true positives, false positives, true negatives, and false negatives.

## Model Comparison Visualizations

These visualizations compare the performance of different GNN models:

### 1. Metric Comparison Bar Charts
- **Description**: Bar charts comparing key metrics (accuracy, precision, recall, F1, AUC-ROC, AUC-PR) across models.
- **Purpose**: Provides a clear visual comparison of model performance.

### 2. ROC Curve Comparison
- **Description**: Multiple ROC curves plotted on the same axes for different models.
- **Purpose**: Compares the trade-off between true positive rate and false positive rate across models.

### 3. Precision-Recall Curve Comparison
- **Description**: Multiple precision-recall curves plotted on the same axes for different models.
- **Purpose**: Compares the trade-off between precision and recall across models.

## How to Generate Visualizations

1. **Graph Structure Visualizations**:
   ```
   python generate_graph_visualizations.py
   ```

2. **Transaction Analysis Visualizations**:
   ```
   python simple_inference.py --model_type tgn --transaction_idx <index> --visualize
   ```

3. **Performance Metric Visualizations**:
   These are automatically generated during model training and evaluation:
   ```
   python -m src.main --model_type gat --train --evaluate
   python -m src.main --model_type tgn --train --evaluate
   ```

4. **Model Comparison Visualizations**:
   ```
   python compare_gnn_models.py
   ```

All visualizations are saved in the `results` directory.
