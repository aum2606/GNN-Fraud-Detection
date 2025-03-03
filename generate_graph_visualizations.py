"""
Generate graph visualizations for the fraud detection dataset.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from torch_geometric.utils import subgraph

from src.config import RESULTS_DIR
from src.data import load_elliptic_bitcoin_dataset
from src.utils.visualization import visualize_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_elliptic_bitcoin_dataset(root="./data")
    data = dataset.data
    
    # Get train/val/test masks
    train_mask, val_mask, test_mask = dataset.get_train_val_test_masks()
    
    # Visualize full graph (with a sample of nodes)
    logger.info("Generating full graph visualization...")
    visualize_graph(
        data,
        node_colors=data.y,
        title="Bitcoin Transaction Graph (Sample)",
        save_path=os.path.join(RESULTS_DIR, "full_graph_visualization.png"),
        max_nodes=500,
        node_size=30,
        cmap='coolwarm'
    )
    
    # Visualize fraudulent subgraph
    logger.info("Generating fraudulent subgraph visualization...")
    # Get indices of fraudulent nodes
    fraud_indices = torch.where(data.y == 1)[0]
    # Sample a subset if there are too many
    if len(fraud_indices) > 500:
        fraud_indices = fraud_indices[:500]
    
    # Get subgraph containing only fraudulent nodes and their neighbors
    edge_index, _ = subgraph(
        fraud_indices, 
        data.edge_index, 
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )
    
    # Create a new data object for the subgraph
    fraud_data = data.clone()
    fraud_data.edge_index = edge_index
    
    # Create node colors for the subgraph (1 for fraudulent, 0 for licit)
    node_colors = torch.zeros(len(fraud_indices))
    for i, idx in enumerate(fraud_indices):
        if data.y[idx] == 1:
            node_colors[i] = 1
    
    visualize_graph(
        fraud_data,
        node_colors=node_colors,
        title="Fraudulent Transaction Subgraph",
        save_path=os.path.join(RESULTS_DIR, "fraudulent_subgraph_visualization.png"),
        max_nodes=500,
        node_size=50,
        cmap='RdYlBu_r'
    )
    
    # Visualize temporal graph (nodes colored by time step)
    logger.info("Generating temporal graph by class visualization...")
    # Use class labels instead of time step since time_step is not available
    visualize_graph(
        data,
        node_colors=data.y,
        title="Bitcoin Transaction Graph by Class (Sample)",
        save_path=os.path.join(RESULTS_DIR, "class_graph_visualization.png"),
        max_nodes=500,
        node_size=30,
        cmap='viridis'
    )
    
    # Visualize test set predictions
    logger.info("Generating prediction visualization...")
    # Load the best model (TGN)
    model_path = os.path.join(RESULTS_DIR, "tgn_model.pt")
    if os.path.exists(model_path):
        from src.models import TemporalGNN
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TemporalGNN(
            in_channels=data.x.size(1),
            hidden_channels=64,
            out_channels=1,
            num_layers=3,
            dropout=0.2,
            residual=True,
            use_batch_norm=True
        )
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Get predictions for test set
        test_indices = torch.where(test_mask)[0]
        # Sample a subset if there are too many
        if len(test_indices) > 500:
            test_indices = test_indices[:500]
        
        # Get subgraph containing only test nodes
        edge_index, _ = subgraph(
            test_indices, 
            data.edge_index, 
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )
        
        # Create a new data object for the test subgraph
        test_data = data.clone()
        test_data.edge_index = edge_index
        
        # Get predictions
        with torch.no_grad():
            test_data = test_data.to(device)
            logits, _ = model(test_data.x, test_data.edge_index)
            predictions = torch.sigmoid(logits)
        
        # Create node colors based on predictions
        node_colors = predictions.cpu().squeeze()
        
        visualize_graph(
            test_data,
            node_colors=node_colors,
            title="Test Set Predictions (Probability of Fraud)",
            save_path=os.path.join(RESULTS_DIR, "prediction_visualization.png"),
            max_nodes=500,
            node_size=50,
            cmap='RdYlBu_r'
        )
    else:
        logger.warning(f"Model file {model_path} not found. Skipping prediction visualization.")
    
    logger.info("Graph visualizations generated successfully.")

if __name__ == "__main__":
    main()
