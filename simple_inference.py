"""
Simple inference script for the GNN Fraud Detection model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from torch_geometric.utils import subgraph

from src.config import RESULTS_DIR
from src.data import load_elliptic_bitcoin_dataset
from src.models import GAT, TemporalGNN
from src.utils.visualization import visualize_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_type, data, device):
    """
    Load a trained model.
    
    Args:
        model_type: Type of model to load (gat or tgn).
        data: Dataset object.
        device: Device to use.
        
    Returns:
        Loaded model.
    """
    # Initialize model
    if model_type == 'gat':
        model = GAT(
            in_channels=data.x.size(1),
            hidden_channels=64,
            out_channels=1,
            num_layers=3,
            dropout=0.2,
            residual=True,
            use_batch_norm=True
        )
    elif model_type == 'tgn':
        model = TemporalGNN(
            in_channels=data.x.size(1),
            hidden_channels=64,
            out_channels=1,
            num_layers=3,
            dropout=0.2,
            residual=True,
            use_batch_norm=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model_path = os.path.join(RESULTS_DIR, f"{model_type}_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def predict_transaction(model, data, transaction_idx, device, model_type='gat'):
    """
    Predict whether a transaction is fraudulent.
    
    Args:
        model: Trained model.
        data: Dataset object.
        transaction_idx: Index of the transaction to predict.
        device: Device to use.
        model_type: Type of model (gat or tgn).
        
    Returns:
        Prediction probability.
    """
    # Get the transaction and its neighbors
    # Extract a subgraph centered around the transaction
    k_hop_neighbors = [transaction_idx]
    visited = {transaction_idx}
    
    # Get 1-hop neighbors
    for i in range(2):  # 2-hop neighborhood
        new_neighbors = []
        for node in k_hop_neighbors:
            # Get outgoing edges
            out_edges = (data.edge_index[0] == node).nonzero().view(-1)
            out_neighbors = data.edge_index[1, out_edges]
            
            # Get incoming edges
            in_edges = (data.edge_index[1] == node).nonzero().view(-1)
            in_neighbors = data.edge_index[0, in_edges]
            
            # Add all neighbors
            for neighbor in torch.cat([out_neighbors, in_neighbors]):
                if neighbor.item() not in visited:
                    new_neighbors.append(neighbor.item())
                    visited.add(neighbor.item())
        
        k_hop_neighbors.extend(new_neighbors)
    
    # Convert to tensor
    subgraph_nodes = torch.tensor(k_hop_neighbors, dtype=torch.long)
    
    # Get subgraph
    edge_index, _ = subgraph(
        subgraph_nodes, 
        data.edge_index, 
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )
    
    # Create a new data object for the subgraph
    subgraph_data = data.clone()
    subgraph_data.x = data.x[subgraph_nodes]
    subgraph_data.edge_index = edge_index
    subgraph_data.y = data.y[subgraph_nodes]
    
    # Find the new index of the target transaction
    target_idx = (subgraph_nodes == transaction_idx).nonzero().item()
    
    # Move to device
    subgraph_data = subgraph_data.to(device)
    
    # Get prediction
    with torch.no_grad():
        if model_type == 'tgn':
            logits, _ = model(subgraph_data.x, subgraph_data.edge_index, batch=None)
        else:
            logits = model(subgraph_data.x, subgraph_data.edge_index, batch=None)
        
        # Get prediction for the target transaction
        target_logit = logits[target_idx]
        if target_logit.dim() > 0:
            target_logit = target_logit.squeeze()
        
        prediction = torch.sigmoid(target_logit).item()
    
    return prediction, subgraph_data, subgraph_nodes, target_idx

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_elliptic_bitcoin_dataset(root="./data")
    data = dataset.data
    
    # Get train/val/test masks
    train_mask, val_mask, test_mask = dataset.get_train_val_test_masks()
    
    # Load model
    logger.info(f"Loading {args.model_type} model...")
    model = load_model(args.model_type, data, device)
    
    # Get test transactions
    test_indices = torch.where(test_mask)[0]
    
    # If transaction_idx is not specified, select a random test transaction
    if args.transaction_idx is None:
        # Try to find a fraudulent transaction
        fraudulent_test_indices = torch.where(data.y[test_indices] == 1)[0]
        if len(fraudulent_test_indices) > 0:
            # Select a random fraudulent transaction
            idx = torch.randint(0, len(fraudulent_test_indices), (1,)).item()
            transaction_idx = test_indices[fraudulent_test_indices[idx]].item()
        else:
            # Select a random test transaction
            idx = torch.randint(0, len(test_indices), (1,)).item()
            transaction_idx = test_indices[idx].item()
    else:
        transaction_idx = args.transaction_idx
    
    # Get the true label
    true_label = data.y[transaction_idx].item()
    label_str = "Fraudulent" if true_label == 1 else "Legitimate"
    
    logger.info(f"Analyzing transaction {transaction_idx} (True label: {label_str})")
    
    # Predict
    prediction, subgraph_data, subgraph_nodes, target_idx = predict_transaction(
        model, data, transaction_idx, device, args.model_type
    )
    
    # Print result
    logger.info(f"Prediction: {prediction:.4f} (Threshold: {args.threshold:.2f})")
    if prediction >= args.threshold:
        logger.info(f"Transaction {transaction_idx} is predicted as FRAUDULENT")
    else:
        logger.info(f"Transaction {transaction_idx} is predicted as LEGITIMATE")
    
    # Visualize the transaction and its neighborhood
    if args.visualize:
        # Create node colors based on predictions
        with torch.no_grad():
            if args.model_type == 'tgn':
                logits, _ = model(subgraph_data.x.to(device), subgraph_data.edge_index.to(device), batch=None)
            else:
                logits = model(subgraph_data.x.to(device), subgraph_data.edge_index.to(device), batch=None)
            
            predictions = torch.sigmoid(logits).cpu()
        
        # Create node colors: red for the target transaction, gradient for others based on fraud probability
        node_colors = predictions.numpy()
        
        # Highlight the target transaction
        plt.figure(figsize=(12, 10))
        visualize_graph(
            subgraph_data,
            node_colors=node_colors,
            title=f"Transaction {transaction_idx} Analysis (Fraud Probability)",
            save_path=os.path.join(RESULTS_DIR, f"transaction_{transaction_idx}_analysis.png"),
            node_size=50,
            cmap='RdYlBu_r'
        )
        
        # Plot feature importance
        if args.feature_importance:
            # Get feature names (placeholder - replace with actual feature names if available)
            feature_names = [f"Feature {i}" for i in range(data.x.size(1))]
            
            # Get feature values for the transaction
            feature_values = data.x[transaction_idx].cpu().numpy()
            
            # Sort features by absolute value
            sorted_indices = np.argsort(np.abs(feature_values))[::-1]
            top_features = sorted_indices[:10]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            plt.bar(
                range(len(top_features)),
                feature_values[top_features],
                tick_label=[feature_names[i] for i in top_features]
            )
            plt.xlabel('Feature')
            plt.ylabel('Value')
            plt.title(f'Top 10 Features for Transaction {transaction_idx}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"transaction_{transaction_idx}_features.png"))
            plt.show()
    
    logger.info("Inference completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a specific transaction")
    parser.add_argument("--model_type", type=str, default="tgn", choices=["gat", "tgn"],
                        help="Type of model to use (gat or tgn)")
    parser.add_argument("--transaction_idx", type=int, default=None,
                        help="Index of the transaction to analyze (default: random test transaction)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the transaction and its neighborhood")
    parser.add_argument("--feature_importance", action="store_true",
                        help="Plot feature importance for the transaction")
    
    args = parser.parse_args()
    main(args)
