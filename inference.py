"""
Script to demonstrate how to use the trained model for inference.
"""

import os
import torch
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from src.config import (
    MODEL_CONFIG,
    RESULTS_DIR,
    EVAL_CONFIG
)
from src.data import (
    load_elliptic_bitcoin_dataset,
    create_dataloaders,
    create_temporal_subgraph
)
from src.models import GAT, TemporalGNN
from src.utils import (
    calculate_metrics,
    print_metrics,
    visualize_graph
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_type: str, device: torch.device, in_channels: int) -> torch.nn.Module:
    """
    Load a trained model.
    
    Args:
        model_type: Type of model to load.
        device: Device to load the model on.
        in_channels: Number of input features.
        
    Returns:
        Loaded model.
    """
    logger.info(f"Loading {model_type} model...")
    
    # Initialize model
    if model_type == 'gat':
        model = GAT(
            in_channels=in_channels,
            hidden_channels=MODEL_CONFIG["hidden_channels"],
            num_layers=MODEL_CONFIG["num_layers"],
            heads=MODEL_CONFIG["heads"],
            dropout=MODEL_CONFIG["dropout"],
            edge_dim=MODEL_CONFIG["edge_dim"]
        )
    elif model_type == 'tgn':
        model = TemporalGNN(
            in_channels=in_channels,
            hidden_channels=MODEL_CONFIG["hidden_channels"],
            num_layers=MODEL_CONFIG["num_layers"],
            heads=MODEL_CONFIG["heads"],
            dropout=MODEL_CONFIG["dropout"],
            edge_dim=MODEL_CONFIG["edge_dim"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model_path = os.path.join(RESULTS_DIR, f"{model_type}_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_transaction(
    model: torch.nn.Module,
    data,
    transaction_id: str,
    device: torch.device,
    threshold: float = EVAL_CONFIG["threshold"],
    visualize: bool = False
) -> Dict:
    """
    Predict whether a transaction is fraudulent.
    
    Args:
        model: Trained model.
        data: Graph data.
        transaction_id: ID of the transaction to predict.
        device: Device to run inference on.
        threshold: Classification threshold.
        visualize: Whether to visualize the subgraph.
        
    Returns:
        Dictionary with prediction results.
    """
    logger.info(f"Predicting transaction {transaction_id}...")
    
    # Find the transaction in the dataset
    node_ids = data.node_ids if hasattr(data, 'node_ids') else None
    
    if node_ids is None:
        # Try to get node IDs from the dataset
        try:
            features_df = pd.read_csv(os.path.join(data.root, 'raw', 'elliptic_txs_features.csv'))
            node_ids = features_df.iloc[:, 0].astype(str).values
        except Exception as e:
            logger.error(f"Error loading node IDs: {e}")
            return {"error": "Could not load node IDs"}
    
    # Find the index of the transaction
    try:
        node_idx = np.where(node_ids == transaction_id)[0][0]
    except IndexError:
        logger.error(f"Transaction {transaction_id} not found in the dataset")
        return {"error": f"Transaction {transaction_id} not found in the dataset"}
    
    # Get the transaction's time step
    time_step = data.time_steps[node_idx].item() if hasattr(data, 'time_steps') else None
    
    if time_step is None:
        logger.error(f"Could not determine time step for transaction {transaction_id}")
        return {"error": f"Could not determine time step for transaction {transaction_id}"}
    
    # Create a subgraph centered around the transaction
    # This is a simplified approach; in a real-world scenario, you would need to
    # extract a relevant subgraph that includes the transaction's neighborhood
    
    # For demonstration purposes, we'll use the entire graph
    # In a real implementation, you would extract a k-hop neighborhood
    batch = data.clone()
    batch = batch.to(device)
    
    # Make prediction
    with torch.no_grad():
        if isinstance(model, TemporalGNN):
            # For temporal model
            logits = model(
                batch.x,
                batch.edge_index,
                time_step,
                torch.tensor([node_idx], device=device),
                edge_attr=getattr(batch, 'edge_attr', None)
            )
        else:
            # For static model
            logits = model(
                batch.x,
                batch.edge_index,
                edge_attr=getattr(batch, 'edge_attr', None)
            )
    
    # Get prediction for the target transaction
    score = torch.sigmoid(logits[node_idx]).item()
    prediction = 1 if score >= threshold else 0
    
    # Get ground truth if available
    ground_truth = batch.y[node_idx].item() if hasattr(batch, 'y') else None
    
    # Visualize subgraph if requested
    if visualize:
        # For simplicity, we'll visualize the entire graph
        # In a real implementation, you would visualize the k-hop neighborhood
        visualize_graph(
            batch,
            node_colors=torch.sigmoid(logits).cpu().numpy(),
            title=f'Prediction for Transaction {transaction_id}',
            save_path=os.path.join(RESULTS_DIR, f"prediction_{transaction_id}.png")
        )
    
    # Return results
    results = {
        "transaction_id": transaction_id,
        "score": score,
        "prediction": "Illicit" if prediction == 1 else "Licit",
        "confidence": max(score, 1 - score),
        "time_step": time_step
    }
    
    if ground_truth is not None and ground_truth != -1:
        results["ground_truth"] = "Illicit" if ground_truth == 1 else "Licit"
        results["correct"] = (prediction == ground_truth)
    
    return results


def main(args):
    """
    Main function for inference.
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_elliptic_bitcoin_dataset(root=args.data_root)
    data = dataset.data
    
    # Load model
    model = load_model(args.model_type, device, data.x.size(1))
    
    # Predict transaction
    results = predict_transaction(
        model,
        data,
        args.transaction_id,
        device,
        threshold=args.threshold,
        visualize=args.visualize
    )
    
    # Print results
    print("\n===== Prediction Results =====")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("==============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use trained model for inference")
    parser.add_argument("--model_type", type=str, default="gat", choices=["gat", "tgn"],
                        help="Type of model to use (gat or tgn)")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for the dataset")
    parser.add_argument("--transaction_id", type=str, required=True,
                        help="ID of the transaction to predict")
    parser.add_argument("--threshold", type=float, default=EVAL_CONFIG["threshold"],
                        help="Classification threshold")
    parser.add_argument("--visualize", action="store_true",
                        help="Whether to visualize the subgraph")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    
    args = parser.parse_args()
    main(args)
