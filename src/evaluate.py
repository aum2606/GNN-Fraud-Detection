"""
Evaluation script for the GNN Fraud Detection model.
"""

import os
import torch
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, List, Tuple, Optional, Union

from src.config import (
    EVAL_CONFIG,
    MODEL_CONFIG,
    RESULTS_DIR
)
from src.data import (
    load_elliptic_bitcoin_dataset,
    create_dataloaders
)
from src.models import GAT, TemporalGNN
from src.utils import (
    calculate_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    visualize_graph
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = EVAL_CONFIG["threshold"]
) -> Dict[str, float]:
    """
    Evaluate the model on the test set.
    
    Args:
        model: The trained model.
        test_loader: Test data loader.
        device: Device to use for evaluation.
        threshold: Classification threshold.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("Evaluating model on test set...")
    
    model.eval()
    test_preds = []
    test_labels = []
    test_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Get node indices with known labels
            mask = batch.y != -1
            if mask.sum() == 0:
                continue  # Skip batches with no labeled nodes
            
            # Forward pass
            if isinstance(model, TemporalGNN):
                # For temporal model
                logits, _ = model(
                    batch.x,
                    batch.edge_index,
                    batch=batch.batch
                )
            else:
                # For static model
                logits = model(
                    batch.x,
                    batch.edge_index,
                    batch=batch.batch
                )
            
            # Apply mask to get predictions and labels for known nodes
            logits_masked = logits[mask]
            labels_masked = batch.y[mask]
            
            # Reshape logits if needed
            if logits_masked.dim() > 1 and logits_masked.size(1) == 1:
                logits_masked = logits_masked.squeeze(1)
            
            # Get predictions
            scores = torch.sigmoid(logits_masked)
            preds = (scores >= threshold).int()
            
            # Update metrics
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels_masked.cpu().numpy())
            test_scores.extend(scores.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(test_labels),
        np.array(test_preds),
        np.array(test_scores)
    )
    
    return metrics, test_labels, test_preds, test_scores


def main(args):
    """
    Main evaluation function.
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_elliptic_bitcoin_dataset(root=args.data_root)
    data = dataset.data
    
    # Get train/val/test masks
    _, _, test_mask = dataset.get_train_val_test_masks()
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloaders = create_dataloaders(
        data,
        torch.zeros_like(test_mask),  # Dummy train mask
        torch.zeros_like(test_mask),  # Dummy val mask
        test_mask,
        batch_size=256,
        use_subgraph_sampling=args.use_subgraph_sampling
    )
    
    test_loader = dataloaders["test"]
    
    # Initialize model
    logger.info(f"Initializing {args.model_type} model...")
    if args.model_type == 'gat':
        model = GAT(
            in_channels=data.x.size(1),
            hidden_channels=MODEL_CONFIG["hidden_channels"],
            out_channels=1,  # Binary classification
            num_layers=MODEL_CONFIG["num_layers"],
            dropout=MODEL_CONFIG["dropout"],
            residual=True,
            use_batch_norm=True
        )
    elif args.model_type == 'tgn':
        model = TemporalGNN(
            in_channels=data.x.size(1),
            hidden_channels=MODEL_CONFIG["hidden_channels"],
            out_channels=1,  # Binary classification
            num_layers=MODEL_CONFIG["num_layers"],
            dropout=MODEL_CONFIG["dropout"],
            residual=True,
            use_batch_norm=True
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load model weights
    model_path = os.path.join(RESULTS_DIR, f"{args.model_type}_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"Model file not found at {model_path}")
        return
    
    model = model.to(device)
    
    # Evaluate model
    metrics, test_labels, test_preds, test_scores = evaluate_model(
        model,
        test_loader,
        device,
        threshold=EVAL_CONFIG["threshold"]
    )
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics to file
    metrics_path = os.path.join(RESULTS_DIR, f"{args.model_type}_metrics.txt")
    with open(metrics_path, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Plot confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    cm_path = os.path.join(RESULTS_DIR, f"{args.model_type}_confusion_matrix.png")
    plot_confusion_matrix(
        cm,
        class_names=['Licit', 'Illicit'],
        title=f'Confusion Matrix - {args.model_type.upper()}',
        save_path=cm_path
    )
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    roc_path = os.path.join(RESULTS_DIR, f"{args.model_type}_roc_curve.png")
    plot_roc_curve(
        fpr,
        tpr,
        metrics['auc'],
        title=f'ROC Curve - {args.model_type.upper()}',
        save_path=roc_path
    )
    
    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(test_labels, test_scores)
    pr_path = os.path.join(RESULTS_DIR, f"{args.model_type}_pr_curve.png")
    plot_precision_recall_curve(
        precision,
        recall,
        metrics['auprc'],
        title=f'Precision-Recall Curve - {args.model_type.upper()}',
        save_path=pr_path
    )
    
    # Visualize graph with predictions if requested
    if args.visualize_graph:
        # Get a small subgraph for visualization
        subgraph_data = data.clone()
        subgraph_data.y = torch.tensor(test_labels)
        
        # Visualize ground truth
        gt_path = os.path.join(RESULTS_DIR, f"{args.model_type}_graph_ground_truth.png")
        visualize_graph(
            subgraph_data,
            node_colors=subgraph_data.y,
            title=f'Ground Truth Labels - {args.model_type.upper()}',
            save_path=gt_path
        )
        
        # Visualize predictions
        pred_path = os.path.join(RESULTS_DIR, f"{args.model_type}_graph_predictions.png")
        visualize_graph(
            subgraph_data,
            node_colors=np.array(test_scores),
            title=f'Prediction Scores - {args.model_type.upper()}',
            save_path=pred_path
        )
    
    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GNN Fraud Detection model")
    parser.add_argument("--model_type", type=str, default="gat", choices=["gat", "tgn"],
                        help="Type of model to evaluate (gat or tgn)")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for the dataset")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--use_subgraph_sampling", action="store_true",
                        help="Whether to use subgraph sampling for evaluation")
    parser.add_argument("--visualize_graph", action="store_true",
                        help="Whether to visualize the graph with predictions")
    
    args = parser.parse_args()
    main(args)
