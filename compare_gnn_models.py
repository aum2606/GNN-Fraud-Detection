"""
Compare the performance of different GNN models for fraud detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch
import logging

from src.config import RESULTS_DIR
from src.models import GAT, TemporalGNN
from src.data import load_elliptic_bitcoin_dataset, create_dataloaders
from src.utils import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_type, data):
    """
    Load a trained model.
    
    Args:
        model_type: Type of model to load (gat or tgn).
        data: Dataset object.
        
    Returns:
        Loaded model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

def get_predictions(model, data_loader, device):
    """
    Get predictions from a model.
    
    Args:
        model: Trained model.
        data_loader: Data loader.
        device: Device to use.
        
    Returns:
        Predictions, scores, and true labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in data_loader:
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
            preds = (scores >= 0.5).int()
            
            # Update metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_masked.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    return np.array(all_preds), np.array(all_scores), np.array(all_labels)

def compare_models():
    """
    Compare the performance of different models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_elliptic_bitcoin_dataset(root="./data")
    data = dataset.data
    
    # Get train/val/test masks
    train_mask, val_mask, test_mask = dataset.get_train_val_test_masks()
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloaders = create_dataloaders(
        data,
        train_mask,
        val_mask,
        test_mask,
        batch_size=256,
        use_subgraph_sampling=False
    )
    
    test_loader = dataloaders["test"]
    
    # Load models
    model_types = ['gat', 'tgn']
    models = {}
    for model_type in model_types:
        logger.info(f"Loading {model_type} model...")
        models[model_type] = load_model(model_type, data)
    
    # Get predictions
    predictions = {}
    for model_type, model in models.items():
        logger.info(f"Getting predictions for {model_type}...")
        preds, scores, labels = get_predictions(model, test_loader, device)
        predictions[model_type] = {
            'preds': preds,
            'scores': scores,
            'labels': labels
        }
    
    # Calculate metrics
    metrics = {}
    for model_type, pred_data in predictions.items():
        logger.info(f"Calculating metrics for {model_type}...")
        metrics[model_type] = calculate_metrics(
            pred_data['labels'],
            pred_data['preds'],
            pred_data['scores']
        )
        print(f"\nMetrics for {model_type.upper()}:")
        print(f"Accuracy: {metrics[model_type]['accuracy']:.4f}")
        print(f"Precision: {metrics[model_type]['precision']:.4f}")
        print(f"Recall: {metrics[model_type]['recall']:.4f}")
        print(f"F1 Score: {metrics[model_type]['f1']:.4f}")
        print(f"AUC: {metrics[model_type]['auc']:.4f}")
        print(f"AUPRC: {metrics[model_type]['auprc']:.4f}")
    
    # Compare metrics
    logger.info("Comparing metrics...")
    metrics_df = pd.DataFrame({
        model_type: {
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1 Score': m['f1'],
            'AUC': m['auc'],
            'AUPRC': m['auprc']
        }
        for model_type, m in metrics.items()
    })
    
    print("\nMetrics Comparison:")
    print(metrics_df)
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'))
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for model_type, pred_data in predictions.items():
        fpr, tpr, _ = roc_curve(pred_data['labels'], pred_data['scores'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_type.upper()} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve_comparison.png'), dpi=300)
    
    # Plot PR curves
    plt.figure(figsize=(10, 8))
    for model_type, pred_data in predictions.items():
        precision, recall, _ = precision_recall_curve(pred_data['labels'], pred_data['scores'])
        pr_auc = metrics[model_type]['auprc']
        plt.plot(recall, precision, lw=2, label=f'{model_type.upper()} (AP = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(RESULTS_DIR, 'pr_curve_comparison.png'), dpi=300)
    
    logger.info("Comparison completed successfully.")

if __name__ == "__main__":
    compare_models()
