"""
Evaluation metrics for fraud detection.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)
from typing import Dict, List, Tuple, Union, Optional


def calculate_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    y_score: Optional[Union[torch.Tensor, np.ndarray]] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_score: Prediction scores (probabilities).
        threshold: Classification threshold.
        
    Returns:
        Dictionary of metrics.
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_score is not None and isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    
    # Convert probabilities to binary predictions if y_pred contains probabilities
    if y_score is None and np.any((y_pred > 0) & (y_pred < 1)):
        y_score = y_pred.copy()
        y_pred = (y_pred >= threshold).astype(int)
    
    # Calculate basic metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    
    # Calculate AUC and AUPRC if scores are provided
    if y_score is not None:
        # ROC AUC
        try:
            metrics["auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["auc"] = 0.5  # Default value when only one class is present
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics["auprc"] = auc(recall, precision)
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics.
    """
    print("\n===== Evaluation Metrics =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    if "auc" in metrics:
        print(f"ROC AUC: {metrics['auc']:.4f}")
    
    if "auprc" in metrics:
        print(f"PR AUC: {metrics['auprc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print("==============================\n")
