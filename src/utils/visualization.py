"""
Visualization utilities for fraud detection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from typing import Dict, List, Tuple, Optional, Union
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        train_losses: Training losses.
        val_losses: Validation losses.
        train_metrics: Training metrics.
        val_metrics: Validation metrics.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    if 'accuracy' in train_metrics and 'accuracy' in val_metrics:
        plt.subplot(2, 2, 2)
        plt.plot(train_metrics['accuracy'], label='Train Accuracy')
        plt.plot(val_metrics['accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
    
    # Plot F1 score
    if 'f1' in train_metrics and 'f1' in val_metrics:
        plt.subplot(2, 2, 3)
        plt.plot(train_metrics['f1'], label='Train F1')
        plt.plot(val_metrics['f1'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()
        plt.grid(True)
    
    # Plot AUC
    if 'auc' in train_metrics and 'auc' in val_metrics:
        plt.subplot(2, 2, 4)
        plt.plot(train_metrics['auc'], label='Train AUC')
        plt.plot(val_metrics['auc'], label='Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Training and Validation AUC')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def visualize_graph(
    data,
    node_colors: Optional[Union[List[float], torch.Tensor]] = None,
    title: str = "Graph Visualization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    node_size: int = 50,
    with_labels: bool = False,
    cmap: str = 'coolwarm',
    max_nodes: int = 500
) -> None:
    """
    Visualize a graph.
    
    Args:
        data: PyTorch Geometric Data object.
        node_colors: Node colors (e.g., labels or predictions).
        title: Plot title.
        save_path: Path to save the plot.
        figsize: Figure size.
        node_size: Size of nodes in the plot.
        with_labels: Whether to show node labels.
        cmap: Colormap.
        max_nodes: Maximum number of nodes to visualize.
    """
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Limit the number of nodes for visualization
    if len(G.nodes) > max_nodes:
        logger.warning(f"Graph has {len(G.nodes)} nodes, which is too many to visualize clearly. "
                      f"Sampling {max_nodes} nodes for visualization.")
        G = nx.subgraph(G, list(G.nodes)[:max_nodes])
    
    # Set up colors
    if node_colors is not None:
        if isinstance(node_colors, torch.Tensor):
            node_colors = node_colors.cpu().numpy()
        
        if len(node_colors) > max_nodes:
            node_colors = node_colors[:max_nodes]
    else:
        node_colors = 'skyblue'
    
    # Set up figure
    plt.figure(figsize=figsize)
    
    # Use spring layout for node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    nx.draw(
        G,
        pos=pos,
        with_labels=with_labels,
        node_color=node_colors,
        cmap=plt.get_cmap(cmap),
        node_size=node_size,
        edge_color='gray',
        alpha=0.8
    )
    
    # Add a colorbar if node_colors is not a string
    if not isinstance(node_colors, str):
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap))
        sm.set_array([])
        plt.colorbar(sm)
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graph visualization saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['Licit', 'Illicit'],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues'
) -> None:
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix.
        class_names: Names of the classes.
        title: Plot title.
        save_path: Path to save the plot.
        figsize: Figure size.
        cmap: Colormap.
    """
    plt.figure(figsize=figsize)
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = 'ROC Curve',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot a ROC curve.
    
    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_score: AUC score.
        title: Plot title.
        save_path: Path to save the plot.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"ROC curve plot saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    auprc_score: float,
    title: str = 'Precision-Recall Curve',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot a precision-recall curve.
    
    Args:
        precision: Precision values.
        recall: Recall values.
        auprc_score: AUPRC score.
        title: Plot title.
        save_path: Path to save the plot.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUPRC = {auprc_score:.3f})')
    plt.axhline(y=0.5, color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Precision-recall curve plot saved to {save_path}")
    
    plt.show()
