"""
Training script for the GNN Fraud Detection model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import argparse
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import (
    TRAIN_CONFIG,
    MODEL_CONFIG,
    RESULTS_DIR,
    WANDB_CONFIG
)
from src.data import (
    load_elliptic_bitcoin_dataset,
    create_dataloaders
)
from src.models import GAT, TemporalGNN
from src.utils import (
    calculate_metrics,
    print_metrics,
    plot_training_history
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = TRAIN_CONFIG["epochs"],
    patience: int = TRAIN_CONFIG["patience"],
    use_wandb: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the model.
    
    Args:
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to use for training.
        epochs: Number of epochs.
        patience: Early stopping patience.
        use_wandb: Whether to use Weights & Biases for logging.
        
    Returns:
        Trained model and training history.
    """
    logger.info(f"Starting training for {epochs} epochs...")
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        },
        'val_metrics': {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        train_scores = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            # Move batch to device
            batch = batch.to(device)
            
            # Get node indices with known labels
            mask = batch.y != -1
            if mask.sum() == 0:
                continue  # Skip batches with no labeled nodes
            
            # Forward pass
            optimizer.zero_grad()
            
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
            
            # Calculate loss
            loss = criterion(logits_masked, labels_masked.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * mask.sum().item()
            train_preds.extend((torch.sigmoid(logits_masked) >= 0.5).cpu().numpy())
            train_labels.extend(labels_masked.cpu().numpy())
            train_scores.extend(torch.sigmoid(logits_masked).detach().cpu().numpy())
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        
        # Calculate training metrics
        train_metrics = calculate_metrics(
            np.array(train_labels),
            np.array(train_preds),
            np.array(train_scores)
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
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
                
                # Calculate loss
                loss = criterion(logits_masked, labels_masked.float())
                
                # Update metrics
                val_loss += loss.item() * mask.sum().item()
                val_preds.extend((torch.sigmoid(logits_masked) >= 0.5).cpu().numpy())
                val_labels.extend(labels_masked.cpu().numpy())
                val_scores.extend(torch.sigmoid(logits_masked).cpu().numpy())
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(
            np.array(val_labels),
            np.array(val_preds),
            np.array(val_scores)
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        for metric in train_metrics:
            if metric in history['train_metrics']:
                history['train_metrics'][metric].append(train_metrics[metric])
        
        for metric in val_metrics:
            if metric in history['val_metrics']:
                history['val_metrics'][metric].append(val_metrics[metric])
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        logger.info(f"Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # Log to wandb
        if use_wandb:
            wandb_log = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            for metric in train_metrics:
                wandb_log[f'train_{metric}'] = train_metrics[metric]
            
            for metric in val_metrics:
                wandb_log[f'val_{metric}'] = val_metrics[metric]
            
            wandb.log(wandb_log)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def main(args):
    """
    Main training function.
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=WANDB_CONFIG["project"],
            entity=WANDB_CONFIG["entity"],
            config={
                "model_type": args.model_type,
                "learning_rate": TRAIN_CONFIG["learning_rate"],
                "epochs": TRAIN_CONFIG["epochs"],
                "batch_size": TRAIN_CONFIG["batch_size"],
                "hidden_channels": MODEL_CONFIG["hidden_channels"],
                "num_layers": MODEL_CONFIG["num_layers"],
                "dropout": MODEL_CONFIG["dropout"],
                "patience": TRAIN_CONFIG["patience"],
                "weight_decay": TRAIN_CONFIG["weight_decay"],
                "pos_weight": TRAIN_CONFIG["pos_weight"]
            }
        )
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_elliptic_bitcoin_dataset(root=args.data_root)
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
        batch_size=TRAIN_CONFIG["batch_size"],
        use_subgraph_sampling=args.use_subgraph_sampling
    )
    
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    
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
    
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    
    # Initialize loss function with class weights
    pos_weight = torch.tensor([TRAIN_CONFIG["pos_weight"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Train model
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=TRAIN_CONFIG["epochs"],
        patience=TRAIN_CONFIG["patience"],
        use_wandb=args.use_wandb
    )
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, f"{args.model_type}_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Plot training history
    plot_path = os.path.join(RESULTS_DIR, f"{args.model_type}_training_history.png")
    plot_training_history(
        history['train_loss'],
        history['val_loss'],
        history['train_metrics'],
        history['val_metrics'],
        save_path=plot_path
    )
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN Fraud Detection model")
    parser.add_argument("--model_type", type=str, default="gat", choices=["gat", "tgn"],
                        help="Type of model to train (gat or tgn)")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for the dataset")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--use_subgraph_sampling", action="store_true",
                        help="Whether to use subgraph sampling for training")
    
    args = parser.parse_args()
    main(args)
