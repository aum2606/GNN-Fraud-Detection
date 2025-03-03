"""
Data loaders for the Elliptic Bitcoin dataset.
"""

import torch
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.data import Data
import logging
from typing import Dict, Tuple, List, Optional

from src.config import TRAIN_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dataloaders(
    data: Data, 
    train_mask: torch.Tensor, 
    val_mask: torch.Tensor, 
    test_mask: torch.Tensor,
    batch_size: int = TRAIN_CONFIG["batch_size"],
    num_neighbors: List[int] = [10, 10, 10],
    use_subgraph_sampling: bool = True
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data: The graph data.
        train_mask: Mask for training nodes.
        val_mask: Mask for validation nodes.
        test_mask: Mask for test nodes.
        batch_size: Batch size for the data loaders.
        num_neighbors: Number of neighbors to sample for each node in each layer.
        use_subgraph_sampling: Whether to use subgraph sampling.
        
    Returns:
        Dictionary containing the data loaders.
    """
    logger.info("Creating data loaders...")
    
    if use_subgraph_sampling:
        # Use NeighborLoader for subgraph sampling
        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_mask,
            shuffle=True
        )
        
        val_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=val_mask,
            shuffle=False
        )
        
        test_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=test_mask,
            shuffle=False
        )
    else:
        # Use the entire graph for each batch
        train_loader = DataLoader(
            [data],
            batch_size=1,
            shuffle=False
        )
        
        val_loader = DataLoader(
            [data],
            batch_size=1,
            shuffle=False
        )
        
        test_loader = DataLoader(
            [data],
            batch_size=1,
            shuffle=False
        )
    
    logger.info(f"Created data loaders with batch size {batch_size}")
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    logger.info(f"Test loader: {len(test_loader)} batches")
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


def create_temporal_dataloaders(
    dataset,
    time_steps: List[int],
    batch_size: int = TRAIN_CONFIG["batch_size"]
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create data loaders for temporal graph analysis.
    
    Args:
        dataset: The dataset containing methods to create temporal subgraphs.
        time_steps: List of time steps to include.
        batch_size: Batch size for the data loaders.
        
    Returns:
        Dictionary mapping time steps to their respective data loaders.
    """
    logger.info(f"Creating temporal data loaders for time steps {time_steps}...")
    
    loaders = {}
    data = dataset.data
    
    for ts in time_steps:
        # Create subgraph for the time step
        subgraph = dataset.create_temporal_subgraph(data, ts)
        
        # Create loader for the subgraph
        loader = DataLoader(
            [subgraph],
            batch_size=1,
            shuffle=False
        )
        
        loaders[ts] = loader
        
    logger.info(f"Created temporal data loaders for {len(loaders)} time steps")
    
    return loaders
