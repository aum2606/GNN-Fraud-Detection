"""
Dataset loading and preprocessing for the Elliptic Bitcoin dataset.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from typing import List, Tuple, Dict, Optional, Union
import logging

from src.config import (
    NODE_FEATURES_FILE, 
    EDGE_LIST_FILE, 
    CLASSES_FILE, 
    CLASS_MAPPING,
    TRAIN_TIME_STEPS,
    VAL_TIME_STEPS,
    TEST_TIME_STEPS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EllipticBitcoinDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for the Elliptic Bitcoin dataset.
    """
    
    def __init__(self, root: str, transform=None, pre_transform=None):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory where the dataset should be saved.
            transform: A function/transform that takes in a `torch_geometric.data.Data` 
                       object and returns a transformed version.
            pre_transform: A function/transform that takes in a `torch_geometric.data.Data` 
                          object and returns a transformed version.
        """
        super(EllipticBitcoinDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        # Store time steps for each node
        self.node_time_steps = {}
        features_df = pd.read_csv(NODE_FEATURES_FILE)
        for idx, row in features_df.iterrows():
            self.node_time_steps[str(int(row.iloc[0]))] = int(row.iloc[1])
    
    @property
    def raw_file_names(self) -> List[str]:
        """
        The names of the raw files.
        """
        return [
            os.path.basename(NODE_FEATURES_FILE),
            os.path.basename(EDGE_LIST_FILE),
            os.path.basename(CLASSES_FILE)
        ]
    
    @property
    def processed_file_names(self) -> List[str]:
        """
        The name of the processed files.
        """
        return ['data.pt']
    
    def download(self):
        """
        Download the dataset. In this case, we assume the dataset is already downloaded.
        """
        pass
    
    def process(self):
        """
        Process the raw data into the PyTorch Geometric format.
        """
        logger.info("Processing Elliptic Bitcoin dataset...")
        
        # Load the data
        features_df = pd.read_csv(NODE_FEATURES_FILE)
        edges_df = pd.read_csv(EDGE_LIST_FILE)
        classes_df = pd.read_csv(CLASSES_FILE)
        
        # Extract node IDs, features, and time steps
        node_ids = features_df.iloc[:, 0].astype(str).values
        time_steps = features_df.iloc[:, 1].values
        features = features_df.iloc[:, 2:].values
        
        # Create a mapping from node ID to index
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Process edges
        edge_index = []
        for _, row in edges_df.iterrows():
            src_id, dst_id = str(int(row['txId1'])), str(int(row['txId2']))
            if src_id in node_id_to_idx and dst_id in node_id_to_idx:
                edge_index.append([node_id_to_idx[src_id], node_id_to_idx[dst_id]])
        
        if not edge_index:
            raise ValueError("No valid edges found after mapping node IDs to indices.")
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Process node labels
        y = torch.full((len(node_ids),), -1, dtype=torch.long)  # Default to unknown (-1)
        
        for _, row in classes_df.iterrows():
            node_id, label = str(int(row['txId'])), row['class']
            if node_id in node_id_to_idx:
                if label in ['1', '2']:  # Only consider known labels
                    y[node_id_to_idx[node_id]] = CLASS_MAPPING[label]
        
        # Convert features to tensor
        x = torch.tensor(features, dtype=torch.float)
        
        # Create time step tensor
        time_steps = torch.tensor(time_steps, dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y, time_steps=time_steps)
        
        # Save to disk
        if self.pre_transform is not None:
            data = self.pre_transform(data)
            
        torch.save(self.collate([data]), self.processed_paths[0])
        logger.info(f"Processed dataset saved to {self.processed_paths[0]}")
    
    def get_train_val_test_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masks for temporal train/validation/test split.
        
        Returns:
            Tuple of train, validation, and test masks.
        """
        time_steps = self.data.time_steps
        
        train_mask = torch.zeros(len(time_steps), dtype=torch.bool)
        val_mask = torch.zeros(len(time_steps), dtype=torch.bool)
        test_mask = torch.zeros(len(time_steps), dtype=torch.bool)
        
        for i, ts in enumerate(time_steps):
            ts_int = ts.item()
            if ts_int in TRAIN_TIME_STEPS:
                train_mask[i] = True
            elif ts_int in VAL_TIME_STEPS:
                val_mask[i] = True
            elif ts_int in TEST_TIME_STEPS:
                test_mask[i] = True
        
        # Only include nodes with known labels in the masks
        known_mask = self.data.y != -1
        train_mask = train_mask & known_mask
        val_mask = val_mask & known_mask
        test_mask = test_mask & known_mask
        
        return train_mask, val_mask, test_mask
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights to handle class imbalance.
        
        Returns:
            Tensor of class weights.
        """
        # Only consider nodes with known labels
        known_mask = self.data.y != -1
        labels = self.data.y[known_mask]
        
        # Count occurrences of each class
        class_counts = torch.bincount(labels)
        
        # Calculate weights as inverse of frequency
        class_weights = 1.0 / class_counts.float()
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        return class_weights


def load_elliptic_bitcoin_dataset(root: str = "./data") -> EllipticBitcoinDataset:
    """
    Load the Elliptic Bitcoin dataset.
    
    Args:
        root: Root directory where the processed dataset should be saved.
        
    Returns:
        The processed dataset.
    """
    dataset = EllipticBitcoinDataset(root=root)
    return dataset


def create_temporal_subgraph(data: Data, time_step: int) -> Data:
    """
    Create a subgraph for a specific time step.
    
    Args:
        data: The full graph data.
        time_step: The time step to extract.
        
    Returns:
        A subgraph containing only nodes and edges from the specified time step.
    """
    # Get nodes from the specified time step
    mask = data.time_steps == time_step
    node_indices = torch.nonzero(mask).squeeze()
    
    # Create a mapping from original indices to new indices
    idx_mapping = {int(idx): i for i, idx in enumerate(node_indices)}
    
    # Filter edges
    edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
    for i in range(data.edge_index.size(1)):
        src, dst = data.edge_index[:, i]
        if src in idx_mapping and dst in idx_mapping:
            edge_mask[i] = True
    
    filtered_edges = data.edge_index[:, edge_mask]
    
    # Remap edge indices
    remapped_edges = torch.zeros_like(filtered_edges)
    for i in range(filtered_edges.size(1)):
        src, dst = filtered_edges[:, i]
        remapped_edges[0, i] = idx_mapping[int(src)]
        remapped_edges[1, i] = idx_mapping[int(dst)]
    
    # Create new data object
    subgraph = Data(
        x=data.x[node_indices],
        edge_index=remapped_edges,
        y=data.y[node_indices],
        time_steps=data.time_steps[node_indices]
    )
    
    return subgraph
