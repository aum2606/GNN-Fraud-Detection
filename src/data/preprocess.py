"""
Script to preprocess the Elliptic Bitcoin dataset.
"""

import os
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from src.config import (
    DATA_DIR,
    NODE_FEATURES_FILE,
    EDGE_LIST_FILE,
    CLASSES_FILE,
    CLASS_MAPPING
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the raw Elliptic Bitcoin dataset.
    
    Returns:
        Tuple of DataFrames (features, edges, classes).
    """
    logger.info("Loading raw data...")
    
    features_df = pd.read_csv(NODE_FEATURES_FILE)
    edges_df = pd.read_csv(EDGE_LIST_FILE)
    classes_df = pd.read_csv(CLASSES_FILE)
    
    logger.info(f"Loaded {len(features_df)} nodes, {len(edges_df)} edges, and {len(classes_df)} class labels")
    
    return features_df, edges_df, classes_df


def analyze_dataset(features_df: pd.DataFrame, edges_df: pd.DataFrame, classes_df: pd.DataFrame) -> None:
    """
    Analyze the dataset and print statistics.
    
    Args:
        features_df: Node features DataFrame.
        edges_df: Edge list DataFrame.
        classes_df: Node classes DataFrame.
    """
    logger.info("Analyzing dataset...")
    
    # Node statistics
    logger.info(f"Number of nodes: {len(features_df)}")
    
    # Feature statistics
    num_features = features_df.shape[1] - 2  # Subtract txId and time_step columns
    logger.info(f"Number of features per node: {num_features}")
    
    # Time step statistics
    time_steps = features_df.iloc[:, 1].unique()
    logger.info(f"Number of time steps: {len(time_steps)}")
    logger.info(f"Time steps: {sorted(time_steps)}")
    
    # Edge statistics
    logger.info(f"Number of edges: {len(edges_df)}")
    
    # Class statistics
    class_counts = classes_df['class'].value_counts()
    logger.info(f"Class distribution: {class_counts.to_dict()}")
    
    # Calculate percentage of illicit transactions
    total_known = sum(count for label, count in class_counts.items() if label in ['1', '2'])
    illicit_count = class_counts.get('1', 0)
    illicit_percentage = (illicit_count / total_known) * 100 if total_known > 0 else 0
    logger.info(f"Percentage of illicit transactions: {illicit_percentage:.2f}%")
    
    # Check for missing values
    logger.info("Checking for missing values...")
    logger.info(f"Missing values in features: {features_df.isnull().sum().sum()}")
    logger.info(f"Missing values in edges: {edges_df.isnull().sum().sum()}")
    logger.info(f"Missing values in classes: {classes_df.isnull().sum().sum()}")
    
    # Check for duplicate nodes
    logger.info(f"Duplicate node IDs: {features_df.iloc[:, 0].duplicated().sum()}")
    
    # Check for self-loops in edges
    self_loops = edges_df[edges_df['txId1'] == edges_df['txId2']]
    logger.info(f"Number of self-loops: {len(self_loops)}")
    
    # Check for nodes without labels
    node_ids = set(features_df.iloc[:, 0].astype(str).values)
    labeled_node_ids = set(classes_df['txId'].astype(str).values)
    unlabeled_nodes = node_ids - labeled_node_ids
    logger.info(f"Number of nodes without labels: {len(unlabeled_nodes)}")
    
    # Check for isolated nodes (nodes with no edges)
    edge_nodes = set(edges_df['txId1'].astype(str).values) | set(edges_df['txId2'].astype(str).values)
    isolated_nodes = node_ids - edge_nodes
    logger.info(f"Number of isolated nodes: {len(isolated_nodes)}")
    
    # Check for nodes in different time steps
    nodes_per_time_step = features_df.groupby(features_df.iloc[:, 1]).size()
    logger.info(f"Nodes per time step: {nodes_per_time_step.to_dict()}")


def preprocess_data(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    classes_df: pd.DataFrame,
    output_dir: str = "./processed_data",
    normalize_features: bool = True,
    remove_isolated_nodes: bool = False,
    remove_self_loops: bool = True
) -> None:
    """
    Preprocess the dataset and save the processed data.
    
    Args:
        features_df: Node features DataFrame.
        edges_df: Edge list DataFrame.
        classes_df: Node classes DataFrame.
        output_dir: Directory to save the processed data.
        normalize_features: Whether to normalize the features.
        remove_isolated_nodes: Whether to remove isolated nodes.
        remove_self_loops: Whether to remove self-loops.
    """
    logger.info("Preprocessing dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process node features
    logger.info("Processing node features...")
    node_ids = features_df.iloc[:, 0].astype(str).values
    time_steps = features_df.iloc[:, 1].values
    features = features_df.iloc[:, 2:].values
    
    # Normalize features if requested
    if normalize_features:
        logger.info("Normalizing features...")
        features_mean = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        features_std[features_std == 0] = 1  # Avoid division by zero
        features = (features - features_mean) / features_std
    
    # Process edges
    logger.info("Processing edges...")
    if remove_self_loops:
        logger.info("Removing self-loops...")
        edges_df = edges_df[edges_df['txId1'] != edges_df['txId2']]
    
    # Create a mapping from node ID to index
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Process edges
    edge_list = []
    for _, row in edges_df.iterrows():
        src_id, dst_id = str(int(row['txId1'])), str(int(row['txId2']))
        if src_id in node_id_to_idx and dst_id in node_id_to_idx:
            edge_list.append((node_id_to_idx[src_id], node_id_to_idx[dst_id]))
    
    # Process node labels
    logger.info("Processing node labels...")
    labels = np.full(len(node_ids), -1)  # Default to unknown (-1)
    
    for _, row in classes_df.iterrows():
        node_id, label = str(int(row['txId'])), row['class']
        if node_id in node_id_to_idx:
            if label in ['1', '2']:  # Only consider known labels
                labels[node_id_to_idx[node_id]] = CLASS_MAPPING[label]
    
    # Remove isolated nodes if requested
    if remove_isolated_nodes:
        logger.info("Removing isolated nodes...")
        edge_nodes = set()
        for src, dst in edge_list:
            edge_nodes.add(src)
            edge_nodes.add(dst)
        
        # Create mask for non-isolated nodes
        non_isolated_mask = np.zeros(len(node_ids), dtype=bool)
        for idx in edge_nodes:
            non_isolated_mask[idx] = True
        
        # Filter data
        node_ids = node_ids[non_isolated_mask]
        time_steps = time_steps[non_isolated_mask]
        features = features[non_isolated_mask]
        labels = labels[non_isolated_mask]
        
        # Update node ID to index mapping
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Update edge list
        new_edge_list = []
        for src, dst in edge_list:
            if src in edge_nodes and dst in edge_nodes:
                new_edge_list.append((node_id_to_idx[node_ids[src]], node_id_to_idx[node_ids[dst]]))
        edge_list = new_edge_list
    
    # Save processed data
    logger.info("Saving processed data...")
    
    # Save node features
    processed_features_df = pd.DataFrame(features)
    processed_features_df.insert(0, 'time_step', time_steps)
    processed_features_df.insert(0, 'node_id', node_ids)
    processed_features_df.to_csv(os.path.join(output_dir, 'processed_features.csv'), index=False)
    
    # Save edge list
    processed_edges_df = pd.DataFrame(edge_list, columns=['source', 'target'])
    processed_edges_df.to_csv(os.path.join(output_dir, 'processed_edges.csv'), index=False)
    
    # Save node labels
    processed_labels_df = pd.DataFrame({'node_id': node_ids, 'label': labels})
    processed_labels_df.to_csv(os.path.join(output_dir, 'processed_labels.csv'), index=False)
    
    logger.info(f"Processed data saved to {output_dir}")
    logger.info(f"Final dataset: {len(node_ids)} nodes, {len(edge_list)} edges")
    
    # Print class distribution
    class_counts = np.bincount(labels[labels != -1])
    logger.info(f"Class distribution: {class_counts}")
    
    # Calculate percentage of illicit transactions
    total_known = sum(class_counts)
    illicit_percentage = (class_counts[1] / total_known) * 100 if total_known > 0 else 0
    logger.info(f"Percentage of illicit transactions: {illicit_percentage:.2f}%")


def main(args):
    """
    Main function to preprocess the dataset.
    """
    # Load raw data
    features_df, edges_df, classes_df = load_raw_data()
    
    # Analyze dataset
    if args.analyze:
        analyze_dataset(features_df, edges_df, classes_df)
    
    # Preprocess data
    if args.preprocess:
        preprocess_data(
            features_df,
            edges_df,
            classes_df,
            output_dir=args.output_dir,
            normalize_features=args.normalize_features,
            remove_isolated_nodes=args.remove_isolated_nodes,
            remove_self_loops=args.remove_self_loops
        )
    
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Elliptic Bitcoin dataset")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze the dataset and print statistics")
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess the dataset and save the processed data")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                        help="Directory to save the processed data")
    parser.add_argument("--normalize_features", action="store_true",
                        help="Whether to normalize the features")
    parser.add_argument("--remove_isolated_nodes", action="store_true",
                        help="Whether to remove isolated nodes")
    parser.add_argument("--remove_self_loops", action="store_true",
                        help="Whether to remove self-loops")
    
    args = parser.parse_args()
    
    # Ensure at least one of analyze or preprocess is specified
    if not args.analyze and not args.preprocess:
        parser.error("At least one of --analyze or --preprocess must be specified")
    
    main(args)
