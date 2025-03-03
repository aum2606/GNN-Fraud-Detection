"""
Preprocess the Elliptic Bitcoin dataset without PyTorch Geometric dependencies.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import time
import argparse
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = Path("./elliptic_bitcoin_dataset")
PROCESSED_DIR = Path("./processed_data")
PROCESSED_DIR.mkdir(exist_ok=True)

def load_dataset():
    """
    Load the Elliptic Bitcoin dataset.
    """
    logger.info("Loading dataset...")
    
    # Load features
    features_path = DATA_DIR / "elliptic_txs_features.csv"
    features_df = pd.read_csv(features_path, header=None)
    
    # Load classes
    classes_path = DATA_DIR / "elliptic_txs_classes.csv"
    classes_df = pd.read_csv(classes_path)
    
    # Load edges
    edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"
    edges_df = pd.read_csv(edges_path)
    
    logger.info(f"Loaded {len(features_df)} nodes, {len(edges_df)} edges")
    
    return features_df, classes_df, edges_df

def analyze_dataset(features_df, classes_df, edges_df):
    """
    Analyze the dataset.
    """
    logger.info("Analyzing dataset...")
    
    # Extract node IDs and features
    node_ids = features_df.iloc[:, 0].values
    features = features_df.iloc[:, 1:].values
    
    # Extract time steps (first feature)
    time_steps = features_df.iloc[:, 1].values
    unique_time_steps = np.unique(time_steps)
    
    # Extract classes
    class_mapping = {"unknown": -1, "1": 1, "2": 0}  # 1: illicit, 2: licit, unknown: -1
    classes_df["class"] = classes_df["class"].astype(str).map(class_mapping)
    
    # Create a mapping from node ID to class
    node_to_class = dict(zip(classes_df["txId"], classes_df["class"]))
    
    # Count classes
    class_counts = classes_df["class"].value_counts()
    illicit_count = class_counts.get(1, 0)
    licit_count = class_counts.get(0, 0)
    unknown_count = class_counts.get(-1, 0)
    
    # Print statistics
    print("\n===== Dataset Statistics =====")
    print(f"Number of nodes: {len(node_ids)}")
    print(f"Number of edges: {len(edges_df)}")
    print(f"Number of features: {features.shape[1]}")
    print(f"Number of time steps: {len(unique_time_steps)}")
    print(f"Class distribution:")
    print(f"  - Illicit transactions: {illicit_count} ({illicit_count / len(node_ids) * 100:.2f}%)")
    print(f"  - Licit transactions: {licit_count} ({licit_count / len(node_ids) * 100:.2f}%)")
    print(f"  - Unknown transactions: {unknown_count} ({unknown_count / len(node_ids) * 100:.2f}%)")
    print("=============================\n")
    
    # Save statistics to JSON
    stats = {
        "num_nodes": int(len(node_ids)),
        "num_edges": int(len(edges_df)),
        "num_features": int(features.shape[1]),
        "num_time_steps": int(len(unique_time_steps)),
        "class_distribution": {
            "illicit": int(illicit_count),
            "licit": int(licit_count),
            "unknown": int(unknown_count)
        }
    }
    
    with open(PROCESSED_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    logger.info("Dataset analysis completed")
    
    return node_ids, features, time_steps, node_to_class

def preprocess_dataset(features_df, classes_df, edges_df, args):
    """
    Preprocess the dataset.
    """
    logger.info("Preprocessing dataset...")
    
    # Extract node IDs and features
    node_ids = features_df.iloc[:, 0].values
    features = features_df.iloc[:, 1:].values
    
    # Extract time steps (first feature)
    time_steps = features_df.iloc[:, 1].values
    
    # Extract classes
    class_mapping = {"unknown": -1, "1": 1, "2": 0}  # 1: illicit, 2: licit, unknown: -1
    classes_df["class"] = classes_df["class"].astype(str).map(class_mapping)
    
    # Create a mapping from node ID to class
    node_to_class = dict(zip(classes_df["txId"], classes_df["class"]))
    
    # Create labels array
    labels = np.array([node_to_class.get(node_id, -1) for node_id in node_ids])
    
    # Normalize features if requested
    if args.normalize_features:
        logger.info("Normalizing features...")
        # Skip the time step feature (first column)
        time_feature = features[:, 0].copy().reshape(-1, 1)
        other_features = features[:, 1:]
        
        # Normalize other features
        feature_mean = np.mean(other_features, axis=0)
        feature_std = np.std(other_features, axis=0)
        feature_std[feature_std == 0] = 1  # Avoid division by zero
        
        normalized_features = (other_features - feature_mean) / feature_std
        
        # Combine time feature and normalized features
        features = np.hstack((time_feature, normalized_features))
        
        # Save normalization parameters
        np.save(PROCESSED_DIR / "feature_mean.npy", feature_mean)
        np.save(PROCESSED_DIR / "feature_std.npy", feature_std)
    
    # Remove self-loops if requested
    if args.remove_self_loops:
        logger.info("Removing self-loops...")
        edges_df = edges_df[edges_df["txId1"] != edges_df["txId2"]]
        logger.info(f"After removing self-loops: {len(edges_df)} edges")
    
    # Create node ID to index mapping
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Convert edge list to use indices
    edge_indices = []
    for _, edge in edges_df.iterrows():
        source = edge["txId1"]
        target = edge["txId2"]
        
        if source in node_id_to_idx and target in node_id_to_idx:
            source_idx = node_id_to_idx[source]
            target_idx = node_id_to_idx[target]
            edge_indices.append((source_idx, target_idx))
    
    edge_indices = np.array(edge_indices)
    
    # Save preprocessed data
    logger.info("Saving preprocessed data...")
    np.save(PROCESSED_DIR / "node_ids.npy", node_ids)
    np.save(PROCESSED_DIR / "features.npy", features)
    np.save(PROCESSED_DIR / "labels.npy", labels)
    np.save(PROCESSED_DIR / "time_steps.npy", time_steps)
    np.save(PROCESSED_DIR / "edge_indices.npy", edge_indices)
    
    # Save mapping for future reference
    with open(PROCESSED_DIR / "node_id_to_idx.json", "w") as f:
        json.dump({str(k): int(v) for k, v in node_id_to_idx.items()}, f)
    
    logger.info("Preprocessing completed")
    
    # Create train/val/test splits based on time steps
    create_temporal_splits(time_steps, labels)

def create_temporal_splits(time_steps, labels):
    """
    Create temporal train/validation/test splits.
    """
    logger.info("Creating temporal splits...")
    
    # Get unique time steps
    unique_time_steps = np.unique(time_steps)
    num_time_steps = len(unique_time_steps)
    
    # Define split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Calculate split indices
    train_end = int(num_time_steps * train_ratio)
    val_end = train_end + int(num_time_steps * val_ratio)
    
    # Get time steps for each split
    train_time_steps = unique_time_steps[:train_end]
    val_time_steps = unique_time_steps[train_end:val_end]
    test_time_steps = unique_time_steps[val_end:]
    
    # Create masks
    train_mask = np.isin(time_steps, train_time_steps)
    val_mask = np.isin(time_steps, val_time_steps)
    test_mask = np.isin(time_steps, test_time_steps)
    
    # Only include labeled nodes in the masks
    train_mask = train_mask & (labels != -1)
    val_mask = val_mask & (labels != -1)
    test_mask = test_mask & (labels != -1)
    
    # Save masks
    np.save(PROCESSED_DIR / "train_mask.npy", train_mask)
    np.save(PROCESSED_DIR / "val_mask.npy", val_mask)
    np.save(PROCESSED_DIR / "test_mask.npy", test_mask)
    
    # Print split statistics
    print("\n===== Split Statistics =====")
    print(f"Train set: {np.sum(train_mask)} nodes ({np.sum(train_mask) / np.sum(labels != -1) * 100:.2f}% of labeled nodes)")
    print(f"Validation set: {np.sum(val_mask)} nodes ({np.sum(val_mask) / np.sum(labels != -1) * 100:.2f}% of labeled nodes)")
    print(f"Test set: {np.sum(test_mask)} nodes ({np.sum(test_mask) / np.sum(labels != -1) * 100:.2f}% of labeled nodes)")
    print("============================\n")
    
    logger.info("Temporal splits created")

def main(args):
    """
    Main function to run the preprocessing.
    """
    start_time = time.time()
    
    # Load dataset
    features_df, classes_df, edges_df = load_dataset()
    
    # Analyze dataset if requested
    if args.analyze:
        node_ids, features, time_steps, node_to_class = analyze_dataset(features_df, classes_df, edges_df)
    
    # Preprocess dataset if requested
    if args.preprocess:
        preprocess_dataset(features_df, classes_df, edges_df, args)
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    logger.info("Preprocessing completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Elliptic Bitcoin dataset")
    
    parser.add_argument("--analyze", action="store_true",
                        help="Whether to analyze the dataset")
    parser.add_argument("--preprocess", action="store_true",
                        help="Whether to preprocess the dataset")
    parser.add_argument("--normalize_features", action="store_true",
                        help="Whether to normalize features")
    parser.add_argument("--remove_self_loops", action="store_true",
                        help="Whether to remove self-loops")
    
    args = parser.parse_args()
    
    # Ensure at least one action is specified
    if not any([args.analyze, args.preprocess]):
        parser.error("At least one of --analyze or --preprocess must be specified")
    
    main(args)
