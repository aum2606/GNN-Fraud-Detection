"""
Simple demonstration of the GNN Fraud Detection project.
This script demonstrates the basic functionality without requiring PyTorch Geometric.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = Path("./elliptic_bitcoin_dataset")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

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
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(["Illicit", "Licit", "Unknown"], [illicit_count, licit_count, unknown_count])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig(RESULTS_DIR / "class_distribution.png")
    
    # Plot time step distribution
    plt.figure(figsize=(12, 6))
    plt.hist(time_steps, bins=len(unique_time_steps), alpha=0.7)
    plt.title("Transactions per Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Transactions")
    plt.savefig(RESULTS_DIR / "time_step_distribution.png")
    
    # Plot feature statistics
    plt.figure(figsize=(12, 6))
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    plt.errorbar(range(len(feature_means)), feature_means, yerr=feature_stds, fmt='o', capsize=5)
    plt.title("Feature Statistics")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Value (with Standard Deviation)")
    plt.savefig(RESULTS_DIR / "feature_statistics.png")
    
    logger.info("Dataset analysis completed")
    
    return node_ids, features, time_steps, node_to_class

def analyze_graph_structure(edges_df, node_ids, node_to_class):
    """
    Analyze the graph structure.
    """
    logger.info("Analyzing graph structure...")
    
    # Create a set of all nodes in the graph
    all_nodes = set(node_ids)
    
    # Count nodes with edges
    nodes_in_edges = set(edges_df["txId1"]).union(set(edges_df["txId2"]))
    nodes_with_edges = len(nodes_in_edges)
    isolated_nodes = len(all_nodes) - nodes_with_edges
    
    # Count edges between different classes
    edges_with_classes = 0
    illicit_to_illicit = 0
    illicit_to_licit = 0
    licit_to_licit = 0
    
    for _, edge in edges_df.iterrows():
        source = edge["txId1"]
        target = edge["txId2"]
        
        if source in node_to_class and target in node_to_class:
            source_class = node_to_class[source]
            target_class = node_to_class[target]
            
            if source_class != -1 and target_class != -1:
                edges_with_classes += 1
                
                if source_class == 1 and target_class == 1:
                    illicit_to_illicit += 1
                elif source_class == 0 and target_class == 0:
                    licit_to_licit += 1
                else:
                    illicit_to_licit += 1
    
    # Print statistics
    print("\n===== Graph Structure Statistics =====")
    print(f"Number of nodes with edges: {nodes_with_edges} ({nodes_with_edges / len(all_nodes) * 100:.2f}%)")
    print(f"Number of isolated nodes: {isolated_nodes} ({isolated_nodes / len(all_nodes) * 100:.2f}%)")
    print(f"Edges between known classes: {edges_with_classes}")
    print(f"  - Illicit to illicit: {illicit_to_illicit} ({illicit_to_illicit / edges_with_classes * 100:.2f}%)")
    print(f"  - Licit to licit: {licit_to_licit} ({licit_to_licit / edges_with_classes * 100:.2f}%)")
    print(f"  - Illicit to licit: {illicit_to_licit} ({illicit_to_licit / edges_with_classes * 100:.2f}%)")
    print("=====================================\n")
    
    # Plot edge distribution
    plt.figure(figsize=(10, 6))
    plt.bar(["Illicit-Illicit", "Licit-Licit", "Illicit-Licit"], 
            [illicit_to_illicit, licit_to_licit, illicit_to_licit])
    plt.title("Edge Distribution by Node Classes")
    plt.xlabel("Edge Type")
    plt.ylabel("Count")
    plt.savefig(RESULTS_DIR / "edge_distribution.png")
    
    logger.info("Graph structure analysis completed")

def main():
    """
    Main function to run the demo.
    """
    start_time = time.time()
    
    # Load dataset
    features_df, classes_df, edges_df = load_dataset()
    
    # Analyze dataset
    node_ids, features, time_steps, node_to_class = analyze_dataset(features_df, classes_df, edges_df)
    
    # Analyze graph structure
    analyze_graph_structure(edges_df, node_ids, node_to_class)
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    logger.info("Demo completed successfully")
    print("\nResults have been saved to the 'results' directory.")

if __name__ == "__main__":
    main()
