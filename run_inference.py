"""
Run inference on the trained model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse
from pathlib import Path
import pickle
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROCESSED_DIR = Path("./processed_data")
RESULTS_DIR = Path("./results")
MODEL_PATH = RESULTS_DIR / "rf_model.pkl"

def load_model():
    """
    Load the trained model.
    """
    logger.info("Loading model...")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    logger.info("Model loaded successfully")
    
    return model

def load_data():
    """
    Load the preprocessed data.
    """
    logger.info("Loading data...")
    
    # Load data
    features = np.load(PROCESSED_DIR / "features.npy")
    labels = np.load(PROCESSED_DIR / "labels.npy")
    node_ids = np.load(PROCESSED_DIR / "node_ids.npy")
    
    logger.info(f"Loaded {features.shape[0]} nodes with {features.shape[1]} features")
    
    return features, labels, node_ids

def predict_transaction(model, features, transaction_idx):
    """
    Predict whether a transaction is illicit or licit.
    """
    # Get transaction features
    transaction_features = features[transaction_idx:transaction_idx+1]
    
    # Check for NaN values
    if np.isnan(transaction_features).any():
        logger.warning("Found NaN values in transaction features. Replacing with 0.")
        transaction_features = np.nan_to_num(transaction_features, nan=0.0)
    
    # Make prediction
    pred_proba = model.predict_proba(transaction_features)[0, 1]
    pred_class = 1 if pred_proba >= 0.5 else 0
    
    return pred_class, pred_proba

def predict_batch(model, features, indices):
    """
    Predict whether a batch of transactions are illicit or licit.
    """
    # Get batch features
    batch_features = features[indices]
    
    # Check for NaN values
    if np.isnan(batch_features).any():
        logger.warning("Found NaN values in batch features. Replacing with 0.")
        batch_features = np.nan_to_num(batch_features, nan=0.0)
    
    # Make predictions
    pred_proba = model.predict_proba(batch_features)[:, 1]
    pred_class = (pred_proba >= 0.5).astype(int)
    
    return pred_class, pred_proba

def analyze_high_risk_transactions(model, features, labels, node_ids, threshold=0.8):
    """
    Analyze high-risk transactions.
    """
    logger.info(f"Analyzing high-risk transactions (threshold: {threshold})...")
    
    # Check for NaN values
    if np.isnan(features).any():
        logger.warning("Found NaN values in features. Replacing with 0.")
        features = np.nan_to_num(features, nan=0.0)
    
    # Make predictions for all transactions
    pred_proba = model.predict_proba(features)[:, 1]
    
    # Find high-risk transactions
    high_risk_indices = np.where(pred_proba >= threshold)[0]
    high_risk_node_ids = node_ids[high_risk_indices]
    high_risk_proba = pred_proba[high_risk_indices]
    high_risk_labels = labels[high_risk_indices]
    
    # Count true positives and false positives
    true_positives = np.sum((high_risk_labels == 1) & (high_risk_labels != -1))
    false_positives = np.sum((high_risk_labels == 0) & (high_risk_labels != -1))
    unknown_labels = np.sum(high_risk_labels == -1)
    
    # Print results
    print(f"\n===== High-Risk Transactions (Threshold: {threshold}) =====")
    print(f"Number of high-risk transactions: {len(high_risk_indices)}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"Unknown labels: {unknown_labels}")
    print(f"Precision (excluding unknown): {true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0:.4f}")
    print("=================================================\n")
    
    # Save high-risk transactions to CSV
    high_risk_df = pd.DataFrame({
        "node_id": high_risk_node_ids,
        "probability": high_risk_proba,
        "true_label": high_risk_labels
    })
    
    high_risk_df.to_csv(RESULTS_DIR / f"high_risk_transactions_{threshold}.csv", index=False)
    
    logger.info(f"High-risk transactions analysis completed. Results saved to {RESULTS_DIR / f'high_risk_transactions_{threshold}.csv'}")
    
    return high_risk_indices, high_risk_proba

def main(args):
    """
    Main function to run inference.
    """
    # Load model
    model = load_model()
    
    # Load data
    features, labels, node_ids = load_data()
    
    # Run inference on specific transaction if provided
    if args.transaction_idx is not None:
        transaction_idx = args.transaction_idx
        true_label = labels[transaction_idx]
        node_id = node_ids[transaction_idx]
        
        pred_class, pred_proba = predict_transaction(model, features, transaction_idx)
        
        print(f"\n===== Transaction Analysis =====")
        print(f"Transaction ID: {node_id}")
        print(f"Predicted class: {'Illicit' if pred_class == 1 else 'Licit'}")
        print(f"Probability of being illicit: {pred_proba:.4f}")
        print(f"True label: {'Illicit' if true_label == 1 else 'Licit' if true_label == 0 else 'Unknown'}")
        print("==============================\n")
    
    # Analyze high-risk transactions
    if args.analyze_high_risk:
        threshold = args.threshold
        analyze_high_risk_transactions(model, features, labels, node_ids, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on the trained model")
    
    parser.add_argument("--transaction_idx", type=int, default=None,
                        help="Index of the transaction to analyze")
    parser.add_argument("--analyze_high_risk", action="store_true",
                        help="Whether to analyze high-risk transactions")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Threshold for high-risk transactions")
    
    args = parser.parse_args()
    
    # Ensure at least one action is specified
    if args.transaction_idx is None and not args.analyze_high_risk:
        parser.error("At least one of --transaction_idx or --analyze_high_risk must be specified")
    
    main(args)
