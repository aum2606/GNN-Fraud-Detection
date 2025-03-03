"""
Train a simple model on the preprocessed Elliptic Bitcoin dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse
from pathlib import Path
import time
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROCESSED_DIR = Path("./processed_data")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_preprocessed_data():
    """
    Load the preprocessed data.
    """
    logger.info("Loading preprocessed data...")
    
    # Load data
    features = np.load(PROCESSED_DIR / "features.npy")
    labels = np.load(PROCESSED_DIR / "labels.npy")
    train_mask = np.load(PROCESSED_DIR / "train_mask.npy")
    val_mask = np.load(PROCESSED_DIR / "val_mask.npy")
    test_mask = np.load(PROCESSED_DIR / "test_mask.npy")
    
    logger.info(f"Loaded {features.shape[0]} nodes with {features.shape[1]} features")
    
    # Filter out unknown labels (-1)
    train_mask = train_mask & (labels != -1)
    val_mask = val_mask & (labels != -1)
    test_mask = test_mask & (labels != -1)
    
    # Check for NaN values
    if np.isnan(features).any():
        logger.warning("Found NaN values in features. Replacing with 0.")
        features = np.nan_to_num(features, nan=0.0)
    
    if np.isnan(labels).any():
        logger.warning("Found NaN values in labels. This should not happen.")
    
    return features, labels, train_mask, val_mask, test_mask

def train_model(features, labels, train_mask, val_mask):
    """
    Train a Random Forest model.
    """
    logger.info("Training Random Forest model...")
    
    # Get training data
    X_train = features[train_mask]
    y_train = labels[train_mask]
    
    # Get validation data
    X_val = features[val_mask]
    y_val = labels[val_mask]
    
    # Check for NaN values
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        logger.warning("Found NaN values in training data. Replacing with 0.")
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0).astype(int)
    
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        logger.warning("Found NaN values in validation data. Replacing with 0.")
        X_val = np.nan_to_num(X_val, nan=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0).astype(int)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val, val_pred)
    val_precision = precision_score(y_val, val_pred)
    val_recall = recall_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    # Print validation metrics
    print("\n===== Validation Metrics =====")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    print(f"AUC-ROC: {val_auc:.4f}")
    print("==============================\n")
    
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, features, labels, test_mask):
    """
    Evaluate the model on the test set.
    """
    logger.info("Evaluating model on test set...")
    
    # Get test data
    X_test = features[test_mask]
    y_test = labels[test_mask]
    
    # Check for NaN values
    if np.isnan(X_test).any() or np.isnan(y_test).any():
        logger.warning("Found NaN values in test data. Replacing with 0.")
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0).astype(int)
    
    # Make predictions
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_pred_proba)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    
    # Print test metrics
    print("\n===== Test Metrics =====")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print("========================\n")
    
    # Save metrics
    metrics = {
        "accuracy": float(test_accuracy),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1),
        "auc": float(test_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }
    
    with open(RESULTS_DIR / "rf_metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, test_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Licit', 'Illicit'])
    plt.yticks([0, 1], ['Licit', 'Illicit'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rf_confusion_matrix.png")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, test_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {test_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(RESULTS_DIR / "rf_roc_curve.png")
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_test, test_pred_proba)
    avg_precision = average_precision_score(y_test, test_pred_proba)
    plt.plot(recall, precision, label=f'AP = {avg_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(RESULTS_DIR / "rf_pr_curve.png")
    
    logger.info("Model evaluation completed")

def analyze_feature_importance(model, features):
    """
    Analyze feature importance.
    """
    logger.info("Analyzing feature importance...")
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Sort feature indices by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.bar(range(20), feature_importance[sorted_idx[:20]])
    plt.xticks(range(20), sorted_idx[:20])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.savefig(RESULTS_DIR / "rf_feature_importance.png")
    
    # Save feature importance
    with open(RESULTS_DIR / "rf_feature_importance.txt", "w") as f:
        for i, idx in enumerate(sorted_idx[:50]):
            f.write(f"Rank {i+1}: Feature {idx} - Importance: {feature_importance[idx]:.6f}\n")
    
    logger.info("Feature importance analysis completed")

def main(args):
    """
    Main function to run the training.
    """
    start_time = time.time()
    
    # Load preprocessed data
    features, labels, train_mask, val_mask, test_mask = load_preprocessed_data()
    
    # Train model
    model = train_model(features, labels, train_mask, val_mask)
    
    # Evaluate model
    evaluate_model(model, features, labels, test_mask)
    
    # Analyze feature importance
    analyze_feature_importance(model, features)
    
    # Save model
    logger.info("Saving model...")
    with open(RESULTS_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the preprocessed Elliptic Bitcoin dataset")
    
    args = parser.parse_args()
    main(args)
