"""
Configuration parameters for the GNN Fraud Detection project.
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "elliptic_bitcoin_dataset")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Dataset files
NODE_FEATURES_FILE = os.path.join(DATA_DIR, "elliptic_txs_features.csv")
EDGE_LIST_FILE = os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv")
CLASSES_FILE = os.path.join(DATA_DIR, "elliptic_txs_classes.csv")

# Data processing parameters
TRAIN_TIME_STEPS = list(range(1, 31))  # Time steps 1-30 for training
VAL_TIME_STEPS = list(range(31, 41))   # Time steps 31-40 for validation
TEST_TIME_STEPS = list(range(41, 50))  # Time steps 41-49 for testing

# Class mapping
CLASS_MAPPING = {
    "unknown": -1,
    "1": 1,  # Illicit
    "2": 0   # Licit
}

# Model parameters
MODEL_CONFIG = {
    "hidden_channels": 64,
    "num_layers": 3,
    "heads": 8,
    "dropout": 0.2,
    "edge_dim": None,  # Set to None if edges don't have features
}

# Training parameters
TRAIN_CONFIG = {
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    "epochs": 100,
    "patience": 10,  # Early stopping patience
    "pos_weight": 50,  # Weight for positive class (illicit) to handle imbalance
}

# Evaluation parameters
EVAL_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "auc"],
    "threshold": 0.5,  # Threshold for binary classification
}

# Experiment tracking
WANDB_CONFIG = {
    "project": "gnn-fraud-detection",
    "entity": None,  # Set to your wandb username or team name
    "log_model": True,
}
