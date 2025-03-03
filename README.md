# GNN Fraud Detection

A comprehensive system for detecting fraudulent transactions in blockchain networks using Graph Neural Networks (GNNs).

## Overview

This project implements a fraud detection system using Graph Neural Networks (GNNs) to identify illicit transactions in a blockchain network. The implementation leverages the Elliptic Bitcoin dataset, which contains features of Bitcoin transactions and their associated labels (licit, illicit, or unknown).

The project includes:
- Data preprocessing and analysis
- Implementation of Graph Attention Networks (GAT) and Temporal Graph Neural Networks (TGN)
- Training and evaluation scripts
- Visualization tools for model performance
- Reporting and model comparison utilities

## Latest Updates

- Fixed compatibility issues with PyTorch Geometric
- Successfully trained and evaluated both GAT and TGN models
- Added model comparison functionality
- Generated comprehensive performance reports
- TGN model outperforms GAT across most metrics (see `gnn_model_comparison_report.md`)

## Project Structure

```
.
├── data/                      # Data directory
├── elliptic_bitcoin_dataset/  # Raw dataset
├── results/                   # Results directory
├── src/                       # Source code
│   ├── config.py              # Configuration parameters
│   ├── data/                  # Data processing modules
│   │   ├── dataset.py         # Dataset class
│   │   ├── dataloader.py      # Data loaders
│   │   └── preprocess.py      # Preprocessing script
│   ├── models/                # Model implementations
│   │   ├── gat.py             # Graph Attention Network
│   │   └── tgn.py             # Temporal Graph Neural Network
│   ├── utils/                 # Utility functions
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── visualization.py   # Visualization tools
│   │   └── report.py          # Reporting utilities
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── main.py                # Main script
├── run_pipeline.py            # Script to run the entire pipeline
├── generate_report.py         # Script to generate a report
├── compare_models.py          # Script to compare different ML models
├── compare_gnn_models.py      # Script to compare GAT and TGN models
├── gnn_model_comparison_report.md # Detailed comparison of GNN models
├── inference.py               # Script for inference
├── setup_and_run.bat          # Setup script for Windows
├── setup.py                   # Package setup script
└── requirements.txt           # Dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda (recommended for environment management)
- CUDA-compatible GPU (optional, but recommended for faster training)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gnn-fraud-detection.git
   cd gnn-fraud-detection
   ```

2. Create and activate a conda environment:
   ```
   conda create -n fraud_detection python=3.8
   conda activate fraud_detection
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install PyTorch Geometric:
   ```
   # For CUDA support (replace xx with your CUDA version)
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cuxx.html
   
   # For CPU only
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cpu.html
   ```

5. For Windows users, you can use the provided setup script:
   ```
   setup_and_run.bat
   ```

## Usage

### Data Preprocessing

Preprocess the Elliptic Bitcoin dataset:

```
python -m src.data.preprocess --analyze --preprocess --normalize_features --remove_self_loops
```

### Training

Train a Graph Attention Network (GAT) model:

```
python -m src.main --model_type gat --train
```

Train a Temporal Graph Neural Network (TGN) model:

```
python -m src.main --model_type tgn --train
```

### Evaluation

Evaluate a trained model:

```
python -m src.main --model_type gat --evaluate
```

### Running the Entire Pipeline

Run the entire pipeline (preprocessing, training, and evaluation):

```
python run_pipeline.py --preprocess --train --evaluate --model_type gat
```

### Generating Reports

Generate a performance report for a trained model:

```
python generate_report.py --model_type gat
```

### Comparing Models

Compare different models:

```
python compare_models.py --model_types gat tgn
```

### Inference

Use a trained model for inference on a specific transaction:

```
python inference.py --model_type gat --transaction_id <transaction_id> --visualize
```

## Dataset

This project uses the Elliptic Bitcoin dataset, which contains:
- Transaction features (elliptic_txs_features.csv)
- Transaction classes (elliptic_txs_classes.csv)
- Transaction edges (elliptic_txs_edgelist.csv)

The dataset is organized into time steps, with each transaction belonging to a specific time step.

## Models

### Graph Attention Network (GAT)

The GAT model uses attention mechanisms to learn the importance of neighboring nodes. It aggregates information from neighbors based on learned attention weights.

### Temporal Graph Neural Network (TGN)

The TGN model extends GAT by incorporating temporal information, allowing it to capture the dynamics of transactions over time.

## Evaluation Metrics

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Area Under the ROC Curve (AUC)
- Area Under the Precision-Recall Curve (AUPRC)

## Visualization

The project includes visualization tools for:
- Training history
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Graph visualization with predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Elliptic Bitcoin dataset is provided by Elliptic (https://www.elliptic.co/)
- PyTorch Geometric library for GNN implementations
- PyTorch for deep learning framework
