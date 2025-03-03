"""
Setup script for the GNN Fraud Detection package.
"""

from setuptools import setup, find_packages

setup(
    name="gnn_fraud_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "torch-scatter>=2.0.9",
        "torch-sparse>=0.6.12",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
    ],
    author="GNN Fraud Detection Team",
    author_email="example@example.com",
    description="Graph Neural Networks for Fraud Detection in Blockchain Networks",
    keywords="gnn, fraud-detection, blockchain, pytorch-geometric",
    python_requires=">=3.8",
)
