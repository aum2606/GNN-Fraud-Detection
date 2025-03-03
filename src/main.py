"""
Main script to run the GNN Fraud Detection pipeline.
"""

import os
import argparse
import logging
from pathlib import Path

from src.config import RESULTS_DIR
from src.train import main as train_main
from src.evaluate import main as evaluate_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    """
    Main function to run the GNN Fraud Detection pipeline.
    """
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Set up arguments for training
    train_args = argparse.Namespace(
        model_type=args.model_type,
        data_root=args.data_root,
        use_wandb=args.use_wandb,
        cpu=args.cpu,
        use_subgraph_sampling=args.use_subgraph_sampling
    )
    
    # Set up arguments for evaluation
    eval_args = argparse.Namespace(
        model_type=args.model_type,
        data_root=args.data_root,
        cpu=args.cpu,
        use_subgraph_sampling=args.use_subgraph_sampling,
        visualize_graph=args.visualize_graph
    )
    
    # Run training
    if args.train:
        logger.info(f"Starting training for model type: {args.model_type}")
        train_main(train_args)
    
    # Run evaluation
    if args.evaluate:
        logger.info(f"Starting evaluation for model type: {args.model_type}")
        evaluate_main(eval_args)
    
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN Fraud Detection pipeline")
    parser.add_argument("--model_type", type=str, default="gat", choices=["gat", "tgn"],
                        help="Type of model to use (gat or tgn)")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for the dataset")
    parser.add_argument("--train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--evaluate", action="store_true",
                        help="Whether to run evaluation")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--use_subgraph_sampling", action="store_true",
                        help="Whether to use subgraph sampling")
    parser.add_argument("--visualize_graph", action="store_true",
                        help="Whether to visualize the graph with predictions")
    
    args = parser.parse_args()
    
    # Ensure at least one of train or evaluate is specified
    if not args.train and not args.evaluate:
        parser.error("At least one of --train or --evaluate must be specified")
    
    main(args)
