"""
Script to run the entire GNN Fraud Detection pipeline.
This script demonstrates how to use the project for fraud detection.
"""

import os
import argparse
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(args):
    """
    Run the entire GNN Fraud Detection pipeline.
    
    Args:
        args: Command-line arguments.
    """
    start_time = time.time()
    
    # Step 1: Preprocess the dataset
    if args.preprocess:
        logger.info("Step 1: Preprocessing the dataset...")
        preprocess_cmd = (
            f"python -m src.data.preprocess "
            f"--analyze "
            f"--preprocess "
            f"--output_dir {args.processed_data_dir} "
            f"--normalize_features "
            f"--remove_self_loops"
        )
        
        if args.remove_isolated_nodes:
            preprocess_cmd += " --remove_isolated_nodes"
        
        logger.info(f"Running command: {preprocess_cmd}")
        os.system(preprocess_cmd)
    
    # Step 2: Train the model
    if args.train:
        logger.info("Step 2: Training the model...")
        train_cmd = (
            f"python -m src.main "
            f"--model_type {args.model_type} "
            f"--data_root {args.data_root} "
            f"--train "
        )
        
        if args.use_wandb:
            train_cmd += " --use_wandb"
        
        if args.use_subgraph_sampling:
            train_cmd += " --use_subgraph_sampling"
        
        if args.cpu:
            train_cmd += " --cpu"
        
        logger.info(f"Running command: {train_cmd}")
        os.system(train_cmd)
    
    # Step 3: Evaluate the model
    if args.evaluate:
        logger.info("Step 3: Evaluating the model...")
        eval_cmd = (
            f"python -m src.main "
            f"--model_type {args.model_type} "
            f"--data_root {args.data_root} "
            f"--evaluate "
        )
        
        if args.use_subgraph_sampling:
            eval_cmd += " --use_subgraph_sampling"
        
        if args.visualize_graph:
            eval_cmd += " --visualize_graph"
        
        if args.cpu:
            eval_cmd += " --cpu"
        
        logger.info(f"Running command: {eval_cmd}")
        os.system(eval_cmd)
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GNN Fraud Detection pipeline")
    
    # Pipeline steps
    parser.add_argument("--preprocess", action="store_true",
                        help="Whether to preprocess the dataset")
    parser.add_argument("--train", action="store_true",
                        help="Whether to train the model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Whether to evaluate the model")
    
    # Preprocessing options
    parser.add_argument("--processed_data_dir", type=str, default="./processed_data",
                        help="Directory to save the processed data")
    parser.add_argument("--remove_isolated_nodes", action="store_true",
                        help="Whether to remove isolated nodes during preprocessing")
    
    # Model options
    parser.add_argument("--model_type", type=str, default="gat", choices=["gat", "tgn"],
                        help="Type of model to use (gat or tgn)")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for the dataset")
    
    # Training options
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--use_subgraph_sampling", action="store_true",
                        help="Whether to use subgraph sampling")
    
    # Evaluation options
    parser.add_argument("--visualize_graph", action="store_true",
                        help="Whether to visualize the graph with predictions")
    
    # General options
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    
    args = parser.parse_args()
    
    # Ensure at least one step is specified
    if not any([args.preprocess, args.train, args.evaluate]):
        parser.error("At least one of --preprocess, --train, or --evaluate must be specified")
    
    # Run the pipeline
    run_pipeline(args)
