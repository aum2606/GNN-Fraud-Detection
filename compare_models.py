"""
Script to compare different models.
"""

import os
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict

from src.utils.report import load_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_models(model_types: List[str], output_path: str = None):
    """
    Compare different models and generate a comparison chart.
    
    Args:
        model_types: List of model types to compare.
        output_path: Path to save the comparison chart.
    """
    logger.info(f"Comparing models: {', '.join(model_types)}")
    
    # Load metrics for each model
    all_metrics = {}
    for model_type in model_types:
        metrics = load_metrics(model_type)
        if metrics:
            all_metrics[model_type] = metrics
        else:
            logger.warning(f"No metrics found for model type: {model_type}")
    
    if not all_metrics:
        logger.error("No metrics found for any model")
        return
    
    # Select common metrics for comparison
    common_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Create a DataFrame for comparison
    comparison_data = []
    for model_type, metrics in all_metrics.items():
        row = {'Model': model_type.upper()}
        for metric in common_metrics:
            if metric in metrics:
                row[metric.capitalize()] = metrics[metric]
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\n===== Model Comparison =====")
    print(comparison_df.to_string(index=False))
    print("============================\n")
    
    # Plot comparison chart
    plt.figure(figsize=(12, 8))
    
    # Set up bar positions
    bar_width = 0.15
    x = np.arange(len(common_metrics))
    
    # Plot bars for each model
    for i, (model_type, metrics) in enumerate(all_metrics.items()):
        values = [metrics.get(metric, 0) for metric in common_metrics]
        plt.bar(x + i * bar_width, values, width=bar_width, label=model_type.upper())
    
    # Set up chart
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + bar_width * (len(all_metrics) - 1) / 2, [m.capitalize() for m in common_metrics])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save or show the chart
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Comparison chart saved to {output_path}")
    else:
        plt.show()


def main(args):
    """
    Main function to compare models.
    """
    compare_models(
        args.model_types,
        output_path=args.output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different models")
    parser.add_argument("--model_types", type=str, nargs='+', default=["gat", "tgn"],
                        help="Types of models to compare")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the comparison chart")
    
    args = parser.parse_args()
    main(args)
