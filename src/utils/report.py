"""
Generate a comprehensive report of the model's performance.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import datetime

from src.config import RESULTS_DIR
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_metrics(model_type: str) -> Dict[str, float]:
    """
    Load metrics from file.
    
    Args:
        model_type: Type of model.
        
    Returns:
        Dictionary of metrics.
    """
    metrics_path = os.path.join(RESULTS_DIR, f"{model_type}_metrics.txt")
    
    if not os.path.exists(metrics_path):
        logger.error(f"Metrics file not found at {metrics_path}")
        return {}
    
    metrics = {}
    with open(metrics_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = int(value)
    
    return metrics


def generate_html_report(
    model_type: str,
    metrics: Dict[str, float],
    output_path: Optional[str] = None
) -> None:
    """
    Generate an HTML report of the model's performance.
    
    Args:
        model_type: Type of model.
        metrics: Dictionary of metrics.
        output_path: Path to save the HTML report.
    """
    if not output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_DIR, f"{model_type}_report_{timestamp}.html")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GNN Fraud Detection Report - {model_type.upper()}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                background-color: #3498db;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 30px;
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .metrics-table th, .metrics-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .metrics-table th {{
                background-color: #f2f2f2;
            }}
            .metrics-table tr:hover {{
                background-color: #f5f5f5;
            }}
            .image-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                margin-top: 20px;
            }}
            .image-item {{
                margin: 10px;
                text-align: center;
            }}
            .image-item img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                background-color: #f2f2f2;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>GNN Fraud Detection Report</h1>
                <h2>Model: {model_type.upper()}</h2>
                <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
    """
    
    # Add metrics to the table
    for metric, value in metrics.items():
        if isinstance(value, float):
            html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{value:.4f}</td>
                    </tr>
            """
        else:
            html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{value}</td>
                    </tr>
            """
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="image-container">
    """
    
    # Add visualizations
    cm_path = os.path.join(RESULTS_DIR, f"{model_type}_confusion_matrix.png")
    roc_path = os.path.join(RESULTS_DIR, f"{model_type}_roc_curve.png")
    pr_path = os.path.join(RESULTS_DIR, f"{model_type}_pr_curve.png")
    
    if os.path.exists(cm_path):
        html_content += f"""
                    <div class="image-item">
                        <img src="{os.path.relpath(cm_path, os.path.dirname(output_path))}" alt="Confusion Matrix">
                        <p>Confusion Matrix</p>
                    </div>
        """
    
    if os.path.exists(roc_path):
        html_content += f"""
                    <div class="image-item">
                        <img src="{os.path.relpath(roc_path, os.path.dirname(output_path))}" alt="ROC Curve">
                        <p>ROC Curve (AUC: {metrics.get('auc', 'N/A')})</p>
                    </div>
        """
    
    if os.path.exists(pr_path):
        html_content += f"""
                    <div class="image-item">
                        <img src="{os.path.relpath(pr_path, os.path.dirname(output_path))}" alt="Precision-Recall Curve">
                        <p>Precision-Recall Curve (AUPRC: {metrics.get('auprc', 'N/A')})</p>
                    </div>
        """
    
    # Add graph visualizations if they exist
    gt_path = os.path.join(RESULTS_DIR, f"{model_type}_graph_ground_truth.png")
    pred_path = os.path.join(RESULTS_DIR, f"{model_type}_graph_predictions.png")
    
    if os.path.exists(gt_path):
        html_content += f"""
                    <div class="image-item">
                        <img src="{os.path.relpath(gt_path, os.path.dirname(output_path))}" alt="Ground Truth Graph">
                        <p>Ground Truth Labels</p>
                    </div>
        """
    
    if os.path.exists(pred_path):
        html_content += f"""
                    <div class="image-item">
                        <img src="{os.path.relpath(pred_path, os.path.dirname(output_path))}" alt="Prediction Graph">
                        <p>Prediction Scores</p>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Interpretation</h2>
                <p>
                    This report shows the performance of the GNN-based fraud detection model on the Elliptic Bitcoin dataset.
                    The model aims to identify illicit transactions in the Bitcoin network by leveraging graph structure and node features.
                </p>
                <h3>Key Findings:</h3>
                <ul>
    """
    
    # Add interpretations based on metrics
    if 'accuracy' in metrics:
        html_content += f"""
                    <li>The model achieved an accuracy of {metrics['accuracy']:.4f}, indicating its overall correctness in classifying transactions.</li>
        """
    
    if 'precision' in metrics and 'recall' in metrics:
        html_content += f"""
                    <li>Precision of {metrics['precision']:.4f} shows that {metrics['precision']*100:.1f}% of transactions flagged as illicit were actually illicit.</li>
                    <li>Recall of {metrics['recall']:.4f} indicates that the model detected {metrics['recall']*100:.1f}% of all illicit transactions in the dataset.</li>
        """
    
    if 'f1' in metrics:
        html_content += f"""
                    <li>F1 score of {metrics['f1']:.4f} represents the harmonic mean of precision and recall, providing a balanced measure of the model's performance.</li>
        """
    
    if 'auc' in metrics:
        html_content += f"""
                    <li>ROC AUC of {metrics['auc']:.4f} demonstrates the model's ability to distinguish between illicit and licit transactions across different threshold settings.</li>
        """
    
    if 'true_positives' in metrics and 'false_positives' in metrics:
        html_content += f"""
                    <li>The model correctly identified {metrics['true_positives']} illicit transactions (true positives) while mistakenly flagging {metrics['false_positives']} licit transactions as illicit (false positives).</li>
        """
    
    if 'false_negatives' in metrics:
        html_content += f"""
                    <li>The model missed {metrics['false_negatives']} illicit transactions (false negatives), which represents potential fraud that went undetected.</li>
        """
    
    html_content += """
                </ul>
                <h3>Recommendations:</h3>
                <ul>
                    <li>Consider adjusting the classification threshold to balance precision and recall based on the specific business requirements.</li>
                    <li>Explore feature engineering to improve the model's ability to detect complex fraud patterns.</li>
                    <li>Investigate the false negatives to understand what types of illicit transactions the model is missing.</li>
                    <li>Implement continuous monitoring and retraining as new transaction data becomes available.</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>GNN Fraud Detection Project</p>
                <p>© 2025</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML content to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated at {output_path}")


def main(args):
    """
    Main function to generate a report.
    """
    # Load metrics
    metrics = load_metrics(args.model_type)
    
    if not metrics:
        logger.error(f"No metrics found for model type: {args.model_type}")
        return
    
    # Generate HTML report
    generate_html_report(
        args.model_type,
        metrics,
        output_path=args.output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a report of the model's performance")
    parser.add_argument("--model_type", type=str, default="gat", choices=["gat", "tgn"],
                        help="Type of model to generate report for (gat or tgn)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the HTML report")
    
    args = parser.parse_args()
    main(args)
