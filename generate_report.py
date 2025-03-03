"""
Script to generate a comprehensive report of the model's performance.
"""

import argparse
import logging
from src.utils.report import load_metrics, generate_html_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    
    logger.info("Report generation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a report of the model's performance")
    parser.add_argument("--model_type", type=str, default="gat", choices=["gat", "tgn"],
                        help="Type of model to generate report for (gat or tgn)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the HTML report")
    
    args = parser.parse_args()
    main(args)
