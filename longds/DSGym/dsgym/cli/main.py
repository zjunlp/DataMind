"""
Main CLI entry point for DSGym.
"""

import sys
import argparse
from typing import List, Optional

from .eval import add_eval_parser, run_eval
from .generate import add_generate_parser, run_generate
from .train import add_train_parser, run_train


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="dsgym",
        description="DSGym: Unified Data Science Benchmark Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a model on QRData
  dsgym eval --model gpt-4o --dataset qrdata --limit 10

  # Generate trajectories for synthetic data
  dsgym generate --model gpt-4o --dataset qrdata --k 8 --temperature 0.8

  # Train a model (coming soon)
  dsgym train --base-model llama2-7b --train-data ./trajectories
        """
    )
    
    parser.add_argument(
        "--version",
        action="version", 
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )
    
    # Add command parsers
    add_eval_parser(subparsers)
    add_generate_parser(subparsers)
    add_train_parser(subparsers)
    
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    if parsed_args.command == "eval":
        return run_eval(parsed_args)
    elif parsed_args.command == "generate":
        return run_generate(parsed_args)
    elif parsed_args.command == "train":
        return run_train(parsed_args)
    else:
        print(f"❌ Unknown command: {parsed_args.command}")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())