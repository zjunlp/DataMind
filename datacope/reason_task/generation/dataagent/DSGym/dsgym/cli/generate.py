#!/usr/bin/env python3
"""
DSGym trajectory generation command.

This command provides trajectory generation functionality for synthetic data creation
and pass@k evaluation.
"""

import argparse
from typing import Optional


def add_generate_parser(subparsers):
    """Add generate command parser."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate trajectories for synthetic data creation",
        description="Generate multiple trajectories per sample for pass@k evaluation"
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'gpt-4', 'together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput')")
    parser.add_argument("--backend", type=str, default="litellm",
                       choices=["litellm", "vllm", "sglang"],
                       help="Backend to use for model inference")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["daeval", "discoverybench", "qrdata", "dabstep", "dspredict", "bio"],
                       help="Dataset to use")
    parser.add_argument("--synthetic-path", type=str, default=None,
                       help="Path to synthetic dataset (optional)")
    
    # Trajectory generation
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature for trajectory generation")
    parser.add_argument("--k", type=int, default=8,
                       help="Number of trajectories to generate per sample")
    parser.add_argument("--limit", type=int, default=None,
                       help="Number of samples to process")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./trajectory_results",
                       help="Output directory for results")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name for output files")
    parser.add_argument("--no-metrics", action="store_true",
                       help="Skip metric computation (faster generation)")
    
    # Infrastructure
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000",
                       help="Code sandbox manager URL")
    parser.add_argument("--max-turns", type=int, default=15,
                       help="Maximum turns per sample")
    parser.add_argument("--max-workers", type=int, default=24,
                       help="Maximum number of parallel workers")
    
    # API keys
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (uses environment variable if not provided)")
    
    return parser


def run_generate(args) -> int:
    """Run trajectory generation command."""
    print("🎯 DSGym Trajectory Generation")
    print("⚠️ Generate functionality is not yet implemented.")
    print("Please use examples/generate_trajectories.py for now.")
    
    # TODO: Implement trajectory generation CLI
    # This will use the TrajectoryGenerator from dsgym.synth
    
    return 0


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="DSGym trajectory generation")
    add_generate_parser(parser._subparsers_action.add_parser if hasattr(parser, '_subparsers_action') else parser.add_subparsers())
    args = parser.parse_args()
    sys.exit(run_generate(args))