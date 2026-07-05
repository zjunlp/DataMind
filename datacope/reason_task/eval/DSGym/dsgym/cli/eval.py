#!/usr/bin/env python3
"""
DSGym evaluation command.

This command provides a unified interface for evaluating agents on datasets,
similar to the example scripts but with a standardized CLI.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add DSGym to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dsgym.datasets import DatasetRegistry
from dsgym.agents import ReActDSAgent, DSPredictReActAgent
from dsgym.eval import Evaluator
from dsgym.eval.utils import EvaluationConfig


def add_eval_parser(subparsers):
    """Add eval command parser."""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate agent on dataset",
        description="Evaluate LLM agents on data science datasets"
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'gpt-4', 'together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput')")
    parser.add_argument("--backend", type=str, default="litellm",
                       choices=["litellm", "vllm", "sglang"],
                       help="Backend to use for model inference")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["daeval", "discoverybench", "qrdata", "dabstep", "dspredict-easy", "dspredict-hard","bio"],
                       help="Dataset to evaluate on")
    parser.add_argument("--limit", type=int, default=None,
                       help="Number of samples to evaluate")
    parser.add_argument("--synthetic-path", type=str, default=None,
                       help="Path to synthetic dataset (optional)")
    
    # Evaluation configuration
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name for output files")
    parser.add_argument("--max-turns", type=int, default=15,
                       help="Maximum turns per sample")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    
    # Infrastructure
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000",
                       help="Code sandbox manager URL")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (auto-set based on backend if not provided)")
    
    # API keys
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (uses environment variable if not provided)")
    
    return parser


def run_eval(args) -> int:
    """Run evaluation command."""
    print("🚀 Starting DSGym Evaluation")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.limit if args.limit is not None else 'all'}")
    
    # Set default max_workers based on backend
    if args.max_workers is None:
        if args.backend == "litellm":
            args.max_workers = 24
        else:  # vllm or sglang
            args.max_workers = 1
    
    print(f"Max workers: {args.max_workers}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize agent
    print("🤖 Initializing agent...")
    agent_config = {
        "manager_url": args.manager_url,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "output_dir": args.output_dir,
    }
    
    if args.backend == "litellm" and args.api_key:
        agent_config["api_key"] = args.api_key
    
    try:
        if "dspredict" in args.dataset:
            agent = DSPredictReActAgent(
                backend=args.backend,
                model=args.model,
                submission_dir="./submissions",
                **agent_config
            )
        else:
            agent = ReActDSAgent(
            backend=args.backend,
            model=args.model,
            **agent_config
        )
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return 1
    
    # Load dataset
    print(f"📊 Loading {args.dataset} dataset...")
    try:
        dataset_config = {}
        if args.synthetic_path:
            dataset_config["synthetic_path"] = args.synthetic_path
            
        load_config = {
            "limit": args.limit
        }
        dataset_name = args.dataset
        if "dspredict" in dataset_name:
            dataset_config["split"] = dataset_name.split("-")[-1]
            dataset_name = dataset_name.split("-")[0]
            dataset_config["virtual_data_root"] = "/data"
            load_config["split"] = dataset_config["split"]
        dataset = DatasetRegistry.load(dataset_name, **dataset_config)
        samples = dataset.load(**load_config)
        print(f"✅ Loaded {len(samples)} samples from {args.dataset}")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Create evaluator with dataset-specific metrics
    print("⚖️ Setting up evaluator...")
    evaluator = Evaluator(
        protocol="multi_turn",
        dataset=dataset,  # This will automatically use dataset metrics and configs
        parallel_workers=args.max_workers
    )
    
    # Create evaluation config
    run_name = args.run_name or f"{args.dataset}_{args.backend}_{args.model.replace('/', '_')}"
    config = EvaluationConfig(
        model_name=args.model,
        backend_type=args.backend,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        run_name=run_name,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_workers=args.max_workers
    )
    
    # Run evaluation
    print("🏃 Starting evaluation...")
    try:
        results = evaluator.evaluate(
            agent=agent,
            tasks=samples,
            config=config,
            save_results=True
        )
        
        print("✅ Evaluation completed!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Print summary statistics
        if "metrics" in results:
            print("\n📈 Results Summary:")
            print("-" * 30)
            metrics = results["metrics"]
            
            # Show key metrics
            key_metrics = [
                "success_rate", "total_samples", "successful_samples", 
                "average_execution_time", "average_turns"
            ]
            
            for metric_name in key_metrics:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, float):
                        print(f"{metric_name}: {value:.3f}")
                    else:
                        print(f"{metric_name}: {value}")
            
            # Show dataset-specific metrics
            metric_scores = {}
            for key, value in metrics.items():
                if key.endswith("_mean") and not key.startswith("error"):
                    metric_name = key.replace("_mean", "")
                    metric_scores[metric_name] = value
            
            if metric_scores:
                print("\n📊 Dataset Metrics:")
                print("-" * 20)
                for metric_name, score in metric_scores.items():
                    print(f"{metric_name}: {score:.3f}")
        if "dspredict" in args.dataset:
            dataset.print_dspredict_results_overview(results["results"])
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="DSGym evaluation")
    add_eval_parser(parser._subparsers_action.add_parser if hasattr(parser, '_subparsers_action') else parser.add_subparsers())
    args = parser.parse_args()
    sys.exit(run_eval(args))