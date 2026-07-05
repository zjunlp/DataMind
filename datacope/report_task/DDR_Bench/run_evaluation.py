#!/usr/bin/env python3
"""
Unified Evaluation Script for DDR_Bench.

Single entry point for evaluating agent results across all scenarios:
- MIMIC: Evaluate medical insights against QA pairs
- 10-K: Evaluate financial insights against QA pairs
- GLOBEM: Evaluate behavioral insights against QA pairs

See README.md for detailed usage instructions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_cli_path(raw_path: str | None, original_cwd: Path, prefer_existing: bool = False) -> str | None:
    """Resolve explicit CLI paths from either caller cwd or DDR_Bench root."""
    if not raw_path:
        return raw_path
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_candidate = original_cwd / path
    if prefer_existing and cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    project_candidate = PROJECT_ROOT / path
    if prefer_existing and project_candidate.exists():
        return str(project_candidate.resolve())

    return str(cwd_candidate.resolve())


def derive_split_artifact_path(base_path: str, split: str) -> str:
    """Derive a split-specific artifact path by appending _<split> to the stem."""
    path = Path(base_path)
    return str(path.with_name(f"{path.stem}_{split}{path.suffix}"))


def resolve_split_evaluation_paths(scenario_config, split: str | None) -> tuple[str, str]:
    """Resolve split-aware QA/log paths for all scenarios."""
    qa_file = scenario_config.qa_file
    log_dir = scenario_config.log_dir

    if not split:
        return qa_file, log_dir

    qa_file = derive_split_artifact_path(qa_file, split)
    log_dir = f"{log_dir}_{split}"
    return qa_file, log_dir


def resolve_10k_evaluation_paths(scenario_config, split: str | None) -> tuple[str, str]:
    """Backward-compatible wrapper for older 10-K-only callers."""
    return resolve_split_evaluation_paths(scenario_config, split)


def resolve_logs_dir(raw_logs: str) -> str:
    """Resolve a log directory path, accepting names under ./logs."""
    path = Path(raw_logs).expanduser()
    if path.exists() or path.is_absolute() or len(path.parts) > 1:
        return str(path)

    logs_candidate = Path("logs") / path
    if logs_candidate.exists():
        return str(logs_candidate)

    return str(path)


def derive_default_output_path(scenario: str, log_dir: str) -> str:
    """Derive the original-framework default evaluation output path."""
    log_dir_name = Path(log_dir).name
    return f"./{scenario}_{log_dir_name}_evaluation_result.json"


def format_metric_summary(output_file: str) -> str:
    """Return a labeled summary of the main evaluation metrics."""
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    stats = data["overall_statistics"]
    metric_rows = [
        ("Sample average, message-wise context", stats["message_wise_context"]["average_correct_percentage"]),
        ("Sample average, chat-wise context", stats["chat_wise_context"]["average_correct_percentage"]),
        ("Item overall, message-wise context", stats["message_wise_context"]["overall_correct_percentage"]),
        ("Item overall, chat-wise context", stats["chat_wise_context"]["overall_correct_percentage"]),
    ]
    average_score = sum(value for _, value in metric_rows) / len(metric_rows)
    metric_rows.append(("Average of the four scores", average_score))

    lines = ["Metric summary:"]
    lines.extend(f"  {label}: {value:.2f}" for label, value in metric_rows)
    return "\n".join(lines)


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="DDR_Bench Unified Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate MIMIC results (using config settings)
  python run_evaluation.py --scenario mimic

  # Evaluate 10-K results with custom logs
  python run_evaluation.py --scenario 10k --logs ./logs/10k_baseline_0426_skill_test
        """
    )

    parser.add_argument("--scenario", required=True, choices=["mimic", "10k", "globem"],
                        help="Evaluation scenario")
    
    # Path overrides
    # parser.add_argument("--qa-file", help="Path to QA file (overrides config)")  # Removed to enforce config usage
    parser.add_argument(
        "--logs",
        "--log-dir",
        dest="logs",
        help="Agent logs directory to evaluate; a bare name is resolved under ./logs when present",
    )
    parser.add_argument("--output", "-o", help="Output file path for results")
    
    # Execution options
    parser.add_argument("--test-mode", "-t", action="store_true",
                        help="Run in test mode (process only first entity)")
    parser.add_argument("--split", choices=["explore", "test"], help="Explore/test split to evaluate")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of entities to evaluate in parallel (default: 1)")
    
    # Configuration file
    parser.add_argument("--config", help="Path to config.yaml file")
    
    args = parser.parse_args()

    original_cwd = Path.cwd()
    if args.config:
        args.config = resolve_cli_path(args.config, original_cwd, prefer_existing=True)
    if args.logs:
        logs_arg = Path(args.logs).expanduser()
        if logs_arg.is_absolute() or len(logs_arg.parts) > 1:
            args.logs = resolve_cli_path(args.logs, original_cwd, prefer_existing=True)
        else:
            cwd_candidate = original_cwd / logs_arg
            project_candidate = PROJECT_ROOT / logs_arg
            if cwd_candidate.exists():
                args.logs = str(cwd_candidate.resolve())
            elif project_candidate.exists():
                args.logs = str(project_candidate.resolve())
    if args.output:
        args.output = resolve_cli_path(args.output, original_cwd)
    os.chdir(PROJECT_ROOT)

    try:
        from .config import get_config
        from .evaluate import UnifiedEvaluator
    except ImportError:
        from config import get_config
        from evaluate import UnifiedEvaluator
    
    # Load configuration
    config = get_config(args.config)
    scenario_config = config.get_scenario(args.scenario)

    # Get paths from config (CLI overrides removed)
    qa_file, log_dir = resolve_split_evaluation_paths(scenario_config, args.split)

    if args.logs:
        log_dir = resolve_logs_dir(args.logs)
    
    if not qa_file:
        parser.error(f"qa_file not found in config.yaml for scenario {args.scenario}. Please check your config.")
    if not log_dir:
        parser.error(f"log_dir not found in config.yaml for scenario {args.scenario}. Please check your config.")

    if not Path(qa_file).exists():
        if args.split:
            parser.error(
                f"Split QA file not found: {qa_file}. "
                "Please generate split artifacts first."
            )
        parser.error(f"QA file not found: {qa_file}")
    if not Path(log_dir).exists():
        if args.split:
            parser.error(
                f"Split log directory not found: {log_dir}. "
                "Please run the split analysis first."
            )
        parser.error(f"Log directory not found: {log_dir}")
    
    # Resolve evaluation parameters from CONFIG (no CLI overrides for these)
    provider = config.evaluation.provider or "azure"
    model = config.evaluation.model or "gpt-5-mini"
    max_retries = config.evaluation.max_retries or 5
    retry_delay = config.evaluation.retry_delay or 2.0
    log_level = config.agent.log_level or "INFO"

    output_file = args.output or derive_default_output_path(args.scenario, log_dir)
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    
    # Set log level for current process
    os.environ["DDR_LOG_LEVEL"] = log_level
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    vllm_host = "localhost" # Assuming usage of configured VLLM credentials if needed, or default
    vllm_port = config.provider.vllm_port or 8000
    
    # Build vLLM URL
    vllm_url = f"http://{vllm_host}:{vllm_port}/v1/chat/completions"
    
    print(f"\n{'='*60}")
    print(f"DDR_Bench Evaluation")
    print(f"Scenario: {args.scenario}")
    if args.split:
        print(f"Split: {args.split}")
    print(f"QA File: {qa_file}")
    print(f"Log Directory: {log_dir}")
    print(f"Output: {output_file}")
    print(f"Judge Provider: {provider}")
    print(f"Judge Model: {model}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Config File: {args.config or 'config.yaml'}")
    if args.test_mode:
        print("Mode: TEST (first entity only)")
    print(f"{'='*60}\n")
    
    # Create unified evaluator
    evaluator = UnifiedEvaluator(
        scenario=args.scenario,
        vllm_url=vllm_url,
        provider=provider,
        openai_model=model,
        azure_model=model,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    # Run evaluation
    evaluator.run_evaluation(
        qa_file=qa_file,
        logs_dir=log_dir,
        output_file=output_file,
        test_mode=args.test_mode,
        max_workers=args.max_workers
    )
    
    print(f"\nEvaluation complete. Results saved to: {output_file}")
    print(format_metric_summary(output_file))


if __name__ == "__main__":
    main()
