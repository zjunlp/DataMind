"""
DSGym Evaluation Framework

A comprehensive evaluation system for data science tasks with pluggable metrics
and reproducible evaluation workflows.

Example usage:
    from dsgym.eval import Evaluator, EvaluationConfig
    
    # Create evaluator with multiple metrics
    evaluator = Evaluator(
        metrics=["exact_match", "equivalence_by_llm", "semantic_similarity"]
    )
    
    # Evaluate agent on samples
    results = evaluator.evaluate(agent, samples, config=config)
"""

from .evaluator import Evaluator, create_evaluator_from_config
from .metric_registry import (
    MetricRegistry, 
    get_registry, 
    get_metric, 
    list_metrics,
    get_recommended_metrics
)
from .metrics.base import BaseMetric, MetricResult
from .utils import (
    EvaluationResult, 
    EvaluationConfig,
    save_evaluation_results,
    compute_aggregated_metrics,
    extract_sample_info
)
from .dataset_integration import (
    evaluate_agent_on_dataset,
    batch_evaluate_datasets,
    get_available_datasets,
    create_evaluation_suite,
    get_dataset_default_metrics
)

__all__ = [
    # Main evaluator
    "Evaluator",
    "create_evaluator_from_config",
    
    # Metric registry
    "MetricRegistry", 
    "get_registry",
    "get_metric",
    "list_metrics", 
    "get_recommended_metrics",
    
    # Base classes
    "BaseMetric",
    "MetricResult",
    
    # Data structures and utilities
    "EvaluationResult",
    "EvaluationConfig",
    "save_evaluation_results",
    "compute_aggregated_metrics", 
    "extract_sample_info",
    
    # Dataset integration
    "evaluate_agent_on_dataset",
    "batch_evaluate_datasets", 
    "get_available_datasets",
    "create_evaluation_suite",
    "get_dataset_default_metrics",
]

__version__ = "0.1.0"