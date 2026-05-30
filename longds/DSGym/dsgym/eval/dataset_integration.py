"""
Dataset integration utilities for DSGym evaluation framework.

Provides seamless integration between the datasets module and evaluation framework.
"""

from typing import List, Dict, Any, Optional, Union
from dsgym.datasets import DatasetRegistry, BaseDataset
from .utils import EvaluationConfig, EvaluationResult
from .evaluator import Evaluator


def evaluate_agent_on_dataset(
    agent,
    dataset_name: str,
    evaluator: Optional[Evaluator] = None,
    dataset_config: Optional[Dict[str, Any]] = None,
    eval_config: Optional[EvaluationConfig] = None,
    **kwargs
) -> EvaluationResult:
    """
    Evaluate an agent on a specific dataset.
    
    Args:
        agent: Agent instance to evaluate
        dataset_name: Name of dataset to load
        evaluator: Evaluator instance (creates default if None)
        dataset_config: Configuration for dataset loading
        eval_config: Evaluation configuration
        **kwargs: Additional arguments passed to dataset.load()
        
    Returns:
        Evaluation results
    """
    # Load dataset
    dataset_config = dataset_config or {}
    dataset = DatasetRegistry.load(dataset_name, **dataset_config)
    
    # Load samples
    samples = dataset.load(**kwargs)
    
    # Create default evaluator if not provided
    if evaluator is None:
        # Get recommended metrics for this dataset
        metadata = dataset.get_metadata()
        dataset_source = metadata.get('source', dataset_name.lower())
        
        # Use dataset-specific default metrics
        default_metrics = get_dataset_default_metrics(dataset_source)
        evaluator = Evaluator(metrics=default_metrics)
    
    # Create default eval config if not provided
    if eval_config is None:
        eval_config = EvaluationConfig(
            dataset_name=dataset_name,
            dataset_metadata=dataset.get_metadata()
        )
    
    # Run evaluation
    return evaluator.evaluate(agent, samples, config=eval_config)


def get_dataset_default_metrics(dataset_source: str) -> List[str]:
    """
    Get default metrics for a dataset based on its source.
    
    Args:
        dataset_source: Dataset source identifier
        
    Returns:
        List of recommended metric names
    """
    dataset_metrics = {
        'discoverybench': ['exact_match', 'semantic_similarity'],
        'dabstep': ['exact_match', 'equivalence_by_llm'],
        'qrdata': ['exact_match', 'semantic_similarity', 'equivalence_by_llm'],
        'kaggle': ['code_execution', 'submission_format'],
        'daeval': ['exact_match', 'equivalence_by_llm'],
        'daeval_original': ['exact_match', 'equivalence_by_llm'],
    }
    
    return dataset_metrics.get(dataset_source, ['exact_match'])


def batch_evaluate_datasets(
    agent,
    dataset_names: List[str],
    evaluator: Optional[Evaluator] = None,
    dataset_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, EvaluationResult]:
    """
    Evaluate an agent on multiple datasets.
    
    Args:
        agent: Agent instance to evaluate
        dataset_names: List of dataset names to evaluate on
        evaluator: Evaluator instance (uses default if None)
        dataset_configs: Per-dataset configuration
        **kwargs: Additional arguments passed to all datasets
        
    Returns:
        Dictionary mapping dataset names to evaluation results
    """
    results = {}
    dataset_configs = dataset_configs or {}
    
    for dataset_name in dataset_names:
        dataset_config = dataset_configs.get(dataset_name, {})
        
        try:
            result = evaluate_agent_on_dataset(
                agent=agent,
                dataset_name=dataset_name,
                evaluator=evaluator,
                dataset_config=dataset_config,
                **kwargs
            )
            results[dataset_name] = result
            
        except Exception as e:
            print(f"Error evaluating on {dataset_name}: {e}")
            results[dataset_name] = None
    
    return results


def get_available_datasets() -> List[str]:
    """
    Get list of all available datasets.
    
    Returns:
        List of dataset names
    """
    return DatasetRegistry.list_datasets()


def create_evaluation_suite(
    datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> Evaluator:
    """
    Create an evaluation suite with specified datasets and metrics.
    
    Args:
        datasets: List of dataset names (uses all if None)
        metrics: List of metric names (uses defaults if None)
        
    Returns:
        Configured evaluator instance
    """
    if datasets is None:
        datasets = get_available_datasets()
    
    if metrics is None:
        # Combine all default metrics from all datasets
        all_metrics = set()
        for dataset_name in datasets:
            dataset_metrics = get_dataset_default_metrics(dataset_name)
            all_metrics.update(dataset_metrics)
        metrics = list(all_metrics)
    
    return Evaluator(metrics=metrics)