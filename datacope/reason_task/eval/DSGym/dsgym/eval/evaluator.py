"""
Main evaluator orchestrator for DSGym evaluation framework.
"""

import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .utils import (
    EvaluationResult, 
    EvaluationConfig, 
    save_evaluation_results,
    compute_aggregated_metrics,
    extract_sample_info
)
from .metric_registry import get_registry, MetricRegistry
from .metrics.base import BaseMetric


class Evaluator:
    """
    Main evaluator class that orchestrates evaluation with protocols and metrics.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        metric_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        registry: Optional[MetricRegistry] = None,
        parallel_workers: Optional[int] = None,
        dataset: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of metric names to compute
            metric_configs: Configuration for each metric
            registry: Metric registry (uses default if None)
            parallel_workers: Number of parallel workers for evaluation
            dataset: Dataset object to get metrics from (optional)
            **kwargs: Additional configuration
        """
        # Setup metrics - use dataset metrics if available
        self.registry = registry or get_registry()
        
        if dataset is not None and hasattr(dataset, 'get_metrics'):
            # Use dataset-specific metrics and configs
            self.metric_names = metrics or dataset.get_metrics()
            dataset_configs = dataset.get_metric_configs() if hasattr(dataset, 'get_metric_configs') else {}
            # Merge user-provided configs with dataset configs (user configs take precedence)
            self.metric_configs = {**dataset_configs, **(metric_configs or {})}
        else:
            # Fallback to provided metrics or default
            self.metric_names = metrics or ["exact_match"]
            self.metric_configs = metric_configs or {}
        
        # Initialize metrics
        self.metrics = self.registry.get_metrics(self.metric_names, self.metric_configs)
        
        # Parallel evaluation settings
        self.parallel_workers = parallel_workers
        
        # Validation
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate evaluator setup."""
        if not self.metrics:
            raise ValueError("At least one metric must be specified")
        
        # Validate metric names
        available_metrics = self.registry.list_metrics()
        for name in self.metric_names:
            if name not in available_metrics:
                raise ValueError(f"Unknown metric: {name}. Available: {available_metrics}")
    
    def evaluate(
        self, 
        agent, 
        tasks: List[Dict[str, Any]],
        config: Optional[EvaluationConfig] = None,
        save_results: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate agent on tasks.
        
        Args:
            agent: Agent instance to evaluate
            tasks: List of task dictionaries
            config: Evaluation configuration
            save_results: Whether to save results to files
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary containing results and metrics
            
        Examples:
            # Single task
            task = create_custom_task("Analyze data trends", ["/path/to/data.csv"])
            result = evaluator.evaluate(agent, [task])
            
            # Multiple tasks from dataset
            tasks = dataset.load(limit=10)
            results = evaluator.evaluate(agent, tasks)
            
            # Specific tasks from dataset
            tasks = load_tasks_from_dataset("discoverybench", indices=[0, 5, 10])
            results = evaluator.evaluate(agent, tasks)
        """
        if not tasks:
            raise ValueError("No tasks provided for evaluation")
        
        start_time = time.time()
        
        # Use parallel evaluation if specified
        if self.parallel_workers and self.parallel_workers > 1:
            evaluation_results = self._evaluate_parallel(
                agent, tasks, show_progress
            )
        else:
            evaluation_results = self._evaluate_sequential(
                agent, tasks, show_progress
            )
        
        # Metrics are already computed per sample, no need for batch computation
        
        total_time = time.time() - start_time
        
        # Aggregate results
        aggregated_metrics = compute_aggregated_metrics(evaluation_results)
        aggregated_metrics["total_evaluation_time"] = total_time
        aggregated_metrics["metrics_used"] = self.metric_names
        
        # Prepare return data
        result_data = {
            "results": evaluation_results,
            "metrics": aggregated_metrics,
            "config": config.to_dict() if config else {},
            "total_time": total_time,
        }
        
        # Save results if requested
        if save_results and config:
            file_paths = save_evaluation_results(
                evaluation_results, 
                config, 
                config.output_dir, 
                config.run_name or "evaluation"
            )
            result_data["file_paths"] = file_paths
        
        return result_data
    
    def _evaluate_sequential(
        self, 
        agent, 
        tasks: List[Dict[str, Any]], 
        show_progress: bool
    ) -> List[EvaluationResult]:
        """Evaluate tasks sequentially."""
        results = []
        
        iterator = tqdm(tasks, desc="Evaluating") if show_progress else tasks
        
        for task in iterator:
            try:
                result = self._evaluate_single_sample(agent, task)
                results.append(result)
                
            except Exception as e:
                # Create error result
                task_info = extract_sample_info(task)
                
                error_result = EvaluationResult(
                    sample_id=f"error_{len(results)}",
                    dataset_name=task_info.get("dataset_name", "unknown"),
                    query=task_info.get("query", ""),
                    ground_truth=task_info.get("ground_truth"),
                    success=False,
                    error_info={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "error_category": "EVALUATION_ERROR"
                    }
                )
                results.append(error_result)
        
        return results
    
    def _evaluate_parallel(
        self, 
        agent, 
        tasks: List[Dict[str, Any]], 
        show_progress: bool
    ) -> List[EvaluationResult]:
        """Evaluate tasks in parallel."""
        results = [None] * len(tasks)
        
        def evaluate_single(index_task):
            index, task = index_task
            try:
                result = self._evaluate_single_sample(agent, task)
                return index, result
            except Exception as e:
                # Create error result
                task_info = extract_sample_info(task)
                
                error_result = EvaluationResult(
                    sample_id=f"error_{index}",
                    dataset_name=task_info.get("dataset_name", "unknown"),
                    query=task_info.get("query", ""),
                    ground_truth=task_info.get("ground_truth"),
                    success=False,
                    error_info={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "error_category": "EVALUATION_ERROR"
                    }
                )
                return index, error_result
        
        # Execute sequentially if max_workers=1 to avoid threading issues
        if self.parallel_workers == 1:
            # Sequential execution to avoid multiprocessing issues
            for i, task in enumerate(tqdm(tasks, desc="Evaluating") if show_progress else tasks):
                try:
                    index, result = evaluate_single((i, task))
                    results[index] = result
                except Exception as e:
                    print(f"Failed to process task {i}: {e}")
                    task_info = extract_sample_info(task)
                    error_result = EvaluationResult(
                        sample_id=f"error_{i}",
                        dataset_name=task_info.get("dataset_name", "unknown"),
                        query=task_info.get("query", ""),
                        ground_truth=task_info.get("ground_truth"),
                        success=False,
                        error_info={
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "error_category": "EVALUATION_ERROR"
                        }
                    )
                    results[i] = error_result
        else:
            # Execute in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                futures = [
                    executor.submit(evaluate_single, (i, task)) 
                    for i, task in enumerate(tasks)
                ]
                
                # Collect results with progress bar
                if show_progress:
                    futures = tqdm(
                        concurrent.futures.as_completed(futures), 
                        total=len(futures),
                        desc="Evaluating (parallel)"
                    )
                else:
                    futures = concurrent.futures.as_completed(futures)
                
                for future in futures:
                    index, result = future.result()
                    results[index] = result
        
        return results
    
    def _evaluate_single_sample(
        self, 
        agent, 
        sample: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate a single sample using agent (integrated protocol functionality).
        
        Args:
            agent: Agent instance to evaluate
            sample: Sample data to evaluate on
            
        Returns:
            EvaluationResult containing evaluation outcome
        """
        start_time = time.time()
        
        # Extract sample information
        sample_info = extract_sample_info(sample)
        query = sample_info["query"]
        ground_truth = sample_info["ground_truth"]
        dataset_name = sample_info["dataset_name"]
        sample_id = f"{dataset_name}_{sample_info['metadata_id']}_{sample_info['query_id']}"
        
        try:
            # Use agent's solve_task method (unified interface)
            if hasattr(agent, 'solve_task'):
                # Use standard agent evaluation method
                agent_result = agent.solve_task(sample)
                
                # Extract information from agent result
                execution_time = time.time() - start_time
                total_turns = agent_result.get("total_turns", agent_result.get("turns", 0))
                trajectory = agent_result.get("trajectory", [])
                done = agent_result.get("done", False)
                
                # Extract final prediction from agent result
                prediction = ""
                raw_response = ""
                
                if "solution" in agent_result:
                    prediction = agent_result["solution"]
                elif "raw_result" in agent_result and "prediction" in agent_result["raw_result"]:
                    prediction = agent_result["raw_result"]["prediction"]
                elif trajectory:
                    # Fallback: try to extract from trajectory
                    for step in reversed(trajectory):
                        if step.get("postprocessed_action"):
                            prediction = step["postprocessed_action"]
                            break
                
                # Extract raw response (full agent response)
                if "conversation" in agent_result and agent_result["conversation"]:
                    # Get the last assistant response from conversation
                    for msg in reversed(agent_result["conversation"]):
                        if msg.get("role") == "assistant" and msg.get("content"):
                            raw_response = msg["content"]
                            break
                elif trajectory:
                    # Fallback: get the last response from trajectory
                    for step in reversed(trajectory):
                        if step.get("response"):
                            raw_response = step["response"]
                            break
                
                # If still no raw_response, use prediction as fallback
                if not raw_response:
                    raw_response = prediction
                
                # Check for errors and success
                if "error" in agent_result and agent_result["error"]:
                    error_info = agent_result["error"]
                    success = False
                else:
                    error_info = None
                    # Always set success=True unless there's an explicit error
                    # We want to evaluate metrics even for empty predictions
                    success = agent_result.get("success", True)
                
                result = EvaluationResult(
                    sample_id=sample_id,
                    dataset_name=dataset_name,
                    query=query,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    raw_response=raw_response,
                    execution_time=execution_time,
                    total_turns=total_turns,
                    success=success,
                    error_info=error_info,
                    trajectory=trajectory,
                    extra_info={
                        **sample_info,
                        "done": done
                    }
                )
                
            else:
                raise ValueError(f"Agent {type(agent)} does not implement solve_task() method. Please inherit from BaseAgent.")
            
            # Compute metrics for all results regardless of success status
            # This ensures empty predictions are also evaluated (likely getting score 0)
            for metric in self.metrics:
                metric_result = metric._safe_evaluate(
                    result.prediction,
                    result.ground_truth,
                    query=result.query,
                    extra_info=sample.get("extra_info", {}),
                )
                result.metrics[metric.name] = metric_result.to_dict()
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create error result
            result = EvaluationResult(
                sample_id=sample_id,
                dataset_name=dataset_name,
                query=query,
                ground_truth=ground_truth,
                execution_time=execution_time,
                total_turns=0,
                success=False,
                error_info={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_category": "AGENT_ERROR"
                },
                extra_info=sample_info
            )
            
            return result
    
    def _compute_metrics_batch(self, results: List[EvaluationResult]):
        """Compute metrics for all results."""
        for metric in self.metrics:
            if metric.supports_batch_evaluation:
                # Batch evaluation
                predictions = [r.prediction for r in results]
                ground_truths = [r.ground_truth for r in results]
                queries = [r.query for r in results]
                
                metric_results = metric.evaluate_batch(
                    predictions, ground_truths, queries
                )
                
                # Assign results back
                for result, metric_result in zip(results, metric_results):
                    # Add metrics for all results regardless of success status
                    # This ensures empty predictions are also evaluated (likely getting score 0)
                    result.metrics[metric.name] = metric_result.to_dict()
            else:
                # Individual evaluation
                for result in results:
                    # Compute metrics for all results regardless of success status
                    # This ensures empty predictions are also evaluated (likely getting score 0)
                    metric_result = metric._safe_evaluate(
                        result.prediction, 
                        result.ground_truth,
                        query=result.query
                    )
                    result.metrics[metric.name] = metric_result.to_dict()
    
    def evaluate_single(
        self, 
        agent, 
        sample: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate a single sample.
        
        Args:
            agent: Agent instance
            sample: Sample to evaluate
            
        Returns:
            EvaluationResult for the sample
        """
        return self._evaluate_single_sample(agent, sample)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this evaluator.
        
        Returns:
            Dictionary with evaluator information
        """
        return {
            "metrics": [
                {
                    "name": metric.name,
                    "requires_ground_truth": metric.requires_ground_truth,
                    "supports_batch": metric.supports_batch_evaluation,
                    "config": getattr(metric, "config", {}),
                }
                for metric in self.metrics
            ],
            "parallel_workers": self.parallel_workers,
            "total_metrics": len(self.metrics),
        }


def create_evaluator_from_config(config: EvaluationConfig) -> Evaluator:
    """
    Create evaluator from configuration.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Configured Evaluator instance
    """
    return Evaluator(
        metrics=config.metrics,
        parallel_workers=config.max_workers,
        **config.extra_config
    )
