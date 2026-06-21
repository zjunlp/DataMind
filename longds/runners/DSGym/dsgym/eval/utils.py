"""
Evaluation utilities and data structures.
"""

import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    # Basic identification
    sample_id: str
    dataset_name: str
    query: str
    ground_truth: Optional[str] = None  # Allow None for cases without ground truth
    
    # Predictions and metrics
    prediction: str = ""
    raw_response: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Execution details
    execution_time: float = 0.0
    total_turns: int = 0
    success: bool = True
    error_info: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_ground_truth(self) -> bool:
        """Check if this result has ground truth for evaluation."""
        return self.ground_truth is not None and self.ground_truth.strip() != ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "raw_response": self.raw_response,
            "metrics": self.metrics,
            "execution_time": self.execution_time,
            "total_turns": self.total_turns,
            "success": self.success,
            "extra_info": self.extra_info,
        }
        
        if self.error_info:
            result["error_info"] = self.error_info
            
        if self.trajectory:
            result["trajectory"] = self.trajectory
            
        return result
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for summary (without raw_response and trajectory)."""
        result = {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "metrics": self.metrics,
            "execution_time": self.execution_time,
            "total_turns": self.total_turns,
            "success": self.success,
            "extra_info": self.extra_info,
        }
        
        if self.error_info:
            result["error_info"] = self.error_info
        
        # Note: trajectory and raw_response are excluded from summary
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create from dictionary."""
        return cls(
            sample_id=data["sample_id"],
            dataset_name=data["dataset_name"], 
            query=data["query"],
            ground_truth=data.get("ground_truth"),  # Allow None
            prediction=data.get("prediction", ""),
            raw_response=data.get("raw_response", ""),
            metrics=data.get("metrics", {}),
            execution_time=data.get("execution_time", 0.0),
            total_turns=data.get("total_turns", 0),
            success=data.get("success", True),
            error_info=data.get("error_info"),
            trajectory=data.get("trajectory", []),
            extra_info=data.get("extra_info", {}),
        )


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    
    # Model configuration
    model_name: str
    backend_type: str = "litellm"
    
    # Dataset configuration  
    dataset_name: str = "discovery"
    dataset_split: str = "test"
    limit: Optional[int] = None
    start_index: int = 0
    
    # Evaluation protocol
    protocol: str = "multi_turn"
    max_turns: int = 20
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: ["exact_match"])
    
    # Generation parameters
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1524
    
    # Output configuration
    output_dir: str = "./evaluation_results"
    run_name: Optional[str] = None
    save_trajectories: bool = True
    
    # Performance
    max_workers: Optional[int] = None
    batch_size: int = 1
    
    # Additional configuration
    seed: int = 42
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "backend_type": self.backend_type,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "limit": self.limit,
            "start_index": self.start_index,
            "protocol": self.protocol,
            "max_turns": self.max_turns,
            "metrics": self.metrics,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "save_trajectories": self.save_trajectories,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "extra_config": self.extra_config,
        }


def extract_sample_info(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract standardized information from a sample.
    Handles cases where ground truth might be missing or empty.
    
    Args:
        sample: Sample data dictionary
        
    Returns:
        Dictionary with standardized fields
    """
    query = ""
    ground_truth = None  # Start with None
    dataset_name = ""
    metadata_id = ""
    query_id = ""
    
    # Extract ground truth from various possible locations
    if "reward_spec" in sample and "ground_truth" in sample["reward_spec"]:
        gt = sample["reward_spec"]["ground_truth"]
        ground_truth = gt if gt and str(gt).strip() else None
    else:
        gt = sample.get("ground_truth", sample.get("answer", ""))
        ground_truth = gt if gt and str(gt).strip() else None
    
    if "extra_info" in sample:
        extra_info = sample["extra_info"]
        query = extra_info.get("question", "")
        dataset_name = extra_info.get("source", extra_info.get("dataset", extra_info.get("dataset_name", "")))
        metadata_id = str(extra_info.get("metadata_id", ""))
        query_id = str(extra_info.get("query_id", extra_info.get("id", "")))
    
    if not query and "prompt" in sample and isinstance(sample["prompt"], list):
        for msg in sample["prompt"]:
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = msg.get("content", "")
                break
    
    if not query:
        query = sample.get("query", "")
    if not dataset_name:
        dataset_name = sample.get("dataset", sample.get("dataset_name", ""))
    if not metadata_id:
        metadata_id = str(sample.get("metadata_id", ""))
    if not query_id:
        query_id = str(sample.get("query_id", sample.get("id", "")))
        
    return {
        "query": query, 
        "ground_truth": ground_truth,  # Can be None
        "dataset_name": dataset_name, 
        "metadata_id": metadata_id, 
        "query_id": query_id
    }


def compute_aggregated_metrics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Compute aggregated metrics across all results.
    Handles cases where some results don't have ground truth.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not results:
        return {}
    
    # Basic statistics
    total_samples = len(results)
    successful_samples = sum(1 for r in results if r.success)
    error_samples = total_samples - successful_samples
    samples_with_ground_truth = sum(1 for r in results if r.has_ground_truth)
    
    metrics = {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "error_samples": error_samples,
        "samples_with_ground_truth": samples_with_ground_truth,
        "success_rate": successful_samples / total_samples if total_samples > 0 else 0.0,
        "average_execution_time": sum(r.execution_time for r in results) / total_samples,
        "average_turns": sum(r.total_turns for r in results) / total_samples,
    }
    
    # Aggregate metric scores (only for successful results)
    successful_results = [r for r in results if r.success]
    if successful_results:
        # Find all metric names
        all_metric_names = set()
        for result in successful_results:
            all_metric_names.update(result.metrics.keys())
        
        # Compute averages for each metric
        for metric_name in all_metric_names:
            values = []
            evaluable_samples = 0  # Samples that could be evaluated for this metric
            
            for result in successful_results:
                if metric_name in result.metrics:
                    metric_value = result.metrics[metric_name]
                    if isinstance(metric_value, (int, float)):
                        values.append(metric_value)
                        evaluable_samples += 1
                    elif isinstance(metric_value, dict):
                        if "score" in metric_value:
                            values.append(metric_value["score"])
                            evaluable_samples += 1
                        # Also track evaluation status
                        if "evaluated" in metric_value:
                            evaluable_samples += 1
            
            if values and all(v is not None for v in values):
                # Use total samples as denominator (including error samples get 0 score)
                metrics[f"{metric_name}_mean"] = sum(values) / total_samples
                metrics[f"{metric_name}_min"] = min(values)
                metrics[f"{metric_name}_max"] = max(values)
                metrics[f"{metric_name}_count"] = len(values)
                metrics[f"{metric_name}_evaluable_count"] = evaluable_samples
    
    # Error analysis
    if error_samples > 0:
        error_results = [r for r in results if not r.success]
        error_categories = {}
        for result in error_results:
            if result.error_info and "error_category" in result.error_info:
                category = result.error_info["error_category"]
                error_categories[category] = error_categories.get(category, 0) + 1
        
        metrics["error_breakdown"] = error_categories
    
    return metrics


def save_evaluation_results(
    results: List[EvaluationResult],
    config: EvaluationConfig,
    output_dir: Union[str, Path],
    run_name: str,
    additional_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Save evaluation results to files.
    
    Args:
        results: List of evaluation results
        config: Evaluation configuration
        output_dir: Output directory
        run_name: Name for this evaluation run
        additional_metrics: Additional metrics to include
        
    Returns:
        Dictionary mapping file types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Save detailed results (with raw_response and trajectory)
    results_file = output_dir / f"{run_name}_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
    file_paths["results"] = str(results_file)
    
    # Save summary results (without raw_response and trajectory for readability)
    summary_file = output_dir / f"{run_name}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_summary_dict() for r in results], f, indent=2, ensure_ascii=False)
    file_paths["summary"] = str(summary_file)
    
    # Save configuration
    config_file = output_dir / f"{run_name}_config.json"
    config_dict = config.to_dict()
    config_dict["timestamp"] = datetime.now().isoformat()
    config_dict["total_samples"] = len(results)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    file_paths["config"] = str(config_file)
    
    # Compute and save aggregated metrics
    aggregated_metrics = compute_aggregated_metrics(results)
    if additional_metrics:
        aggregated_metrics.update(additional_metrics)
    
    metrics_file = output_dir / f"{run_name}_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_metrics, f, indent=2, ensure_ascii=False)
    file_paths["metrics"] = str(metrics_file)
    
    # Save summary CSV for easy analysis
    import pandas as pd
    summary_data = []
    for result in results:
        summary_row = {
            "sample_id": result.sample_id,
            "dataset_name": result.dataset_name,
            "query": result.query[:100] + "..." if len(result.query) > 100 else result.query,
            "ground_truth": result.ground_truth if result.has_ground_truth else "N/A",
            "prediction": result.prediction,
            "success": result.success,
            "execution_time": result.execution_time,
            "total_turns": result.total_turns,
            "has_ground_truth": result.has_ground_truth,
        }
        
        # Add metric values as separate columns
        for metric_name, metric_value in result.metrics.items():
            if isinstance(metric_value, (int, float, bool)):
                summary_row[f"metric_{metric_name}"] = metric_value
            elif isinstance(metric_value, dict) and "score" in metric_value:
                summary_row[f"metric_{metric_name}"] = metric_value["score"]
        
        summary_data.append(summary_row)
    
    summary_file = output_dir / f"{run_name}_summary.csv"
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    file_paths["summary"] = str(summary_file)
    
    return file_paths


def format_metric_display(metric_value: Any) -> str:
    """
    Format metric value for display.
    
    Args:
        metric_value: Metric value to format
        
    Returns:
        Formatted string representation
    """
    if isinstance(metric_value, float):
        return f"{metric_value:.4f}"
    elif isinstance(metric_value, dict):
        if "score" in metric_value:
            return f"{metric_value['score']:.4f}"
        else:
            return str(metric_value)
    else:
        return str(metric_value)