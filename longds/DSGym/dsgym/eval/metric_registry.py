"""
Metric registry for automatic discovery and management of evaluation metrics.
"""

from typing import Dict, List, Type, Any, Optional
from .metrics.base import BaseMetric
from .metrics.exact_match import ExactMatchMetric, FuzzyExactMatchMetric, ListMatchMetric
from .metrics.code_execution import CodeExecutionMetric, CodeCorrectnessMetric
from .metrics.equivalence_by_llm import EquivalenceByLLMMetric, FastEquivalenceByLLMMetric
from .metrics.semantic_similarity import SemanticSimilarityMetric, BinarySemanticSimilarityMetric
from .metrics.domain_specific import (
    DABStepMetric,
    LLMScoreMetric,
    HMSScoreMetric,
)
from .metrics.dspredict import KaggleSubmissionMetric as DSPredictSubmissionMetric
from .metrics.mlebench import MLEBenchSubmissionMetric


class MetricRegistry:
    """
    Registry for evaluation metrics with automatic discovery and configuration.
    """
    
    def __init__(self):
        """Initialize metric registry with default metrics."""
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register all default metrics."""
        # Basic metrics
        self.register("exact_match", ExactMatchMetric)
        self.register("fuzzy_exact_match", FuzzyExactMatchMetric)
        self.register("list_match", ListMatchMetric)
        
        # Code evaluation
        self.register("code_execution", CodeExecutionMetric)
        self.register("code_correctness", CodeCorrectnessMetric)
        
        # LLM-based evaluation
        self.register("equivalence_by_llm", EquivalenceByLLMMetric)
        self.register("fast_equivalence_by_llm", FastEquivalenceByLLMMetric)
        
        # Semantic similarity
        self.register("semantic_similarity", SemanticSimilarityMetric)
        self.register("binary_semantic_similarity", BinarySemanticSimilarityMetric)
        
        # Domain-specific metrics
        self.register("dabstep", DABStepMetric)
        # DSPredict submission + leaderboard scoring
        self.register("dspredict_submission", DSPredictSubmissionMetric)
        # MLE-Bench offline grading metric
        self.register("mlebench_submission", MLEBenchSubmissionMetric)
        self.register("llm_score", LLMScoreMetric)
        self.register("hms_score", HMSScoreMetric)
    
    def register(self, name: str, metric_class: Type[BaseMetric]):
        """
        Register a metric class.
        
        Args:
            name: Metric name for lookup
            metric_class: Metric class to register
        """
        if not issubclass(metric_class, BaseMetric):
            raise ValueError(f"Metric class {metric_class} must inherit from BaseMetric")
        
        self._metrics[name] = metric_class
    
    def unregister(self, name: str):
        """
        Unregister a metric.
        
        Args:
            name: Metric name to remove
        """
        if name in self._metrics:
            del self._metrics[name]
    
    def get_metric(self, name: str, **kwargs) -> BaseMetric:
        """
        Get metric instance by name.
        
        Args:
            name: Metric name
            **kwargs: Metric configuration parameters
            
        Returns:
            Configured metric instance
        """
        if name not in self._metrics:
            raise ValueError(f"Unknown metric: {name}. Available metrics: {self.list_metrics()}")
        
        metric_class = self._metrics[name]
        return metric_class(**kwargs)
    
    def get_metrics(self, names: List[str], configs: Optional[Dict[str, Dict[str, Any]]] = None) -> List[BaseMetric]:
        """
        Get multiple metric instances.
        
        Args:
            names: List of metric names
            configs: Optional dict mapping metric names to their configurations
            
        Returns:
            List of configured metric instances
        """
        metrics = []
        configs = configs or {}
        
        for name in names:
            metric_config = configs.get(name, {})
            metric = self.get_metric(name, **metric_config)
            metrics.append(metric)
        
        return metrics
    
    def list_metrics(self) -> List[str]:
        """
        Get list of available metric names.
        
        Returns:
            List of registered metric names
        """
        return list(self._metrics.keys())
    
    def get_metric_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with metric information
        """
        if name not in self._metrics:
            raise ValueError(f"Unknown metric: {name}")
        
        metric_class = self._metrics[name]
        
        # Create temporary instance to get properties
        try:
            temp_metric = metric_class()
            requires_ground_truth = temp_metric.requires_ground_truth
            supports_batch = temp_metric.supports_batch_evaluation
        except Exception:
            # If instantiation fails, use defaults
            requires_ground_truth = True
            supports_batch = False
        
        return {
            "name": name,
            "class": metric_class.__name__,
            "module": metric_class.__module__,
            "requires_ground_truth": requires_ground_truth,
            "supports_batch_evaluation": supports_batch,
            "docstring": metric_class.__doc__,
        }
    
    def get_all_metric_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered metrics.
        
        Returns:
            Dictionary mapping metric names to their information
        """
        return {name: self.get_metric_info(name) for name in self.list_metrics()}
    
    def filter_metrics(
        self, 
        requires_ground_truth: Optional[bool] = None,
        supports_batch: Optional[bool] = None,
        dataset_specific: Optional[str] = None
    ) -> List[str]:
        """
        Filter metrics by criteria.
        
        Args:
            requires_ground_truth: Filter by ground truth requirement
            supports_batch: Filter by batch evaluation support
            dataset_specific: Filter by dataset name (e.g., "discovery", "dabstep")
            
        Returns:
            List of metric names matching criteria
        """
        matching_metrics = []
        
        for name in self.list_metrics():
            try:
                info = self.get_metric_info(name)
                
                # Check ground truth requirement
                if requires_ground_truth is not None:
                    if info["requires_ground_truth"] != requires_ground_truth:
                        continue
                
                # Check batch support
                if supports_batch is not None:
                    if info["supports_batch_evaluation"] != supports_batch:
                        continue
                
                # Check dataset specificity
                if dataset_specific is not None:
                    if dataset_specific.lower() not in name.lower():
                        continue
                
                matching_metrics.append(name)
                
            except Exception:
                # Skip metrics that can't be introspected
                continue
        
        return matching_metrics
    
    def get_recommended_metrics(self, dataset_name: str) -> List[str]:
        """
        Get recommended metrics for a dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            List of recommended metric names
        """
        dataset_name = dataset_name.lower()
        
        # Base metrics recommended for all datasets
        recommended = ["exact_match"]
        
        # Add dataset-specific metrics
        if "discovery" in dataset_name:
            recommended.extend(["discovery_bench", "equivalence_by_llm"])
        elif "dabstep" in dataset_name:
            recommended.extend(["dabstep", "equivalence_by_llm"])
        elif "qrdata" in dataset_name:
            recommended.extend(["qrdata", "semantic_similarity"])
        elif "dspredict" in dataset_name:
            recommended.extend(["dspredict", "code_execution"])
        else:
            # General recommendations
            recommended.extend(["semantic_similarity", "equivalence_by_llm"])
        
        # Filter to only include registered metrics
        available_metrics = set(self.list_metrics())
        recommended = [m for m in recommended if m in available_metrics]
        
        return recommended


# Global registry instance
_default_registry = MetricRegistry()


def get_registry() -> MetricRegistry:
    """Get the default metric registry."""
    return _default_registry


def register_metric(name: str, metric_class: Type[BaseMetric]):
    """
    Register a metric in the default registry.
    
    Args:
        name: Metric name
        metric_class: Metric class
    """
    _default_registry.register(name, metric_class)


def get_metric(name: str, **kwargs) -> BaseMetric:
    """
    Get metric from default registry.
    
    Args:
        name: Metric name
        **kwargs: Metric configuration
        
    Returns:
        Configured metric instance
    """
    return _default_registry.get_metric(name, **kwargs)


def list_metrics() -> List[str]:
    """Get list of available metrics from default registry."""
    return _default_registry.list_metrics()


def get_recommended_metrics(dataset_name: str) -> List[str]:
    """
    Get recommended metrics for a dataset from default registry.
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        List of recommended metric names
    """
    return _default_registry.get_recommended_metrics(dataset_name)
