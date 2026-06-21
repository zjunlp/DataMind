"""
Base metric interface and utilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import time


@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    
    metric_name: str
    score: Optional[float] = None  # Main score (None if cannot be computed)
    details: Dict[str, Any] = None  # Additional details
    error: Optional[str] = None  # Error message if evaluation failed
    evaluation_time: float = 0.0  # Time taken to compute metric
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    @property
    def success(self) -> bool:
        """Check if metric evaluation was successful."""
        return self.error is None and self.score is not None
    
    @property 
    def skipped(self) -> bool:
        """Check if metric evaluation was skipped (e.g., no ground truth)."""
        return self.error is None and self.score is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details,
            "evaluation_time": self.evaluation_time,
            "success": self.success,
            "skipped": self.skipped,
        }
        
        if self.error:
            result["error"] = self.error
            
        return result


class BaseMetric(ABC):
    """
    Base class for all evaluation metrics.
    
    Metrics should be stateless and thread-safe to support parallel evaluation.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize metric with configuration parameters.
        
        Args:
            **kwargs: Metric-specific configuration parameters
        """
        self.config = kwargs
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass
    
    @property
    def requires_ground_truth(self) -> bool:
        """Return True if this metric requires ground truth for evaluation."""
        return True
    
    @property 
    def supports_batch_evaluation(self) -> bool:
        """Return True if this metric supports batch evaluation for efficiency."""
        return False
        
    def can_evaluate(self, prediction: str, ground_truth: Optional[str] = None, **kwargs) -> bool:
        """
        Check if this metric can evaluate the given prediction.
        
        Args:
            prediction: Model prediction to evaluate
            ground_truth: Ground truth answer (if available)
            **kwargs: Additional context information
            
        Returns:
            True if metric can be evaluated, False otherwise
        """
        if self.requires_ground_truth:
            return ground_truth is not None and ground_truth.strip() != ""
        return prediction is not None
    
    @abstractmethod
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate prediction against ground truth.
        
        Args:
            prediction: Model prediction to evaluate
            ground_truth: Ground truth answer (if available)
            query: Original query/question (if available)
            **kwargs: Additional context information
            
        Returns:
            MetricResult containing score and details
        """
        pass
    
    def evaluate_batch(
        self,
        predictions: list,
        ground_truths: list,
        queries: Optional[list] = None,
        **kwargs
    ) -> list:
        """
        Evaluate multiple predictions in batch (default implementation uses individual evaluation).
        
        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth answers
            queries: List of original queries (optional)
            **kwargs: Additional context information
            
        Returns:
            List of MetricResult objects
        """
        if not self.supports_batch_evaluation:
            # Default implementation: evaluate individually
            results = []
            for i, prediction in enumerate(predictions):
                ground_truth = ground_truths[i] if i < len(ground_truths) else None
                query = queries[i] if queries and i < len(queries) else None
                result = self.evaluate(prediction, ground_truth, query, **kwargs)
                results.append(result)
            return results
        else:
            # Subclasses should override this for efficient batch evaluation
            raise NotImplementedError("Batch evaluation not implemented")
    
    def _safe_evaluate(self, prediction: str, ground_truth: Optional[str] = None, **kwargs) -> MetricResult:
        """
        Safely evaluate with error handling and timing.
        
        Args:
            prediction: Model prediction to evaluate
            ground_truth: Ground truth answer (if available)
            **kwargs: Additional context information
            
        Returns:
            MetricResult with error handling
        """
        start_time = time.time()
        
        try:
            # Check if evaluation is possible
            if not self.can_evaluate(prediction, ground_truth, **kwargs):
                return MetricResult(
                    metric_name=self.name,
                    score=None,
                    details={"reason": "Cannot evaluate - missing required inputs"},
                    evaluation_time=time.time() - start_time
                )
            
            # Perform evaluation
            # Pass through useful context if available on the caller result
            # e.g., extra_info populated by the protocol/dataset
            result = self.evaluate(
                prediction,
                ground_truth,
                query=kwargs.get("query"),
                extra_info=kwargs.get("extra_info", {}),
            )
            result.evaluation_time = time.time() - start_time
            return result
            
        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                score=None,
                error=str(e),
                evaluation_time=time.time() - start_time
            )


class ExactMatchMixin:
    """Mixin for exact match functionality."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        return text.strip().lower()
    
    @staticmethod
    def exact_match_score(prediction: str, ground_truth: str) -> float:
        """Compute exact match score."""
        norm_pred = ExactMatchMixin.normalize_text(prediction)
        norm_truth = ExactMatchMixin.normalize_text(ground_truth)
        return 1.0 if norm_pred == norm_truth else 0.0


class NumericMixin:
    """Mixin for numeric comparison functionality."""
    
    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """Extract first number from text."""
        import re
        if not text:
            return None
        
        # Remove common prefixes/suffixes
        text = text.strip()
        
        # Try to find number patterns
        patterns = [
            r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',  # Scientific notation
            r'[-+]?\d+\.?\d*',  # Regular numbers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def relative_error(prediction: float, ground_truth: float) -> float:
        """Compute relative error between two numbers."""
        if ground_truth == 0:
            return abs(prediction)
        return abs(prediction - ground_truth) / abs(ground_truth)
    
    @staticmethod
    def numeric_match_score(
        prediction: str, 
        ground_truth: str, 
        tolerance: float = 0.01
    ) -> Optional[float]:
        """
        Compute numeric match score with tolerance.
        
        Args:
            prediction: Predicted text
            ground_truth: Ground truth text
            tolerance: Relative error tolerance
            
        Returns:
            1.0 if match within tolerance, 0.0 if not, None if not numeric
        """
        pred_num = NumericMixin.extract_number(prediction)
        truth_num = NumericMixin.extract_number(ground_truth)
        
        if pred_num is None or truth_num is None:
            return None
        
        rel_error = NumericMixin.relative_error(pred_num, truth_num)
        return 1.0 if rel_error < tolerance else 0.0
