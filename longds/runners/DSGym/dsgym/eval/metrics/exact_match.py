"""
Exact match metric implementation.
"""

import re
import ast
from typing import Optional, List
from .base import BaseMetric, MetricResult, ExactMatchMixin, NumericMixin


class ExactMatchMetric(BaseMetric, ExactMatchMixin, NumericMixin):
    """
    Exact match metric with support for numeric tolerance.
    """
    
    def __init__(self, numeric_tolerance: float = 0.03, **kwargs):
        """
        Initialize exact match metric.
        
        Args:
            numeric_tolerance: Relative error tolerance for numeric values
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.numeric_tolerance = numeric_tolerance
    
    @property
    def name(self) -> str:
        return "exact_match"
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate exact match between prediction and ground truth.
        
        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            query: Original query (unused)
            **kwargs: Additional context
            
        Returns:
            MetricResult with exact match score
        """
        if ground_truth is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No ground truth available"}
            )
        
        # Try exact string match first
        exact_score = self.exact_match_score(prediction, ground_truth)
        
        details = {
            "prediction_normalized": self.normalize_text(prediction),
            "ground_truth_normalized": self.normalize_text(ground_truth),
            "exact_string_match": exact_score == 1.0,
        }
        
        # If exact match fails, try other matching strategies
        if exact_score == 0.0:
            # First try list comparison if both can be parsed as lists
            list_score = self._try_list_match(prediction, ground_truth)
            if list_score is not None:
                details["list_match_attempted"] = True
                details["list_match_score"] = list_score
                return MetricResult(
                    metric_name=self.name,
                    score=list_score,
                    details=details
                )
            else:
                details["list_match_attempted"] = False
            
            # Then try numeric match if both can be converted to float
            if self._can_convert_to_float(prediction) and self._can_convert_to_float(ground_truth):
                numeric_score = self.numeric_match_score(
                    prediction, ground_truth, self.numeric_tolerance
                )
                
                if numeric_score is not None:
                    details["numeric_match_attempted"] = True
                    details["numeric_match_score"] = numeric_score
                    details["numeric_tolerance"] = self.numeric_tolerance
                    
                    # Extract numbers for debugging
                    pred_num = self.extract_number(prediction)
                    truth_num = self.extract_number(ground_truth)
                    if pred_num is not None and truth_num is not None:
                        details["prediction_number"] = pred_num
                        details["ground_truth_number"] = truth_num
                        details["relative_error"] = self.relative_error(pred_num, truth_num)
                    
                    return MetricResult(
                        metric_name=self.name,
                        score=numeric_score,
                        details=details
                    )
                else:
                    details["numeric_match_attempted"] = False
            else:
                details["numeric_match_attempted"] = False
        
        return MetricResult(
            metric_name=self.name,
            score=exact_score,
            details=details
        )
    
    def _can_convert_to_float(self, text: str) -> bool:
        """Check if text can be directly converted to float."""
        try:
            float(text.strip())
            return True
        except ValueError:
            return False
    
    def _try_list_match(self, prediction: str, ground_truth: str) -> Optional[float]:
        """Try to parse both as lists and compare as sets."""
        try:
            pred_list = ast.literal_eval(prediction.strip())
            truth_list = ast.literal_eval(ground_truth.strip())
            
            if isinstance(pred_list, list) and isinstance(truth_list, list):
                # Convert to sets for order-independent comparison
                pred_set = set(str(item).lower() for item in pred_list)
                truth_set = set(str(item).lower() for item in truth_list)
                return 1.0 if pred_set == truth_set else 0.0
        except (ValueError, SyntaxError):
            pass
        
        return None


class FuzzyExactMatchMetric(BaseMetric, ExactMatchMixin):
    """
    Fuzzy exact match metric with character-level similarity.
    """
    
    def __init__(self, similarity_threshold: float = 0.8, **kwargs):
        """
        Initialize fuzzy exact match metric.
        
        Args:
            similarity_threshold: Minimum similarity score to consider a match
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
    
    @property
    def name(self) -> str:
        return "fuzzy_exact_match"
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute character-level similarity between two texts.
        Uses Levenshtein distance normalized by max length.
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
            
        # Simple Levenshtein distance implementation
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
        return similarity
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate fuzzy exact match between prediction and ground truth.
        
        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            query: Original query (unused)
            **kwargs: Additional context
            
        Returns:
            MetricResult with fuzzy match score
        """
        if ground_truth is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No ground truth available"}
            )
        
        # Normalize texts
        norm_pred = self.normalize_text(prediction)
        norm_truth = self.normalize_text(ground_truth)
        
        # Compute similarity
        similarity = self._compute_similarity(norm_pred, norm_truth)
        
        # Determine if it's a match based on threshold
        is_match = similarity >= self.similarity_threshold
        score = 1.0 if is_match else 0.0
        
        details = {
            "prediction_normalized": norm_pred,
            "ground_truth_normalized": norm_truth,
            "character_similarity": similarity,
            "similarity_threshold": self.similarity_threshold,
            "is_fuzzy_match": is_match,
        }
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details=details
        )


class ListMatchMetric(BaseMetric, NumericMixin):
    """
    Metric for evaluating daeval predictions that use @key[value] format.
    Converts both prediction and ground truth to lists and compares them,
    ignoring order of elements. Supports tolerance for numerical values.
    """
    
    def __init__(self, numeric_tolerance: float = 0.0, **kwargs):
        """
        Initialize list match metric.
        
        Args:
            numeric_tolerance: Tolerance for numerical value comparison
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.numeric_tolerance = numeric_tolerance
    
    @property
    def name(self) -> str:
        return "list_match"
    
    def _parse_agent_output(self, output: str) -> List[List[str]]:
        """
        Parse agent output in format @key[value] to list of [key, value] pairs.
        
        Args:
            output: String with format like "@mean_fare[34.65]" or 
                   "@importance_score_mean[mean] @importance_score_std[std_dev]"
                   
        Returns:
            List of [key, value] pairs
        """
        if not output:
            return []
            
        # Pattern to match @key[value] format (allows empty values)
        pattern = r'@([^[]+)\[([^\]]*)\]'
        matches = re.findall(pattern, output)
        
        # Convert to list of [key, value] pairs with string values
        result = []
        for key, value in matches:
            result.append([key.strip(), str(value.strip())])
        
        return result
    
    def _parse_ground_truth(self, ground_truth: str) -> List[List[str]]:
        """
        Parse ground truth which should be in list format like [['key', 'value']].
        
        Args:
            ground_truth: String representation of list
            
        Returns:
            List of [key, value] pairs
        """
        if not ground_truth:
            return []
            
        try:
            # Try to parse as Python literal
            parsed = ast.literal_eval(ground_truth.strip())
            
            # Ensure it's a list of lists and convert values to strings
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, list) and len(item) >= 2:
                        # Convert both key and value to strings for comparison
                        result.append([str(item[0]), str(item[1])])
                return result
            
        except (ValueError, SyntaxError):
            # If parsing fails, return empty list
            pass
            
        return []
    
    def _lists_match(self, pred_list: List[List[str]], truth_list: List[List[str]]) -> bool:
        """
        Check if two lists of [key, value] pairs match using dictionary comparison
        with tolerance for numerical values.
        
        Args:
            pred_list: Predicted list of pairs
            truth_list: Ground truth list of pairs
            
        Returns:
            True if lists contain the same pairs with tolerance for numerical values
        """
        # Step 1: Compare lengths
        if len(pred_list) != len(truth_list):
            return False
            
        # Step 2: Convert lists to dictionaries
        pred_dict = {pair[0]: pair[1] for pair in pred_list}
        truth_dict = {pair[0]: pair[1] for pair in truth_list}
        
        # Check if keys match
        if set(pred_dict.keys()) != set(truth_dict.keys()):
            return False
            
        # Step 3: Compare each value with tolerance for numerical values
        for key in pred_dict.keys():
            pred_value = pred_dict[key]
            truth_value = truth_dict[key]
            
            # Try to parse as numbers first
            pred_num = self.extract_number(pred_value)
            truth_num = self.extract_number(truth_value)
            
            if pred_num is not None and truth_num is not None:
                # Both are numerical - use tolerance comparison
                if not self._numbers_match_with_tolerance(pred_num, truth_num):
                    return False
            else:
                # At least one is non-numerical - use exact match
                if pred_value != truth_value:
                    return False
                    
        return True
    
    def _numbers_match_with_tolerance(self, num1: float, num2: float) -> bool:
        """
        Check if two numbers match within tolerance.
        
        Args:
            num1: First number
            num2: Second number
            
        Returns:
            True if numbers match within tolerance
        """
        if num2 == 0:
            # Avoid division by zero - use absolute tolerance
            return abs(num1 - num2) <= self.numeric_tolerance
        else:
            # Use relative tolerance
            relative_error = abs(num1 - num2) / abs(num2)
            return relative_error <= self.numeric_tolerance
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate prediction by parsing both prediction and ground truth to lists
        and checking if they match (ignoring order).
        
        Args:
            prediction: Model prediction in @key[value] format
            ground_truth: Ground truth in list format
            query: Original query (unused)
            **kwargs: Additional context
            
        Returns:
            MetricResult with list match score
        """
        if ground_truth is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No ground truth available"}
            )
        
        # Parse prediction and ground truth
        pred_list = self._parse_agent_output(prediction)
        truth_list = self._parse_ground_truth(ground_truth)
        
        # Check if lists match
        matches = self._lists_match(pred_list, truth_list)
        score = 1.0 if matches else 0.0
        
        # Generate detailed comparison info
        comparison_details = {}
        if len(pred_list) == len(truth_list):
            pred_dict = {pair[0]: pair[1] for pair in pred_list}
            truth_dict = {pair[0]: pair[1] for pair in truth_list}
            
            for key in pred_dict.keys():
                if key in truth_dict:
                    pred_value = pred_dict[key]
                    truth_value = truth_dict[key]
                    pred_num = self.extract_number(pred_value)
                    truth_num = self.extract_number(truth_value)
                    
                    if pred_num is not None and truth_num is not None:
                        comparison_details[key] = {
                            "type": "numerical",
                            "pred_value": pred_value,
                            "truth_value": truth_value,
                            "pred_number": pred_num,
                            "truth_number": truth_num,
                            "tolerance": self.numeric_tolerance,
                            "matches": self._numbers_match_with_tolerance(pred_num, truth_num)
                        }
                    else:
                        comparison_details[key] = {
                            "type": "string",
                            "pred_value": pred_value,
                            "truth_value": truth_value,
                            "matches": pred_value == truth_value
                        }
        
        details = {
            "prediction_raw": prediction,
            "ground_truth_raw": ground_truth,
            "prediction_parsed": pred_list,
            "ground_truth_parsed": truth_list,
            "lists_match": matches,
            "pred_length": len(pred_list),
            "truth_length": len(truth_list),
            "numeric_tolerance": self.numeric_tolerance,
            "comparison_details": comparison_details,
        }
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details=details
        )