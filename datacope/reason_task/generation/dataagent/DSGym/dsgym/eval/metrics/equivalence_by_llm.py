"""
LLM-based equivalence evaluation metric.
"""

import re
from typing import Optional, List
from .base import BaseMetric, MetricResult


class EquivalenceByLLMMetric(BaseMetric):
    """
    LLM-based metric that evaluates answer equivalence using a judge model.
    Based on the existing discovery_llm_judge.py implementation.
    """
    
    def __init__(
        self, 
        judge_model: str = "gpt-4o",
        numeric_tolerance: float = 0.01,
        **kwargs
    ):
        """
        Initialize LLM equivalence metric.
        
        Args:
            judge_model: Model to use for judging equivalence
            numeric_tolerance: Tolerance for numeric comparisons
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.judge_model = judge_model
        self.numeric_tolerance = numeric_tolerance
    
    @property
    def name(self) -> str:
        return "equivalence_by_llm"
    
    @property
    def supports_batch_evaluation(self) -> bool:
        return True  # Can batch API calls for efficiency
    
    def _extract_judgment(self, response_text: str) -> Optional[float]:
        """
        Extract judgment from LLM response.
        Based on extract_answer function from discovery_llm_judge.py.
        """
        if not response_text:
            return None
            
        # Look for the final answer pattern
        if "## The final answer is:" in response_text:
            answer_str = response_text.split("## The final answer is:")[1].strip()
            if answer_str == "True":
                return 1.0
            elif answer_str == "False":
                return 0.0
        
        return None
    
    def _create_judgment_prompt(self, query: str, prediction: str, ground_truth: str) -> str:
        """
        Create prompt for LLM judgment.
        Based on compute_llm_score_discrete from discovery_llm_judge.py.
        """
        return f"""Please judge whether the generated answer is right or wrong.

Query: {query}

Predicted answer: {prediction}

True answer: {ground_truth}

Rules for judgment:
- If the answer is numerical, treat it as correct if the relative error < {self.numeric_tolerance * 100}% compared with the ground-truth value.
- Otherwise, judge correctness against the provided ground-truth answer. 
- The answer should be clear and complete.
- Calculation process alone is not considered correct.

Please reply in this format:

Thoughts: <your thoughts here>

## The final answer is: <Output only True or False>"""
    
    def _call_judge_model(self, prompt: str) -> Optional[str]:
        """
        Call the judge model with the given prompt.
        """
        try:
            from litellm import completion
            
            response = completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Log error but don't fail completely
            print(f"Error calling judge model {self.judge_model}: {e}")
            return None
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate equivalence using LLM judge.
        
        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            query: Original query
            **kwargs: Additional context
            
        Returns:
            MetricResult with LLM judgment score
        """
        if ground_truth is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No ground truth available"}
            )
        
        if query is None:
            query = "N/A"  # Use placeholder if query not provided
        
        # Create judgment prompt
        prompt = self._create_judgment_prompt(query, prediction, ground_truth)
        
        # Call judge model
        response = self._call_judge_model(prompt)
        
        if response is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                error="Failed to get response from judge model",
                details={
                    "judge_model": self.judge_model,
                    "prompt": prompt
                }
            )
        
        # Extract judgment
        score = self._extract_judgment(response)
        
        details = {
            "judge_model": self.judge_model,
            "judge_response": response,
            "prompt": prompt,
            "numeric_tolerance": self.numeric_tolerance,
        }
        
        if score is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                error="Could not extract valid judgment from response",
                details=details
            )
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details=details
        )
    
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        queries: Optional[List[str]] = None,
        **kwargs
    ) -> List[MetricResult]:
        """
        Evaluate multiple predictions in batch.
        
        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth answers
            queries: List of original queries
            **kwargs: Additional context
            
        Returns:
            List of MetricResult objects
        """
        results = []
        
        # For now, use individual evaluation
        # TODO: Implement true batch API calls for efficiency
        for i, prediction in enumerate(predictions):
            ground_truth = ground_truths[i] if i < len(ground_truths) else None
            query = queries[i] if queries and i < len(queries) else None
            
            result = self.evaluate(prediction, ground_truth, query, **kwargs)
            results.append(result)
        
        return results


class FastEquivalenceByLLMMetric(EquivalenceByLLMMetric):
    """
    Faster version using a smaller/cheaper model for equivalence checking.
    """
    
    def __init__(self, judge_model: str = "gpt-4o-mini", **kwargs):
        """
        Initialize fast LLM equivalence metric.
        
        Args:
            judge_model: Faster/cheaper model for judging
            **kwargs: Additional configuration
        """
        super().__init__(judge_model=judge_model, **kwargs)
    
    @property
    def name(self) -> str:
        return "fast_equivalence_by_llm"