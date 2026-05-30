"""
Base agent interface for DSGym.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseAgent(ABC):
    """Base interface for all DSGym agents."""
    
    def __init__(self, backend: str, model: str, **kwargs):
        """
        Initialize agent.
        
        Args:
            backend: Backend type (vllm, litellm, openai, anthropic)
            model: Model name/identifier
            **kwargs: Additional configuration
        """
        self.backend = backend
        self.model = model
        self.config = kwargs
    
    @abstractmethod
    def solve_task(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve a given task using a sample dictionary.
        
        Args:
            sample: Task sample dictionary containing:
                - prompt: Task description or conversation history
                - ground_truth: Expected answer (optional)
                - extra_info: Additional metadata (optional)
                - reward_spec: Reward specification (optional)
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary containing solution and metadata:
                - solution: Final answer/solution
                - success: Whether task was completed successfully
                - turns: Number of interaction turns used
                - error: Error message if failed (None if success)
                - metadata: Additional metadata about execution
                - conversation: Full conversation history
                - raw_result: Raw result from agent execution
        """
        pass
    
    def evaluate_batch(self, samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of samples.
        
        Default implementation processes samples sequentially.
        Subclasses can override for parallel processing or optimizations.
        
        Args:
            samples: List of samples to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, sample in enumerate(samples):
            try:
                result = self.solve_task(sample, **kwargs)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = {
                    'solution': '',
                    'success': False,
                    'turns': 0,
                    'error': str(e),
                    'metadata': {
                        'model': self.model,
                        'backend': self.backend,
                        'sample_index': i
                    },
                    'conversation': [],
                    'raw_result': None
                }
                results.append(error_result)
        
        return results
    
    def reset(self):
        """Reset agent state for new task."""
        # Default implementation - subclasses can override if needed
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            'backend': self.backend,
            'model': self.model,
            **self.config
        }