"""
Base backend interface for DSGym agents.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseBackend(ABC):
    """Base interface for all model backends."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize backend.
        
        Args:
            model_name: Name/path of the model
            **kwargs: Backend-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response from messages.
        
        Args:
            messages: List of conversation messages
            **kwargs: Generation parameters
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def is_api_based(self) -> bool:
        """
        Check if this backend uses API calls.
        
        Returns:
            True if API-based, False if local
        """
        pass
    
    def apply_chat_template(self, conversation: List[Dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = False) -> Optional[str]:
        """
        Apply chat template to conversation.
        
        Args:
            conversation: List of conversation messages
            add_generation_prompt: Whether to add generation prompt
            tokenize: Whether to tokenize the result
            
        Returns:
            Formatted conversation string or None if not supported
        """
        return None
    
    def get_config(self) -> Dict[str, Any]:
        """Get backend configuration."""
        return {
            'model_name': self.model_name,
            'backend_type': self.__class__.__name__,
            **self.config
        }