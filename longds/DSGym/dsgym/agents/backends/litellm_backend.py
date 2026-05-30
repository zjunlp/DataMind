"""
LiteLLM backend for DSGym agents.

Supports various API providers through LiteLLM.
"""

import os
import time
from typing import List, Dict, Any, Optional

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None

from .base import BaseBackend


class LiteLLMBackend(BaseBackend):
    """LiteLLM backend for calling various API providers."""
    
    def __init__(
        self, 
        model_name: str, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        temperature: float = 0.0, 
        top_p: float = 1.0, 
        max_tokens: int = 8192, 
        timeout: int = 300, 
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize LiteLLM backend.
        
        Args:
            model_name: Model name (e.g., 'gpt-4', 'together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput')
            api_key: API key (uses environment variable if None)
            base_url: Base URL for API
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            **kwargs: Additional configuration
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for LiteLLMBackend. Please install: pip install litellm")
        
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.is_reasoning_model = self._is_reasoning_model(model_name)

        # Set generation parameters based on model type
        if self.is_reasoning_model:
            # Reasoning models don't support temperature/top_p and use max_completion_tokens
            # For reasoning models, we need much higher token limits since they use tokens for internal reasoning
            # reasoning_max_tokens = max(max_tokens * 4, 4096)  # At least 4x the requested tokens, minimum 4096
            # self.generation_params = {"max_completion_tokens": reasoning_max_tokens}
            self.generation_params = {"max_tokens": max_tokens,
                                      "max_completion_tokens": max_tokens}  # At least 4x the requested tokens, minimum 4096
        else:
            # Only include top_p if explicitly set to non-default value or when temperature is 0
            # This avoids parameter conflicts with models that don't support both
            self.generation_params = {
                "temperature": temperature, 
                "max_tokens": max_tokens,
                "max_completion_tokens": max_tokens
            }
            if self.model_name.startswith("gpt-5"):
                self.generation_params["reasoning_effort"] = "medium"
            if top_p != 1.0:
                self.generation_params["top_p"] = top_p
        print(self.generation_params)
        self._setup_litellm()
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if model is a reasoning model (o1, o3, o4, gpt-5 series)."""
        reasoning_model_prefixes = ["o1-", "o3-", "o4-", "gpt-5"]
        return any(model_name.startswith(prefix) for prefix in reasoning_model_prefixes)
    
    def _setup_litellm(self):
        """Setup LiteLLM configuration and API keys."""
        # Set API key based on model provider
        if self.api_key:
            if self.model_name.startswith("together_ai/"):
                os.environ["TOGETHER_API_KEY"] = self.api_key
            elif self.model_name.startswith("gpt-") or self._is_reasoning_model(self.model_name):
                os.environ["OPENAI_API_KEY"] = self.api_key
            elif self.model_name.startswith("claude-") or self.model_name.startswith("anthropic/"):
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
            else:
                litellm.api_key = self.api_key
        else:
            # Check for required environment variables
            if self.model_name.startswith("together_ai/"):
                self.api_key = os.environ.get("TOGETHER_API_KEY")
                if not self.api_key:
                    raise ValueError("TOGETHER_API_KEY environment variable is required for Together AI models")
            elif self.model_name.startswith("gpt-") or self._is_reasoning_model(self.model_name):
                self.api_key = os.environ.get("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
            elif self.model_name.startswith("claude-") or self.model_name.startswith("anthropic/"):
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not self.api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic Claude models")
            elif self.model_name.startswith("openai/gemini"):
                self.api_key = os.environ.get("GEMINI_API_KEY")
                self.base_url = os.environ.get("GEMINI_BASE_URL")
                litellm.api_key = self.api_key
                self.generation_params.update({"maxOutputTokens": 8092}, reasoning_effort="high")
                if not self.api_key:
                    raise ValueError("GEMINI_API_KEY environment variable is required for Gemini models")
            
        # Set base URL if provided
        if self.base_url:
            litellm.api_base = self.base_url
        
        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported parameters
        litellm.set_verbose = False  # Disable verbose logging
        print(f" url={self.base_url}, api_key={self.api_key}")
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using LiteLLM.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if messages is None:
            raise RuntimeError("Messages is None in LiteLLM generate - this indicates an upstream error")
        elif not isinstance(messages, list):
            raise TypeError(f"Messages should be a list, got {type(messages)}")
        
        # Merge generation parameters
        params = self.generation_params.copy()
        params.update(kwargs)
        
        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    model=self.model_name, 
                    messages=messages, 
                    timeout=self.timeout, 
                    **params,
                    # stop=["</python>"],
                
                    
                )
                content = response.choices[0].message.content
                return content if content is not None else "<reasoning>no content</reasoning>"
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    raise e
                else:
                    # Wait and retry
                    wait_time = 2 ** attempt
                    print(f"LiteLLM API call failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
        
        raise RuntimeError(f"LiteLLM API call failed after {self.max_retries} attempts")
    
    def is_api_based(self) -> bool:
        """LiteLLM backend is always API-based."""
        return True
    
    def apply_chat_template(self, conversation: List[Dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = False) -> Optional[str]:
        """
        LiteLLM handles chat templating internally.
        
        Returns:
            None (LiteLLM handles templating automatically)
        """
        return None