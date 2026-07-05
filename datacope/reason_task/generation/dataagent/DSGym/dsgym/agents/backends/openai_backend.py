"""
LiteLLM backend for DSGym agents.

Supports various API providers through LiteLLM.
"""

import os
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .base import BaseBackend


class OpenAIBackend(BaseBackend):
    """LiteLLM backend for calling various API providers."""
    
    def __init__(
        self, 
        model_name: str, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        temperature: float = 0.0, 
        top_p: float = 1.0, 
        max_tokens: int = 1524, 
        timeout: int = 180, 
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
        
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.is_reasoning_model = self._is_reasoning_model(model_name)
        self.client = OpenAI(
            api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            timeout=self.timeout
        )
        
        if self.is_reasoning_model:
            self.generation_params = {}
        else:
            self.generation_params = {
                "temperature": temperature, 
                "max_tokens": max_tokens
            }
            if self.model_name.startswith("gpt-5"):
                self.generation_params["reasoning_effort"] = "medium"
            if top_p != 1.0:
                self.generation_params["top_p"] = top_p
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if model is a reasoning model (o1, o3, o4, gpt-5 series)."""
        reasoning_model_prefixes = ["o1-", "o3-", "o4-", "gpt-5"]
        return any(model_name.startswith(prefix) for prefix in reasoning_model_prefixes)
    
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
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=messages,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": False
                        }
                    },
                    stop=["\n<information>"],
                    **params
                )

                content = response.choices[0].message.content

                # qwen3 reasoning trace
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)

                usage = response.usage
                token_usage = {
                    "input_tokens": getattr(usage, 'prompt_tokens', 0) if usage else 0,
                    "output_tokens": getattr(usage, 'completion_tokens', 0) if usage else 0,
                    "total_tokens": getattr(usage, 'total_tokens', 0) if usage else 0,
                }

                return content if content is not None else "", reasoning_content if reasoning_content is not None else "", token_usage
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    raise e
                else:
                    # Wait and retry
                    wait_time = 2 ** attempt
                    print(f"OpenAI API call failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s: {str(e)}")
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