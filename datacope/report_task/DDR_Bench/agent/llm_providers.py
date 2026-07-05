#!/usr/bin/env python3
"""
LLM Providers Router

Factory module for creating LLM provider instances.
Supports: Gemini, VLLM, MiniMax, OpenAI/Azure OpenAI
"""

import os
from typing import Any

from .providers import LLMProvider
from .providers.base import (
    log_api_stats_summary,
    system_logger
)


def create_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """
    Create LLM provider based on type.
    
    Args:
        provider_type: Provider type string (gemini, vllm, minimax, openai)
        **kwargs: Provider-specific arguments
        
    Returns:
        Configured LLM provider instance
        
    Raises:
        ValueError: If provider type is unsupported or API key is missing
    """
    provider_lower = provider_type.lower()
    
    if provider_lower == "openai":
        from .providers import OpenAIProvider
        api_key = kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable")
        return OpenAIProvider(
            api_key=api_key,
            model=kwargs.get("model") or "gpt-4o",
            azure_endpoint=kwargs.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_api_version=kwargs.get("azure_api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            use_azure=kwargs.get("use_azure", True)
        )
    
    elif provider_lower == "gemini":
        from .providers import GeminiProvider
        api_key = kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
        model = kwargs.get("model") or "gemini-2.5-flash"
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable")
        try:
            return GeminiProvider(
                api_key=api_key,
                model=model,
                thinking_budget=kwargs.get("thinking_budget")
            )
        except Exception as e:
            system_logger.error(f"Failed to create Gemini provider: {e}")
            raise
    
    elif provider_lower == "vllm":
        from .providers import VLLMProvider
        base_url = kwargs.get("base_url")
        port = kwargs.get("port")
        return VLLMProvider(
            base_url=base_url,
            model=kwargs.get("model") or "",
            api_key=kwargs.get("api_key", "EMPTY"),
            port=port
        )
    
    elif provider_lower == "minimax":
        from .providers import MiniMaxProvider
        api_key = kwargs.get("api_key") or os.getenv("MINIMAX_API_KEY") or os.getenv("MINIMAX_APIKEY")
        model = kwargs.get("model") or os.getenv("MINIMAX_MODEL", "MiniMax-M2")
        if not api_key:
            raise ValueError("MiniMax API key required. Set MINIMAX_API_KEY environment variable")
        return MiniMaxProvider(api_key=api_key, model=model)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_type}. Supported: gemini, vllm, minimax, openai")


# Export for backwards compatibility
__all__ = [
    'create_llm_provider',
    'LLMProvider',
    'log_api_stats_summary'
]


def __getattr__(name):
    """Lazy import of provider classes."""
    if name == "OpenAIProvider":
        from .providers import OpenAIProvider
        return OpenAIProvider
    elif name == "GeminiProvider":
        from .providers import GeminiProvider
        return GeminiProvider
    elif name == "VLLMProvider":
        from .providers import VLLMProvider
        return VLLMProvider
    elif name == "MiniMaxProvider":
        from .providers import MiniMaxProvider
        return MiniMaxProvider
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
