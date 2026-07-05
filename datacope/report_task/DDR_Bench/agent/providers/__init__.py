"""
LLM Providers Package

Provides unified interface for multiple LLM providers:
- GeminiProvider: Google Gemini API
- VLLMProvider: Local VLLM server
- OpenAIProvider: OpenAI/Azure OpenAI API
- MiniMaxProvider: MiniMax API
"""

from .base import LLMProvider


def __getattr__(name):
    """Lazy import of provider classes to avoid loading all dependencies."""
    if name == "OpenAIProvider":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider
    elif name == "GeminiProvider":
        from .gemini_provider import GeminiProvider
        return GeminiProvider
    elif name == "VLLMProvider":
        from .vllm_provider import VLLMProvider
        return VLLMProvider
    elif name == "MiniMaxProvider":
        from .minimax_provider import MiniMaxProvider
        return MiniMaxProvider
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'LLMProvider',
    'OpenAIProvider',
    'GeminiProvider',
    'VLLMProvider',
    'MiniMaxProvider'
]
