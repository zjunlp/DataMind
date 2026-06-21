"""
Backend implementations for DSGym agents.
"""

from importlib import import_module
from typing import Dict, Tuple, Type

from .base import BaseBackend

__all__ = [
    'BaseBackend',
    'LiteLLMBackend',
    'NativeAPIBackend',
    'VLLMBackend',
    'SGLangBackend',
    'MultiVLLMBackend'
]

_BACKEND_SPECS: Dict[str, Tuple[str, str]] = {
    'litellm': ('.litellm_backend', 'LiteLLMBackend'),
    'native': ('.native_backend', 'NativeAPIBackend'),
    'native-api': ('.native_backend', 'NativeAPIBackend'),
    'openai': ('.native_backend', 'NativeAPIBackend'),
    'anthropic': ('.native_backend', 'NativeAPIBackend'),
    'together': ('.native_backend', 'NativeAPIBackend'),
    'openai-compatible': ('.native_backend', 'NativeAPIBackend'),
    'vllm': ('.vllm_backend', 'VLLMBackend'),
    'sglang': ('.sglang_backend', 'SGLangBackend'),
    'multi-vllm': ('.multi_vllm_backend', 'MultiVLLMBackend'),
}

_EXPORT_SPECS: Dict[str, Tuple[str, str]] = {
    'LiteLLMBackend': ('.litellm_backend', 'LiteLLMBackend'),
    'NativeAPIBackend': ('.native_backend', 'NativeAPIBackend'),
    'VLLMBackend': ('.vllm_backend', 'VLLMBackend'),
    'SGLangBackend': ('.sglang_backend', 'SGLangBackend'),
    'MultiVLLMBackend': ('.multi_vllm_backend', 'MultiVLLMBackend'),
}


def __getattr__(name: str):
    """Lazy-load backend classes to avoid importing optional dependencies early."""
    if name in _EXPORT_SPECS:
        module_name, class_name = _EXPORT_SPECS[name]
        backend_class = getattr(import_module(module_name, __name__), class_name)
        globals()[name] = backend_class
        return backend_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_backend_class(backend_type: str) -> Type[BaseBackend]:
    """Load backend class for a backend type."""
    module_name, class_name = _BACKEND_SPECS[backend_type]
    return getattr(import_module(module_name, __name__), class_name)


def get_backend(backend_type: str, model_name: str, **kwargs) -> BaseBackend:
    """
    Get backend instance by type.
    
    Args:
        backend_type: Type of backend ('litellm', 'native', 'openai',
            'anthropic', 'together', 'vllm', 'sglang', 'multi-vllm')
        model_name: Model name/path
        **kwargs: Backend-specific configuration
        
    Returns:
        Backend instance
    """
    backend_type = backend_type.lower()
    if backend_type not in _BACKEND_SPECS:
        available = list(_BACKEND_SPECS.keys())
        raise ValueError(f"Unknown backend type '{backend_type}'. Available: {available}")
    
    provider_aliases = {
        'openai': 'openai',
        'anthropic': 'anthropic',
        'together': 'together',
        'openai-compatible': 'openai-compatible',
    }
    if backend_type in provider_aliases and 'provider' not in kwargs:
        kwargs['provider'] = provider_aliases[backend_type]

    backend_class = _load_backend_class(backend_type)
    return backend_class(model_name, **kwargs)
