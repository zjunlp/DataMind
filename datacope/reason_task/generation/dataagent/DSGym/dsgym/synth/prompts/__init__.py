"""
Prompts for synthetic data generation.
"""

from .system_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    QRDATA_SYSTEM_PROMPT,
    DAEVAL_SYSTEM_PROMPT,
    get_system_prompt
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "QRDATA_SYSTEM_PROMPT", 
    "DAEVAL_SYSTEM_PROMPT",
    "get_system_prompt"
]