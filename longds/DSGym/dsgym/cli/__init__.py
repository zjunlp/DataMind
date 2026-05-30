"""
Command Line Interface for DSGym.

This module provides CLI commands for evaluating models,
generating synthetic data, and training.
"""

from .main import main
from .eval import run_eval
from .generate import run_generate  
from .train import run_train

__version__ = "0.1.0"

__all__ = [
    "main",
    "run_eval",
    "run_generate", 
    "run_train",
]

def run():
    """Entry point for the CLI."""
    import sys
    return main(sys.argv[1:])