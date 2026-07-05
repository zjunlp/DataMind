"""
DSGym: Unified Data Science Benchmark Framework

A comprehensive platform for data science task evaluation and training.
"""

__version__ = "0.1.0"

# Import main modules
from . import datasets
from . import agents
from . import eval

# Import commonly used classes and functions
from .datasets import DatasetRegistry
from .agents import ReActDSAgent
from .eval import Evaluator

__all__ = [
    # Modules
    'datasets',
    'agents', 
    'eval',
    
    # Main classes
    'DatasetRegistry',
    'ReActDSAgent',
    'Evaluator',
    
    # Version
    '__version__'
]