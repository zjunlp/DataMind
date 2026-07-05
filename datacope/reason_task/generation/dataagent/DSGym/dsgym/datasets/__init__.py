"""
DSGym Datasets Module

Provides unified abstractions for all supported datasets with consistent interfaces.
"""

from .base import BaseDataset
from .registry import DatasetRegistry
from .utils import create_custom_task, load_tasks_from_dataset, create_standard_task
from .loaders import *

__all__ = ['BaseDataset', 'DatasetRegistry', 'create_custom_task', 'load_tasks_from_dataset', 'create_standard_task']