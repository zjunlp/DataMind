"""
Dataset registration system for DSGym.

Provides automatic dataset discovery and loading.
"""

from typing import Dict, Type, Optional, Any, List
from .base import BaseDataset


class DatasetRegistry:
    """Registry for dataset classes with automatic discovery."""
    
    _datasets: Dict[str, Type[BaseDataset]] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: Type[BaseDataset]):
        """
        Register a dataset class.
        
        Args:
            name: Dataset name (e.g., "DiscoveryBench")
            dataset_class: Dataset class implementing BaseDataset
        """
        cls._datasets[name.lower()] = dataset_class
    
    @classmethod
    def load(cls, name: str, **kwargs) -> BaseDataset:
        """
        Load a dataset by name.
        
        Args:
            name: Dataset name
            **kwargs: Arguments passed to dataset constructor
            
        Returns:
            Initialized dataset instance
        """
        name_lower = name.lower()
        if name_lower not in cls._datasets:
            available = list(cls._datasets.keys())
            raise ValueError(f"Dataset '{name}' not found. Available: {available}")
        
        dataset_class = cls._datasets[name_lower]
        return dataset_class(**kwargs)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """
        List all registered datasets.
        
        Returns:
            List of dataset names
        """
        return list(cls._datasets.keys())
    
    @classmethod
    def get_dataset_class(cls, name: str) -> Type[BaseDataset]:
        """
        Get dataset class by name.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset class
        """
        name_lower = name.lower()
        if name_lower not in cls._datasets:
            available = list(cls._datasets.keys())
            raise ValueError(f"Dataset '{name}' not found. Available: {available}")
        
        return cls._datasets[name_lower]


# Auto-register decorator
def register_dataset(name: str):
    """
    Decorator to automatically register dataset classes.
    
    Args:
        name: Dataset name
    """
    def decorator(cls: Type[BaseDataset]):
        DatasetRegistry.register(name, cls)
        return cls
    return decorator