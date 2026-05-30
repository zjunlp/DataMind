"""
Base dataset interface for DSGym.

All datasets should implement this interface for consistent usage.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseDataset(ABC):
    """Base interface for all DSGym datasets."""
    
    def __init__(self, data_dir: Optional[str] = None, virtual_data_root: Optional[str] = None, **kwargs):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing dataset files
            virtual_data_root: Root path for virtual/docker paths (default: None = use absolute paths)
            **kwargs: Additional dataset-specific configuration
        """
        self.data_dir = data_dir
        self.virtual_data_root = virtual_data_root if virtual_data_root is not None else ""
        self.config = kwargs
        self._metadata = None
        self._samples = None
    
    @abstractmethod
    def load(self, split: str = "test", **kwargs) -> List[Dict[str, Any]]:
        """
        Load dataset samples.
        
        Args:
            split: Dataset split (train/validation/test)
            **kwargs: Additional loading parameters
            
        Returns:
            List of dataset samples with consistent format
        """
        pass
    
    @abstractmethod
    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        Get a single sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Single dataset sample
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary containing dataset metadata
        """
        pass
    
    def get_samples(self, limit: Optional[int] = None, start_index: int = 0) -> List[Dict[str, Any]]:
        """
        Get multiple samples with optional limit and start index.
        
        Args:
            limit: Maximum number of samples to return
            start_index: Starting index for sample selection
            
        Returns:
            List of dataset samples
        """
        if self._samples is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        end_index = len(self._samples)
        if limit is not None:
            end_index = min(start_index + limit, len(self._samples))
        
        return self._samples[start_index:end_index]
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        if self._samples is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return len(self._samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get sample by index (for direct indexing)."""
        return self.get_sample(index)
    
    def get_metrics(self) -> List[str]:
        """
        Get metrics for this dataset.
        
        Returns:
            List of metric names for this dataset
        """
        # Default metrics that work for most datasets
        return ["exact_match"]
    
    def get_metric_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metric-specific configurations for this dataset.
        
        Returns:
            Dictionary mapping metric names to their configurations
        """
        # Default configurations
        return {}