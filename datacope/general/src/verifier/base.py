from abc import ABC, abstractmethod

class BaseVerifier(ABC):
    """Base interface for all verifiers."""
    
    def __init__(self, input_file_dirs: list, output_dir: str, category_list: list = None, **kwargs):
        """
        Initialize verifier.
        
        Args:
            input_file_dirs: List of directories containing prediction files to verify, index representing iteration order
            output_dir: Directory to save organized verification results
            category_list: List of categories for verification
            **kwargs: Additional verifier-specific configuration
        """
        self.input_file_dirs = input_file_dirs
        self.output_dir = output_dir
        self.category_list = category_list
        self.kwargs = kwargs

    @abstractmethod
    def init_run(self) -> None:
        pass

    @abstractmethod
    def iterate_run(self) -> None:
        pass