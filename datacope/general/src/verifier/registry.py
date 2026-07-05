from typing import Dict, Type, Optional, Any, List
from src.verifier.base import BaseVerifier


class VerifierRegistry:
    """Registry for verifier classes with automatic discovery."""
    
    _verifiers: Dict[str, Type[BaseVerifier]] = {}
    
    @classmethod
    def register(cls, name: str, verifier_class: Type[BaseVerifier]):
        """
        Register a verifier class.
        
        Args:
            name: Verifier name (e.g., "DiscoveryBench")
            verifier_class: Verifier class implementing BaseVerifier
        """
        cls._verifiers[name.lower()] = verifier_class
    
    @classmethod
    def load(cls, name: str, **kwargs) -> BaseVerifier:
        """
        Load a verifier by name.
        
        Args:
            name: Verifier name
            **kwargs: Arguments passed to verifier constructor
            
        Returns:
            Initialized verifier instance
        """
        name_lower = name.lower()
        if name_lower not in cls._verifiers:
            available = list(cls._verifiers.keys())
            raise ValueError(f"Verifier '{name}' not found. Available: {available}")
        
        verifier_class = cls._verifiers[name_lower]
        return verifier_class(**kwargs)
    
    @classmethod
    def list_verifiers(cls) -> List[str]:
        """
        List all registered verifiers.
        
        Returns:
            List of verifier names
        """
        return list(cls._verifiers.keys())
    
    @classmethod
    def get_verifier_class(cls, name: str) -> Type[BaseVerifier]:
        """
        Get verifier class by name.
        
        Args:
            name: Verifier name
            
        Returns:
            Verifier class
        """
        name_lower = name.lower()
        if name_lower not in cls._verifiers:
            available = list(cls._verifiers.keys())
            raise ValueError(f"Verifier '{name}' not found. Available: {available}")
        
        return cls._verifiers[name_lower]


# Auto-register decorator
def register_verifier(name: str):
    """
    Decorator to automatically register verifier classes.
    
    Args:
        name: Verifier name
    """
    def decorator(cls: Type[BaseVerifier]):
        VerifierRegistry.register(name, cls)
        return cls
    return decorator