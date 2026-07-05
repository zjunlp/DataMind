from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from src.core.schema import AgentResult

class BaseAgent(ABC):
    def __init__(self, agent_type: str, model: str, backend: str = "", **kwargs):
        """
        Initialize agent.
        """
        self.agent_type = agent_type
        self.model = model
        self.backend = backend
        self.config = kwargs
    
    @abstractmethod
    def solve_task(self, prompt, system, **kwargs) -> AgentResult:
        """
        Solve a given task using a sample dictionary.
        
        Args:
            prompt: Task prompt or input data
            system: System message or context
            **kwargs: Additional task-specific parameters
            
        Returns:
            AgentResult: Dataclass containing solution, conversation, and metadata
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            'agent_type': self.agent_type,
            'model': self.model,
            'backend': self.backend,
            **self.config
        }
    
    def get_agent_type(self) -> str:
        """Get agent type."""
        return self.agent_type