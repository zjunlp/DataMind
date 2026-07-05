from typing import Dict, Type, Optional, Any, List
from src.runtimes.base_agent import BaseAgent

class AgentRegistry:
    """Registry for agent classes with automatic discovery."""
    
    _agents: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]):
        """
        Register an agent class.
        
        Args:
            name: Agent name (e.g., "DiscoveryBench")
            agent_class: Agent class implementing BaseAgent
        """
        cls._agents[name.lower()] = agent_class
    
    @classmethod
    def load(cls, name: str, **kwargs) -> BaseAgent:
        """
        Load an agent by name.
        
        Args:
            name: Agent name
            **kwargs: Arguments passed to agent constructor
            
        Returns:
            Initialized agent instance
        """
        name_lower = name.lower()
        if name_lower not in cls._agents:
            available = list(cls._agents.keys())
            raise ValueError(f"Agent '{name}' not found. Available: {available}")
        
        agent_class = cls._agents[name_lower]
        return agent_class(**kwargs)
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """
        List all registered agents.
        
        Returns:
            List of agent names
        """
        return list(cls._agents.keys())
    
    @classmethod
    def get_agent_class(cls, name: str) -> Type[BaseAgent]:
        """
        Get agent class by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent class
        """
        name_lower = name.lower()
        if name_lower not in cls._agents:
            available = list(cls._agents.keys())
            raise ValueError(f"Agent '{name}' not found. Available: {available}")
        
        return cls._agents[name_lower]


# Auto-register decorator
def register_agent(name: str):
    """
    Decorator to automatically register agent classes.
    
    Args:
        name: Agent name
    """
    def decorator(cls: Type[BaseAgent]):
        AgentRegistry.register(name, cls)
        return cls
    return decorator