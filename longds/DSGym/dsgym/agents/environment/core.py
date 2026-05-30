"""
Core environment functionality for DSGym agents.

Simplified reimplementation of skyrl_gym components with only needed features.
"""

from typing import Any, Dict, List, Optional, TypedDict, Tuple
from abc import ABC, abstractmethod

# Type definitions
MessageType = Dict[str, str]
ConversationType = List[MessageType]


class BaseEnvStepOutput(TypedDict):
    """Standard output format for environment steps."""
    observations: ConversationType  # OpenAI API Messages Format
    reward: float
    done: bool
    metadata: Dict[str, Any]
    postprocessed_action: Optional[str]


class BaseEnv(ABC):
    """
    Base environment class for text-in / text-out environments.
    
    Simplified version of skyrl_gym BaseTextEnv with only essential features.
    """
    
    def __init__(self):
        """Initialize base environment."""
        self.turns = 0
        self.max_turns = 10
        self.tool_groups = []
        self.tool_to_toolgroup = {}
    
    def init_tool_groups(self, tool_groups: List = []) -> None:
        """
        Initialize the tool groups for the environment.
        
        Args:
            tool_groups: List of tool group instances
        """
        self.tool_groups = tool_groups
        self.tool_to_toolgroup = {}
        for tool_group in self.tool_groups:
            if hasattr(tool_group, 'get_tool_to_group_mapping'):
                self.tool_to_toolgroup.update(tool_group.get_tool_to_group_mapping())
    
    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        """
        Find the right ToolGroup and Tool and execute it.
        
        Args:
            tool_group_name: Name of the tool group
            tool_name: Name of the tool
            tool_input: Input for the tool
            
        Returns:
            Tool execution result
        """
        for group in self.tool_groups:
            if hasattr(group, 'name') and group.name == tool_group_name:
                if hasattr(group, 'execute_tool'):
                    return group.execute_tool(tool_name, *tool_input)
        
        raise ValueError(f"ToolGroup '{tool_group_name}' not found.")
    
    @abstractmethod
    def step(self, action: str) -> BaseEnvStepOutput:
        """
        Runs one environment step.
        
        Args:
            action: Action to take (usually LLM response)
            
        Returns:
            Step output with observations, reward, done flag, and metadata
        """
        pass
    
    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Return the first prompt to be given to the model and optional metadata.
        
        Args:
            prompt: Initial conversation prompt
            
        Returns:
            Tuple of (prompt, metadata)
        """
        return prompt, {}
    
    def reset(self, **kwargs) -> ConversationType:
        """
        Reset environment to initial state.
        
        Args:
            **kwargs: Reset parameters
            
        Returns:
            Initial observations
        """
        self.turns = 0
        return []
    
    def close(self):
        """Close the environment, override if needed by subclasses."""
        pass