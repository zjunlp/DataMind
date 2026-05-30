"""
Base text environment for DSGym agents.

Simplified reimplementation of skyrl_gym BaseTextEnv.
"""

from typing import Any, Dict, List, Optional, Tuple
from .core import BaseEnv, BaseEnvStepOutput, ConversationType


class BaseTextEnv(BaseEnv):
    """
    Base environment class for all text-in / text-out environments.
    Supports tool-calling and multi-turn trajectories.
    
    Input Types:
        - ObsType: ConversationType (tool output, LLM input)
        - ActType: str (LLM output)
    """
    
    def __init__(self):
        """Initialize base text environment."""
        super().__init__()
        self.chat_history: ConversationType = []
    
    def step(self, action: str) -> BaseEnvStepOutput:
        """
        Runs one environment step.
        
        Args:
            action: LLM response/action
            
        Returns:
            - observations: [{"role": "user", "content": observation}]
            - reward: float
            - done: bool
            - postprocessed_action: Optional[str]
            - metadata: Dict[str, Any] any metadata
        """
        self.turns += 1
        
        # Add assistant response to chat history
        if action:
            self.chat_history.append({"role": "assistant", "content": action})
        
        # Check if done
        done = self.turns >= self.max_turns
        
        # Basic implementation - subclasses should override
        return BaseEnvStepOutput(
            observations=self.chat_history,
            reward=0.0,
            done=done,
            metadata={"turns": self.turns},
            postprocessed_action=action
        )
    
    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize environment with prompt.
        
        Args:
            prompt: Initial conversation prompt
            
        Returns:
            Tuple of (prompt, metadata)
        """
        self.turns = 0
        self.chat_history = prompt.copy() if prompt else []
        return prompt, {"initialized": True}
    
    def reset(self, **kwargs) -> ConversationType:
        """
        Reset environment to initial state.
        
        Args:
            **kwargs: Reset parameters
            
        Returns:
            Initial observations
        """
        self.turns = 0
        self.chat_history = []
        return self.chat_history
    
    def add_user_message(self, content: str):
        """
        Add a user message to the chat history.
        
        Args:
            content: User message content
        """
        self.chat_history.append({"role": "user", "content": content})
    
    def add_system_message(self, content: str):
        """
        Add a system message to the chat history.
        
        Args:
            content: System message content
        """
        self.chat_history.append({"role": "system", "content": content})
    
    def get_conversation_history(self) -> ConversationType:
        """
        Get the current conversation history.
        
        Returns:
            Current conversation as list of messages
        """
        return self.chat_history.copy()