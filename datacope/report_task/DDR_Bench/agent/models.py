#!/usr/bin/env python3
"""
Data models for the autonomous data analysis agent
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    script_path: str
    description: str
    args: List[str] = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = []

@dataclass
class AgentMessage:
    """A message in the agent-environment conversation"""
    role: str  # 'agent', 'environment', 'system'
    content: str
    timestamp: str = None
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class AutonomousSession:
    """Autonomous exploration session data"""
    session_id: str
    start_time: datetime
    task: str
    messages: List[AgentMessage]
    available_tools: Dict[str, Dict[str, Any]]
    mcp_servers: Dict[str, Any]
    completed: bool = False
