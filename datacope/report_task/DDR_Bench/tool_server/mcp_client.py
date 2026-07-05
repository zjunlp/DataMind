#!/usr/bin/env python3
"""
MCP client management for the autonomous data analysis agent
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.models import MCPServerConfig

logger = logging.getLogger("mcp-client")

class MCPClientManager:
    """Manages connections to MCP servers"""
    
    def __init__(self):
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.stdio_clients: Dict[str, Any] = {}
    
    async def connect_to_servers(self, configs: list[MCPServerConfig]) -> bool:
        """Connect to all configured MCP servers"""
        success = False
        
        for config in configs:
            try:
                # Validate server script exists
                script_path = Path(config.script_path)
                if not script_path.exists():
                    logger.error(f"MCP server script not found: {script_path}")
                    continue
                
                # Create server parameters with script path and arguments
                args = [str(script_path)] + (config.args or [])
                server_params = StdioServerParameters(
                    command=sys.executable,
                    args=args,
                    env=os.environ.copy()  # Pass current environment variables to MCP server
                )
                
                # Connect to MCP server
                stdio_client_instance = stdio_client(server_params)
                read_stream, write_stream = await stdio_client_instance.__aenter__()
                
                # Create client session
                mcp_session = ClientSession(read_stream, write_stream)
                await mcp_session.__aenter__()
                await mcp_session.initialize()
                
                self.mcp_sessions[config.name] = mcp_session
                self.stdio_clients[config.name] = stdio_client_instance
                
                logger.info(f"Connected to MCP server: {config.name}")
                success = True
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {config.name}: {e}")
        
        return success
    
    async def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get available tools from all MCP servers"""
        all_tools = {}
        
        for server_name, session in self.mcp_sessions.items():
            try:
                tools_result = await session.list_tools()
                server_tools = {}
                
                for tool in tools_result.tools:
                    server_tools[tool.name] = {
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    }
                
                all_tools[server_name] = server_tools
                logger.info(f"Got {len(server_tools)} tools from {server_name}")
                
            except Exception as e:
                logger.error(f"Failed to get tools from {server_name}: {e}")
        
        return all_tools
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool call"""
        # Find appropriate MCP server for the tool
        for server_name, session in self.mcp_sessions.items():
            # This would need to be updated to check against available_tools
            # For now, we'll try all servers
            try:
                result = await session.call_tool(tool_name, arguments)
                return self._format_tool_result(result)
            except Exception as e:
                continue
        
        return {"error": f"Tool '{tool_name}' not found or execution failed"}
    
    def _format_tool_result(self, result: Any) -> Any:
        """Format tool execution result"""
        if hasattr(result, 'content') and result.content:
            content = result.content[0]
            if hasattr(content, 'text'):
                try:
                    # Try to parse as JSON
                    import json
                    return json.loads(content.text)
                except:
                    # Return as string if not JSON
                    return content.text
        
        return result
    
    async def close(self):
        """Close all MCP connections"""
        # Close all MCP sessions
        for name, session in self.mcp_sessions.items():
            try:
                await session.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing MCP session {name}: {e}")
        
        # Close all stdio clients
        for name, client in self.stdio_clients.items():
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing stdio client {name}: {e}")
