#!/usr/bin/env python3
"""
Base MCP Server with common functionality
Shared components for all MCP servers to eliminate code duplication
"""

import asyncio
import csv
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions


def setup_logging(log_level: str = "INFO", log_file: str = None, server_name: str = "mcp"):
    """Setup enhanced logging configuration"""
    
    # If no log file specified, create one with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check for custom log directory from environment variable
        custom_log_dir = os.getenv('CUSTOM_LOG_DIR')
        if custom_log_dir:
            log_file = f"{custom_log_dir}/{server_name}_server_{timestamp}.log"
        else:
            log_file = f"./logs/{server_name}_server_{timestamp}.log"
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter with more details
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Only use file handler - no console output
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log session start separator
    parent_pid = os.getppid()
    root_logger.info("=" * 80)
    root_logger.info(f"{server_name.upper()} MCP SERVER SESSION STARTED - PID: {os.getpid()}, PPID: {parent_pid}")
    root_logger.info(f"Timestamp: {datetime.now().isoformat()}")
    root_logger.info(f"Log file: {log_file}")
    root_logger.info("=" * 80)
    
    return root_logger


class CSVLogger:
    """Simple CSV logger for MCP server operations"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.csv_path = Path(csv_file)
        
        # Ensure directory exists
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV headers
        self.headers = [
            'timestamp',
            'tool_name', 
            'params',
            'result_content',
            'success'
        ]
        
        # Initialize CSV file with headers if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
    
    def log_tool_call(self, tool_name: str, params: Dict[str, Any], result_content: str, success: bool):
        """Log a tool call to CSV"""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    tool_name,
                    json.dumps(params, default=str, ensure_ascii=False),
                    result_content,
                    success
                ])
        except Exception as e:
            # Fallback to basic logging if CSV fails
            print(f"CSV logging failed: {e}")


class BaseMCPServer(ABC):
    """Base class for MCP servers with common functionality"""
    
    def __init__(self, server_name: str, enable_detailed_logging: bool = True):
        self.server_name = server_name
        self.server = Server(server_name)
        self.enable_detailed_logging = enable_detailed_logging
        self.session_start_time = datetime.now()
        self.call_counter = 0
        self.client_stats = {
            "total_calls": 0,
            "tool_calls": {},
            "resource_reads": {},
            "errors": 0,
            "start_time": self.session_start_time.isoformat()
        }
        self.qa_submission_counter = 0
        self.qa_submission_file = None
        
        # Initialize logger - get the root logger that was set up by setup_logging
        self.logger = logging.getLogger()
        
        # Initialize CSV logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check for custom log directory from environment variable
        custom_log_dir = os.getenv('CUSTOM_LOG_DIR')
        if custom_log_dir:
            csv_file = f"{custom_log_dir}/{server_name.replace('_', '-')}_calls_{timestamp}.csv"
        else:
            csv_file = f"./logs/{server_name.replace('_', '-')}_calls_{timestamp}.csv"
            
        self.csv_logger = CSVLogger(csv_file)
        
        self._setup_common_handlers()
        self._setup_specific_handlers()
    
    def _log_client_call(self, call_type: str, details: Dict[str, Any] = None):
        """Log client call with details"""
        self.call_counter += 1
        self.client_stats["total_calls"] += 1
        
        if self.enable_detailed_logging:
            log_msg = f"📞 Client Call #{self.call_counter}: {call_type}"
            if details:
                log_msg += f" | Details: {json.dumps(details, default=str)}"
    
    def _log_tool_call(self, tool_name: str, arguments: Dict[str, Any], result_content: str = "", error: str = None):
        """Log tool call with execution details"""
        # Update statistics
        if tool_name not in self.client_stats["tool_calls"]:
            self.client_stats["tool_calls"][tool_name] = {"count": 0, "errors": 0}
        
        self.client_stats["tool_calls"][tool_name]["count"] += 1
        if error:
            self.client_stats["tool_calls"][tool_name]["errors"] += 1
            self.client_stats["errors"] += 1
        
        # Log to CSV
        success = error is None
        content_to_log = error if error else result_content
        self.csv_logger.log_tool_call(tool_name, arguments, content_to_log, success)
    
    def _log_resource_read(self, uri: str, success: bool = True, error: str = None):
        """Log resource read operation"""
        # Update statistics
        if uri not in self.client_stats["resource_reads"]:
            self.client_stats["resource_reads"][uri] = {"count": 0, "errors": 0}
        
        self.client_stats["resource_reads"][uri]["count"] += 1
        if error:
            self.client_stats["resource_reads"][uri]["errors"] += 1
            self.client_stats["errors"] += 1
        
        # Log details
        if self.enable_detailed_logging:
            status = "❌ ERROR" if error else "✅ SUCCESS"
            error_info = f" | Error: {error}" if error else ""
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        current_time = datetime.now()
        session_duration = (current_time - self.session_start_time).total_seconds()
        
        # Ensure all values are JSON serializable
        serializable_stats = {
            "total_calls": self.client_stats["total_calls"],
            "tool_calls": {},
            "resource_reads": {},
            "errors": self.client_stats["errors"],
            "start_time": self.client_stats["start_time"],
            "session_duration_seconds": session_duration,
            "calls_per_minute": (self.client_stats["total_calls"] / session_duration * 60) if session_duration > 0 else 0,
            "current_time": current_time.isoformat()
        }
        
        # Convert tool_calls to serializable format
        for tool_name, stats in self.client_stats["tool_calls"].items():
            serializable_stats["tool_calls"][str(tool_name)] = {
                "count": stats["count"],
                "errors": stats["errors"]
            }
        
        # Convert resource_reads to serializable format
        for uri, stats in self.client_stats["resource_reads"].items():
            serializable_stats["resource_reads"][str(uri)] = {
                "count": stats["count"],
                "errors": stats["errors"]
            }
        
        return serializable_stats
    
    def _setup_common_handlers(self):
        """Setup common MCP handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            """List available resources"""
            self._log_client_call("list_resources")
            
            resources = [
                types.Resource(
                    uri=f"{self.server_name}://stats",
                    name="Session Statistics",
                    description="Current session statistics and metrics",
                    mimeType="application/json"
                )
            ]
            
            # Add server-specific resources
            specific_resources = self._get_specific_resources()
            resources.extend(specific_resources)
            
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> types.ReadResourceResult:
            """Read resource content"""
            start_time = time.time()
            
            try:
                if uri == f"{self.server_name}://stats":
                    stats = self.get_session_stats()
                    content = json.dumps(stats, indent=2, ensure_ascii=False, default=str)
                else:
                    # Delegate to specific implementation
                    content = await self._read_specific_resource(uri)
                
                return types.ReadResourceResult(
                    contents=[
                        types.TextContent(
                            type="text",
                            text=content
                        )
                    ]
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)
                self._log_resource_read(uri, success=False, error=error_msg)
                raise

        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available tools"""
            self._log_client_call("list_tools")
            
            tools = self._get_specific_tools()
            if self._qa_submission_enabled():
                tools.append(self._get_qa_submission_tool())
            
            # Add common session stats tool
            # tools.append(
            #     types.Tool(
            #         name="get_session_stats",
            #         description="Get current session statistics and metrics",
            #         inputSchema={
            #             "type": "object",
            #             "properties": {},
            #             "required": []
            #         }
            #     )
            # )
            
            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls"""
            # Add detailed debug log
            
            try:
                if name == "get_session_stats":
                    result = self.get_session_stats()
                elif name == "submit_qa_pair" and self._qa_submission_enabled():
                    result = self._handle_qa_submission(arguments)
                else:
                    # Delegate to specific implementation
                    result = await self._handle_specific_tool_call(name, arguments)
                
                # Prepare result text
                result_text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
                
                # Log successful call
                self._log_tool_call(name, arguments, result_text)
                
                return [
                    types.TextContent(
                        type="text",
                        text=result_text
                    )
                ]
                
            except Exception as e:
                error_msg = str(e)
                self._log_tool_call(name, arguments, "", error_msg)
                
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps({"error": error_msg}, ensure_ascii=False)
                    )
                ]

    def _qa_submission_enabled(self) -> bool:
        """Whether to expose the QA submission tool for this server session."""
        return os.getenv("DDR_AGENT_MODE", "").lower() == "checklist"

    def _get_qa_submission_tool(self) -> types.Tool:
        """Get the shared QA submission tool definition."""
        return types.Tool(
            name="submit_qa_pair",
            description=(
                "Submit one synthesized QA pair. "
                "Use this only when the conclusion requires combining multiple evidence signals."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "The question that would naturally elicit the synthesized conclusion."
                    },
                    "a": {
                        "type": "string",
                        "description": "The concise answer containing the synthesized conclusion."
                    }
                },
                "required": ["q", "a"]
            }
        )

    def _get_qa_submission_file(self) -> Path:
        """Return the JSONL file used for QA submissions."""
        if self.qa_submission_file is None:
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            custom_log_dir = os.getenv("CUSTOM_LOG_DIR")
            log_dir = Path(custom_log_dir) if custom_log_dir else Path("./logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            self.qa_submission_file = log_dir / f"qa_submissions_{timestamp}.jsonl"
        return self.qa_submission_file

    def _handle_qa_submission(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a submitted QA pair to the task log directory."""
        q = arguments.get("q") or arguments.get("question")
        a = arguments.get("a") or arguments.get("answer")

        if not isinstance(q, str) or not q.strip():
            raise ValueError("submit_qa_pair requires a non-empty string field: q")
        if not isinstance(a, str) or not a.strip():
            raise ValueError("submit_qa_pair requires a non-empty string field: a")

        record = {
            "q": q.strip(),
            "a": a.strip(),
        }

        submission_file = self._get_qa_submission_file()
        with open(submission_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        self.qa_submission_counter += 1
        return {
            "status": "submitted",
            "submission_id": self.qa_submission_counter,
            "q": record["q"],
            "a": record["a"],
            "path": str(submission_file)
        }
    
    @abstractmethod
    def _setup_specific_handlers(self):
        """Setup server-specific handlers - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _get_specific_resources(self) -> List[types.Resource]:
        """Get server-specific resources - implemented by subclasses"""
        pass
    
    @abstractmethod
    async def _read_specific_resource(self, uri: str) -> str:
        """Read server-specific resource - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _get_specific_tools(self) -> List[types.Tool]:
        """Get server-specific tools - implemented by subclasses"""
        pass
    
    @abstractmethod
    async def _handle_specific_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle server-specific tool call - implemented by subclasses"""
        pass
    
    async def run(self):
        """Run MCP server"""
        
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name=self.server_name,
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )
        except Exception as e:
            raise
        finally:
            # Log final statistics
            # final_stats = self.get_session_stats()
            pass
