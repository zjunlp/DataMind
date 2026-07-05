#!/usr/bin/env python3
"""
Base classes and common utilities for LLM providers
"""

import asyncio
import csv
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def setup_llm_file_logging():
    """Setup minimal logging for LLM providers (disabled file logging)"""
    
    # Get log level from environment or default to WARNING
    log_level_str = os.getenv("DDR_LOG_LEVEL", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)
    
    # Setup system logger (no file handler - only in-memory)
    system_logger = logging.getLogger("llm.system")
    system_logger.setLevel(log_level)
    system_logger.propagate = False  # Don't propagate to root logger
    
    # Clear existing handlers
    system_logger.handlers.clear()
    
    # Setup API call logger (no file handler - only in-memory)
    llm_logger = logging.getLogger("llm.api")
    llm_logger.setLevel(log_level)
    llm_logger.propagate = False  # Don't propagate to root logger
    
    # Clear existing handlers
    llm_logger.handlers.clear()
    
    return system_logger, llm_logger


# Setup file-based logging and get loggers
system_logger, llm_logger = setup_llm_file_logging()

# Global API call statistics
api_call_stats = {
    "total_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "total_duration": 0.0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "providers": {},
    "session_start": datetime.now()
}


class LLMCSVLogger:
    """CSV logger for LLM API calls with complete raw input/output"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.csv_path = Path(csv_file)
        
        # Ensure directory exists
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV headers
        self.headers = [
            'timestamp',
            'provider_name',
            'call_type',
            'duration_seconds',
            'success',
            'raw_input_json',
            'raw_output_json',
            'metadata_json'
        ]
        
        # Initialize CSV file with headers if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
    
    def log_api_call(self, provider_name: str, call_type: str, raw_input: Any, 
                    raw_output: Any, metadata: Dict[str, Any], start_time: float, 
                    end_time: float, success: bool):
        """Log a complete LLM API call to CSV"""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.fromtimestamp(start_time).isoformat(),
                    provider_name,
                    call_type,
                    end_time - start_time,
                    success,
                    json.dumps(raw_input, default=str, ensure_ascii=False),
                    json.dumps(raw_output, default=str, ensure_ascii=False),
                    json.dumps(metadata, default=str, ensure_ascii=False)
                ])
        except Exception as e:
            print(f"LLM CSV logging failed: {e}")


# Global CSV logger instance
_llm_csv_logger = None

def get_llm_csv_logger() -> LLMCSVLogger:
    """Get or create the global LLM CSV logger"""
    global _llm_csv_logger
    
    if _llm_csv_logger is None:
        # Check for custom log directory from environment variable
        custom_log_dir = os.getenv('CUSTOM_LOG_DIR')
        if custom_log_dir:
            csv_file = f"{custom_log_dir}/llm_api_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            csv_file = f"./logs/llm_api_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        _llm_csv_logger = LLMCSVLogger(csv_file)
    
    return _llm_csv_logger


def update_api_stats(provider_name: str, call_type: str, duration: float, 
                    metadata: Dict[str, Any], success: bool):
    """Update global API call statistics"""
    global api_call_stats
    
    api_call_stats["total_calls"] += 1
    api_call_stats["total_duration"] += duration
    
    if success:
        api_call_stats["successful_calls"] += 1
    else:
        api_call_stats["failed_calls"] += 1
    
    # Extract token counts from metadata
    if "prompt_tokens" in metadata:
        api_call_stats["total_input_tokens"] += metadata.get("prompt_tokens", 0)
        api_call_stats["total_output_tokens"] += metadata.get("completion_tokens", 0)
    elif "input_tokens" in metadata:
        api_call_stats["total_input_tokens"] += metadata.get("input_tokens", 0)
        api_call_stats["total_output_tokens"] += metadata.get("output_tokens", 0)
    elif "prompt_token_count" in metadata:
        api_call_stats["total_input_tokens"] += metadata.get("prompt_token_count", 0)
        api_call_stats["total_output_tokens"] += metadata.get("candidates_token_count", 0)
    
    # Track per-provider stats
    if provider_name not in api_call_stats["providers"]:
        api_call_stats["providers"][provider_name] = {
            "calls": 0,
            "successful": 0,
            "failed": 0,
            "duration": 0.0
        }
    
    provider_stats = api_call_stats["providers"][provider_name]
    provider_stats["calls"] += 1
    provider_stats["duration"] += duration
    if success:
        provider_stats["successful"] += 1
    else:
        provider_stats["failed"] += 1


def log_api_stats_summary():
    """Log summary of API call statistics"""
    session_duration = (datetime.now() - api_call_stats["session_start"]).total_seconds()
    
    summary = f"""
{'='*80}
LLM API CALL STATISTICS SUMMARY
{'='*80}
Session Duration: {session_duration:.1f}s
Total API Calls: {api_call_stats['total_calls']}
Successful Calls: {api_call_stats['successful_calls']}
Failed Calls: {api_call_stats['failed_calls']}
Success Rate: {(api_call_stats['successful_calls']/max(api_call_stats['total_calls'], 1)*100):.1f}%
Total API Duration: {api_call_stats['total_duration']:.3f}s
Average Call Duration: {(api_call_stats['total_duration']/max(api_call_stats['total_calls'], 1)):.3f}s
Total Input Tokens: {api_call_stats['total_input_tokens']:,}
Total Output Tokens: {api_call_stats['total_output_tokens']:,}
Total Tokens: {(api_call_stats['total_input_tokens'] + api_call_stats['total_output_tokens']):,}

PROVIDER BREAKDOWN:
"""
    
    for provider, stats in api_call_stats["providers"].items():
        summary += f"""
{provider}:
  - Calls: {stats['calls']} ({(stats['calls']/max(api_call_stats['total_calls'], 1)*100):.1f}%)
  - Success Rate: {(stats['successful']/max(stats['calls'], 1)*100):.1f}%
  - Total Duration: {stats['duration']:.3f}s
  - Avg Duration: {(stats['duration']/max(stats['calls'], 1)):.3f}s
"""
    
    summary += "="*80
    llm_logger.info(summary)


def log_llm_call(provider_name: str, call_type: str, input_data: Any, output_data: Any, 
                 metadata: Dict[str, Any], start_time: float, end_time: float):
    """Log LLM API call with essential information (no verbose chat messages)"""
    duration = end_time - start_time
    success = "error" not in call_type.lower()
    
    # Update statistics
    update_api_stats(provider_name, call_type, duration, metadata, success)
    
    # Log complete raw data to CSV
    try:
        csv_logger = get_llm_csv_logger()
        csv_logger.log_api_call(
            provider_name=provider_name,
            call_type=call_type,
            raw_input=input_data,
            raw_output=output_data,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
            success=success
        )
    except Exception as e:
        print(f"Failed to log to CSV: {e}")
    
#     # Create summary input data without verbose messages
#     input_summary = {}
#     if isinstance(input_data, dict):
#         for key, value in input_data.items():
#             if key == "messages" and isinstance(value, list):
#                 # Summarize messages instead of logging full content
#                 input_summary[key] = f"[{len(value)} messages: {', '.join([msg.get('role', 'unknown') for msg in value if isinstance(msg, dict)])}]"
#             elif key == "payload" and isinstance(value, dict):
#                 # Summarize payload without messages
#                 payload_summary = {k: v for k, v in value.items() if k != "messages"}
#                 if "messages" in value:
#                     payload_summary["messages"] = f"[{len(value['messages'])} messages]"
#                 input_summary[key] = payload_summary
#             elif key == "tools" and isinstance(value, list):
#                 # Summarize tools
#                 input_summary[key] = f"[{len(value)} tools: {', '.join([tool.get('name', 'unknown') if isinstance(tool, dict) else str(tool) for tool in value])}]"
#             else:
#                 input_summary[key] = value
#     else:
#         input_summary = input_data
    
#     # Create summary output data without verbose responses
#     output_summary = {}
#     if isinstance(output_data, dict):
#         for key, value in output_data.items():
#             if key == "response" and isinstance(value, str) and len(value) > 200:
#                 # Truncate long responses
#                 output_summary[key] = f"{value[:200]}... [truncated, total length: {len(value)}]"
#             elif key == "content" and isinstance(value, str) and len(value) > 200:
#                 # Truncate long content
#                 output_summary[key] = f"{value[:200]}... [truncated, total length: {len(value)}]"
#             else:
#                 output_summary[key] = value
#     else:
#         output_summary = output_data
    
#     # Format the log entry concisely
#     log_message = f"""
# {'='*80}
# PROVIDER: {provider_name}
# CALL_TYPE: {call_type}
# TIMESTAMP: {datetime.fromtimestamp(start_time).isoformat()}
# DURATION: {duration:.3f}s
# METADATA: {json.dumps(metadata, indent=2, ensure_ascii=False)}
# INPUT: {json.dumps(input_summary, indent=2, ensure_ascii=False)}
# OUTPUT: {json.dumps(output_summary, indent=2, ensure_ascii=False)}
# {'='*80}
# """
    
#     llm_logger.info(log_message)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable"""
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Network/connection errors - always retryable
    if any(keyword in error_str for keyword in [
        "connection", "timeout", "network", "dns", "socket", 
        "temporary", "temporarily", "503", "502", "500", "429"
    ]):
        return True
    
    # Rate limiting errors - retryable
    if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
        return True
    
    # Server errors - retryable
    if any(keyword in error_str for keyword in ["server error", "internal error", "service unavailable"]):
        return True
    
    # OpenAI specific errors
    if hasattr(error, 'status_code'):
        return error.status_code in [429, 500, 502, 503, 504]
    
    # Anthropic specific errors
    if error_type in ['RateLimitError', 'InternalServerError', 'APIConnectionError']:
        return True
    
    # Authentication/permission errors - not retryable
    if any(keyword in error_str for keyword in [
        "unauthorized", "forbidden", "invalid api key", "authentication", 
        "permission", "401", "403"
    ]):
        return False
    
    # Client errors (4xx except 429) - generally not retryable
    if any(keyword in error_str for keyword in ["400", "404", "422"]):
        return False
    
    # Default: retry for unknown errors
    return True


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True
):
    """Execute function with exponential backoff retry"""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                # Last attempt failed
                break
            
            if not is_retryable_error(e):
                # Non-retryable error, don't retry
                system_logger.warning(f"Non-retryable error encountered: {e}")
                raise
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            
            system_logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.2f} seconds..."
            )
            
            await asyncio.sleep(delay)
    
    # All retries exhausted
    system_logger.error(f"All {max_retries + 1} attempts failed. Last error: {last_exception}")
    raise last_exception


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self):
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 10.0
        self.backoff_factor = 2.0
        self.jitter = True
    
    @abstractmethod
    async def _generate_response_impl(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Implementation of generate response (without retry logic)"""
        pass
    
    @abstractmethod
    async def _generate_response_with_tools_impl(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Implementation of generate response with tools (without retry logic)"""
        pass
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response with retry mechanism"""
        # Extract retry parameters from kwargs
        max_retries = kwargs.pop('max_retries', self.max_retries)
        base_delay = kwargs.pop('base_delay', self.base_delay)
        max_delay = kwargs.pop('max_delay', self.max_delay)
        backoff_factor = kwargs.pop('backoff_factor', self.backoff_factor)
        jitter = kwargs.pop('jitter', self.jitter)
        
        async def _call():
            return await self._generate_response_impl(messages, **kwargs)
        
        return await retry_with_backoff(
            _call,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            jitter=jitter
        )
    
    async def generate_response_with_tools(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate response with tools using retry mechanism"""
        # Extract retry parameters from kwargs
        max_retries = kwargs.pop('max_retries', self.max_retries)
        base_delay = kwargs.pop('base_delay', self.base_delay)
        max_delay = kwargs.pop('max_delay', self.max_delay)
        backoff_factor = kwargs.pop('backoff_factor', self.backoff_factor)
        jitter = kwargs.pop('jitter', self.jitter)
        
        async def _call():
            return await self._generate_response_with_tools_impl(messages, tools, **kwargs)
        
        return await retry_with_backoff(
            _call,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            jitter=jitter
        )
    
    def convert_mcp_tools_to_provider_format(self, mcp_tools: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to provider-specific format. Override in subclasses."""
        return []
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name"""
        pass
