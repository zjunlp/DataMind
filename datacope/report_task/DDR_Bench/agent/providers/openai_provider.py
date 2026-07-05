#!/usr/bin/env python3
"""
OpenAI API Provider
"""

import json
import os
import time
from typing import Any, Dict, List

try:
    # Prefer modern OpenAI SDK classes
    from openai import AsyncOpenAI, AsyncAzureOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore
    AsyncAzureOpenAI = None  # type: ignore

from .base import LLMProvider, log_llm_call, system_logger


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str = "", model: str = "gpt-4o", azure_endpoint: str = "", azure_api_version: str = "2024-12-01-preview", use_azure: bool = True):
        super().__init__()
        if AsyncOpenAI is None or AsyncAzureOpenAI is None:
            raise RuntimeError("OpenAI SDK not available. Please install openai>=1.0.0.")
        
        # Resolve credentials from args or environment
        resolved_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        resolved_azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        resolved_azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        # Default to Azure OpenAI if no explicit choice and we have Azure credentials
        if use_azure and resolved_azure_endpoint and resolved_api_key:
            self.is_azure = True
        else:
            if not resolved_api_key:
                raise RuntimeError("API key is required. Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable.")
            self.is_azure = False
        
        self.model = model
        # Store resolved creds for per-call client construction to ensure clean aclose
        self._resolved_api_key = resolved_api_key
        self._resolved_azure_endpoint = resolved_azure_endpoint
        self._resolved_azure_api_version = resolved_azure_api_version
    
    async def _generate_response_impl(self, messages: List[Dict[str, str]], **kwargs) -> str:
        start_time = time.time()
        input_data = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": kwargs.get("max_completion_tokens", 16384),
        }
        
        try:
            # Create a short-lived async client and ensure proper close within active loop
            if self.is_azure:
                async with AsyncAzureOpenAI(
                    api_version=self._resolved_azure_api_version,
                    azure_endpoint=self._resolved_azure_endpoint,
                    api_key=self._resolved_api_key
                ) as client:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=kwargs.get("max_completion_tokens", 16384),
                        top_p=kwargs.get("top_p", 1),
                        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                        presence_penalty=kwargs.get("presence_penalty", 0.0)
                    )
            else:
                async with AsyncOpenAI(api_key=self._resolved_api_key) as client:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=kwargs.get("max_completion_tokens", 16384),
                        top_p=kwargs.get("top_p", 1),
                        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                        presence_penalty=kwargs.get("presence_penalty", 0.0)
                    )
            
            end_time = time.time()
            output_text = response.choices[0].message.content
            
            # Extract metadata
            metadata = {
                "model": self.model,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
                "created": response.created
            }
            
            # Log the API call
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="chat_completion",
                input_data=input_data,
                output_data={"response": output_text},
                metadata=metadata,
                start_time=start_time,
                end_time=end_time
            )
            
            return output_text
            
        except Exception as e:
            end_time = time.time()
            error_metadata = {
                "model": self.model,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            # Log the failed API call
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="chat_completion_error",
                input_data=input_data,
                output_data={"error": str(e)},
                metadata=error_metadata,
                start_time=start_time,
                end_time=end_time
            )
            
            system_logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _generate_response_with_tools_impl(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        # Convert tools to OpenAI format
        openai_tools = self.convert_mcp_tools_to_provider_format(tools)
        
        input_data = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": kwargs.get("max_completion_tokens", 16384),
            "tools": openai_tools,
            "tool_choice": "auto"
        }
        
        try:
            if self.is_azure:
                async with AsyncAzureOpenAI(
                    api_version=self._resolved_azure_api_version,
                    azure_endpoint=self._resolved_azure_endpoint,
                    api_key=self._resolved_api_key
                ) as client:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=kwargs.get("max_completion_tokens", 16384),
                        top_p=kwargs.get("top_p", 1),
                        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                        presence_penalty=kwargs.get("presence_penalty", 0.0),
                        tools=openai_tools,
                        tool_choice=kwargs.get("tool_choice", "auto")
                    )
            else:
                async with AsyncOpenAI(api_key=self._resolved_api_key) as client:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=kwargs.get("max_completion_tokens", 16384),
                        top_p=kwargs.get("top_p", 1),
                        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                        presence_penalty=kwargs.get("presence_penalty", 0.0),
                        tools=openai_tools,
                        tool_choice=kwargs.get("tool_choice", "auto")
                    )
            
            end_time = time.time()
            
            # Extract message and tool calls
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = []
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            result = {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
            
            # Extract metadata
            metadata = {
                "model": self.model,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
                "created": response.created,
                "tool_calls_count": len(tool_calls)
            }
            
            # Log the API call
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="chat_completion_with_tools",
                input_data=input_data,
                output_data=result,
                metadata=metadata,
                start_time=start_time,
                end_time=end_time
            )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            error_metadata = {
                "model": self.model,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            # Log the failed API call
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="chat_completion_with_tools_error",
                input_data=input_data,
                output_data={"error": str(e)},
                metadata=error_metadata,
                start_time=start_time,
                end_time=end_time
            )
            
            system_logger.error(f"OpenAI API error: {e}")
            raise
    
    def convert_mcp_tools_to_provider_format(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format"""
        openai_tools = []
        
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                }
            }
            
            # Add parameters if available
            if "inputSchema" in tool and tool["inputSchema"]:
                openai_tool["function"]["parameters"] = tool["inputSchema"]
            else:
                # Default empty parameters
                openai_tool["function"]["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    def get_provider_name(self) -> str:
        provider_type = "Azure-OpenAI" if self.is_azure else "OpenAI"
        return f"{provider_type}-{self.model}"
