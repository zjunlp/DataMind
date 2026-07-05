#!/usr/bin/env python3
"""
MiniMax API Provider (OpenAI-compatible Chat Completions API)
"""

import json
import time
from typing import Any, Dict, List

try:
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

from .base import LLMProvider, log_llm_call, system_logger


class MiniMaxProvider(LLMProvider):
    """MiniMax API provider via OpenAI-compatible Chat Completions API"""
    
    def __init__(self, api_key: str, model: str = "MiniMax-M2"):
        super().__init__()
        self.api_key = api_key
        # Hardcoded MiniMax OpenAI-compatible base URL
        self.base_url = "https://api.minimaxi.com/v1"
        self.model = model
        if AsyncOpenAI is None:
            raise RuntimeError("OpenAI SDK not available. Please install openai>=1.0.0.")
    
    def _debug_dump_response_meta(self, response: Any, label: str = "") -> None:
        try:
            print(f"[MiniMax DEBUG] === META {label} ===")
            core = {
                "id": getattr(response, "id", None),
                "type": getattr(response, "type", None),
                "role": getattr(response, "role", None),
                "model": getattr(response, "model", None),
                "stop_reason": getattr(response, "stop_reason", None),
                "stop_sequence": getattr(response, "stop_sequence", None),
            }
            print("[MiniMax DEBUG] core:", core)
            usage = getattr(response, "usage", None)
            if usage is not None:
                usage_dict = {}
                for k in [
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                    "server_tool_use",
                    "service_tier",
                ]:
                    usage_dict[k] = getattr(usage, k, None)
                print("[MiniMax DEBUG] usage:", usage_dict)
            else:
                print("[MiniMax DEBUG] usage: None")
            base_resp = getattr(response, "base_resp", None)
            if base_resp is not None:
                base_dict = {
                    "status_code": getattr(base_resp, "status_code", None),
                    "status_msg": getattr(base_resp, "status_msg", None),
                }
                print("[MiniMax DEBUG] base_resp:", base_dict)
            else:
                print("[MiniMax DEBUG] base_resp: None")
        except Exception as e:
            print(f"[MiniMax DEBUG] meta dump failed: {e}")
    
    async def _generate_response_impl(self, messages: List[Dict[str, str]], **kwargs) -> str:
        start_time = time.time()
        # Ensure at least one non-system message (MiniMax OpenAI API requires it)
        if not any((m.get("role") or "").lower() != "system" for m in messages):
            print("[MiniMax DEBUG] No non-system messages; injecting minimal user message.")
            messages = messages + [{"role": "user", "content": "Please begin."}]
        
        input_data = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 16384)),
        }
        try:
            async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 16384)),
                    top_p=kwargs.get("top_p", 1),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                    presence_penalty=kwargs.get("presence_penalty", 0.0)
                )
            end_time = time.time()
            if not getattr(response, "choices", None):
                raise RuntimeError("MiniMax returned empty choices for non-tools request")
            output_text = response.choices[0].message.content or ""
            metadata = {
                "model": self.model,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
                "created": response.created,
                "base_url": self.base_url
            }
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
            error_metadata = {"model": self.model, "error": str(e), "error_type": type(e).__name__, "base_url": self.base_url}
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="chat_completion_error",
                input_data=input_data,
                output_data={"error": str(e)},
                metadata=error_metadata,
                start_time=start_time,
                end_time=end_time
            )
            system_logger.error(f"MiniMax API error: {e}")
            raise
    
    async def _generate_response_with_tools_impl(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        # Convert tools to OpenAI format
        openai_tools = self.convert_mcp_tools_to_provider_format(tools)
        # Ensure at least one non-system message
        if not any((m.get("role") or "").lower() != "system" for m in messages):
            print("[MiniMax DEBUG] No non-system messages (tools); injecting minimal user message.")
            messages = messages + [{"role": "user", "content": "Please begin and decide which tool to use first."}]
        input_data = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 16384)),
            "tools": openai_tools,
            "tool_choice": kwargs.get("tool_choice", "auto")
        }
        try:
            async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 16384)),
                    top_p=kwargs.get("top_p", 1),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                    presence_penalty=kwargs.get("presence_penalty", 0.0),
                    tools=openai_tools,
                    tool_choice=kwargs.get("tool_choice", "auto")
                )
            end_time = time.time()
            if not getattr(response, "choices", None):
                raise RuntimeError("MiniMax returned empty choices for tools request")
            message = response.choices[0].message
            content = message.content or ""
            tool_calls: List[Dict[str, Any]] = []
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
            metadata = {
                "model": self.model,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
                "created": response.created,
                "tool_calls_count": len(tool_calls),
                "base_url": self.base_url
            }
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
            error_metadata = {"model": self.model, "error": str(e), "error_type": type(e).__name__, "base_url": self.base_url}
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="chat_completion_with_tools_error",
                input_data=input_data,
                output_data={"error": str(e)},
                metadata=error_metadata,
                start_time=start_time,
                end_time=end_time
            )
            system_logger.error(f"MiniMax API error: {e}")
            raise
    
    def convert_mcp_tools_to_provider_format(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format"""
        openai_tools: List[Dict[str, Any]] = []
        for tool in mcp_tools:
            openai_tool: Dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                }
            }
            if "inputSchema" in tool and tool["inputSchema"]:
                openai_tool["function"]["parameters"] = tool["inputSchema"]
            else:
                openai_tool["function"]["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            openai_tools.append(openai_tool)
        return openai_tools
    
    def get_provider_name(self) -> str:
        return f"MiniMax-{self.model}"


