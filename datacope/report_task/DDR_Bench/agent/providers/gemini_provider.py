#!/usr/bin/env python3
"""
Google Gemini API Provider
"""

import asyncio
import json
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types

from .base import LLMProvider, log_llm_call, system_logger


class GeminiProvider(LLMProvider):
    """
    Google Gemini API provider.
    
    Supports both Gemini 2.x and Gemini 3.x models with optional thinking mode.
    
    Args:
        api_key: Google API key
        model: Model name (e.g., 'gemini-2.5-flash', 'gemini-3-flash-preview')
        thinking_budget: Optional thinking token budget for Gemini 3 models
    """
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", thinking_budget: int = None):
        super().__init__()
        try:
            self.client = genai.Client(api_key=api_key)
            self.model_name = model
            self.thinking_budget = thinking_budget
            system_logger.info(f"Gemini provider initialized: {model}")
            if thinking_budget:
                system_logger.info(f"Thinking budget: {thinking_budget}")
        except Exception as e:
            system_logger.error(f"Failed to initialize Gemini provider: {e}")
            raise
    
    async def _generate_response_impl(self, messages: List[Dict[str, str]], **kwargs) -> str:
        start_time = time.time()
        
        # Convert message format to Gemini format
        contents = self._convert_messages_to_contents(messages)
        
        # Create configuration for simple text generation
        config = types.GenerateContentConfig(
            max_output_tokens=kwargs.get("max_tokens", 16384),
            temperature=kwargs.get("temperature", 0.7),
            # Disable automatic function calling
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
            # No thinking_config for simple text generation
        )
        
        input_data = {
            "model": self.model_name,
            # JSON-safe full copy of original messages for logging visibility
            "contents": self._summarize_messages(messages),
            "config": {
                "max_output_tokens": kwargs.get("max_tokens", 16384),
                "temperature": kwargs.get("temperature", 0.7)
            }
        }
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            end_time = time.time()
            
            # Safely get response text
            output_text = ""
            try:
                if hasattr(response, 'text') and response.text:
                    output_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    # Try to get text from first candidate
                    first_candidate = response.candidates[0]
                    
                    # Extract main content
                    if hasattr(first_candidate, 'content') and first_candidate.content:
                        if hasattr(first_candidate.content, 'parts') and first_candidate.content.parts:
                            for part in first_candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    output_text += part.text
            except Exception as e:
                system_logger.warning(f"Failed to extract text from response: {e}")
                output_text = ""
            
            # Extract metadata with safe null checks
            metadata = {
                "model": self.model_name,
                "prompt_token_count": 0,
                "candidates_token_count": 0, 
                "total_token_count": 0,
                "finish_reason": "UNKNOWN",
                "safety_ratings": []
            }
            
            # Safely extract usage metadata
            try:
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    # Coerce None to 0 to avoid int += NoneType errors downstream
                    metadata["prompt_token_count"] = int((getattr(usage, 'prompt_token_count', 0) or 0))
                    metadata["candidates_token_count"] = int((getattr(usage, 'candidates_token_count', 0) or 0))
                    metadata["total_token_count"] = int((getattr(usage, 'total_token_count', 0) or 0))
            except Exception as e:
                system_logger.warning(f"Failed to extract usage metadata: {e}")
            
            # Safely extract finish reason
            try:
                if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        if hasattr(candidate.finish_reason, 'name'):
                            metadata["finish_reason"] = candidate.finish_reason.name
                        else:
                            metadata["finish_reason"] = str(candidate.finish_reason)
            except Exception as e:
                system_logger.warning(f"Failed to extract finish reason: {e}")
            
            # Safely extract safety ratings
            try:
                if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        metadata["safety_ratings"] = [
                            {
                                "category": getattr(rating.category, 'name', str(rating.category)) if hasattr(rating, 'category') else "UNKNOWN",
                                "probability": getattr(rating.probability, 'name', str(rating.probability)) if hasattr(rating, 'probability') else "UNKNOWN"
                            } for rating in candidate.safety_ratings
                        ]
            except Exception as e:
                system_logger.warning(f"Failed to extract safety ratings: {e}")
            
            # Log the API call
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="generate_content",
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
                "model": self.model_name,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            # Log the failed API call
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="generate_content_error",
                input_data=input_data,
                output_data={"error": str(e)},
                metadata=error_metadata,
                start_time=start_time,
                end_time=end_time
            )
            
            system_logger.error(f"Gemini API error: {e}")
            raise
    
    async def _generate_response_with_tools_impl(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        # Convert message format to Gemini format
        contents = self._convert_messages_to_contents(messages)
        
        # Convert tools to Gemini format
        gemini_tools = self.convert_mcp_tools_to_provider_format(tools)
        try:
            tool_names = [t.name for t in gemini_tools] if gemini_tools else []
            system_logger.info(f"Gemini tools prepared: {tool_names}")
        except Exception:
            system_logger.info("Gemini tools prepared: [unavailable]")
        
        # Create configuration with tools
        config = types.GenerateContentConfig(
            max_output_tokens=kwargs.get("max_tokens", 16384),
            temperature=kwargs.get("temperature", 0.7),
            tools=gemini_tools if gemini_tools else None,
            # Disable automatic function calling - tools are managed manually
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
            # No thinking_config - thinking mode disabled
        )
        
        # Build JSON-safe input summary for logging (avoid non-serializable Tool objects)
        input_data = {
            "model": self.model_name,
            "contents": self._summarize_messages(messages),
            "config": {
                "max_output_tokens": kwargs.get("max_tokens", 16384),
                "temperature": kwargs.get("temperature", 0.7),
                "tools": self._summarize_gemini_tools(gemini_tools)
            }
        }
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            end_time = time.time()
            
            # Extract content and function calls
            content = ""
            tool_calls = []
            
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    first_candidate = response.candidates[0]
                    
                    # Extract main content and function calls from content parts
                    if hasattr(first_candidate, 'content') and first_candidate.content:
                        if hasattr(first_candidate.content, 'parts') and first_candidate.content.parts:
                            for part in first_candidate.content.parts:
                                # Skip thinking content (thought=True)
                                if hasattr(part, 'thought') and part.thought:
                                    content += part.thought
                                if hasattr(part, 'text') and part.text:
                                    content += part.text
                                if hasattr(part, 'function_call') and part.function_call:
                                    # Handle function call
                                    func_call = part.function_call
                                    system_logger.info(f"Gemini function_call detected: {getattr(func_call, 'name', '')}")
                                    tool_calls.append({
                                        "id": f"call_{len(tool_calls)}",
                                        "type": "function",
                                        "function": {
                                            "name": func_call.name,
                                            "arguments": json.dumps(dict(func_call.args)) if getattr(func_call, 'args', None) else "{}"
                                        }
                                    })
            except Exception as e:
                system_logger.warning(f"Failed to extract content/function calls from response: {e}")
            
            # Determine finish signal (for auto-finish workflows)
            finish_flag = False
            try:
                if content and "FINISH" in content:
                    finish_flag = True
            except Exception:
                finish_flag = False
            
            result = {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": self._get_finish_reason(response),
                "finish": finish_flag
            }
            
            # Add usage information if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                result["usage"] = {
                    # Coerce None to 0 for all token counts
                    "prompt_tokens": int((getattr(usage, 'prompt_token_count', 0) or 0)),
                    "completion_tokens": int((getattr(usage, 'candidates_token_count', 0) or 0)),
                    "total_tokens": int((getattr(usage, 'total_token_count', 0) or 0))
                }
            else:
                result["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            
            # Extract metadata with safe null checks
            metadata = {
                "model": self.model_name,
                "prompt_token_count": 0,
                "candidates_token_count": 0, 
                "total_token_count": 0,
                "finish_reason": result["finish_reason"],
                "safety_ratings": [],
                "tool_calls_count": len(tool_calls)
            }
            
            # Safely extract usage metadata
            try:
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    metadata["prompt_token_count"] = int((getattr(usage, 'prompt_token_count', 0) or 0))
                    metadata["candidates_token_count"] = int((getattr(usage, 'candidates_token_count', 0) or 0))
                    metadata["total_token_count"] = int((getattr(usage, 'total_token_count', 0) or 0))
            except Exception as e:
                system_logger.warning(f"Failed to extract usage metadata: {e}")
            
            # Log the API call
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="generate_content_with_tools",
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
                "model": self.model_name,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            # Log the failed API call
            log_llm_call(
                provider_name=self.get_provider_name(),
                call_type="generate_content_with_tools_error",
                input_data=input_data,
                output_data={"error": str(e)},
                metadata=error_metadata,
                start_time=start_time,
                end_time=end_time
            )
            
            system_logger.error(f"Gemini API error: {e}")
            raise
    
    def convert_mcp_tools_to_provider_format(self, mcp_tools: List[Dict[str, Any]]) -> List[types.Tool]:
        """Convert MCP tools to Gemini function declarations format"""
        if not mcp_tools:
            return []
        
        function_declarations = []
        
        for tool in mcp_tools:
            func_decl = types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            )
            function_declarations.append(func_decl)
        
        # Return a single Tool containing all function declarations
        return [types.Tool(function_declarations=function_declarations)]
    
    def _get_finish_reason(self, response) -> str:
        """Safely extract finish reason from response"""
        try:
            if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    if hasattr(candidate.finish_reason, 'name'):
                        return candidate.finish_reason.name
                    else:
                        return str(candidate.finish_reason)
        except Exception as e:
            system_logger.warning(f"Failed to extract finish reason: {e}")
        return "UNKNOWN"
    
    def _convert_messages_to_contents(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        """Convert messages to Gemini contents format with proper roles and tool calls"""
        contents = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            # Map roles to Gemini format
            if role in ("assistant", "model"):
                gemini_role = "model"
            else:
                # Treat system, user, and tool as user role
                gemini_role = "user"
            
            # Build parts list
            parts = []
            
            # Handle tool role messages specially - only add function_response, no text
            if role == "tool" and "name" in msg and "tool_call_id" in msg:
                # Extract tool name from the message
                tool_name = msg.get("name", "")
                tool_result_str = msg.get("content", "{}")
                
                # Parse result
                try:
                    tool_result = json.loads(tool_result_str) if isinstance(tool_result_str, str) else tool_result_str
                except:
                    tool_result = tool_result_str
                
                # Create function response part
                parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=tool_name,
                        response=tool_result
                    )
                ))
            else:
                # For non-tool messages, add text content if present
                if content:
                    parts.append(types.Part(text=content))
                
                # Add tool calls if present (for assistant messages)
                if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                    for tool_call in msg["tool_calls"]:
                        func_name = tool_call.get("function", {}).get("name", "")
                        func_args_str = tool_call.get("function", {}).get("arguments", "{}")
                        
                        # Parse arguments
                        try:
                            func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                        except:
                            func_args = {}
                        
                        # Create function call part
                        parts.append(types.Part(
                            function_call=types.FunctionCall(
                                name=func_name,
                                args=func_args
                            )
                        ))
            
            # Only add content if there are parts
            if parts:
                contents.append(types.Content(
                    role=gemini_role,
                    parts=parts
                ))
        
        return contents
    
    def get_provider_name(self) -> str:
        return f"Gemini-{self.model_name}"

    def _summarize_gemini_tools(self, gemini_tools: List[Any]) -> Any:
        """Create a JSON-serializable summary for Gemini tools for logging."""
        try:
            if not gemini_tools:
                return []
            summaries = []
            for tool in gemini_tools:
                try:
                    fns = []
                    fdecls = getattr(tool, 'function_declarations', None)
                    if fdecls:
                        for fd in fdecls:
                            try:
                                fns.append({
                                    "name": getattr(fd, 'name', None),
                                    "has_parameters": bool(getattr(fd, 'parameters', None))
                                })
                            except Exception:
                                continue
                    summaries.append({
                        "tool": getattr(tool, 'name', None),
                        "function_declarations": fns
                    })
                except Exception:
                    summaries.append({"tool": None, "function_declarations": []})
            return summaries
        except Exception:
            return "[tools summary unavailable]"

    def _summarize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Return a JSON-serializable, full copy of messages for logging (no truncation)."""
        summary: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            summary.append({
                "role": role,
                "content": content
            })
        return summary
