import time
import asyncio
import traceback
import json
from typing import Dict, Any, List, Optional

from src.runtimes.base_agent import BaseAgent
from src.runtimes.registry import register_agent
from src.core.schema import AgentResult

AGENT_TYPE = "claude_code"

DEFAULT_ALLOWED_TOOLS = [
    "Read", "Write", "Bash", "Glob", "Grep", "Edit",
    "TodoWrite", "BashOutput", "Skill"
]

DEFAULT_DISALLOWED_TOOLS = ["Task", "WebFetch", "WebSearch",]

@register_agent(AGENT_TYPE)
class ClaudeCodeAgent(BaseAgent):
    """Autonomous data science agent using Claude Agent SDK.

    Claude Code is a full agent with built-in bash/file/edit tools. Instead of
    the manual backend + environment loop used by ReActDSAgent, this agent sends
    the task to Claude and lets it run autonomously.
    """

    def __init__(self, backend: str, model: str, **kwargs):
        super().__init__(AGENT_TYPE, model, backend, **kwargs)

        self.working_dir = kwargs.get('working_dir', '.')
        self.output_schema = kwargs.get('output_schema', {})
        self.timeout = kwargs.get('timeout', None)

        self.setting_sources: List[str] = kwargs.get(
            'setting_sources', ["user", "project"]
        )
        self.permission_mode: str = kwargs.get(
            'permission_mode', 'acceptEdits'
        )
        self.max_buffer_size: Optional[int] = kwargs.get(
            'max_buffer_size', None
        )
        self.allowed_tools: List[str] = kwargs.get('tools', DEFAULT_ALLOWED_TOOLS)

        self.disallowed_tools: List[str] = kwargs.get('disallowed_tools', DEFAULT_DISALLOWED_TOOLS)

        self.max_turns: Optional[int] = kwargs.get('max_turns', None)

    # ------------------------------------------------------------------
    # Claude SDK interaction
    # ------------------------------------------------------------------

    def _build_claude_options(self, system: str = "") -> Any:
        """Build ClaudeAgentOptions for a Claude SDK session."""
        from claude_agent_sdk import ClaudeAgentOptions

        system_prompt: Dict[str, Any] = {
            "type": "preset",
            "preset": "claude_code",
        }

        if system:
            system_prompt["append"] = system

        kwargs: Dict[str, Any] = {
            "system_prompt": system_prompt,
            "allowed_tools": list(self.allowed_tools),
            "disallowed_tools": list(self.disallowed_tools),
            "cwd": str(self.working_dir),
            "setting_sources": self.setting_sources,
            "permission_mode": self.permission_mode,
            "max_turns": self.max_turns,
        }
        if self.output_schema:
            kwargs["output_format"] = {
                "type": "json_schema",
                "schema": self.output_schema,
            }
        if self.max_buffer_size is not None:
            kwargs["max_buffer_size"] = self.max_buffer_size

        options = ClaudeAgentOptions(**kwargs)
        if self.model:
            model_id = self.model
            if not model_id.startswith("claude-") and "claude" in model_id.lower():
                pass
            options.model = model_id
        return options

    async def _run_claude(self, prompt: str, system: str) -> Dict[str, Any]:
        """Run a single Claude autonomous session and return parsed results."""

        from claude_agent_sdk import ClaudeSDKClient

        options = self._build_claude_options(system=system)

        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(prompt)
                messages = [msg async for msg in client.receive_response()]
        except Exception as e:
            raise RuntimeError(f"Claude SDK error: {e}")

        if not messages:
            return {
                "final_response": "",
                "uuid": "",
                "session_id": "",
                "model": self.model or "",
                "tools": [],
                "duration_ms": 0,
                "total_cost_usd": 0.0,
                "num_turns": 0,
                "usage": {},
                "result": "",
                "is_error": True,
                "structured_output": None,
                "messages": [],
            }

        first = messages[0]
        last = messages[-1]

        return {
            "final_response": getattr(last, 'result', '') or '',
            "uuid": first.data.get("uuid", "") if hasattr(first, 'data') else "",
            "session_id": getattr(last, 'session_id', '') or "",
            "model": first.data.get("model", "") if hasattr(first, 'data') else "",
            "tools": first.data.get("tools", []) if hasattr(first, 'data') else [],
            "duration_ms": getattr(last, 'duration_ms', 0) or 0,
            "total_cost_usd": getattr(last, 'total_cost_usd', 0.0) or 0.0,
            "num_turns": getattr(last, 'num_turns', 0) or 0,
            "usage": getattr(last, 'usage', {}) or {},
            "result": getattr(last, 'result', '') or '',
            "is_error": getattr(last, 'is_error', False),
            "structured_output": getattr(last, 'structured_output', None),
            "messages": messages,
        }


    def _run_claude_sync(self, prompt: str, system: str) -> Dict[str, Any]:
        async def runner():
            if self.timeout is None:
                return await self._run_claude(prompt, system)
            return await asyncio.wait_for(
                self._run_claude(prompt, system),
                timeout=self.timeout,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            outer_timeout = None if self.timeout is None else self.timeout + 30
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(lambda: asyncio.run(runner()))
                return future.result(timeout=outer_timeout)

        return asyncio.run(runner())

    @staticmethod
    def _extract_answer(structured_output: Any) -> Optional[str]:
        """Extract the final answer from Claude's structured output."""
        if structured_output is None:
            return None

        if isinstance(structured_output, dict):
            answer = structured_output.get("answer", "")
            return answer.strip() if answer and answer.strip() else None

        if isinstance(structured_output, str):
            try:
                parsed = json.loads(structured_output)
                answer = parsed.get("answer", "")
                return answer.strip() if answer and answer.strip() else None
            except (json.JSONDecodeError, AttributeError):
                return structured_output.strip() or None

        return str(structured_output).strip() or None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def solve_task(self, prompt: str, system: str = "",  **kwargs) -> AgentResult:
        start_time = time.time()
        actual_turns = 0

        try:
            result = self._run_claude_sync(prompt, system)
            messages = result.get("messages", [])
            actual_turns = result.get("num_turns", 0)

            agent_conversation = []
            for msg in messages:
                msg_type = type(msg).__name__
                if msg_type in ["SystemMessage"]:
                    continue
                elif msg_type in ["AssistantMessage"]:
                    content = getattr(msg, 'content', [])
                    for block in content:
                        block_type = type(block).__name__
                        if block_type == "ThinkingBlock":
                            thinking = getattr(block, 'thinking', '')
                            agent_conversation.append({
                                "type": msg_type,
                                "block_type": block_type,
                                "text": thinking.strip() if thinking else '',
                            })
                        elif block_type == "TextBlock":
                            text = getattr(block, 'text', '')
                            agent_conversation.append({
                                "type": msg_type,
                                "block_type": block_type,
                                "text": text.strip() if text else '',
                            })
                        elif block_type == "ToolUseBlock":
                            tool_name = getattr(block, 'name', '')
                            tool_input = getattr(block, 'input', {})
                            agent_conversation.append({
                                "type": msg_type,
                                "block_type": block_type,
                                "tool_name": tool_name,
                                "tool_input": tool_input,
                            })
                        else:
                            agent_conversation.append({
                                "type": msg_type,
                                "block_type": block_type,
                                "content": getattr(block, 'content', f"The block type {block_type} of {msg_type} is not recognized"),
                            })
                elif msg_type in ["UserMessage"]:
                    content = getattr(msg, 'content', [])
                    for block in content:
                        block_type = type(block).__name__
                        if block_type == "ToolResultBlock":
                            tool_result = getattr(block, 'content', '')
                            is_error = getattr(block, 'is_error', False)
                            agent_conversation.append({
                                "type": msg_type,
                                "block_type": block_type,
                                "tool_result": tool_result,
                                "is_error": is_error,
                            })
                        elif block_type == "TextBlock":
                            text = getattr(block, 'text', '')
                            agent_conversation.append({
                                "type": msg_type,
                                "block_type": block_type,
                                "text": text.strip() if text else '',
                            })
                        else:
                            agent_conversation.append({
                                "type": msg_type,
                                "block_type": block_type,
                                "content": getattr(block, 'content', f"The block type {block_type} of {msg_type} is not recognized"),
                            })
                elif msg_type in ["ResultMessage"]:
                    structured_output = getattr(msg, 'structured_output', None)
                    final_response = getattr(msg, 'result', '')
                    agent_conversation.append({
                        "type": msg_type,
                        "structured_output": structured_output,
                        "final_response": final_response.strip() if final_response else '',
                    })
                elif msg_type in ["TaskStartedMessage"]:
                    description = getattr(msg, 'description', '')
                    task_type = getattr(msg, 'task_type', '')
                    agent_conversation.append({
                        "type": msg_type,
                        "description": description.strip() if description else '',
                        "task_type": task_type.strip() if task_type else '',
                    })
                elif msg_type in ["TaskNotificationMessage"]:
                    status = getattr(msg, 'status', '')
                    output_file = getattr(msg, 'output_file', '')
                    summary = getattr(msg, 'summary', '')
                    agent_conversation.append({
                        "type": msg_type,
                        "status": status.strip() if status else '',
                        "output_file": output_file.strip() if output_file else '',
                        "summary": summary.strip() if summary else '',
                    })
                else:
                    agent_conversation.append({
                        "type": msg_type,
                        "content": getattr(msg, 'content', f"The message type {msg_type} is not recognized"),
                    })

            structured_output = result.get("structured_output")
            response_text = result.get("final_response", "")

            raw_conversation = [
                {"type": "SystemMessage", "text": system},
                {"type": "UserMessage", "text": prompt}
            ] + agent_conversation

            execution_time = time.time() - start_time

            result_fields = {
                'conversation': raw_conversation,
                'response': self._parse_results(structured_output) if structured_output else {},
                'raw_response': response_text,
                'turns': actual_turns,
                'error': None,
                'metadata': {
                    'model': result.get("model", self.model),
                    'agent_type': 'claude_code',
                    'execution_time': execution_time,
                    'conversation_length': len(raw_conversation),
                    'claude_uuid': result.get("uuid", ""),
                    'claude_session_id': result.get("session_id", ""),
                    'duration_ms': result.get("duration_ms", 0),
                    'total_cost_usd': result.get("total_cost_usd", 0.0),
                    'token_usage': result.get("usage", {}),
                    'is_error': result.get("is_error", False),
                },
                'raw_result': {
                    'raw_conversation': raw_conversation,
                    'raw_response': response_text,
                    'turns': actual_turns,
                    'messages': messages,
                },
            }
            return AgentResult(**result_fields)

        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            print(f"ClaudeCodeAgent error: {e}")
            print(f"ClaudeCodeAgent error trace: {error_trace}")

            result_fields = {
                'conversation': [],
                'response': {},
                'raw_response': '',
                'turns': actual_turns if 'actual_turns' in locals() else 0,
                'error': str(e),
                'metadata': {
                    'model': self.model,
                    'agent_type': 'claude_code',
                    'execution_time': execution_time,
                    'error_trace': error_trace,
                },
                'raw_result': None,
            }
            return AgentResult(**result_fields)
        
    def _parse_results(self, structured_output: dict) -> Dict[str, Any]:
        """Parse the raw result from Codex SDK into a structured format."""
        if structured_output:
            return structured_output