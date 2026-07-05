import anyio
import json
import logging
import sys
import tempfile
from pathlib import Path
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    CLINotFoundError,
    CLIConnectionError,
)
from pydantic import BaseModel
from typing import Any, Optional
from .prompt import SKILL_GENERATOR_SYSTEM_PROMPT, SKILL_GENERATOR_MODIFY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()

class _C:
    """ANSI color codes; disabled when stdout is not a TTY."""
    RST   = "\033[0m"    if _USE_COLOR else ""
    BOLD  = "\033[1m"    if _USE_COLOR else ""
    DIM   = "\033[2m"    if _USE_COLOR else ""
    CYAN  = "\033[36m"   if _USE_COLOR else ""
    BLUE  = "\033[34m"   if _USE_COLOR else ""
    YELL  = "\033[33m"   if _USE_COLOR else ""
    GREEN = "\033[32m"   if _USE_COLOR else ""
    RED   = "\033[31m"   if _USE_COLOR else ""
    MAG   = "\033[35m"   if _USE_COLOR else ""


_TRUNC = 800   # chars before/after truncation point


def _trunc(text: str, limit: int = _TRUNC) -> str:
    """Keep first and last `limit` chars if text is too long."""
    if len(text) <= limit * 2 + 40:
        return text
    omitted = len(text) - limit * 2
    return (
        text[:limit]
        + f"\n{_C.DIM}  ... [{omitted} chars omitted] ...{_C.RST}\n"
        + text[-limit:]
    )


def _hr(char: str = "─", width: int = 60) -> str:
    return _C.DIM + char * width + _C.RST


def _header(label: str, color: str) -> str:
    return f"{color}{_C.BOLD}┌─ {label} {_C.RST}"


def _print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


class ToolGeneratorResponse(BaseModel):
    generated_skill: str
    reasoning: str


skill_generator_output_format = {
    "type": "json_schema",
    "schema": ToolGeneratorResponse.model_json_schema()
}


def get_project_root() -> str:
    """Get the project root directory by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    return str(current.parent.parent.parent)


skill_generator_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": SKILL_GENERATOR_SYSTEM_PROMPT.strip()
}

skill_generator_modify_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": SKILL_GENERATOR_MODIFY_SYSTEM_PROMPT.strip()
}

SKILL_GENERATOR_TOOLS = [
    "Read", "Write", "Bash", "Glob", "Grep", "Edit",
    "TodoWrite", "BashOutput", "Skill"
]


class SkillGeneratorAgent:
    """A class to invoke Claude Code (Agent SDK) from Python."""

    def __init__(
        self,
        cwd: str = '/DataCOPE/reason_task/generation/skill_manager',
        allowed_tools: Optional[list] = None,
        permission_mode: str = 'acceptEdits',
        output_format: Optional[dict] = None,
        system_prompt=None,
        max_turns: Optional[int] = None,
        setting_sources: Optional[list] = ['project', 'user'],
    ):
        self.options = ClaudeAgentOptions(
            cwd=cwd,
            allowed_tools=allowed_tools or SKILL_GENERATOR_TOOLS,
            disallowed_tools = ["Task", "Agent", "WebFetch", "WebSearch"],
            permission_mode=permission_mode,
            output_format=output_format or skill_generator_output_format,
            system_prompt=system_prompt or skill_generator_system_prompt,
            max_turns=max_turns,
            setting_sources=setting_sources,
            env={
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"
            }
        )

    def run(self, prompt: str) -> str:
        """Synchronously run a prompt and return the result text."""
        return anyio.run(self._run_async, prompt)

    async def _run_async(self, prompt: str) -> str:
        """Async implementation: stream messages and return final result."""
        result = ""
        try:
            async for message in query(prompt=prompt, options=self.options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ThinkingBlock):
                            _print(_header("Thinking", _C.MAG))
                            _print(_C.MAG + _trunc(block.thinking) + _C.RST)
                            _print(_hr())
                        elif isinstance(block, ToolUseBlock):
                            _print(_header(f"Tool Call  {block.name}", _C.CYAN))
                            _print(f"{_C.DIM}  id: {block.id}{_C.RST}")
                            inp = json.dumps(block.input, ensure_ascii=False, indent=2)
                            _print(_C.CYAN + _trunc(inp) + _C.RST)
                            _print(_hr())
                        elif isinstance(block, ToolResultBlock):
                            ok = block.is_error is not True
                            tag = f"{_C.GREEN}OK{_C.RST}" if ok else f"{_C.RED}ERROR{_C.RST}"
                            _print(_header(f"Tool Result  [{tag}{_C.BOLD}]", _C.BLUE))
                            _print(f"{_C.DIM}  tool_use_id: {block.tool_use_id}{_C.RST}")
                            content = block.content if isinstance(block.content, str) else json.dumps(block.content, ensure_ascii=False, indent=2)
                            color = _C.RED if block.is_error else _C.BLUE
                            _print(color + _trunc(content or "") + _C.RST)
                            _print(_hr())
                        elif isinstance(block, TextBlock):
                            _print(_header("Assistant", _C.GREEN))
                            _print(_C.GREEN + _trunc(block.text) + _C.RST)
                            _print(_hr())
                elif isinstance(message, UserMessage):
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                ok = block.is_error is not True
                                tag = f"{_C.GREEN}OK{_C.RST}" if ok else f"{_C.RED}ERROR{_C.RST}"
                                _print(_header(f"Tool Result  [{tag}{_C.BOLD}]", _C.BLUE))
                                _print(f"{_C.DIM}  tool_use_id: {block.tool_use_id}{_C.RST}")
                                content = block.content if isinstance(block.content, str) else json.dumps(block.content, ensure_ascii=False, indent=2)
                                color = _C.RED if block.is_error else _C.BLUE
                                _print(color + _trunc(content or "") + _C.RST)
                                _print(_hr())
                    else:
                        _print(_header("User", _C.GREEN))
                        _print(_C.GREEN + _trunc(str(message.content)) + _C.RST)
                        _print(_hr())
                elif isinstance(message, SystemMessage):
                    _print(_header(f"System  {message.subtype}", _C.YELL))
                    if message.data:
                        _print(_C.YELL + _trunc(json.dumps(message.data, ensure_ascii=False, indent=2)) + _C.RST)
                    _print(_hr())
                elif isinstance(message, ResultMessage):
                    result = message.result
                    _print(_header("Result", _C.GREEN))
                    _print(
                        f"  {_C.BOLD}turns{_C.RST}={message.num_turns}  "
                        f"{_C.BOLD}cost{_C.RST}=${message.total_cost_usd:.4f}  "
                        f"{_C.BOLD}stop{_C.RST}={message.stop_reason}"
                    )
                    _print(_C.GREEN + _trunc(result or "") + _C.RST)
                    _print(_hr("═"))
        except CLINotFoundError:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: pip install claude-agent-sdk"
            )
        except CLIConnectionError as e:
            raise RuntimeError(f"Connection error: {e}")
        return result

    async def run_async(self, prompt: str) -> str:
        """Async entry point for use in async contexts."""
        return await self._run_async(prompt)

# --- Example usage ---
async def main():
    agent = SkillGeneratorAgent()
    result = await agent.run_async("List the Python files in the current directory.")
    print(result)

if __name__ == "__main__":
    anyio.run(main)
