import json
import logging
import shutil
from pathlib import Path

from .prompts import PROMPT_ORIGIN, PROMPT_ANALYSIS_ITERATION, PROMPT_CHECKLIST_ITERATION, render_prompt

logger = logging.getLogger("ddr_pipeline")

_TRUNC = 800


def _trunc(text: str, limit: int = _TRUNC) -> str:
    if len(text) <= limit * 2 + 40:
        return text
    omitted = len(text) - limit * 2
    return text[:limit] + f"\n  ... [{omitted} chars omitted] ...\n" + text[-limit:]


SKILL_GENERATOR_TOOLS = [
    "Read", "Write", "Bash", "Glob", "Grep", "Edit",
    "TodoWrite", "BashOutput", "Skill",
]


def get_prompt(prompt_type: str, workspace: Path) -> str:
    templates = {
        "origin": PROMPT_ORIGIN,
        "analysis_iteration": PROMPT_ANALYSIS_ITERATION,
        "checklist_iteration": PROMPT_CHECKLIST_ITERATION,
    }
    template = templates.get(prompt_type)
    if not template:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")
    return render_prompt(template, str(workspace))


class SkillGeneratorAgent:

    def __init__(self, workspace: Path, max_turns: int | None = None):
        self.workspace = workspace
        self.max_turns = max_turns

    def run(self, prompt: str, max_retries: int = 3) -> str:
        import anyio
        import time
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(max_retries + 1):
            try:
                return anyio.run(self._run_async, prompt)
            except RuntimeError as e:
                msg = str(e).lower()
                if any(kw in msg for kw in ("connection error", "socket", "timeout", "522", "524")):
                    if attempt < max_retries:
                        wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
                        logger.warning(
                            f"[Skill Gen] Connection error (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {wait}s: {e}"
                        )
                        time.sleep(wait)
                        last_exc = e
                        continue
                raise
        raise last_exc

    async def _run_async(self, prompt: str) -> str:
        from claude_agent_sdk import (
            query,
            ClaudeAgentOptions,
            CLINotFoundError,
            CLIConnectionError,
            ResultMessage,
            AssistantMessage,
            UserMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
            ToolResultBlock,
        )

        options = ClaudeAgentOptions(
            cwd=str(self.workspace),
            allowed_tools=SKILL_GENERATOR_TOOLS,
            disallowed_tools=["Task", "Agent", "WebFetch", "WebSearch"],
            permission_mode="bypassPermissions",
            system_prompt={"type": "preset", "preset": "claude_code"},
            model="claude-sonnet-4-6",
            effort="medium",
            max_turns=self.max_turns,
            setting_sources=["project", "user"],
            env={"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"},
        )

        result = ""
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ThinkingBlock):
                            logger.debug(f"[Thinking] {_trunc(block.thinking)}")
                        elif isinstance(block, ToolUseBlock):
                            inp = json.dumps(block.input, ensure_ascii=False, indent=2)
                            logger.debug(f"[Tool Call] {block.name}: {_trunc(inp)}")
                        elif isinstance(block, ToolResultBlock):
                            tag = "OK" if block.is_error is not True else "ERROR"
                            content = block.content if isinstance(block.content, str) else json.dumps(block.content, ensure_ascii=False, indent=2)
                            logger.debug(f"[Tool Result {tag}] {_trunc(content or '')}")
                        elif isinstance(block, TextBlock):
                            logger.debug(f"[Assistant] {_trunc(block.text)}")
                elif isinstance(message, UserMessage):
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                tag = "OK" if block.is_error is not True else "ERROR"
                                content = block.content if isinstance(block.content, str) else json.dumps(block.content, ensure_ascii=False, indent=2)
                                logger.debug(f"[Tool Result {tag}] {_trunc(content or '')}")
                elif isinstance(message, SystemMessage):
                    logger.debug(f"[System {message.subtype}] {json.dumps(message.data, ensure_ascii=False)[:200] if message.data else ''}")
                elif isinstance(message, ResultMessage):
                    result = message.result
                    logger.info(
                        f"[Skill Gen] turns={message.num_turns} cost=${message.total_cost_usd:.4f} stop={message.stop_reason}"
                    )
        except CLINotFoundError:
            raise RuntimeError(
                "Claude Code CLI not found."
            )
        except CLIConnectionError as e:
            raise RuntimeError(f"Claude Code connection error: {e}")
        return result

    async def run_async(self, prompt: str) -> str:
        return await self._run_async(prompt)


def save_skill_artifacts(
    workspace: Path,
    records_base: Path,
    folder_name: str,
) -> None:
    """Move generated skills, trajectories, and Claude project files into records."""
    record_dir = records_base / folder_name
    record_dir.mkdir(parents=True, exist_ok=True)
    traj_dst = record_dir / "trajectory"
    claude_temp_dst = record_dir / "claude-temp"
    traj_dst.mkdir(exist_ok=True)
    claude_temp_dst.mkdir(exist_ok=True)

    skills_src = workspace / ".claude" / "skills"
    if skills_src.exists():
        for item in sorted(skills_src.iterdir()):
            if item.name == "skill-creator":
                continue
            dst = record_dir / item.name
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.move(str(item), str(dst))
            logger.info(f"Moved skill {item.name} -> {dst}")

    traj_src = workspace / "trajectory"
    if traj_src.exists():
        for item in traj_src.iterdir():
            dst = traj_dst / item.name
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.move(str(item), str(dst))

    from .utils import workspace_cache_dir_name
    cache_dir_name = workspace_cache_dir_name(workspace)
    cache_src = Path.home() / ".claude" / "projects" / cache_dir_name
    if cache_src.exists():
        for item in cache_src.iterdir():
            dst = claude_temp_dst / item.name
            if dst.exists():
                candidate = dst.with_name(dst.name + ".new")
                i = 1
                while candidate.exists():
                    candidate = dst.with_name(f"{dst.name}.new.{i}")
                    i += 1
                dst = candidate
            shutil.move(str(item), str(dst))

    logger.info(f"Saved skill artifacts to {record_dir}")
