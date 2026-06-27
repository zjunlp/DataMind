#!/usr/bin/env python3
"""Run LongDS tasks directly with Claude Code, without importing DSGym."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt import build_turn_prompt


TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Direct answer to the current LongDS turn. Use JSON text for structured outputs.",
        },
        "reasoning_summary": {
            "type": "string",
            "description": "Brief summary of the method used. Do not include hidden chain of thought.",
        },
        "files_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Relative or absolute data files inspected for this turn.",
        },
    },
    "required": ["answer", "reasoning_summary", "files_used"],
    "additionalProperties": False,
}


COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    longds_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run LongDS-Bench directly with Claude Code sessions."
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=longds_root / "dataset" / "task" / "longds",
        help="LongDS task root containing task_list.json.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=longds_root / "dataset" / "data" / "longds",
        help="LongDS data root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results",
        help="Result root. Task runs are saved as <domain>/<dataset>/<task_id>/<run_name>/.",
    )
    parser.add_argument("--claude-bin", default="claude", help="Claude Code CLI executable.")
    parser.add_argument(
        "--claude-model",
        default=None,
        help="Model passed to `claude --model`. Omit to use Claude Code config default.",
    )
    parser.add_argument(
        "--analysis-python",
        default=sys.executable,
        help="Python executable Claude Code should use for data analysis commands.",
    )
    parser.add_argument(
        "--permission-mode",
        default="bypassPermissions",
        choices=["acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"],
        help="Claude Code permission mode. Default: bypassPermissions.",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Run Claude Code in bare mode to reduce external context and hooks.",
    )
    parser.add_argument(
        "--max-budget-usd",
        default=None,
        help="Optional Claude Code API budget cap for each turn.",
    )
    parser.add_argument("--task-limit", type=int, default=1, help="Number of tasks to run.")
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run every task after --start-index. Overrides --task-limit.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index in task_list.json.")
    parser.add_argument("--turn-limit", type=int, default=None, help="Maximum turns per task.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per Claude Code turn, seconds.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name. Defaults to claude_code_YYYYmmdd_HHMMSS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts and metadata without invoking Claude Code.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next task if a turn fails.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "claude_code"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def prepare_workspace_data(source_data_dir: Path, workspace_dir: Path) -> Path:
    """Copy task data into the task workspace and return the local data path."""
    if not source_data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {source_data_dir}")

    local_data_dir = workspace_dir / "data"
    if local_data_dir.exists():
        return local_data_dir

    shutil.copytree(source_data_dir, local_data_dir, symlinks=False)
    return local_data_dir


def claude_command(
    *,
    args: argparse.Namespace,
    session_id: str | None,
    resume: bool,
) -> list[str]:
    cmd = [
        args.claude_bin,
        "-p",
        "--input-format",
        "text",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        args.permission_mode,
        "--json-schema",
        json.dumps(TURN_SCHEMA, ensure_ascii=False),
    ]
    if args.bare:
        cmd.append("--bare")
    if args.claude_model:
        cmd.extend(["--model", args.claude_model])
    if args.max_budget_usd:
        cmd.extend(["--max-budget-usd", str(args.max_budget_usd)])
    if session_id and resume:
        cmd.extend(["--resume", session_id])
    elif session_id:
        cmd.extend(["--session-id", session_id])
    return cmd


def build_manual_resume_metadata(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    session_id: str | None,
) -> dict[str, str | None]:
    if not session_id:
        return {
            "manual_resume_note": None,
            "manual_resume_command": None,
        }

    return {
        "manual_resume_note": "Run manual_resume_command to open the Claude Code interactive session for this task.",
        "manual_resume_command": f"{args.claude_bin} --resume {session_id}",
    }


def parse_claude_output(stdout: str, fallback: str | None) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None, str]:
    session_id = fallback
    usage = None
    final_text = ""
    structured_payload = None
    structured_raw = ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        session_id = event.get("session_id") or session_id
        if isinstance(event.get("usage"), dict):
            usage = event.get("usage")
        message = event.get("message")
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            usage = message.get("usage")

        extracted_structured = extract_structured_output(event)
        if extracted_structured is not None:
            structured_payload = normalize_turn_payload(extracted_structured)
            structured_raw = json.dumps(structured_payload, ensure_ascii=False)

        if "result" in event:
            result = event.get("result")
            if isinstance(result, str):
                final_text = result
            elif isinstance(result, dict):
                payload = normalize_turn_payload(result)
                return session_id, usage, payload, json.dumps(payload, ensure_ascii=False)

        text = extract_text_from_message(message)
        if text:
            final_text = text

    if structured_payload is not None:
        return session_id, usage, structured_payload, structured_raw

    payload, raw_message = parse_structured_message(final_text)
    if payload is not None:
        payload = normalize_turn_payload(payload)
        raw_message = json.dumps(payload, ensure_ascii=False)
    return session_id, usage, payload, raw_message


def normalize_turn_payload(payload: dict[str, Any]) -> dict[str, Any]:
    answer = payload.get("answer", "")
    if not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)

    reasoning_summary = payload.get("reasoning_summary", "")
    if not isinstance(reasoning_summary, str):
        reasoning_summary = json.dumps(reasoning_summary, ensure_ascii=False)

    files_used = payload.get("files_used", [])
    if isinstance(files_used, str):
        files_used = [files_used]
    elif not isinstance(files_used, list):
        files_used = []

    return {
        "answer": answer,
        "reasoning_summary": reasoning_summary,
        "files_used": [str(path) for path in files_used],
    }


def extract_structured_output(event: dict[str, Any]) -> dict[str, Any] | None:
    structured_output = event.get("structured_output")
    if isinstance(structured_output, dict):
        return structured_output

    message = event.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and block.get("name") == "StructuredOutput":
            tool_input = block.get("input")
            if isinstance(tool_input, dict):
                return tool_input
    return None


def extract_text_from_message(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(part for part in parts if part).strip()


def parse_structured_message(text: str) -> tuple[dict[str, Any] | None, str]:
    raw = (text or "").strip()
    if not raw:
        return None, ""
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload, raw
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return payload, raw
        except json.JSONDecodeError:
            pass
    return None, raw


def parse_last_message(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None, ""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload, text
    except json.JSONDecodeError:
        pass
    return None, text


def use_color() -> bool:
    return "NO_COLOR" not in os.environ


def paint(text: str, color_code: str) -> str:
    if not use_color():
        return text
    return f"{color_code}{text}{COLOR_RESET}"


def indent_text(text: str, prefix: str = "") -> str:
    if not text:
        return ""
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())


def print_field(label: str, value: Any, *, color_code: str = COLOR_DIM) -> None:
    if value is None or value == "":
        return
    print(f"{paint(label + ':', color_code)} {value}", flush=True)


def print_text_block(label: str, text: str, *, color_code: str = COLOR_DIM) -> None:
    if not text:
        return
    print(paint(f"{label}:", color_code), flush=True)
    print(indent_text(text), flush=True)


def print_agent_text(text: str) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        print_text_block("message", text, color_code=COLOR_MAGENTA)
        return

    if not isinstance(payload, dict):
        print_text_block("message", text, color_code=COLOR_MAGENTA)
        return

    print_text_block("response", str(payload.get("answer") or ""), color_code=COLOR_MAGENTA)
    print_text_block("reasoning_summary", str(payload.get("reasoning_summary") or ""), color_code=COLOR_MAGENTA)
    files_used = payload.get("files_used")
    if files_used:
        print_text_block("files_used", "\n".join(str(path) for path in files_used), color_code=COLOR_MAGENTA)


def event_color(event_type: str, item_type: str | None) -> str:
    if "error" in event_type:
        return COLOR_RED
    if item_type == "command_execution":
        return COLOR_YELLOW
    if item_type == "agent_message":
        return COLOR_MAGENTA
    if event_type.endswith(".started"):
        return COLOR_BLUE
    if event_type.endswith(".completed"):
        return COLOR_GREEN
    return COLOR_CYAN


class ClaudeEventFormatter:
    def __init__(
        self,
        *,
        turn_label: str,
        user_request: str,
        previous_session_id: str | None = None,
    ) -> None:
        self.next_step = 0
        self.turn_label = turn_label
        self.user_request = user_request
        self.previous_session_id = previous_session_id
        self.user_request_printed = False
        self.session_printed = False
        self.pending_tool_uses: dict[str, dict[str, Any]] = {}

    def __call__(self, line: str) -> None:
        text = line.rstrip("\n")
        if not text:
            return

        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            print(text, flush=True)
            return

        if not isinstance(event, dict):
            print(text, flush=True)
            return

        self.print_event(event)

    def print_user_request(self, color_code: str = COLOR_CYAN) -> None:
        if self.user_request_printed:
            return
        print("", flush=True)
        print(paint(f"turn {self.turn_label}", COLOR_BOLD + color_code), flush=True)
        print_text_block("user_request", self.user_request, color_code=color_code)
        self.user_request_printed = True

    def print_session(self, event: dict[str, Any], color_code: str = COLOR_CYAN) -> None:
        if self.session_printed:
            return
        session_id = event.get("session_id")
        if not session_id:
            return
        if self.previous_session_id is None:
            session_state = "new"
        elif session_id == self.previous_session_id:
            session_state = "same_as_previous_turn"
        else:
            session_state = "changed_from_previous_turn"

        print("", flush=True)
        print(paint("session", COLOR_BOLD + color_code), flush=True)
        print_field("session_id", session_id, color_code=color_code)
        print_field("session_state", session_state, color_code=color_code)
        if session_state == "changed_from_previous_turn":
            print_field("previous_session_id", self.previous_session_id, color_code=color_code)
        self.session_printed = True

    def print_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", "unknown"))
        color_code = COLOR_CYAN
        if event_type == "assistant":
            color_code = COLOR_MAGENTA
        elif event_type == "user":
            color_code = COLOR_YELLOW
        elif event_type == "result":
            color_code = COLOR_GREEN
        elif event_type == "error" or event.get("is_error"):
            color_code = COLOR_RED

        self.print_session(event, color_code=color_code)
        self.print_user_request(color_code=color_code)

        if event_type == "assistant":
            self.print_assistant_event(event, color_code=color_code)
            return
        if event_type == "user":
            self.print_user_event(event, color_code=color_code)
            return
        if event_type == "result":
            self.print_result_event(event, color_code=color_code)
            return
        if event_type not in {"system"}:
            self.print_generic_event(event, event_type, color_code=color_code)

    def print_step_header(self, color_code: str) -> int:
        step = self.next_step
        self.next_step += 1
        print("", flush=True)
        print(paint(f"step {step}", COLOR_BOLD + color_code), flush=True)
        return step

    def print_assistant_event(self, event: dict[str, Any], *, color_code: str) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if isinstance(content, str):
            self.print_step_header(color_code)
            print_agent_text(content)
            return
        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text") or "")
                if text:
                    self.print_step_header(color_code)
                    print_agent_text(text)
            elif block_type == "tool_use":
                tool_id = str(block.get("id") or "")
                if tool_id:
                    self.pending_tool_uses[tool_id] = block
                else:
                    self.print_tool_step(block, output="", is_error=False)

    def print_user_event(self, event: dict[str, Any], *, color_code: str) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = str(block.get("tool_use_id") or "")
            tool_use = self.pending_tool_uses.pop(tool_use_id, None)
            output = self.tool_result_output(block)
            is_error = bool(block.get("is_error"))
            if tool_use is not None:
                self.print_tool_step(tool_use, output=output, is_error=is_error)
            elif output:
                self.print_step_header(color_code)
                print_text_block("output", str(output), color_code=color_code)

    def tool_result_output(self, block: dict[str, Any]) -> str:
        output = block.get("content")
        if isinstance(output, list):
            return "\n".join(
                str(part.get("text") or "")
                for part in output
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if output is None:
            return ""
        return str(output)

    def print_tool_step(self, tool_use: dict[str, Any], *, output: str, is_error: bool) -> None:
        color_code = COLOR_RED if is_error else COLOR_YELLOW
        self.print_step_header(color_code)
        print_field("tool", tool_use.get("name"), color_code=color_code)
        tool_input = tool_use.get("input")
        if isinstance(tool_input, dict) and "command" in tool_input:
            print_text_block("command", str(tool_input.get("command") or ""), color_code=color_code)
        elif tool_input:
            print_text_block("input", json.dumps(tool_input, ensure_ascii=False), color_code=color_code)
        print_text_block("output", output, color_code=color_code)

    def flush_pending_tool_uses(self) -> None:
        for tool_id in list(self.pending_tool_uses):
            tool_use = self.pending_tool_uses.pop(tool_id)
            self.print_tool_step(tool_use, output="", is_error=False)

    def print_result_event(self, event: dict[str, Any], *, color_code: str) -> None:
        self.flush_pending_tool_uses()
        print("", flush=True)
        print(paint(f"turn {self.turn_label} finished", COLOR_BOLD + color_code), flush=True)
        usage = event.get("usage")
        if isinstance(usage, dict):
            for key in ("input_tokens", "cache_read_input_tokens", "output_tokens"):
                print_field(key, usage.get(key), color_code=color_code)
        print_field("total_cost_usd", event.get("total_cost_usd"), color_code=color_code)

        result = event.get("result")
        if isinstance(result, str) and result.strip():
            self.print_step_header(COLOR_MAGENTA)
            print_agent_text(result)

    def print_generic_event(self, event: dict[str, Any], event_type: str, *, color_code: str) -> None:
        self.print_step_header(color_code)
        print_text_block("event", json.dumps({"type": event_type, **event}, ensure_ascii=False), color_code=color_code)


def print_stderr_line(line: str) -> None:
    print(paint(line.rstrip("\n"), COLOR_RED), file=sys.stderr, flush=True)


def stream_pipe(pipe: Any, chunks: list[str], target: Any, *, formatter: Any | None = None) -> None:
    for line in iter(pipe.readline, ""):
        chunks.append(line)
        if formatter is None:
            print(line, end="", file=target, flush=True)
        else:
            formatter(line)
    pipe.close()


def run_command_streaming(
    cmd: list[str],
    *,
    prompt: str,
    turn_label: str,
    previous_session_id: str | None,
    timeout: int,
    env: dict[str, str],
    cwd: Path,
) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
    )
    if proc.stdout is None or proc.stderr is None or proc.stdin is None:
        raise RuntimeError("Failed to open Claude Code subprocess pipes.")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_formatter = ClaudeEventFormatter(
        turn_label=turn_label,
        user_request=prompt,
        previous_session_id=previous_session_id,
    )
    stdout_thread = threading.Thread(
        target=stream_pipe,
        args=(proc.stdout, stdout_chunks, sys.stdout),
        kwargs={"formatter": stdout_formatter},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stream_pipe,
        args=(proc.stderr, stderr_chunks, sys.stderr),
        kwargs={"formatter": print_stderr_line},
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        try:
            proc.stdin.write(prompt)
        except BrokenPipeError:
            pass
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        raise subprocess.TimeoutExpired(cmd, timeout)

    stdout_thread.join()
    stderr_thread.join()
    return returncode, "".join(stdout_chunks), "".join(stderr_chunks)


def run_claude_turn(
    *,
    args: argparse.Namespace,
    prompt: str,
    turn_label: str,
    turn_dir: Path,
    work_dir: Path,
    session_id: str | None,
) -> tuple[str | None, dict[str, Any]]:
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    last_message_path = turn_dir / "last_message.json"
    previous_session_id = session_id
    session_id = session_id or str(uuid.uuid4())

    if args.dry_run:
        ClaudeEventFormatter(
            turn_label=turn_label,
            user_request=prompt,
            previous_session_id=previous_session_id,
        ).print_user_request()
        result = {
            "success": True,
            "dry_run": True,
            "answer": "",
            "reasoning_summary": "",
            "files_used": [],
            "elapsed_seconds": 0.0,
            "returncode": 0,
        }
        write_json(turn_dir / "result.json", result)
        return session_id, result

    cmd = claude_command(
        args=args,
        session_id=session_id,
        resume=previous_session_id is not None,
    )

    start = time.time()
    returncode, stdout, stderr = run_command_streaming(
        cmd,
        timeout=args.timeout,
        env=os.environ.copy(),
        prompt=prompt,
        turn_label=turn_label,
        previous_session_id=previous_session_id,
        cwd=work_dir,
    )
    elapsed = time.time() - start

    (turn_dir / "claude_stdout.jsonl").write_text(stdout, encoding="utf-8")
    (turn_dir / "claude_stderr.txt").write_text(stderr, encoding="utf-8")
    session_id, usage, payload, raw_message = parse_claude_output(stdout, session_id)
    if payload is not None:
        write_json(last_message_path, payload)

    result = {
        "success": returncode == 0 and payload is not None,
        "returncode": returncode,
        "session_id": session_id,
        "usage": usage,
        "elapsed_seconds": elapsed,
        "answer": payload.get("answer", "") if payload else "",
        "reasoning_summary": payload.get("reasoning_summary", "") if payload else "",
        "files_used": payload.get("files_used", []) if payload else [],
        "raw_last_message": raw_message,
    }
    write_json(turn_dir / "result.json", result)

    if returncode != 0:
        raise RuntimeError(f"Claude Code exited with code {returncode}; see {turn_dir}")
    if payload is None:
        raise RuntimeError(f"Claude Code did not produce schema JSON; see {turn_dir}")
    if not session_id:
        raise RuntimeError(f"Could not determine Claude Code session id; see {turn_dir}")

    return session_id, result


def run_task(
    *,
    args: argparse.Namespace,
    task_info: dict[str, str],
    run_name: str,
) -> dict[str, Any]:
    domain = task_info["task_domain"]
    dataset = task_info["dataset_name"]
    task_id = task_info["task_id"]
    task_name = f"{domain}/{dataset}/{task_id}"
    task_json = args.task_root / domain / dataset / task_id / "task.json"
    source_data_dir = args.data_root / domain / dataset / task_id / "data"

    print("", flush=True)
    print(paint(f"======= Start running task {task_name} ... =======", COLOR_RED), flush=True)

    turns = load_json(task_json)
    if args.turn_limit is not None:
        turns = turns[: args.turn_limit]

    model_slug = slugify(args.claude_model or "claude-code-default")
    run_dir = task_run_dir(args, task_info, run_name)
    workspace_dir = run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    local_data_dir = prepare_workspace_data(source_data_dir, workspace_dir)
    schema_path = run_dir / "claude_turn.schema.json"
    write_json(schema_path, TURN_SCHEMA)

    task_result: dict[str, Any] = {
        "task_domain": domain,
        "dataset_name": dataset,
        "task_id": task_id,
        "run_name": run_name,
        "model_slug": model_slug,
        "local_data_dir": str(local_data_dir),
        "workspace_dir": str(workspace_dir),
        "run_dir": str(run_dir),
        "session_id": None,
        **build_manual_resume_metadata(
            args=args,
            run_dir=run_dir,
            session_id=None,
        ),
        "turns": [],
    }
    write_json(run_dir / "task_metadata.json", task_result)

    session_id: str | None = None
    for idx, turn in enumerate(turns):
        turn_id = turn.get("turn_id", idx + 1)
        turn_dir = run_dir / "detail" / f"turn_{int(turn_id)}"
        prompt = build_turn_prompt(
            turn=turn,
            analysis_python=args.analysis_python,
            first_turn=idx == 0,
        )
        print("", flush=True)
        print(paint(f"Start running {task_name} turn {turn_id} ...", COLOR_RED), flush=True)
        session_id, result = run_claude_turn(
            args=args,
            prompt=prompt,
            turn_label=str(turn_id),
            turn_dir=turn_dir,
            work_dir=workspace_dir,
            session_id=session_id,
        )
        task_result["session_id"] = session_id
        task_result.update(
            build_manual_resume_metadata(
                args=args,
                run_dir=run_dir,
                session_id=session_id,
            )
        )
        task_result["turns"].append(
            {
                "turn_id": turn_id,
                "context": turn.get("context", ""),
                "question": turn.get("question", ""),
                "solution": result["answer"],
                "reasoning_summary": result["reasoning_summary"],
                "files_used": result["files_used"],
                "success": result["success"],
                "elapsed_seconds": result["elapsed_seconds"],
                "turn_dir": str(turn_dir),
            }
        )
        write_json(run_dir / "results.json", task_result["turns"])
        write_json(run_dir / "task_metadata.json", task_result)

    results_with_ground_truth = []
    for turn, result in zip(turns, task_result["turns"]):
        result_with_gt = dict(result)
        result_with_gt["ground_truth"] = turn.get("answer")
        results_with_ground_truth.append(result_with_gt)

    source_metadata = {
        **task_result,
        "task_json": str(task_json),
        "source_data_dir": str(source_data_dir),
    }
    write_json(run_dir / "results_with_ground_truth.json", results_with_ground_truth)
    write_json(run_dir / "task_metadata_with_sources.json", source_metadata)

    print("", flush=True)
    print(paint(f"======= Finished task {task_name}. =======", COLOR_RED), flush=True)

    return task_result


def task_run_dir(args: argparse.Namespace, task_info: dict[str, str], run_name: str) -> Path:
    return (
        args.output_dir
        / task_info["task_domain"]
        / task_info["dataset_name"]
        / task_info["task_id"]
        / run_name
    )


def write_summary_to_task_dirs(summary: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for task_result in summary.get("tasks", []):
        run_dir = Path(task_result["run_dir"])
        path = run_dir / "summary.json"
        write_json(path, summary)
        paths.append(path)

    for error in summary.get("errors", []):
        run_dir_text = error.get("run_dir")
        if not run_dir_text:
            continue
        run_dir = Path(run_dir_text)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "summary.json"
        write_json(path, summary)
        paths.append(path)

    return paths


def print_run_config(
    *,
    args: argparse.Namespace,
    run_name: str,
    results_root: Path,
    selected_count: int,
    total_count: int,
) -> None:
    print("Claude Code LongDS run configuration:", flush=True)
    print(f"  run_name: {run_name}", flush=True)
    print(f"  task_root: {args.task_root}", flush=True)
    print(f"  data_root: {args.data_root}", flush=True)
    print(f"  output_dir: {results_root}", flush=True)
    print(f"  claude_bin: {args.claude_bin}", flush=True)
    print(f"  claude_model: {args.claude_model or 'config-default'}", flush=True)
    print(f"  analysis_python: {args.analysis_python}", flush=True)
    print(f"  permission_mode: {args.permission_mode}", flush=True)
    print(f"  bare: {args.bare}", flush=True)
    print(f"  max_budget_usd: {args.max_budget_usd or 'none'}", flush=True)
    print(f"  start_index: {args.start_index}", flush=True)
    print(f"  task_limit: {'all' if args.all_tasks else args.task_limit}", flush=True)
    print(f"  turn_limit: {args.turn_limit if args.turn_limit is not None else 'all'}", flush=True)
    print(f"  timeout: {args.timeout}", flush=True)
    print(f"  selected_tasks: {selected_count} / {total_count}", flush=True)
    print(f"  dry_run: {args.dry_run}", flush=True)
    print(f"  continue_on_error: {args.continue_on_error}", flush=True)


def main() -> int:
    args = parse_args()
    if args.task_limit is not None and args.task_limit < 0:
        raise ValueError("--task-limit must be non-negative")
    if args.turn_limit is not None and args.turn_limit < 0:
        raise ValueError("--turn-limit must be non-negative")

    run_name = args.run_name or datetime.now().strftime("claude_code_%Y%m%d_%H%M%S")
    results_root = args.output_dir
    results_root.mkdir(parents=True, exist_ok=True)

    task_list = load_json(args.task_root / "task_list.json")
    selected = task_list[args.start_index :]
    if not args.all_tasks and args.task_limit is not None:
        selected = selected[: args.task_limit]

    print_run_config(
        args=args,
        run_name=run_name,
        results_root=results_root,
        selected_count=len(selected),
        total_count=len(task_list),
    )

    summary = {
        "run_name": run_name,
        "claude_model": args.claude_model or "config-default",
        "analysis_python": args.analysis_python,
        "task_root": str(args.task_root),
        "data_root": str(args.data_root),
        "results_root": str(results_root),
        "tasks": [],
    }
    for task_info in selected:
        try:
            task_result = run_task(
                args=args,
                task_info=task_info,
                run_name=run_name,
            )
            summary["tasks"].append(task_result)
            write_summary_to_task_dirs(summary)
        except Exception as exc:
            run_dir = task_run_dir(args, task_info, run_name)
            error = {"task": task_info, "run_dir": str(run_dir), "error": str(exc)}
            summary.setdefault("errors", []).append(error)
            write_summary_to_task_dirs(summary)
            print(f"ERROR: {error}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    summary_paths = write_summary_to_task_dirs(summary)
    if summary_paths:
        print(f"Saved Claude Code LongDS run summary under task result directories, e.g. {summary_paths[-1]}")
    else:
        print("No tasks selected; no Claude Code LongDS run summary was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# python run_claude_longds.py --task-limit 1 --turn-limit 1
