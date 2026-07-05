#!/usr/bin/env python3
"""Run LongDS tasks directly with Codex CLI, without importing DSGym."""

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
        description="Run LongDS-Bench directly with Codex CLI sessions."
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
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI executable.")
    parser.add_argument(
        "--codex-model",
        default=None,
        help="Model passed to `codex exec -m`. Omit to use Codex config default.",
    )
    parser.add_argument(
        "--analysis-python",
        default=sys.executable,
        help="Python executable Codex should use for data analysis commands.",
    )
    parser.add_argument(
        "--sandbox",
        default="workspace-write",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help="Sandbox mode for the first Codex turn in each task.",
    )
    parser.add_argument(
        "--approval-policy",
        default="never",
        choices=["untrusted", "on-failure", "on-request", "never"],
        help="Top-level Codex approval policy.",
    )
    parser.add_argument("--task-limit", type=int, default=1, help="Number of tasks to run.")
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run every task after --start-index. Overrides --task-limit.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index in task_list.json.")
    parser.add_argument("--turn-limit", type=int, default=None, help="Maximum turns per task.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per Codex turn, seconds.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name. Defaults to codex_YYYYmmdd_HHMMSS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts and metadata without invoking Codex.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next task if a turn fails.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "codex"


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


def codex_command(
    *,
    args: argparse.Namespace,
    schema_path: Path,
    last_message_path: Path,
    work_dir: Path,
    session_id: str | None,
) -> list[str]:
    cmd = [args.codex_bin]
    if args.approval_policy:
        cmd.extend(["--ask-for-approval", args.approval_policy])
    cmd.extend(["exec"])

    if session_id:
        cmd.extend(["resume", "--json", "--skip-git-repo-check"])
        if args.codex_model:
            cmd.extend(["-m", args.codex_model])
        cmd.extend(
            [
                "--output-schema",
                str(schema_path),
                "-o",
                str(last_message_path),
                session_id,
                "-",
            ]
        )
        return cmd

    cmd.extend(
        [
            "--json",
            "--sandbox",
            args.sandbox,
            "--skip-git-repo-check",
            "-C",
            str(work_dir),
        ]
    )
    if args.codex_model:
        cmd.extend(["-m", args.codex_model])
    cmd.extend(["--output-schema", str(schema_path), "-o", str(last_message_path), "-"])
    return cmd


def build_manual_resume_metadata(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    schema_path: Path,
    session_id: str | None,
) -> dict[str, str | None]:
    if not session_id:
        return {
            "manual_resume_note": None,
            "manual_resume_command": None,
        }

    return {
        "manual_resume_note": "Run manual_resume_command to open the Codex interactive session for this task.",
        "manual_resume_command": f"{args.codex_bin} resume {session_id}",
    }


def parse_thread_id(stdout: str, fallback: str | None) -> tuple[str | None, dict[str, Any] | None]:
    thread_id = fallback
    usage = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id") or thread_id
        elif event.get("type") == "turn.completed":
            usage = event.get("usage")
    return thread_id, usage


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


class CodexEventFormatter:
    def __init__(
        self,
        *,
        turn_label: str,
        user_request: str,
        previous_thread_id: str | None = None,
    ) -> None:
        self.step_by_item_id: dict[str, int] = {}
        self.next_step = 0
        self.turn_label = turn_label
        self.user_request = user_request
        self.previous_thread_id = previous_thread_id
        self.user_request_printed = False
        self.pending_item_events: dict[int, tuple[str, dict[str, Any], str | None, str]] = {}
        self.next_step_to_print = 0

    def step_for_item(self, item: dict[str, Any]) -> int:
        item_id = str(item.get("id") or "")
        if item_id in self.step_by_item_id:
            return self.step_by_item_id[item_id]

        match = re.fullmatch(r"item_(\d+)", item_id)
        if match:
            step = int(match.group(1))
            self.next_step = max(self.next_step, step + 1)
        else:
            step = self.next_step
            self.next_step += 1

        self.step_by_item_id[item_id] = step
        return step

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

        event_type = str(event.get("type", "unknown"))
        item = event.get("item")
        item_type = item.get("type") if isinstance(item, dict) else None
        color_code = event_color(event_type, item_type)

        if not isinstance(item, dict):
            self.print_non_item_event(event, event_type, color_code)
            return

        self.print_item_event(event_type, item, item_type, color_code)

    def print_user_request(self, color_code: str = COLOR_CYAN) -> None:
        if self.user_request_printed:
            return
        print("", flush=True)
        print(paint(f"turn {self.turn_label}", COLOR_BOLD + color_code), flush=True)
        print_text_block("user_request", self.user_request, color_code=color_code)
        self.user_request_printed = True

    def print_non_item_event(self, event: dict[str, Any], event_type: str, color_code: str) -> None:
        if event_type.startswith("thread."):
            thread_id = event.get("thread_id")
            if self.previous_thread_id is None:
                session_state = "new"
            elif thread_id == self.previous_thread_id:
                session_state = "same_as_previous_turn"
            else:
                session_state = "changed_from_previous_turn"

            print("", flush=True)
            print(paint("session", COLOR_BOLD + color_code), flush=True)
            print_field("thread_id", thread_id, color_code=color_code)
            print_field("session_state", session_state, color_code=color_code)
            if session_state == "changed_from_previous_turn":
                print_field("previous_thread_id", self.previous_thread_id, color_code=color_code)
            return

        if event_type == "turn.started":
            self.print_user_request(color_code=color_code)
            return

        if event_type == "turn.completed":
            self.flush_ready_items()
            self.flush_remaining_items()
            print("", flush=True)
            print(paint(f"turn {self.turn_label} finished", COLOR_BOLD + color_code), flush=True)
            usage = event.get("usage")
            if isinstance(usage, dict):
                for key in ("input_tokens", "cached_input_tokens", "reasoning_output_tokens"):
                    print_field(key, usage.get(key), color_code=color_code)
            return

        print("", flush=True)
        print(paint(event_type, COLOR_BOLD + color_code), flush=True)
        extra = {k: v for k, v in event.items() if k not in {"type", "thread_id"}}
        print_field("thread_id", event.get("thread_id"), color_code=color_code)
        if extra:
            print_text_block("event", json.dumps(extra, ensure_ascii=False), color_code=color_code)

    def print_item_event(
        self,
        event_type: str,
        item: dict[str, Any],
        item_type: str | None,
        color_code: str,
    ) -> None:
        if item.get("status") == "in_progress":
            self.step_for_item(item)
            return

        self.print_user_request(color_code=color_code)
        step = self.step_for_item(item)
        self.pending_item_events[step] = (event_type, item, item_type, color_code)
        self.flush_ready_items()

    def flush_ready_items(self) -> None:
        while self.next_step_to_print in self.pending_item_events:
            event_type, item, item_type, color_code = self.pending_item_events.pop(self.next_step_to_print)
            self.print_item_event_now(
                step=self.next_step_to_print,
                event_type=event_type,
                item=item,
                item_type=item_type,
                color_code=color_code,
            )
            self.next_step_to_print += 1

    def flush_remaining_items(self) -> None:
        for step in sorted(self.pending_item_events):
            event_type, item, item_type, color_code = self.pending_item_events[step]
            self.print_item_event_now(
                step=step,
                event_type=event_type,
                item=item,
                item_type=item_type,
                color_code=color_code,
            )
        self.pending_item_events.clear()

    def print_item_event_now(
        self,
        *,
        step: int,
        event_type: str,
        item: dict[str, Any],
        item_type: str | None,
        color_code: str,
    ) -> None:

        print("", flush=True)
        print(paint(f"step {step}", COLOR_BOLD + color_code), flush=True)

        if item_type == "command_execution":
            if event_type.endswith(".completed"):
                print_text_block("command", str(item.get("command", "")), color_code=color_code)
                print_text_block("output", str(item.get("aggregated_output") or ""), color_code=color_code)
            return

        if item_type == "agent_message":
            print_agent_text(str(item.get("text", "")))
            return

        extra = {
            key: value
            for key, value in item.items()
            if key not in {"id", "type", "status", "command", "aggregated_output", "text"}
        }
        if extra:
            print_text_block("item_data", json.dumps(extra, ensure_ascii=False), color_code=color_code)


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
    previous_thread_id: str | None,
    timeout: int,
    env: dict[str, str],
) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    if proc.stdout is None or proc.stderr is None or proc.stdin is None:
        raise RuntimeError("Failed to open Codex subprocess pipes.")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_formatter = CodexEventFormatter(
        turn_label=turn_label,
        user_request=prompt,
        previous_thread_id=previous_thread_id,
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


def run_codex_turn(
    *,
    args: argparse.Namespace,
    prompt: str,
    turn_label: str,
    schema_path: Path,
    turn_dir: Path,
    work_dir: Path,
    session_id: str | None,
) -> tuple[str | None, dict[str, Any]]:
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    last_message_path = turn_dir / "last_message.json"

    if args.dry_run:
        CodexEventFormatter(turn_label=turn_label, user_request=prompt).print_user_request()
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
        return session_id or "dry-run-session", result

    cmd = codex_command(
        args=args,
        schema_path=schema_path,
        last_message_path=last_message_path,
        work_dir=work_dir,
        session_id=session_id,
    )

    start = time.time()
    returncode, stdout, stderr = run_command_streaming(
        cmd,
        timeout=args.timeout,
        env=os.environ.copy(),
        prompt=prompt,
        turn_label=turn_label,
        previous_thread_id=session_id,
    )
    elapsed = time.time() - start

    (turn_dir / "codex_stdout.jsonl").write_text(stdout, encoding="utf-8")
    (turn_dir / "codex_stderr.txt").write_text(stderr, encoding="utf-8")
    thread_id, usage = parse_thread_id(stdout, session_id)
    payload, raw_message = parse_last_message(last_message_path)

    result = {
        "success": returncode == 0 and payload is not None,
        "returncode": returncode,
        "thread_id": thread_id,
        "usage": usage,
        "elapsed_seconds": elapsed,
        "answer": payload.get("answer", "") if payload else "",
        "reasoning_summary": payload.get("reasoning_summary", "") if payload else "",
        "files_used": payload.get("files_used", []) if payload else [],
        "raw_last_message": raw_message,
    }
    write_json(turn_dir / "result.json", result)

    if returncode != 0:
        raise RuntimeError(f"Codex exited with code {returncode}; see {turn_dir}")
    if payload is None:
        raise RuntimeError(f"Codex did not produce schema JSON; see {turn_dir}")
    if not thread_id:
        raise RuntimeError(f"Could not determine Codex thread id; see {turn_dir}")

    return thread_id, result


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

    model_slug = slugify(args.codex_model or "codex-default")
    run_dir = task_run_dir(args, task_info, run_name)
    workspace_dir = run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    local_data_dir = prepare_workspace_data(source_data_dir, workspace_dir)
    schema_path = run_dir / "codex_turn.schema.json"
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
        "thread_id": None,
        **build_manual_resume_metadata(
            args=args,
            run_dir=run_dir,
            schema_path=schema_path,
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
        session_id, result = run_codex_turn(
            args=args,
            prompt=prompt,
            turn_label=str(turn_id),
            schema_path=schema_path,
            turn_dir=turn_dir,
            work_dir=workspace_dir,
            session_id=session_id,
        )
        task_result["thread_id"] = session_id
        task_result.update(
            build_manual_resume_metadata(
                args=args,
                run_dir=run_dir,
                schema_path=schema_path,
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
    print("Codex LongDS run configuration:", flush=True)
    print(f"  run_name: {run_name}", flush=True)
    print(f"  task_root: {args.task_root}", flush=True)
    print(f"  data_root: {args.data_root}", flush=True)
    print(f"  output_dir: {results_root}", flush=True)
    print(f"  codex_bin: {args.codex_bin}", flush=True)
    print(f"  codex_model: {args.codex_model or 'config-default'}", flush=True)
    print(f"  analysis_python: {args.analysis_python}", flush=True)
    print(f"  sandbox: {args.sandbox}", flush=True)
    print(f"  approval_policy: {args.approval_policy}", flush=True)
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

    run_name = args.run_name or datetime.now().strftime("codex_%Y%m%d_%H%M%S")
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
        "codex_model": args.codex_model or "config-default",
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
        print(f"Saved Codex LongDS run summary under task result directories, e.g. {summary_paths[-1]}")
    else:
        print("No tasks selected; no Codex LongDS run summary was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# python run_codex_longds.py    --task-limit 1 --turn-limit 1 
