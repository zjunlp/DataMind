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
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from DataMind.longds.runners.codex.prompt import build_turn_prompt


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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run LongDS-Bench directly with Codex CLI sessions."
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=root / "DSGym" / "data" / "task" / "longds",
        help="LongDS task root containing task_list.json.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=root / "DSGym" / "data" / "data" / "longds",
        help="LongDS data root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "scripts" / "results",
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


def prepare_workspace_data(source_data_dir: Path, run_dir: Path) -> Path:
    """Copy task data into the task run directory and return the local data path."""
    if not source_data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {source_data_dir}")

    local_data_dir = run_dir / "data"
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


def run_codex_turn(
    *,
    args: argparse.Namespace,
    prompt: str,
    schema_path: Path,
    turn_dir: Path,
    work_dir: Path,
    session_id: str | None,
) -> tuple[str | None, dict[str, Any]]:
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    last_message_path = turn_dir / "last_message.json"

    if args.dry_run:
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
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        env=os.environ.copy(),
    )
    elapsed = time.time() - start

    (turn_dir / "codex_stdout.jsonl").write_text(proc.stdout, encoding="utf-8")
    (turn_dir / "codex_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    thread_id, usage = parse_thread_id(proc.stdout, session_id)
    payload, raw_message = parse_last_message(last_message_path)

    result = {
        "success": proc.returncode == 0 and payload is not None,
        "returncode": proc.returncode,
        "thread_id": thread_id,
        "usage": usage,
        "elapsed_seconds": elapsed,
        "answer": payload.get("answer", "") if payload else "",
        "reasoning_summary": payload.get("reasoning_summary", "") if payload else "",
        "files_used": payload.get("files_used", []) if payload else [],
        "raw_last_message": raw_message,
    }
    write_json(turn_dir / "result.json", result)

    if proc.returncode != 0:
        raise RuntimeError(f"Codex exited with code {proc.returncode}; see {turn_dir}")
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
    task_json = args.task_root / domain / dataset / task_id / "task.json"
    source_data_dir = args.data_root / domain / dataset / task_id / "data"

    turns = load_json(task_json)
    if args.turn_limit is not None:
        turns = turns[: args.turn_limit]

    model_slug = slugify(args.codex_model or "codex-default")
    run_dir = task_run_dir(args, task_info, run_name)
    workspace_dir = run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    local_data_dir = prepare_workspace_data(source_data_dir, run_dir)
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
        "turns": [],
    }
    write_json(run_dir / "task_metadata.json", task_result)

    session_id: str | None = None
    for idx, turn in enumerate(turns):
        turn_id = turn.get("turn_id", idx + 1)
        turn_dir = run_dir / f"turn_{int(turn_id):03d}"
        prompt = build_turn_prompt(
            turn=turn,
            analysis_python=args.analysis_python,
            first_turn=idx == 0,
        )
        print(f"Running {domain}/{dataset}/{task_id} turn {turn_id} ...", flush=True)
        session_id, result = run_codex_turn(
            args=args,
            prompt=prompt,
            schema_path=schema_path,
            turn_dir=turn_dir,
            work_dir=run_dir,
            session_id=session_id,
        )
        task_result["thread_id"] = session_id
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
