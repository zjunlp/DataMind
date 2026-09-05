#!/usr/bin/env python3
"""Run LongDS tasks directly with Kimi Code, without importing DSGym."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_summary import RunSummary
from src.longds_dataset import (
    add_dataset_arguments, resolve_dataset, dataset_metadata, load_task_list, load_turns, result_root,
)

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
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"

CONTAINER_WORKSPACE = "/workspace"
CONTAINER_HOME = f"{CONTAINER_WORKSPACE}/.home"
CONTAINER_KIMI_HOME = "/tmp/longds_kimi_home"
CONTAINER_KIMI_CONFIG = f"{CONTAINER_KIMI_HOME}/config.toml"
FIXED_DOCKER_ENV = {
    "HOME": CONTAINER_HOME,
    "KIMI_CODE_HOME": CONTAINER_KIMI_HOME,
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONWARNINGS": "ignore::FutureWarning",
}
DEFAULT_DOCKER_ENV_KEYS = (
    "KIMI_CODE_EXPERIMENTAL_FLAG",
    "KIMI_CODE_LEGACY_FLAG",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run LongDS-Bench directly with Kimi Code sessions."
    )
    add_dataset_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Result base (default: ./results in the current directory). Appends longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/.",
    )
    parser.add_argument("--kimi-bin", default="kimi", help="Kimi Code CLI executable.")
    parser.add_argument(
        "--kimi-model",
        default=None,
        help="Model passed to `kimi --model`. Omit to use Kimi Code config default.",
    )
    parser.add_argument(
        "--kimi-config",
        type=Path,
        default=script_dir / "config.toml",
        help=(
            "Kimi Code config.toml. Defaults to config.toml in this directory. "
            f"With --use-docker it is copied to {CONTAINER_KIMI_CONFIG}."
        ),
    )
    parser.add_argument(
        "--analysis-python",
        default=None,
        help=(
            "Python executable Kimi Code should use for data analysis commands. "
            "Defaults to the current Python locally and /usr/local/bin/python with --use-docker."
        ),
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help=(
            "Run Kimi Code inside one Docker container per task. Task data is copied into "
            f"{CONTAINER_WORKSPACE}; Kimi Code tools are otherwise left unrestricted."
        ),
    )
    parser.add_argument("--docker-bin", default="docker", help="Docker CLI executable.")
    parser.add_argument(
        "--docker-image",
        default="longds-kimi-code:latest",
        help="Docker image used with --use-docker.",
    )
    parser.add_argument(
        "--docker-network",
        default="host",
        help="Docker network mode. Default: host.",
    )
    parser.add_argument(
        "--docker-memory",
        default=None,
        help="Optional docker run --memory value, for example 8g.",
    )
    parser.add_argument(
        "--docker-cpus",
        default=None,
        help="Optional docker run --cpus value.",
    )
    parser.add_argument(
        "--docker-user",
        default=None,
        help="Optional docker run --user value. Defaults to the current uid:gid.",
    )
    parser.add_argument(
        "--docker-container-prefix",
        default="longds-kimi",
        help="Prefix for per-task Docker container names.",
    )
    parser.add_argument(
        "--keep-docker-container",
        action="store_true",
        help="Keep per-task Docker containers after task completion or failure for debugging.",
    )
    parser.add_argument(
        "--docker-env",
        action="append",
        default=[],
        metavar="KEY_OR_KEY=VALUE",
        help=(
            "Additional environment variable to pass to Docker. Repeatable. "
            "Use KEY to pass the current host value or KEY=VALUE to set an explicit value."
        ),
    )
    parser.add_argument(
        "--docker-env-file",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help=(
            "Load Docker environment variables from a .env-style file. Repeatable. "
            "Values are passed through the Docker CLI process environment, not command argv."
        ),
    )
    parser.add_argument(
        "--kimi-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra raw argument appended to every Kimi Code invocation. Repeatable.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Maximum number of tasks to run after --start-index. Defaults to all remaining tasks.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the task list.")
    parser.add_argument("--turn-limit", type=int, default=None, help="Maximum turns per task.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per Kimi Code turn, seconds.")
    parser.add_argument(
        "--run-parallel",
        type=int,
        default=1,
        metavar="N",
        help="Number of tasks to run concurrently. Turns within each task remain sequential. Default: 1.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional run directory name. Defaults to kimi_code_<model>_YYYYmmdd_HHMMSS. "
            "A task is skipped when its run directory already exists."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing task run directory and run the task again. Default: disabled.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep workspace/data/ after a task finishes. By default the copied inputs are deleted "
        "to bound peak disk usage; data of a failed task is always kept for debugging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts and metadata without invoking Kimi Code or copying task data.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run judge.py immediately after each successfully completed task.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "kimi_code"


def validate_run_name(run_name: str) -> None:
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("--run-name must be a single directory name without path separators")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def kimi_config_model(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    with path.open("rb") as f:
        payload = tomllib.load(f)
    if not isinstance(payload, dict):
        return None
    model = payload.get("default_model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def prepare_workspace_data(
    source_data_dir: Path,
    workspace_dir: Path,
    *,
    materialize: bool = True,
) -> Path:
    """Copy task data into the task workspace and return the local data path.

    With materialize=False the source is still validated but nothing is copied, so a dry run
    cannot fill the disk with data it will never read.
    """
    if not source_data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {source_data_dir}")

    local_data_dir = workspace_dir / "data"
    if local_data_dir.exists() or not materialize:
        return local_data_dir

    shutil.copytree(source_data_dir, local_data_dir, symlinks=False)
    return local_data_dir


def directory_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            total += entry.stat().st_size
    return total


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def cleanup_workspace_data(local_data_dir: Path) -> dict[str, Any]:
    """Delete the copied task data so peak disk usage stays bounded by concurrent tasks.

    Only the copied inputs are removed. Helper scripts and intermediate artifacts the agent
    produced stay in the workspace as trajectory evidence.
    """
    if not local_data_dir.exists():
        return {"removed": False, "reason": "missing", "freed_bytes": 0}

    freed_bytes = directory_size(local_data_dir)
    shutil.rmtree(local_data_dir)
    return {"removed": True, "reason": "task_completed", "freed_bytes": freed_bytes}


def finalize_workspace_data(
    *,
    args: argparse.Namespace,
    local_data_dir: Path,
) -> dict[str, Any]:
    """Drop the copied inputs of a completed task unless the run asked to keep them."""
    if args.dry_run:
        return {"removed": False, "reason": "dry_run", "freed_bytes": 0}
    if args.keep_data:
        return {"removed": False, "reason": "keep_data", "freed_bytes": 0}

    cleanup = cleanup_workspace_data(local_data_dir)
    if cleanup["removed"]:
        print(
            paint(
                f"Removed copied data, freed {format_bytes(cleanup['freed_bytes'])}: {local_data_dir}",
                COLOR_DIM,
            ),
            flush=True,
        )
    return cleanup


def kimi_command(
    *,
    args: argparse.Namespace,
    prompt: str,
    session_id: str | None,
) -> list[str]:
    cmd = [
        args.kimi_bin,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
    ]
    if args.kimi_model and getattr(args, "kimi_model_explicit", True):
        cmd.extend(["--model", args.kimi_model])
    if session_id:
        cmd.extend(["--session", session_id])
    cmd.extend(args.kimi_arg)
    return cmd


def docker_user_arg(args: argparse.Namespace) -> str:
    return args.docker_user or f"{os.getuid()}:{os.getgid()}"


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def docker_client_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()

    for path in getattr(args, "docker_env_file", []) or []:
        env.update(parse_env_file(Path(path)))

    for spec in getattr(args, "docker_env", []) or []:
        if "=" not in spec:
            continue
        key, value = spec.split("=", 1)
        key = key.strip()
        if key:
            env[key] = value

    return env


def docker_env_specs(args: argparse.Namespace, client_env: dict[str, str]) -> list[str]:
    specs: list[str] = [f"{key}={value}" for key, value in FIXED_DOCKER_ENV.items()]
    seen = {spec.split("=", 1)[0] for spec in specs}

    for key in DEFAULT_DOCKER_ENV_KEYS:
        if client_env.get(key) and key not in seen:
            specs.append(key)
            seen.add(key)

    for spec in getattr(args, "docker_env", []) or []:
        key = spec.split("=", 1)[0]
        if key and key not in seen:
            specs.append(key)
            seen.add(key)

    return specs


def docker_env_args(args: argparse.Namespace, client_env: dict[str, str]) -> list[str]:
    cmd: list[str] = []
    for spec in docker_env_specs(args, client_env):
        cmd.extend(["--env", spec])
    return cmd


def docker_container_name(
    *,
    args: argparse.Namespace,
    task_info: dict[str, str],
    run_name: str,
) -> str:
    raw = (
        f"{args.output_dir.resolve()}/"
        f"{task_info['task_domain']}/"
        f"{task_info['dataset_name']}/"
        f"{task_info['task_id']}/"
        f"{run_name}"
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    task_slug = slugify(
        f"{task_info['task_domain']}-{task_info['dataset_name']}-{task_info['task_id']}"
    )
    prefix = slugify(args.docker_container_prefix)
    return f"{prefix}-{slugify(run_name)[:32]}-{task_slug[:48]}-{digest}"[:128]


def docker_run_detached_command(
    *,
    args: argparse.Namespace,
    container_name: str,
    client_env: dict[str, str],
) -> list[str]:
    cmd = [args.docker_bin, "run", "-d", "--name", container_name, "--init"]
    cmd.extend(["--user", docker_user_arg(args)])
    cmd.extend(["--workdir", CONTAINER_WORKSPACE])

    if args.docker_network:
        cmd.extend(["--network", args.docker_network])
    if args.docker_memory:
        cmd.extend(["--memory", str(args.docker_memory)])
    if args.docker_cpus:
        cmd.extend(["--cpus", str(args.docker_cpus)])

    cmd.extend(docker_env_args(args, client_env))
    cmd.append(args.docker_image)
    cmd.extend(
        [
            "/bin/sh",
            "-lc",
            (
                f"mkdir -p {CONTAINER_WORKSPACE}/data {CONTAINER_HOME} "
                f"{CONTAINER_KIMI_HOME} && tail -f /dev/null"
            ),
        ]
    )
    return cmd


def docker_exec_command(
    *,
    args: argparse.Namespace,
    container_name: str,
    inner_cmd: list[str],
    interactive: bool = False,
) -> list[str]:
    cmd = [args.docker_bin, "exec", "-it" if interactive else "-i"]
    cmd.extend(["--user", docker_user_arg(args)])
    cmd.extend(["--workdir", CONTAINER_WORKSPACE])
    cmd.append(container_name)
    cmd.extend(inner_cmd)
    return cmd


def docker_chown_user(args: argparse.Namespace) -> str | None:
    user = docker_user_arg(args)
    return user if re.fullmatch(r"\d+(?::\d+)?", user) else None


def run_docker_control(args: argparse.Namespace, cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, env=docker_client_env(args))


def start_task_container(
    *,
    args: argparse.Namespace,
    container_name: str,
) -> None:
    run_docker_control(
        args,
        docker_run_detached_command(
            args=args,
            container_name=container_name,
            client_env=docker_client_env(args),
        ),
    )


def copy_task_data_to_container(
    *,
    args: argparse.Namespace,
    container_name: str,
    source_data_dir: Path,
) -> None:
    run_docker_control(
        args,
        [
            args.docker_bin,
            "exec",
            "--user",
            docker_user_arg(args),
            container_name,
            "mkdir",
            "-p",
            f"{CONTAINER_WORKSPACE}/data",
        ]
    )
    run_docker_control(
        args,
        [
            args.docker_bin,
            "cp",
            f"{source_data_dir.resolve()}/.",
            f"{container_name}:{CONTAINER_WORKSPACE}/data",
        ]
    )
    chown_user = docker_chown_user(args)
    if chown_user is not None:
        run_docker_control(
            args,
            [
                args.docker_bin,
                "exec",
                "--user",
                "0:0",
                container_name,
                "chown",
                "-R",
                chown_user,
                CONTAINER_WORKSPACE,
            ]
        )


def copy_kimi_config_to_container(
    *,
    args: argparse.Namespace,
    container_name: str,
) -> None:
    if args.kimi_config is None:
        return

    run_docker_control(
        args,
        [
            args.docker_bin,
            "cp",
            str(args.kimi_config.resolve()),
            f"{container_name}:{CONTAINER_KIMI_CONFIG}",
        ],
    )
    chown_user = docker_chown_user(args)
    if chown_user is not None:
        run_docker_control(
            args,
            [
                args.docker_bin,
                "exec",
                "--user",
                "0:0",
                container_name,
                "chown",
                chown_user,
                CONTAINER_KIMI_CONFIG,
            ],
        )


def prepare_local_kimi_home(*, args: argparse.Namespace, run_dir: Path) -> Path:
    kimi_home = run_dir / "kimi_home"
    kimi_home.mkdir(parents=True, exist_ok=True)
    if args.kimi_config is not None:
        shutil.copy2(args.kimi_config, kimi_home / "config.toml")
    return kimi_home


def scrub_kimi_config(kimi_home: Path) -> None:
    (kimi_home / "config.toml").unlink(missing_ok=True)


def sync_kimi_home_from_container(
    *,
    args: argparse.Namespace,
    container_name: str,
    kimi_home: Path,
) -> dict[str, Any]:
    kimi_home.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            args.docker_bin,
            "cp",
            f"{container_name}:{CONTAINER_KIMI_HOME}/.",
            str(kimi_home),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=docker_client_env(args),
    )
    scrub_kimi_config(kimi_home)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def sync_workspace_from_container(
    *,
    args: argparse.Namespace,
    container_name: str,
    workspace_dir: Path,
) -> dict[str, Any]:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            args.docker_bin,
            "cp",
            f"{container_name}:{CONTAINER_WORKSPACE}/.",
            str(workspace_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=docker_client_env(args),
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def remove_task_container(
    *,
    args: argparse.Namespace,
    container_name: str,
) -> dict[str, Any]:
    completed = subprocess.run(
        [args.docker_bin, "rm", "-f", container_name],
        check=False,
        capture_output=True,
        text=True,
        env=docker_client_env(args),
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def write_task_metadata_with_sources(
    *,
    task_result: dict[str, Any],
    run_dir: Path,
    task_json: Path,
    source_data_dir: Path,
) -> None:
    write_json(
        run_dir / "task_metadata_with_sources.json",
        {
            **task_result,
            "task_json": str(task_json),
            "source_data_dir": str(source_data_dir),
        },
    )


def build_manual_resume_metadata(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    session_id: str | None,
    container_name: str | None = None,
) -> dict[str, str | None]:
    if not session_id:
        return {
            "manual_resume_note": None,
            "manual_resume_command": None,
        }

    if args.use_docker and container_name and not args.dry_run:
        inner_cmd = [args.kimi_bin, "--session", session_id]
        return {
            "manual_resume_note": (
                "Run manual_resume_command while the per-task Docker container exists. "
                "Pass --keep-docker-container to keep it after the task finishes."
            ),
            "manual_resume_command": shlex.join(
                docker_exec_command(
                    args=args,
                    container_name=container_name,
                    inner_cmd=inner_cmd,
                    interactive=True,
                )
            ),
        }

    kimi_home = run_dir / "kimi_home"
    local_cmd = ["env", f"KIMI_CODE_HOME={kimi_home}", args.kimi_bin, "--session", session_id]
    return {
        "manual_resume_note": (
            "Run manual_resume_command to open the Kimi Code interactive session for this task. "
            "The saved config is removed after the run, so restore config.toml first if needed."
        ),
        "manual_resume_command": shlex.join(local_cmd),
    }


def event_role(event: dict[str, Any]) -> str:
    role = event.get("role") or event.get("type")
    if isinstance(role, str):
        return role.lower()
    message = event.get("message")
    if isinstance(message, dict) and isinstance(message.get("role"), str):
        return message["role"].lower()
    return ""


def event_text(event: dict[str, Any]) -> str:
    content = event.get("content")
    if content is None and isinstance(event.get("message"), dict):
        content = event["message"].get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(part for part in parts if part).strip()


def event_tool_calls(event: dict[str, Any]) -> list[dict[str, Any]]:
    calls = event.get("tool_calls")
    if calls is None and isinstance(event.get("message"), dict):
        calls = event["message"].get("tool_calls")
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def parse_tool_arguments(call: dict[str, Any]) -> tuple[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict):
        function = {}
    name = str(function.get("name") or call.get("name") or "unknown")
    arguments = function.get("arguments", call.get("arguments", {}))
    if isinstance(arguments, str):
        parsed = _loads_or_none(arguments)
        arguments = parsed if parsed is not None else {"raw": arguments}
    return name, arguments


def parse_kimi_output(
    stdout: str,
    fallback: str | None,
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None, str]:
    session_id = fallback
    usage = None
    final_text = ""
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
            usage = message["usage"]

        text = event_text(event)
        if event_role(event) == "assistant" and text:
            final_text = text

    payload, raw_message = parse_structured_message(final_text)
    if payload is not None:
        payload = normalize_turn_payload(payload)
        raw_message = json.dumps(payload, ensure_ascii=False)
    elif final_text.strip():
        payload = {
            "answer": final_text.strip(),
            "reasoning_summary": "none",
            "files_used": ["none"],
        }
        raw_message = final_text.strip()
    return session_id, usage, payload, raw_message


def normalize_turn_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or "answer" not in payload:
        return None

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


def _loads_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_embedded_json(text: str) -> dict[str, Any] | None:
    """Recover a turn-schema object from fenced or inline JSON in a free-form message."""
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.append(text)
    decoder = json.JSONDecoder()

    for candidate in candidates:
        normalized = normalize_turn_payload(_loads_or_none(candidate.strip()))
        if normalized is not None:
            return normalized

        for match in re.finditer(r"\{", candidate):
            try:
                payload, _ = decoder.raw_decode(candidate, match.start())
            except json.JSONDecodeError:
                continue
            normalized = normalize_turn_payload(payload)
            if normalized is not None:
                return normalized
    return None


def parse_structured_message(text: str) -> tuple[dict[str, Any] | None, str]:
    raw = (text or "").strip()
    if not raw:
        return None, ""
    return parse_embedded_json(raw), raw


def describe_agent_text(text: str) -> dict[str, Any]:
    """Render an assistant text block, unpacking it when it is the turn-schema JSON."""
    payload = normalize_turn_payload(_loads_or_none(text.strip())) or parse_embedded_json(text)
    if payload is None:
        return {"message": text}

    formatted: dict[str, Any] = {
        "response": payload["answer"],
        "reasoning_summary": payload["reasoning_summary"],
    }
    if payload["files_used"]:
        formatted["files_used"] = payload["files_used"]
    return formatted


def format_kimi_steps(
    stdout: str,
    *,
    prompt: str,
    turn_label: str,
    previous_session_id: str | None,
) -> dict[str, Any]:
    """Convert raw Kimi Code stream-json events into the compact judge trajectory format."""
    trace: dict[str, Any] = {
        "session": {},
        "turn": int(turn_label) if turn_label.isdigit() else turn_label,
        "user_request": prompt,
        "steps": [],
    }
    steps: list[dict[str, Any]] = []
    pending: dict[str, int] = {}
    result_text = ""

    for line in stdout.splitlines():
        event = _loads_or_none(line)
        if not isinstance(event, dict):
            continue

        role = event_role(event)
        session_id = event.get("session_id")
        if session_id and not trace["session"]:
            trace["session"] = {
                "session_id": session_id,
                "session_state": (
                    "new"
                    if previous_session_id is None
                    else "same_as_previous_turn"
                    if session_id == previous_session_id
                    else "changed_from_previous_turn"
                ),
            }

        if isinstance(event.get("usage"), dict):
            trace["usage"] = event["usage"]

        if role == "assistant":
            text = event_text(event)
            if text:
                result_text = text
                steps.append({"step": len(steps), **describe_agent_text(text)})

            for call in event_tool_calls(event):
                name, arguments = parse_tool_arguments(call)
                formatted: dict[str, Any] = {"step": len(steps), "tool": name}
                if isinstance(arguments, dict) and "command" in arguments:
                    formatted["command"] = str(arguments.get("command") or "")
                elif arguments not in ({}, None):
                    formatted["input"] = arguments
                formatted["output"] = ""
                steps.append(formatted)
                tool_id = str(call.get("id") or "")
                if tool_id:
                    pending[tool_id] = len(steps) - 1
            continue

        if role == "tool":
            tool_use_id = str(event.get("tool_call_id") or event.get("tool_use_id") or "")
            output = event_text(event)
            index = pending.pop(tool_use_id, None)
            if index is not None:
                steps[index]["output"] = output
                if event.get("is_error"):
                    steps[index]["is_error"] = True
            elif output:
                steps.append({"step": len(steps), "output": output})

    if result_text.strip():
        final_step = describe_agent_text(result_text)
        last_step = {key: value for key, value in steps[-1].items() if key != "step"} if steps else {}
        if final_step != last_step:
            steps.append({"step": len(steps), **final_step})

    trace["steps"] = steps
    return trace


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
    described = describe_agent_text(text)
    if "message" in described:
        print_text_block("message", str(described["message"]), color_code=COLOR_MAGENTA)
        return

    print_text_block("response", str(described.get("response") or ""), color_code=COLOR_MAGENTA)
    print_text_block("reasoning_summary", str(described.get("reasoning_summary") or ""), color_code=COLOR_MAGENTA)
    files_used = described.get("files_used")
    if files_used:
        print_text_block("files_used", "\n".join(str(path) for path in files_used), color_code=COLOR_MAGENTA)


class KimiEventFormatter:
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
        self.finished = False
        self.pending_tool_uses: dict[str, tuple[str, Any]] = {}

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
        role = event_role(event)
        color_code = COLOR_CYAN
        if role == "assistant":
            color_code = COLOR_MAGENTA
        elif role == "tool":
            color_code = COLOR_YELLOW
        elif role == "error" or event.get("is_error"):
            color_code = COLOR_RED

        self.print_session(event, color_code=color_code)
        self.print_user_request(color_code=color_code)

        if role == "assistant":
            self.print_assistant_event(event, color_code=color_code)
            return
        if role == "tool":
            self.print_tool_event(event, color_code=color_code)
            return

    def print_step_header(self, color_code: str) -> int:
        step = self.next_step
        self.next_step += 1
        print("", flush=True)
        print(paint(f"step {step}", COLOR_BOLD + color_code), flush=True)
        return step

    def print_assistant_event(self, event: dict[str, Any], *, color_code: str) -> None:
        text = event_text(event)
        if text:
            self.print_step_header(color_code)
            print_agent_text(text)
        for call in event_tool_calls(event):
            tool_id = str(call.get("id") or "")
            name, arguments = parse_tool_arguments(call)
            if tool_id:
                self.pending_tool_uses[tool_id] = (name, arguments)
            else:
                self.print_tool_step(name, arguments, output="", is_error=False)

    def print_tool_event(self, event: dict[str, Any], *, color_code: str) -> None:
        tool_id = str(event.get("tool_call_id") or event.get("tool_use_id") or "")
        tool_use = self.pending_tool_uses.pop(tool_id, None)
        output = event_text(event)
        if tool_use is None:
            if output:
                self.print_step_header(color_code)
                print_text_block("output", output, color_code=color_code)
            return
        name, arguments = tool_use
        self.print_tool_step(name, arguments, output=output, is_error=bool(event.get("is_error")))

    def print_tool_step(self, name: str, arguments: Any, *, output: str, is_error: bool) -> None:
        color_code = COLOR_RED if is_error else COLOR_YELLOW
        self.print_step_header(color_code)
        print_field("tool", name, color_code=color_code)
        if isinstance(arguments, dict) and "command" in arguments:
            print_text_block("command", str(arguments.get("command") or ""), color_code=color_code)
        elif arguments:
            print_text_block("input", json.dumps(arguments, ensure_ascii=False), color_code=color_code)
        print_text_block("output", output, color_code=color_code)

    def flush_pending_tool_uses(self) -> None:
        for tool_id in list(self.pending_tool_uses):
            name, arguments = self.pending_tool_uses.pop(tool_id)
            self.print_tool_step(name, arguments, output="", is_error=False)

    def finish(self, returncode: int) -> None:
        if self.finished:
            return
        self.finished = True
        self.flush_pending_tool_uses()
        print("", flush=True)
        color_code = COLOR_GREEN if returncode == 0 else COLOR_RED
        print(paint(f"turn {self.turn_label} finished", COLOR_BOLD + color_code), flush=True)
        print_field("returncode", returncode, color_code=color_code)


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
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("Failed to open Kimi Code subprocess pipes.")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_formatter = KimiEventFormatter(
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
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        raise subprocess.TimeoutExpired(cmd, timeout)

    stdout_thread.join()
    stderr_thread.join()
    stdout_formatter.finish(returncode)
    return returncode, "".join(stdout_chunks), "".join(stderr_chunks)


def run_kimi_turn(
    *,
    args: argparse.Namespace,
    prompt: str,
    turn_label: str,
    turn_dir: Path,
    work_dir: Path,
    kimi_home: Path,
    session_id: str | None,
    container_name: str | None = None,
) -> tuple[str | None, dict[str, Any]]:
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    last_message_path = turn_dir / "last_message.json"
    previous_session_id = session_id

    if args.dry_run:
        KimiEventFormatter(
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
        return session_id or "dry-run-session", result

    inner_cmd = kimi_command(
        args=args,
        prompt=prompt,
        session_id=session_id,
    )
    if args.use_docker:
        if not container_name:
            raise RuntimeError("Docker mode requires a per-task container name.")
        cmd = docker_exec_command(
            args=args,
            container_name=container_name,
            inner_cmd=inner_cmd,
        )
    else:
        cmd = inner_cmd

    process_env = os.environ.copy()
    if not args.use_docker:
        process_env["KIMI_CODE_HOME"] = str(kimi_home)

    start = time.time()
    returncode, stdout, stderr = run_command_streaming(
        cmd,
        timeout=args.timeout,
        env=process_env,
        prompt=prompt,
        turn_label=turn_label,
        previous_session_id=previous_session_id,
        cwd=work_dir,
    )
    elapsed = time.time() - start

    (turn_dir / "kimi_stdout.jsonl").write_text(stdout, encoding="utf-8")
    (turn_dir / "kimi_stderr.txt").write_text(stderr, encoding="utf-8")
    write_json(
        turn_dir / "formatted_steps.json",
        format_kimi_steps(
            stdout,
            prompt=prompt,
            turn_label=turn_label,
            previous_session_id=previous_session_id,
        ),
    )
    session_id, usage, payload, raw_message = parse_kimi_output(stdout, previous_session_id)
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
        raise RuntimeError(f"Kimi Code exited with code {returncode}; see {turn_dir}")
    if payload is None:
        raise RuntimeError(f"Kimi Code did not produce schema JSON; see {turn_dir}")
    if not session_id:
        raise RuntimeError(f"Could not determine Kimi Code session id; see {turn_dir}")

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
    run_dir = task_run_dir(args, task_info, run_name)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and run_dir.is_symlink():
        raise RuntimeError(f"Refusing to overwrite symlinked run path: {run_dir}")
    if args.overwrite and run_dir.exists():
        if not run_dir.is_dir():
            raise RuntimeError(f"Cannot overwrite non-directory run path: {run_dir}")
        print("", flush=True)
        print(paint(f"Removing existing task run: {run_dir}", COLOR_YELLOW), flush=True)
        shutil.rmtree(run_dir)
    try:
        run_dir.mkdir()
    except FileExistsError:
        print("", flush=True)
        print(
            paint(f"======= Skipped task {task_name}; run already exists: {run_dir} =======", COLOR_YELLOW),
            flush=True,
        )
        return {
            "task_domain": domain,
            "dataset_name": dataset,
            "task_id": task_id,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "skipped": True,
            "skip_reason": "run_directory_exists",
        }

    print("", flush=True)
    print(paint(f"======= Start running task {task_name} ... =======", COLOR_RED), flush=True)

    turns = load_turns(task_json, args.turn_limit)

    model_slug = slugify(args.kimi_model or "kimi-code-default")
    workspace_dir = run_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    kimi_home = run_dir / "kimi_home"
    container_name = (
        docker_container_name(args=args, task_info=task_info, run_name=run_name)
        if args.use_docker
        else None
    )
    if not args.use_docker and not args.dry_run:
        prepare_local_kimi_home(args=args, run_dir=run_dir)
    local_data_dir = prepare_workspace_data(
        source_data_dir,
        workspace_dir,
        materialize=not args.dry_run and not args.use_docker,
    )
    schema_path = run_dir / "kimi_turn.schema.json"
    write_json(schema_path, TURN_SCHEMA)

    task_result: dict[str, Any] = {
        **dataset_metadata(args),
        "task_domain": domain,
        "dataset_name": dataset,
        "task_id": task_id,
        "run_name": run_name,
        "model_slug": model_slug,
        "kimi_model": args.kimi_model,
        "kimi_model_source": getattr(args, "kimi_model_source", None),
        "kimi_model_cli_arg": (
            args.kimi_model if getattr(args, "kimi_model_explicit", False) else None
        ),
        "execution_mode": "docker" if args.use_docker else "local",
        "docker_image": args.docker_image if args.use_docker else None,
        "container_workspace": CONTAINER_WORKSPACE if args.use_docker else None,
        "container_home": CONTAINER_HOME if args.use_docker else None,
        "container_kimi_config": (
            CONTAINER_KIMI_CONFIG
            if args.use_docker and args.kimi_config is not None
            else None
        ),
        "kimi_config_file": str(args.kimi_config) if args.kimi_config else None,
        "docker_container_name": container_name,
        "kimi_home_dir": CONTAINER_KIMI_HOME if args.use_docker else str(kimi_home),
        "saved_kimi_home_dir": str(kimi_home),
        "local_data_dir": str(local_data_dir),
        "workspace_dir": str(workspace_dir),
        "run_dir": str(run_dir),
        "session_id": None,
        **build_manual_resume_metadata(
            args=args,
            run_dir=run_dir,
            session_id=None,
            container_name=container_name,
        ),
        "turns": [],
    }
    write_json(run_dir / "task_metadata.json", task_result)

    container_started = False
    try:
        if args.use_docker and not args.dry_run:
            if container_name is None:
                raise RuntimeError("Docker mode requires a per-task container name.")
            print(paint(f"Starting Docker task container {container_name} ...", COLOR_DIM), flush=True)
            start_task_container(args=args, container_name=container_name)
            container_started = True
            task_result["docker_container_started"] = True
            write_json(run_dir / "task_metadata.json", task_result)
            copy_task_data_to_container(
                args=args,
                container_name=container_name,
                source_data_dir=source_data_dir,
            )
            copy_kimi_config_to_container(args=args, container_name=container_name)
            task_result["data_materialization"] = {
                "mode": "docker_cp",
                "source_data_dir": str(source_data_dir),
                "container_data_dir": f"{CONTAINER_WORKSPACE}/data",
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
            session_id, result = run_kimi_turn(
                args=args,
                prompt=prompt,
                turn_label=str(turn_id),
                turn_dir=turn_dir,
                work_dir=workspace_dir,
                kimi_home=kimi_home,
                session_id=session_id,
                container_name=container_name,
            )
            task_result["session_id"] = session_id
            task_result.update(
                build_manual_resume_metadata(
                    args=args,
                    run_dir=run_dir,
                    session_id=session_id,
                    container_name=container_name,
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

        if args.use_docker and container_started and container_name is not None:
            sync = sync_workspace_from_container(
                args=args,
                container_name=container_name,
                workspace_dir=workspace_dir,
            )
            task_result["workspace_sync"] = sync
            write_json(run_dir / "task_metadata.json", task_result)
            if sync["returncode"] != 0:
                raise RuntimeError(f"Failed to sync Docker workspace: {sync['stderr']}")
            kimi_sync = sync_kimi_home_from_container(
                args=args,
                container_name=container_name,
                kimi_home=kimi_home,
            )
            task_result["kimi_home_sync"] = kimi_sync
            write_json(run_dir / "task_metadata.json", task_result)
            if kimi_sync["returncode"] != 0:
                raise RuntimeError(f"Failed to sync Kimi Code home: {kimi_sync['stderr']}")

        results_with_ground_truth = []
        for turn, result in zip(turns, task_result["turns"]):
            result_with_gt = dict(result)
            result_with_gt["ground_truth"] = turn.get("answer")
            results_with_ground_truth.append(result_with_gt)

        task_result["data_cleanup"] = finalize_workspace_data(args=args, local_data_dir=local_data_dir)
        write_json(run_dir / "task_metadata.json", task_result)

        write_json(run_dir / "results_with_ground_truth.json", results_with_ground_truth)
        write_task_metadata_with_sources(
            task_result=task_result,
            run_dir=run_dir,
            task_json=task_json,
            source_data_dir=source_data_dir,
        )

        print("", flush=True)
        print(paint(f"======= Finished task {task_name}. =======", COLOR_RED), flush=True)

        return task_result
    except Exception:
        if args.use_docker and container_started and container_name is not None:
            task_result["workspace_sync"] = sync_workspace_from_container(
                args=args,
                container_name=container_name,
                workspace_dir=workspace_dir,
            )
            write_json(run_dir / "task_metadata.json", task_result)
            task_result["kimi_home_sync"] = sync_kimi_home_from_container(
                args=args,
                container_name=container_name,
                kimi_home=kimi_home,
            )
            write_json(run_dir / "task_metadata.json", task_result)
        raise
    finally:
        if not args.use_docker:
            scrub_kimi_config(kimi_home)
        if args.use_docker and container_started and container_name is not None:
            if args.keep_docker_container:
                task_result["docker_container_cleanup"] = {
                    "removed": False,
                    "reason": "keep_docker_container",
                    "container_name": container_name,
                }
            else:
                task_result["docker_container_cleanup"] = {
                    "removed": True,
                    "container_name": container_name,
                    **remove_task_container(args=args, container_name=container_name),
                }
            write_json(run_dir / "task_metadata.json", task_result)
            if (run_dir / "task_metadata_with_sources.json").exists():
                write_task_metadata_with_sources(
                    task_result=task_result,
                    run_dir=run_dir,
                    task_json=task_json,
                    source_data_dir=source_data_dir,
                )


def task_run_dir(args: argparse.Namespace, task_info: dict[str, str], run_name: str) -> Path:
    return (
        args.output_dir
        / run_name
        / task_info["task_domain"]
        / task_info["dataset_name"]
        / task_info["task_id"]
    )


def evaluate_completed_task(task_result: dict[str, Any]) -> dict[str, Any]:
    judge_script = Path(__file__).resolve().with_name("judge.py")
    run_dir = Path(task_result["run_dir"])
    print("", flush=True)
    print(paint(f"======= Evaluating {run_dir} ... =======", COLOR_RED), flush=True)
    completed = subprocess.run(
        [sys.executable, str(judge_script), "--run-dir", str(run_dir)],
        cwd=judge_script.parent,
        env=os.environ.copy(),
        check=False,
    )
    return {"run_dir": str(run_dir), "returncode": completed.returncode}


def run_task_pipeline(
    *,
    args: argparse.Namespace,
    task_info: dict[str, str],
    run_name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    task_result = run_task(args=args, task_info=task_info, run_name=run_name)
    if task_result.get("skipped") or not args.judge:
        return task_result, None

    try:
        evaluation = evaluate_completed_task(task_result)
    except Exception as exc:  # noqa: BLE001
        evaluation = {
            "run_dir": task_result["run_dir"],
            "returncode": 1,
            "error": str(exc),
        }
    return task_result, evaluation


def print_run_config(
    *,
    args: argparse.Namespace,
    run_name: str,
    results_root: Path,
    selected_count: int,
    total_count: int,
) -> None:
    print("Kimi Code LongDS run configuration:", flush=True)
    print(f"  run_name: {run_name}", flush=True)
    print(f"  task_root: {args.task_root}", flush=True)
    print(f"  longds_version: {args.longds_version}", flush=True)
    print(f"  split: {args.split}", flush=True)
    print(f"  task_list_name: {args.task_list_name}", flush=True)
    print(f"  data_root: {args.data_root}", flush=True)
    print(f"  output_dir: {results_root}", flush=True)
    print(f"  kimi_bin: {args.kimi_bin}", flush=True)
    print(f"  kimi_model: {args.kimi_model or 'config-default'}", flush=True)
    print(f"  kimi_model_source: {getattr(args, 'kimi_model_source', 'unknown')}", flush=True)
    print(
        "  kimi_model_cli_arg: "
        f"{args.kimi_model if getattr(args, 'kimi_model_explicit', False) else 'none'}",
        flush=True,
    )
    print(f"  kimi_config: {args.kimi_config or 'none'}", flush=True)
    print(f"  analysis_python: {args.analysis_python}", flush=True)
    print("  permission_mode: auto (fixed by kimi --prompt)", flush=True)
    print(f"  use_docker: {args.use_docker}", flush=True)
    if args.use_docker:
        print(f"  docker_bin: {args.docker_bin}", flush=True)
        print(f"  docker_image: {args.docker_image}", flush=True)
        print(f"  docker_user: {docker_user_arg(args)}", flush=True)
        print(f"  docker_network: {args.docker_network or 'default'}", flush=True)
        print(f"  docker_memory: {args.docker_memory or 'default'}", flush=True)
        print(f"  docker_cpus: {args.docker_cpus or 'default'}", flush=True)
        print(f"  docker_container_prefix: {args.docker_container_prefix}", flush=True)
        print(f"  keep_docker_container: {args.keep_docker_container}", flush=True)
        print(f"  docker_env_files: {len(args.docker_env_file)}", flush=True)
        print(f"  docker_extra_env: {len(args.docker_env)}", flush=True)
    print(f"  kimi_extra_args: {len(args.kimi_arg)}", flush=True)
    print(f"  start_index: {args.start_index}", flush=True)
    print(f"  task_limit: {args.task_limit if args.task_limit is not None else 'all'}", flush=True)
    print(f"  turn_limit: {args.turn_limit if args.turn_limit is not None else 'all'}", flush=True)
    print(f"  timeout: {args.timeout}", flush=True)
    print(f"  run_parallel: {args.run_parallel}", flush=True)
    print(f"  overwrite: {args.overwrite}", flush=True)
    print(f"  selected_tasks: {selected_count} / {total_count}", flush=True)
    print(f"  keep_data: {args.keep_data}", flush=True)
    print(f"  dry_run: {args.dry_run}", flush=True)
    print(f"  judge: {args.judge}", flush=True)


def main() -> int:
    args = parse_args()
    args.kimi_model_explicit = args.kimi_model is not None
    resolve_dataset(args)
    args.output_dir = result_root(args)
    if args.kimi_config is not None:
        args.kimi_config = args.kimi_config.resolve()
        if not args.kimi_config.is_file():
            raise FileNotFoundError(f"--kimi-config does not exist: {args.kimi_config}")
    if args.kimi_model is None:
        args.kimi_model = kimi_config_model(args.kimi_config)
        args.kimi_model_source = "config" if args.kimi_model else "config-default"
    else:
        args.kimi_model_source = "cli"
    if args.analysis_python is None:
        args.analysis_python = "/usr/local/bin/python" if args.use_docker else sys.executable
    if args.task_limit is not None and args.task_limit < 0:
        raise ValueError("--task-limit must be non-negative")
    if args.turn_limit is not None and args.turn_limit < 0:
        raise ValueError("--turn-limit must be non-negative")
    if args.run_parallel < 1:
        raise ValueError("--run-parallel must be at least 1")
    if args.dry_run and args.judge:
        raise ValueError("--judge cannot be combined with --dry-run")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(args.kimi_model or "default")
    run_name = args.run_name or f"kimi_code_{model_slug}_{timestamp}"
    validate_run_name(run_name)
    results_root = args.output_dir
    results_root.mkdir(parents=True, exist_ok=True)

    task_list = load_task_list(args)
    selected = task_list[args.start_index :]
    if args.task_limit is not None:
        selected = selected[: args.task_limit]

    print_run_config(
        args=args,
        run_name=run_name,
        results_root=results_root,
        selected_count=len(selected),
        total_count=len(task_list),
    )

    summary = RunSummary(args, "kimi_code", args.kimi_model, run_name, selected, results_root / run_name)
    completed_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0

    def record_success(
        task_result: dict[str, Any],
        evaluation: dict[str, Any] | None,
    ) -> bool:
        nonlocal completed_tasks, failed_tasks, skipped_tasks
        if task_result.get("skipped"):
            summary.record(task_result, "skipped")
            skipped_tasks += 1
            return True
        summary.record(task_result, "skipped" if task_result.get("skipped") else
                       "dry_run" if args.dry_run else "completed")
        completed_tasks += 1
        if evaluation is not None:
            task_result["evaluation"] = evaluation
            write_json(Path(task_result["run_dir"]) / "task_metadata.json", task_result)
        if evaluation is not None and evaluation["returncode"] != 0:
            summary.judge_failed(task_result)
            failed_tasks += 1
            print(f"ERROR: evaluation failed: {evaluation}", file=sys.stderr)
            return False
        return True

    def record_error(task_info: dict[str, str], exc: Exception) -> None:
        nonlocal failed_tasks
        summary.record(task_info, "failed")
        failed_tasks += 1
        run_dir = task_run_dir(args, task_info, run_name)
        error = {**dataset_metadata(args), "task": task_info, "run_dir": str(run_dir), "error": str(exc)}
        write_json(run_dir / "error.json", error)
        print(f"ERROR: {error}", file=sys.stderr)

    def execute(task_info: dict[str, str]) -> bool:
        try:
            task_result, evaluation = run_task_pipeline(
                args=args,
                task_info=task_info,
                run_name=run_name,
            )
            return record_success(task_result, evaluation)
        except Exception as exc:
            record_error(task_info, exc)
            return False

    def print_final_status() -> None:
        print(
            "Finished Kimi Code LongDS run: "
            f"completed_tasks={completed_tasks}, "
            f"failed_tasks={failed_tasks}, "
            f"skipped_tasks={skipped_tasks}, "
            f"selected_tasks={len(selected)}",
            flush=True,
        )

    failed = False
    if args.run_parallel == 1:
        for task_info in selected:
            if not execute(task_info):
                failed = True
    elif selected:
        max_workers = min(args.run_parallel, len(selected))
        next_index = 0
        futures: dict[
            Future[tuple[dict[str, Any], dict[str, Any] | None]],
            tuple[int, dict[str, str]],
        ] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while next_index < max_workers:
                task_info = selected[next_index]
                future = pool.submit(
                    run_task_pipeline,
                    args=args,
                    task_info=task_info,
                    run_name=run_name,
                )
                futures[future] = (next_index, task_info)
                next_index += 1

            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, task_info = futures.pop(future)
                    try:
                        task_result, evaluation = future.result()
                        succeeded = record_success(task_result, evaluation)
                    except Exception as exc:
                        record_error(task_info, exc)
                        succeeded = False

                    if not succeeded:
                        failed = True

                for _ in completed:
                    if next_index >= len(selected):
                        break
                    next_task = selected[next_index]
                    next_future = pool.submit(
                        run_task_pipeline,
                        args=args,
                        task_info=next_task,
                        run_name=run_name,
                    )
                    futures[next_future] = (next_index, next_task)
                    next_index += 1

    summary.save()
    print_final_status()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
# python run_kimi_longds.py --task-limit 1 --turn-limit 1
