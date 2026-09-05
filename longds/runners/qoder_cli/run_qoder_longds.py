#!/usr/bin/env python3
"""Run LongDS tasks directly with Qoder, without importing DSGym."""

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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_summary import RunSummary
from src.longds_dataset import (
    add_dataset_arguments, resolve_dataset, dataset_metadata, load_task_list, load_turns, result_root,
)

from prompt import build_output_contract, build_turn_prompt


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

CONTAINER_WORKSPACE = "/workspace"
CONTAINER_HOME = f"{CONTAINER_WORKSPACE}/.home"
CONTAINER_QODER_CONFIG = "/tmp/longds_qoder_config"
CONTAINER_QODER_SETTINGS = "/tmp/longds_qoder_settings.json"
FIXED_DOCKER_ENV = {
    "HOME": CONTAINER_HOME,
    "QODER_CONFIG_DIR": CONTAINER_QODER_CONFIG,
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONWARNINGS": "ignore::FutureWarning",
}
DEFAULT_DOCKER_ENV_KEYS = (
    "QODER_PERSONAL_ACCESS_TOKEN",
    "QODER_MODEL",
    "QODER_SUBAGENT_MODEL",
    "QODER_APPEND_SYSTEM_PROMPT",
    "QODER_MCP_LAZY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
QODER_CONFIG_SEED_ENTRIES = (
    ".auth",
    ".models",
    "settings.json",
    "state.json",
    "installation_id",
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run LongDS-Bench directly with Qoder headless sessions."
    )
    add_dataset_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Result base (default: ./results in the current directory). Appends longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/.",
    )
    parser.add_argument("--qoder-bin", default="qoder", help="Qoder executable.")
    parser.add_argument(
        "--qoder-model",
        default=None,
        help="Model passed to `qoder --model`, for example auto, lite, performance, or a BYOK key. "
        "Omit to use model.name from --qoder-settings.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Value for `qoder --reasoning-effort`. Omit to use settings.json. Levels include none, "
        "minimal, low, medium, high, max and xhigh, but the accepted set is reported per model, so it "
        "is not validated here.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Value for `qoder --context-window`. Omit to use settings.json. Only supported windows "
        "are honoured by the selected model.",
    )
    parser.add_argument(
        "--analysis-python",
        default=None,
        help=(
            "Python executable Qoder should use for data analysis commands. Defaults to the current "
            "Python locally and /usr/local/bin/python with --use-docker."
        ),
    )
    parser.add_argument(
        "--permission-mode",
        default="bypass_permissions",
        choices=["default", "accept_edits", "bypass_permissions", "dont_ask", "auto"],
        help="Qoder permission mode. Default: bypass_permissions, which imposes no filesystem "
        "boundary. Use auto to confine writes to the workspace, at the cost of false denials on "
        "in-workspace shell redirection.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Pass `--sandbox` to Qoder. The backend comes from QODER_SANDBOX. Off by default "
        "because analysis needs the external Python executable.",
    )
    parser.add_argument(
        "--allowed-tools",
        default=None,
        help="Optional value for `qoder --allowed-tools`, for example 'Read,Bash'.",
    )
    parser.add_argument(
        "--disallowed-tools",
        default=None,
        help="Optional value for `qoder --disallowed-tools`.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Value for `qoder --max-turns`: agent loop cap inside one LongDS turn.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional value for `qoder --max-output-tokens`.",
    )
    parser.add_argument(
        "--session-flag",
        default="resume",
        choices=["resume", "session-id"],
        help="Flag used to continue a task session after the first turn. Default: resume.",
    )
    parser.add_argument(
        "--qoder-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra raw argument appended to every Qoder invocation. Repeatable.",
    )
    parser.add_argument(
        "--qoder-config-dir",
        type=Path,
        default=Path.home() / ".qoder",
        help=(
            "Host Qoder user configuration directory. In Docker mode only the minimal login and "
            f"model settings are copied to {CONTAINER_QODER_CONFIG}; large session/cache data is skipped."
        ),
    )
    parser.add_argument(
        "--qoder-settings",
        type=Path,
        default=script_dir / "settings.json",
        help=(
            "Qoder settings JSON passed through `qoder --settings`. Defaults to settings.json in "
            f"this directory. With --use-docker it is copied to {CONTAINER_QODER_SETTINGS}."
        ),
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help=(
            "Run Qoder inside one Docker container per task. Task data is copied into "
            f"{CONTAINER_WORKSPACE}; all turns of the task share one container and Qoder session."
        ),
    )
    parser.add_argument("--docker-bin", default="docker", help="Docker CLI executable.")
    parser.add_argument(
        "--docker-image",
        default="longds-qoder:latest",
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
        default="longds-qoder",
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
        "--task-limit",
        type=int,
        default=None,
        help="Number of tasks to run after --start-index. Omit to run every task.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the task list.")
    parser.add_argument("--turn-limit", type=int, default=None, help="Maximum LongDS turns per task.")
    parser.add_argument(
        "--timeout", type=int, default=7200
        , help="Timeout per Qoder CLI turn, seconds."
    )
    parser.add_argument(
        "--turn-retries",
        type=int,
        default=2,
        help="Extra attempts for a turn that fails on an infrastructure fault such as a dropped "
        "connection or a 5xx response. Model failures are never retried. 0 disables retrying.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=30.0,
        help="Seconds to wait before the first retry; doubled for each further attempt.",
    )
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
        help="Optional run directory name. Defaults to qoder_<model>_YYYYmmdd_HHMMSS.",
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
        help="Write prompts and metadata without invoking Qoder CLI or copying task data.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run judge.py immediately after each successfully completed task.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "qoder"


def validate_run_name(run_name: str) -> None:
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("--run-name must be a single directory name without path separators")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def qoder_settings_model(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        return {}
    model = payload["model"]
    return {
        "name": model.get("name") if isinstance(model.get("name"), str) else None,
        "reasoning_effort": (
            model.get("reasoningEffort")
            if isinstance(model.get("reasoningEffort"), str)
            else None
        ),
        "context_window": (
            model.get("contextWindow")
            if isinstance(model.get("contextWindow"), int)
            else None
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


def qoder_command(
    *,
    args: argparse.Namespace,
    work_dir: Path | str,
    session_id: str | None,
    config_dir: str | None = None,
    settings_path: str | None = None,
) -> list[str]:
    """Build one headless Qoder invocation.

    Qoder has no output-schema flag, so the turn schema is enforced through the prompt
    and recovered from the final message when parsing.
    """
    cmd = [
        args.qoder_bin,
        "--print",
        "--input-format",
        "text",
        "--output-format",
        "stream-json",
        "--permission-mode",
        args.permission_mode,
        "--cwd",
        str(work_dir),
    ]
    if config_dir:
        cmd.extend(["--config-dir", config_dir])
    if settings_path:
        cmd.extend(["--settings", settings_path])
    if args.qoder_model and getattr(args, "qoder_model_explicit", True):
        cmd.extend(["--model", args.qoder_model])
    if args.reasoning_effort and getattr(args, "reasoning_effort_explicit", True):
        cmd.extend(["--reasoning-effort", args.reasoning_effort])
    if args.context_window is not None and getattr(args, "context_window_explicit", True):
        cmd.extend(["--context-window", str(args.context_window)])
    if args.max_turns is not None:
        cmd.extend(["--max-turns", str(args.max_turns)])
    if args.max_output_tokens is not None:
        cmd.extend(["--max-output-tokens", str(args.max_output_tokens)])
    if args.allowed_tools:
        cmd.extend(["--allowed-tools", args.allowed_tools])
    if args.disallowed_tools:
        cmd.extend(["--disallowed-tools", args.disallowed_tools])
    if args.sandbox:
        cmd.append("--sandbox")
    if session_id:
        cmd.extend([f"--{args.session_flag}", session_id])
    cmd.extend(args.qoder_arg)
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
    for path in args.docker_env_file:
        env.update(parse_env_file(path))
    for spec in args.docker_env:
        if "=" not in spec:
            continue
        key, value = spec.split("=", 1)
        if key.strip():
            env[key.strip()] = value
    return env


def docker_env_specs(args: argparse.Namespace, client_env: dict[str, str]) -> list[str]:
    specs = [f"{key}={value}" for key, value in FIXED_DOCKER_ENV.items()]
    seen = {spec.split("=", 1)[0] for spec in specs}
    for key in DEFAULT_DOCKER_ENV_KEYS:
        if client_env.get(key) and key not in seen:
            specs.append(key)
            seen.add(key)
    for path in args.docker_env_file:
        for key in parse_env_file(path):
            if key and key not in seen:
                specs.append(key)
                seen.add(key)
    for spec in args.docker_env:
        key = spec.split("=", 1)[0].strip()
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
    cmd.extend(["--user", docker_user_arg(args), "--workdir", CONTAINER_WORKSPACE])
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
                f"{CONTAINER_QODER_CONFIG} && tail -f /dev/null"
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
    cmd.extend(["--user", docker_user_arg(args), "--workdir", CONTAINER_WORKSPACE])
    cmd.append(container_name)
    cmd.extend(inner_cmd)
    return cmd


def docker_chown_user(args: argparse.Namespace) -> str | None:
    user = docker_user_arg(args)
    return user if re.fullmatch(r"\d+(?::\d+)?", user) else None


def run_docker_control(args: argparse.Namespace, cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, env=docker_client_env(args))


def start_task_container(*, args: argparse.Namespace, container_name: str) -> None:
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
            "cp",
            f"{source_data_dir.resolve()}/.",
            f"{container_name}:{CONTAINER_WORKSPACE}/data",
        ],
    )
    chown_container_paths(args=args, container_name=container_name, paths=[CONTAINER_WORKSPACE])


def seed_qoder_config_to_container(
    *,
    args: argparse.Namespace,
    container_name: str,
) -> list[str]:
    """Copy only login/model settings, not the host's large caches and old sessions."""
    copied: list[str] = []
    source_root = args.qoder_config_dir
    if source_root is None or not source_root.is_dir():
        return copied
    for name in QODER_CONFIG_SEED_ENTRIES:
        source = source_root / name
        if not source.exists():
            continue
        run_docker_control(
            args,
            [args.docker_bin, "cp", str(source.resolve()), f"{container_name}:{CONTAINER_QODER_CONFIG}/"],
        )
        copied.append(name)
    chown_container_paths(
        args=args,
        container_name=container_name,
        paths=[CONTAINER_QODER_CONFIG],
    )
    return copied


def copy_qoder_settings_to_container(
    *,
    args: argparse.Namespace,
    container_name: str,
) -> None:
    if args.qoder_settings is None:
        return
    run_docker_control(
        args,
        [
            args.docker_bin,
            "cp",
            str(args.qoder_settings.resolve()),
            f"{container_name}:{CONTAINER_QODER_SETTINGS}",
        ],
    )
    chown_container_paths(
        args=args,
        container_name=container_name,
        paths=[CONTAINER_QODER_SETTINGS],
    )


def chown_container_paths(
    *,
    args: argparse.Namespace,
    container_name: str,
    paths: list[str],
) -> None:
    chown_user = docker_chown_user(args)
    if chown_user is None:
        return
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
            *paths,
        ],
    )


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


def build_manual_resume_metadata(
    *,
    args: argparse.Namespace,
    session_id: str | None,
    container_name: str | None = None,
) -> dict[str, str | None]:
    if not session_id:
        return {
            "manual_resume_note": None,
            "manual_resume_command": None,
        }

    if args.use_docker and container_name and not args.dry_run:
        inner_cmd = [
            args.qoder_bin,
            "--config-dir",
            CONTAINER_QODER_CONFIG,
        ]
        if args.qoder_settings:
            inner_cmd.extend(["--settings", CONTAINER_QODER_SETTINGS])
        inner_cmd.extend(["--resume", session_id])
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

    local_cmd = [args.qoder_bin]
    if args.qoder_config_dir:
        local_cmd.extend(["--config-dir", str(args.qoder_config_dir)])
    if args.qoder_settings:
        local_cmd.extend(["--settings", str(args.qoder_settings)])
    local_cmd.extend(["--resume", session_id])
    return {
        "manual_resume_note": "Run manual_resume_command from the task workspace to open this Qoder session.",
        "manual_resume_command": shlex.join(local_cmd),
    }


def normalize_turn_payload(payload: Any) -> dict[str, Any] | None:
    """Coerce a candidate final message into the turn schema, or return None."""
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
        "files_used": [str(file_path) for file_path in files_used],
    }


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


def _loads_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# Qoder CLI leaves the result event's permission_denials empty even when a tool was blocked, so
# denials have to be recognized from the failed tool_result text instead.
PERMISSION_DENIAL_MARKERS = (
    "permission mode",
    "outside workspace boundary",
    "requires permission",
    "automatically denied",
    "permission denied for tool",
)


def is_permission_denial(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in PERMISSION_DENIAL_MARKERS)


def message_content_blocks(message: Any) -> list[dict[str, Any]]:
    """Normalize a stream-json message into a list of content blocks."""
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def describe_tool_failure(
    tool_use: dict[str, Any] | None,
    block: dict[str, Any],
) -> dict[str, Any]:
    message = tool_result_output(block)
    tool_input = (tool_use or {}).get("input")
    target = ""
    if isinstance(tool_input, dict):
        target = str(tool_input.get("command") or tool_input.get("file_path") or "")
    return {
        "tool": (tool_use or {}).get("name"),
        "target": target[:200],
        "message": message[:500],
        "permission_denied": is_permission_denial(message),
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


def parse_qoder_output(stdout: str, fallback: str | None) -> dict[str, Any]:
    """Extract session id, usage, cost and the final turn payload from a stream-json run."""
    session_id = fallback
    usage: dict[str, Any] | None = None
    total_cost_usd: Any = None
    total_credits: Any = None
    num_turns: Any = None
    terminal_reason: Any = None
    cli_permission_denials: list[Any] = []
    permission_denials: list[dict[str, Any]] = []
    tool_error_count = 0
    pending_tool_uses: dict[str, dict[str, Any]] = {}
    cli_error: str | None = None
    cli_error_code: Any = None
    cli_model: str | None = None
    cli_version: str | None = None
    structured: dict[str, Any] | None = None
    result_text = ""
    result_subtype = None
    result_is_error = False
    last_assistant_text = ""

    for line in stdout.splitlines():
        event = _loads_or_none(line)
        if not isinstance(event, dict):
            continue

        session_id = event.get("session_id") or session_id
        if event.get("subtype") == "init":
            # The init event is the only place the CLI reports which model the session resolved to.
            cli_model = event.get("model") or cli_model
            cli_version = event.get("qodercli_version") or cli_version
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        message = event.get("message")
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            usage = message["usage"]
        if isinstance(event.get("error"), str) and event["error"]:
            cli_error = event["error"]
        # A failing result event reports the cause in `errors` and `error_code`, leaving the
        # singular `error` unset, so a server-side failure is invisible without reading both.
        if isinstance(event.get("errors"), list):
            joined = "; ".join(str(item).strip() for item in event["errors"] if str(item).strip())
            if joined:
                cli_error = joined
        if event.get("error_code") is not None:
            cli_error_code = event["error_code"]

        extracted = extract_structured_output(event)
        if extracted is not None:
            structured = extracted

        for block in message_content_blocks(message):
            block_type = block.get("type")
            if block_type == "tool_use":
                tool_id = str(block.get("id") or "")
                if tool_id:
                    pending_tool_uses[tool_id] = block
            elif block_type == "tool_result" and block.get("is_error"):
                tool_use = pending_tool_uses.pop(str(block.get("tool_use_id") or ""), None)
                failure = describe_tool_failure(tool_use, block)
                tool_error_count += 1
                if failure["permission_denied"]:
                    permission_denials.append(failure)

        text = extract_text_from_message(message)
        if text:
            last_assistant_text = text

        if event.get("type") == "result" or "result" in event:
            if event.get("total_cost_usd") is not None:
                total_cost_usd = event.get("total_cost_usd")
            if event.get("total_credits") is not None:
                total_credits = event.get("total_credits")
            if event.get("num_turns") is not None:
                num_turns = event.get("num_turns")
            if event.get("terminal_reason") is not None:
                terminal_reason = event.get("terminal_reason")
            if isinstance(event.get("permission_denials"), list):
                cli_permission_denials = event["permission_denials"]
            result_subtype = event.get("subtype") or result_subtype
            result_is_error = bool(event.get("is_error")) or result_is_error
            raw_result = event.get("result")
            if isinstance(raw_result, str) and raw_result.strip():
                result_text = raw_result
            elif isinstance(raw_result, dict) and structured is None:
                structured = raw_result

    payload, answer_source = resolve_turn_payload(
        structured=structured,
        result_text=result_text,
        assistant_text=last_assistant_text,
    )
    return {
        "session_id": session_id,
        "usage": usage,
        "total_cost_usd": total_cost_usd,
        "total_credits": total_credits,
        "num_turns": num_turns,
        "terminal_reason": terminal_reason,
        "permission_denials": permission_denials,
        "cli_permission_denials": cli_permission_denials,
        "tool_error_count": tool_error_count,
        "cli_error": cli_error,
        "cli_error_code": cli_error_code,
        "cli_model": cli_model,
        "cli_version": cli_version,
        "payload": payload,
        "answer_source": answer_source,
        "result_subtype": result_subtype,
        "result_is_error": result_is_error,
        "raw_final_message": result_text or last_assistant_text,
    }


def resolve_turn_payload(
    *,
    structured: dict[str, Any] | None,
    result_text: str,
    assistant_text: str,
) -> tuple[dict[str, Any] | None, str]:
    """Pick the best available final payload and report where it came from."""
    normalized = normalize_turn_payload(structured)
    if normalized is not None:
        return normalized, "structured_output"

    for text, source in ((result_text, "result"), (assistant_text, "assistant_message")):
        text = (text or "").strip()
        if not text:
            continue
        embedded = parse_embedded_json(text)
        if embedded is not None:
            return embedded, f"{source}_json"
        return {"answer": text, "reasoning_summary": "", "files_used": []}, f"{source}_raw_text"

    return None, "missing"


def format_qoder_steps(
    stdout: str,
    *,
    prompt: str,
    turn_label: str,
    previous_session_id: str | None,
) -> dict[str, Any]:
    """Convert raw Qoder CLI stream-json events into the compact judge trajectory format."""
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

        event_type = str(event.get("type", ""))
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

        if event_type == "result":
            if isinstance(event.get("usage"), dict):
                trace["usage"] = event["usage"]
            if event.get("total_cost_usd") is not None:
                trace["total_cost_usd"] = event.get("total_cost_usd")
            if isinstance(event.get("result"), str):
                result_text = event["result"]
            continue

        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                steps.append({"step": len(steps), **describe_agent_text(str(block.get("text") or ""))})
            elif block_type == "tool_use":
                formatted: dict[str, Any] = {"step": len(steps), "tool": block.get("name")}
                tool_input = block.get("input")
                if isinstance(tool_input, dict) and "command" in tool_input:
                    formatted["command"] = str(tool_input.get("command") or "")
                elif tool_input is not None:
                    formatted["input"] = tool_input
                formatted["output"] = ""
                steps.append(formatted)
                tool_id = str(block.get("id") or "")
                if tool_id:
                    pending[tool_id] = len(steps) - 1
            elif block_type == "tool_result":
                tool_use_id = str(block.get("tool_use_id") or "")
                output = tool_result_output(block)
                index = pending.pop(tool_use_id, None)
                if index is not None:
                    steps[index]["output"] = output
                    if block.get("is_error"):
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


def tool_result_output(block: dict[str, Any]) -> str:
    output = block.get("content")
    if isinstance(output, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in output
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if output is None:
        return ""
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False)
    return str(output)


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
    print_text_block(
        "reasoning_summary", str(described.get("reasoning_summary") or ""), color_code=COLOR_MAGENTA
    )
    files_used = described.get("files_used")
    if files_used:
        print_text_block(
            "files_used",
            "\n".join(str(path) for path in files_used),
            color_code=COLOR_MAGENTA,
        )



class QoderEventFormatter:
    """Render Qoder CLI stream-json events as colorized step blocks while the turn runs."""

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

        event = _loads_or_none(text)
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
        if event_type == "error" or event.get("is_error"):
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
            output = tool_result_output(block)
            is_error = bool(block.get("is_error"))
            if tool_use is not None:
                self.print_tool_step(tool_use, output=output, is_error=is_error)
            elif output:
                self.print_step_header(color_code)
                print_text_block("output", output, color_code=color_code)

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
        print_field("subtype", event.get("subtype"), color_code=color_code)
        print_field("terminal_reason", event.get("terminal_reason"), color_code=color_code)
        print_field("agent_turns", event.get("num_turns"), color_code=color_code)
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
        print_text_block(
            "event",
            json.dumps({"type": event_type, **event}, ensure_ascii=False),
            color_code=color_code,
        )


def print_stderr_line(line: str) -> None:
    print(paint(line.rstrip("\n"), COLOR_RED), file=sys.stderr, flush=True)


def warn_permission_denials(denials: list[dict[str, Any]], *, turn_label: str) -> None:
    """Make blocked tool calls visible: a denied turn still reports subtype success."""
    if not denials:
        return

    print("", flush=True)
    print(
        paint(f"turn {turn_label}: {len(denials)} tool call(s) blocked by permissions", COLOR_RED),
        flush=True,
    )
    for denial in denials:
        target = f" {denial['target']}" if denial["target"] else ""
        print(paint(f"  {denial['tool']}{target}", COLOR_RED), flush=True)
        print(paint(f"    {denial['message'].splitlines()[0]}", COLOR_DIM), flush=True)


def stream_pipe(pipe: Any, chunks: list[str], target: Any, *, formatter: Any | None = None) -> None:
    for line in iter(pipe.readline, ""):
        chunks.append(line)
        if formatter is None:
            print(line, end="", file=target, flush=True)
        else:
            formatter(line)
    pipe.close()


# Only infrastructure faults are worth retrying. Retrying a turn the model simply answered badly
# would hand it extra attempts at the benchmark task and inflate the score, so this is an allowlist
# of transient conditions rather than a list of failures to exclude.
TRANSIENT_ERROR_MARKERS = (
    "connection interrupted",
    "connection error",
    "connection reset",
    "connection closed",
    "socket hang up",
    "econnreset",
    "etimedout",
    "enotfound",
    "eai_again",
    "fetch failed",
    "network error",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "rate limit",
    "too many requests",
    "overloaded",
)


def transient_failure_reason(result: dict[str, Any]) -> str | None:
    """Describe why a failed turn looks transient, or return None when it must not be retried."""
    code = result.get("cli_error_code")
    if isinstance(code, int) and (code >= 500 or code == 429):
        return f"error_code {code}"

    message = str(result.get("cli_error") or "")
    lowered = message.lower()
    for marker in TRANSIENT_ERROR_MARKERS:
        if marker in lowered:
            return message.splitlines()[0]
    return None


def archive_failed_attempt(turn_dir: Path, attempt: int) -> None:
    """Keep a failed attempt's raw logs, which the next attempt would otherwise overwrite."""
    for name in ("qoder_stdout.jsonl", "qoder_stderr.txt", "formatted_steps.json"):
        source = turn_dir / name
        if source.exists():
            source.replace(turn_dir / f"attempt_{attempt}_{name}")


def run_command_streaming(
    cmd: list[str],
    *,
    cwd: Path,
    prompt: str,
    turn_label: str,
    previous_session_id: str | None,
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
        cwd=str(cwd),
        env=env,
    )
    if proc.stdout is None or proc.stderr is None or proc.stdin is None:
        raise RuntimeError("Failed to open Qoder CLI subprocess pipes.")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_formatter = QoderEventFormatter(
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
        proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        raise subprocess.TimeoutExpired(cmd, timeout)

    stdout_thread.join()
    stderr_thread.join()
    return returncode, "".join(stdout_chunks), "".join(stderr_chunks)


def run_qoder_turn(
    *,
    args: argparse.Namespace,
    prompt: str,
    turn_label: str,
    turn_dir: Path,
    work_dir: Path,
    session_id: str | None,
    container_name: str | None = None,
) -> tuple[str | None, dict[str, Any]]:
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    last_message_path = turn_dir / "last_message.json"
    previous_session_id = session_id

    if args.dry_run:
        QoderEventFormatter(
            turn_label=turn_label,
            user_request=prompt,
            previous_session_id=previous_session_id,
        ).print_user_request()
        result = {
            "success": True,
            "dry_run": True,
            "command": qoder_command(
                args=args,
                work_dir=CONTAINER_WORKSPACE if args.use_docker else work_dir,
                session_id=session_id,
                config_dir=(
                    CONTAINER_QODER_CONFIG
                    if args.use_docker
                    else str(args.qoder_config_dir)
                    if args.qoder_config_dir
                    else None
                ),
                settings_path=(
                    CONTAINER_QODER_SETTINGS
                    if args.use_docker and args.qoder_settings
                    else str(args.qoder_settings)
                    if args.qoder_settings
                    else None
                ),
            ),
            "cli_model": None,
            "answer": "",
            "reasoning_summary": "",
            "files_used": [],
            "elapsed_seconds": 0.0,
            "returncode": 0,
        }
        write_json(turn_dir / "result.json", result)
        return session_id or "dry-run-session", result

    attempt_history: list[dict[str, Any]] = []
    max_attempts = max(1, args.turn_retries + 1)

    for attempt in range(1, max_attempts + 1):
        # Always resume the session this turn started from. Retrying with the failed attempt's own
        # session id would stack a partially answered turn on top of itself.
        inner_cmd = qoder_command(
            args=args,
            work_dir=CONTAINER_WORKSPACE if args.use_docker else work_dir,
            session_id=previous_session_id,
            config_dir=(
                CONTAINER_QODER_CONFIG
                if args.use_docker
                else str(args.qoder_config_dir)
                if args.qoder_config_dir
                else None
            ),
            settings_path=(
                CONTAINER_QODER_SETTINGS
                if args.use_docker and args.qoder_settings
                else str(args.qoder_settings)
                if args.qoder_settings
                else None
            ),
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

        start = time.time()
        returncode, stdout, stderr = run_command_streaming(
            cmd,
            cwd=work_dir,
            timeout=args.timeout,
            env=docker_client_env(args) if args.use_docker else os.environ.copy(),
            prompt=prompt,
            turn_label=turn_label,
            previous_session_id=previous_session_id,
        )
        elapsed = time.time() - start

        (turn_dir / "qoder_stdout.jsonl").write_text(stdout, encoding="utf-8")
        (turn_dir / "qoder_stderr.txt").write_text(stderr, encoding="utf-8")
        write_json(
            turn_dir / "formatted_steps.json",
            format_qoder_steps(
                stdout,
                prompt=prompt,
                turn_label=turn_label,
                previous_session_id=previous_session_id,
            ),
        )

        parsed = parse_qoder_output(stdout, previous_session_id)
        payload = parsed["payload"]
        session_id = parsed["session_id"]
        if payload is not None:
            write_json(last_message_path, payload)

        result = {
            "success": returncode == 0 and payload is not None and not parsed["result_is_error"],
            "returncode": returncode,
            "session_id": session_id,
            "cli_model": parsed["cli_model"],
            "cli_version": parsed["cli_version"],
            "usage": parsed["usage"],
            "total_cost_usd": parsed["total_cost_usd"],
            "total_credits": parsed["total_credits"],
            "agent_turns": parsed["num_turns"],
            "terminal_reason": parsed["terminal_reason"],
            "permission_denials": parsed["permission_denials"],
            "cli_permission_denials": parsed["cli_permission_denials"],
            "tool_error_count": parsed["tool_error_count"],
            "cli_error": parsed["cli_error"],
            "cli_error_code": parsed["cli_error_code"],
            "answer_source": parsed["answer_source"],
            "result_subtype": parsed["result_subtype"],
            "result_is_error": parsed["result_is_error"],
            "elapsed_seconds": elapsed,
            "answer": payload.get("answer", "") if payload else "",
            "reasoning_summary": payload.get("reasoning_summary", "") if payload else "",
            "files_used": payload.get("files_used", []) if payload else [],
            "raw_last_message": parsed["raw_final_message"],
        }

        failure: str | None = None
        if returncode != 0:
            failure = f"Qoder CLI exited with code {returncode}"
        elif parsed["result_is_error"]:
            # A failed turn can still exit 0, for example on authentication or permission errors.
            detail = (
                parsed["cli_error"]
                or parsed["raw_final_message"][:200]
                or parsed["result_subtype"]
            )
            failure = f"Qoder CLI reported a failed turn ({detail})"
        elif payload is None:
            failure = "Qoder CLI did not produce a final answer"
        elif not session_id:
            failure = "Could not determine Qoder CLI session id"

        attempt_history.append(
            {
                "attempt": attempt,
                "returncode": returncode,
                "result_subtype": parsed["result_subtype"],
                "cli_error": parsed["cli_error"],
                "cli_error_code": parsed["cli_error_code"],
                "elapsed_seconds": elapsed,
                "failure": failure,
            }
        )
        result["attempt"] = attempt
        result["attempts"] = attempt_history
        write_json(turn_dir / "result.json", result)
        warn_permission_denials(parsed["permission_denials"], turn_label=turn_label)

        if failure is None:
            return session_id, result

        reason = transient_failure_reason(result)
        if reason is None or attempt == max_attempts:
            raise RuntimeError(f"{failure}; see {turn_dir}")

        archive_failed_attempt(turn_dir, attempt)
        delay = args.retry_backoff * (2 ** (attempt - 1))
        print(
            paint(
                f"turn {turn_label}: transient failure ({reason}); retrying in {delay:.0f}s "
                f"[attempt {attempt + 1}/{max_attempts}]",
                COLOR_RED,
            ),
            flush=True,
        )
        time.sleep(delay)

    raise RuntimeError(f"Qoder CLI turn failed after {max_attempts} attempts; see {turn_dir}")


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

    model_slug = slugify(args.qoder_model or "qoder-default")
    workspace_dir = run_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    container_name = (
        docker_container_name(args=args, task_info=task_info, run_name=run_name)
        if args.use_docker
        else None
    )
    local_data_dir = prepare_workspace_data(
        source_data_dir,
        workspace_dir,
        materialize=not args.dry_run and not args.use_docker,
    )
    schema_path = run_dir / "qoder_turn.schema.json"
    write_json(schema_path, TURN_SCHEMA)
    output_contract = build_output_contract(TURN_SCHEMA)

    task_result: dict[str, Any] = {
        **dataset_metadata(args),
        "task_domain": domain,
        "dataset_name": dataset,
        "task_id": task_id,
        "run_name": run_name,
        "model_slug": model_slug,
        "requested_model": (
            args.qoder_model if getattr(args, "qoder_model_explicit", False) else None
        ),
        "qoder_model": args.qoder_model,
        "qoder_model_source": getattr(args, "qoder_model_source", None),
        "qoder_model_cli_arg": (
            args.qoder_model if getattr(args, "qoder_model_explicit", False) else None
        ),
        "cli_model": None,
        "cli_model_by_turn": {},
        "cli_version": None,
        "cli_options": build_cli_options(args),
        "execution_mode": "docker" if args.use_docker else "local",
        "docker_image": args.docker_image if args.use_docker else None,
        "container_workspace": CONTAINER_WORKSPACE if args.use_docker else None,
        "container_home": CONTAINER_HOME if args.use_docker else None,
        "container_qoder_config": CONTAINER_QODER_CONFIG if args.use_docker else None,
        "container_qoder_settings": (
            CONTAINER_QODER_SETTINGS
            if args.use_docker and args.qoder_settings is not None
            else None
        ),
        "qoder_config_dir": str(args.qoder_config_dir) if args.qoder_config_dir else None,
        "qoder_settings_file": str(args.qoder_settings) if args.qoder_settings else None,
        "docker_container_name": container_name,
        "local_data_dir": str(local_data_dir),
        "workspace_dir": str(workspace_dir),
        "run_dir": str(run_dir),
        "session_id": None,
        **build_manual_resume_metadata(
            args=args,
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
            task_result["qoder_config_seed_entries"] = seed_qoder_config_to_container(
                args=args,
                container_name=container_name,
            )
            copy_qoder_settings_to_container(args=args, container_name=container_name)
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
                output_contract=output_contract,
            )
            print("", flush=True)
            print(paint(f"Start running {task_name} turn {turn_id} ...", COLOR_RED), flush=True)
            session_id, result = run_qoder_turn(
                args=args,
                prompt=prompt,
                turn_label=str(turn_id),
                turn_dir=turn_dir,
                work_dir=workspace_dir,
                session_id=session_id,
                container_name=container_name,
            )
            task_result["session_id"] = session_id
            task_result.update(
                build_manual_resume_metadata(
                    args=args,
                    session_id=session_id,
                    container_name=container_name,
                )
            )
            record_cli_model(task_result, turn_id=turn_id, result=result, task_name=task_name)
            task_result["turns"].append(
                {
                    "turn_id": turn_id,
                    "context": turn.get("context", ""),
                    "question": turn.get("question", ""),
                    "solution": result["answer"],
                    "reasoning_summary": result["reasoning_summary"],
                    "files_used": result["files_used"],
                    "answer_source": result.get("answer_source"),
                    "cli_model": result.get("cli_model"),
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

        results_with_ground_truth = []
        for turn, result in zip(turns, task_result["turns"]):
            result_with_gt = dict(result)
            result_with_gt["ground_truth"] = turn.get("answer")
            results_with_ground_truth.append(result_with_gt)

        task_result["data_cleanup"] = finalize_workspace_data(
            args=args,
            local_data_dir=local_data_dir,
        )
        write_json(run_dir / "task_metadata.json", task_result)
        write_json(run_dir / "results_with_ground_truth.json", results_with_ground_truth)
        write_json(
            run_dir / "task_metadata_with_sources.json",
            {
                **task_result,
                "task_json": str(task_json),
                "source_data_dir": str(source_data_dir),
            },
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
        raise
    finally:
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
                write_json(
                    run_dir / "task_metadata_with_sources.json",
                    {
                        **task_result,
                        "task_json": str(task_json),
                        "source_data_dir": str(source_data_dir),
                    },
                )


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


def build_cli_options(args: argparse.Namespace) -> dict[str, Any]:
    """Snapshot every option that changes how Qoder CLI behaves, so a run stays reproducible."""
    return {
        "permission_mode": args.permission_mode,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_effort_source": getattr(args, "reasoning_effort_source", None),
        "context_window": args.context_window,
        "context_window_source": getattr(args, "context_window_source", None),
        "settings_file": str(args.qoder_settings) if args.qoder_settings else None,
        "max_turns": args.max_turns,
        "max_output_tokens": args.max_output_tokens,
        "sandbox": args.sandbox,
        "allowed_tools": args.allowed_tools,
        "disallowed_tools": args.disallowed_tools,
        "session_flag": args.session_flag,
        "extra_args": list(args.qoder_arg),
    }


def record_cli_model(
    task_result: dict[str, Any],
    *,
    turn_id: Any,
    result: dict[str, Any],
    task_name: str,
) -> None:
    """Track the model the CLI reported per turn.

    `model_slug` reflects the effective configured model, but the CLI init event is the authoritative
    record of what served each turn. A session normally keeps one model for the whole task; a change
    mid-task means the turns are not comparable and is called out.
    """
    model = result.get("cli_model")
    if result.get("cli_version"):
        task_result["cli_version"] = result["cli_version"]
    if not model:
        return

    task_result["cli_model_by_turn"][str(turn_id)] = model
    distinct = sorted(set(task_result["cli_model_by_turn"].values()))
    task_result["cli_model"] = distinct[0] if len(distinct) == 1 else None
    if len(distinct) > 1:
        print(
            paint(
                f"WARNING: {task_name} changed model mid-task: "
                f"{json.dumps(task_result['cli_model_by_turn'], ensure_ascii=False)}",
                COLOR_RED,
            ),
            flush=True,
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
    print("Qoder LongDS run configuration:", flush=True)
    print(f"  run_name: {run_name}", flush=True)
    print(f"  task_root: {args.task_root}", flush=True)
    print(f"  longds_version: {args.longds_version}", flush=True)
    print(f"  split: {args.split}", flush=True)
    print(f"  task_list_name: {args.task_list_name}", flush=True)
    print(f"  data_root: {args.data_root}", flush=True)
    print(f"  output_dir: {results_root}", flush=True)
    print(f"  qoder_bin: {args.qoder_bin}", flush=True)
    print(f"  qoder_model: {args.qoder_model or 'config-default'}", flush=True)
    print(f"  qoder_model_source: {getattr(args, 'qoder_model_source', 'unknown')}", flush=True)
    print(
        "  qoder_model_cli_arg: "
        f"{args.qoder_model if getattr(args, 'qoder_model_explicit', False) else 'none'}",
        flush=True,
    )
    print(f"  qoder_settings: {args.qoder_settings or 'none'}", flush=True)
    print(f"  reasoning_effort: {args.reasoning_effort or 'config-default'}", flush=True)
    print(
        f"  context_window: {args.context_window if args.context_window is not None else 'config-default'}",
        flush=True,
    )
    auth_env = docker_client_env(args) if args.use_docker else os.environ
    print(
        "  qoder_personal_access_token: "
        f"{'set' if auth_env.get('QODER_PERSONAL_ACCESS_TOKEN') else 'not set (using Qoder login state)'}",
        flush=True,
    )
    print(f"  analysis_python: {args.analysis_python}", flush=True)
    print(f"  permission_mode: {args.permission_mode}", flush=True)
    print(f"  qoder_config_dir: {args.qoder_config_dir or 'none'}", flush=True)
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
    print(f"  sandbox: {args.sandbox}", flush=True)
    print(f"  allowed_tools: {args.allowed_tools or 'default'}", flush=True)
    print(f"  disallowed_tools: {args.disallowed_tools or 'none'}", flush=True)
    print(f"  max_turns: {args.max_turns if args.max_turns is not None else 'unlimited'}", flush=True)
    print(f"  session_flag: --{args.session_flag}", flush=True)
    print(f"  extra_qoder_args: {args.qoder_arg or 'none'}", flush=True)
    print(f"  start_index: {args.start_index}", flush=True)
    print(f"  task_limit: {args.task_limit if args.task_limit is not None else 'all'}", flush=True)
    print(f"  turn_limit: {args.turn_limit if args.turn_limit is not None else 'all'}", flush=True)
    print(f"  timeout: {args.timeout}", flush=True)
    print(f"  turn_retries: {args.turn_retries} (backoff {args.retry_backoff}s)", flush=True)
    print(f"  run_parallel: {args.run_parallel}", flush=True)
    print(f"  overwrite: {args.overwrite}", flush=True)
    print(f"  selected_tasks: {selected_count} / {total_count}", flush=True)
    print(f"  keep_data: {args.keep_data}", flush=True)
    print(f"  dry_run: {args.dry_run}", flush=True)
    print(f"  judge: {args.judge}", flush=True)


def main() -> int:
    args = parse_args()
    args.qoder_model_explicit = args.qoder_model is not None
    args.reasoning_effort_explicit = args.reasoning_effort is not None
    args.context_window_explicit = args.context_window is not None
    resolve_dataset(args)
    args.output_dir = result_root(args)
    if args.qoder_config_dir is not None:
        args.qoder_config_dir = args.qoder_config_dir.expanduser().resolve()
    if args.qoder_settings is not None:
        args.qoder_settings = args.qoder_settings.expanduser().resolve()
        if not args.qoder_settings.is_file():
            raise FileNotFoundError(f"--qoder-settings does not exist: {args.qoder_settings}")
    settings_model = qoder_settings_model(args.qoder_settings)
    if args.qoder_model is None:
        args.qoder_model = settings_model.get("name")
        args.qoder_model_source = "settings" if args.qoder_model else "config-default"
    else:
        args.qoder_model_source = "cli"
    if args.reasoning_effort is None:
        args.reasoning_effort = settings_model.get("reasoning_effort")
        args.reasoning_effort_source = (
            "settings" if args.reasoning_effort else "config-default"
        )
    else:
        args.reasoning_effort_source = "cli"
    if args.context_window is None:
        args.context_window = settings_model.get("context_window")
        args.context_window_source = "settings" if args.context_window else "config-default"
    else:
        args.context_window_source = "cli"
    args.docker_env_file = [path.expanduser().resolve() for path in args.docker_env_file]
    for path in args.docker_env_file:
        if not path.is_file():
            raise FileNotFoundError(f"--docker-env-file does not exist: {path}")
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
    model_slug = slugify(args.qoder_model or "default")
    run_name = args.run_name or f"qoder_{model_slug}_{timestamp}"
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

    summary = RunSummary(args, "qoder_cli", args.qoder_model, run_name, selected, results_root / run_name)
    completed_tasks = 0
    failed_tasks = 0

    def record_success(
        task_result: dict[str, Any],
        evaluation: dict[str, Any] | None,
    ) -> bool:
        nonlocal completed_tasks, failed_tasks
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
        except Exception as exc:  # noqa: BLE001
            record_error(task_info, exc)
            return False

    def print_final_status() -> None:
        print(
            "Finished Qoder LongDS run: "
            f"completed_tasks={completed_tasks}, "
            f"failed_tasks={failed_tasks}, "
            f"selected_tasks={len(selected)}",
            flush=True,
        )

    # A failed task never stops the run; the failure is recorded and reported in the exit code.
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
                    except Exception as exc:  # noqa: BLE001
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
# python run_qoder_longds.py --task-limit 1 --turn-limit 1
