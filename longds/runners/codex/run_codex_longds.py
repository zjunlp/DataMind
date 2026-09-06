#!/usr/bin/env python3
"""Run LongDS tasks directly with Codex CLI, without importing DSGym."""

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
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"

CONTAINER_WORKSPACE = "/workspace"
CONTAINER_HOME = f"{CONTAINER_WORKSPACE}/.home"
CONTAINER_CODEX_HOME = "/codex-home"
CONTAINER_CODEX_CONFIG = f"{CONTAINER_CODEX_HOME}/config.toml"
CONTAINER_CODEX_AUTH = f"{CONTAINER_CODEX_HOME}/auth.json"
CONTAINER_CODEX_SCHEMA = "/tmp/longds_codex_turn.schema.json"
CONTAINER_CODEX_LAST_MESSAGE_PREFIX = "/tmp/longds_codex_last_message"
FIXED_DOCKER_ENV = {
    "HOME": CONTAINER_HOME,
    "CODEX_HOME": CONTAINER_CODEX_HOME,
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONWARNINGS": "ignore::FutureWarning",
}
DEFAULT_DOCKER_ENV_KEYS = (
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
        description="Run LongDS-Bench directly with Codex CLI sessions."
    )
    add_dataset_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Result base (default: ./results in the current directory). Appends longds_<version>_<split>/<run_name>/<domain>/<dataset>/<task_id>/.",
    )
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI executable.")
    parser.add_argument(
        "--codex-config",
        type=Path,
        default=script_dir / "config.toml",
        help=(
            "Codex config.toml. Defaults to config.toml in this directory. "
            f"With --use-docker it is copied to {CONTAINER_CODEX_CONFIG}."
        ),
    )
    parser.add_argument(
        "--codex-auth",
        type=Path,
        default=None,
        help=(
            "Optional Codex auth.json created by `codex login`. With --use-docker it is "
            f"copied to {CONTAINER_CODEX_AUTH}. Omit for config-based provider auth."
        ),
    )
    parser.add_argument(
        "--codex-model",
        default=None,
        help="Model passed to `codex exec -m`. Omit to use Codex config default.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help=(
            "Codex model reasoning effort. Omit to use the Codex configuration default."
        ),
    )
    parser.add_argument(
        "--codex-base-url",
        default=None,
        help=(
            "Optional base URL override for the provider selected by config.toml. "
            "Omit to use the value from --codex-config."
        ),
    )
    parser.add_argument(
        "--model-supports-reasoning-summaries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Temporarily force Codex reasoning-summary support for this run. "
            "Use --no-model-supports-reasoning-summaries to force-disable it."
        ),
    )
    parser.add_argument(
        "--analysis-python",
        default=None,
        help=(
            "Python executable Codex should use for data analysis commands. "
            "Defaults to the current Python locally and /usr/local/bin/python with --use-docker."
        ),
    )
    parser.add_argument(
        "--sandbox",
        default="workspace-write",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help=(
            "Sandbox mode for local execution. Docker mode uses the task container as the "
            "isolation boundary and leaves Codex tools unrestricted inside it."
        ),
    )
    parser.add_argument(
        "--approval-policy",
        default="never",
        choices=["untrusted", "on-failure", "on-request", "never"],
        help="Top-level Codex approval policy.",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help=(
            "Run Codex inside one Docker container per task. Task data is copied into "
            f"{CONTAINER_WORKSPACE}; Codex tools are unrestricted inside the container."
        ),
    )
    parser.add_argument("--docker-bin", default="docker", help="Docker CLI executable.")
    parser.add_argument(
        "--docker-image",
        default="longds-codex:latest",
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
        default="longds-codex",
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
        help="Maximum number of tasks to run after --start-index. Defaults to all remaining tasks.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the selected task list.")
    parser.add_argument("--turn-limit", type=int, default=None, help="Maximum turns per task.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per Codex turn, seconds.")
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
            "Optional run directory name. Defaults to codex_<model>_YYYYmmdd_HHMMSS. "
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
        help="Write prompts and metadata without invoking Codex or copying task data.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run judge.py immediately after each successfully completed task.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "codex"


def load_codex_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Codex config must contain a TOML table: {path}")
    return config


def toml_key_segment(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return json.dumps(value)


def codex_config_provider(config: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    provider_name = config.get("model_provider")
    providers = config.get("model_providers")
    if not isinstance(provider_name, str) or not isinstance(providers, dict):
        return None, {}
    provider = providers.get(provider_name)
    if not isinstance(provider, dict):
        return provider_name, {}
    return provider_name, provider


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
        key = key.strip()
        if key:
            env[key] = value

    return env


def docker_env_specs(args: argparse.Namespace, client_env: dict[str, str]) -> list[str]:
    specs = [f"{key}={value}" for key, value in FIXED_DOCKER_ENV.items()]
    seen = set(FIXED_DOCKER_ENV)

    for key in DEFAULT_DOCKER_ENV_KEYS:
        if client_env.get(key) and key not in seen:
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
                f"{CONTAINER_CODEX_HOME} && tail -f /dev/null"
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
                CONTAINER_CODEX_HOME,
            ],
        )


def copy_codex_schema_to_container(
    *,
    args: argparse.Namespace,
    container_name: str,
    schema_path: Path,
) -> None:
    run_docker_control(
        args,
        [
            args.docker_bin,
            "cp",
            str(schema_path.resolve()),
            f"{container_name}:{CONTAINER_CODEX_SCHEMA}",
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
                CONTAINER_CODEX_SCHEMA,
            ],
        )


def copy_codex_config_to_container(
    *,
    args: argparse.Namespace,
    container_name: str,
) -> None:
    if args.codex_config is None:
        return
    run_docker_control(
        args,
        [
            args.docker_bin,
            "cp",
            str(args.codex_config.resolve()),
            f"{container_name}:{CONTAINER_CODEX_CONFIG}",
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
                CONTAINER_CODEX_CONFIG,
            ],
        )


def copy_codex_auth_to_container(
    *,
    args: argparse.Namespace,
    container_name: str,
) -> None:
    if args.codex_auth is None:
        return
    run_docker_control(
        args,
        [
            args.docker_bin,
            "cp",
            str(args.codex_auth.resolve()),
            f"{container_name}:{CONTAINER_CODEX_AUTH}",
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
                CONTAINER_CODEX_AUTH,
            ],
        )


def prepare_local_codex_home(*, args: argparse.Namespace, run_dir: Path) -> Path:
    codex_home = run_dir / "codex_home"
    codex_home.mkdir(parents=True, exist_ok=True)
    if args.codex_config is not None:
        shutil.copy2(args.codex_config, codex_home / "config.toml")
    if args.codex_auth is not None:
        shutil.copy2(args.codex_auth, codex_home / "auth.json")
    return codex_home


def copy_codex_last_message_from_container(
    *,
    args: argparse.Namespace,
    container_name: str,
    container_last_message_path: str,
    last_message_path: Path,
) -> None:
    run_docker_control(
        args,
        [
            args.docker_bin,
            "cp",
            f"{container_name}:{container_last_message_path}",
            str(last_message_path),
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


def scrub_codex_home(codex_home: Path) -> None:
    (codex_home / "auth.json").unlink(missing_ok=True)
    (codex_home / "config.toml").unlink(missing_ok=True)
    shutil.rmtree(codex_home / "shell_snapshots", ignore_errors=True)


def sync_codex_home_from_container(
    *,
    args: argparse.Namespace,
    container_name: str,
    codex_home: Path,
) -> dict[str, Any]:
    codex_home.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            args.docker_bin,
            "cp",
            f"{container_name}:{CONTAINER_CODEX_HOME}/.",
            str(codex_home),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=docker_client_env(args),
    )
    scrub_codex_home(codex_home)
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


def codex_command(
    *,
    args: argparse.Namespace,
    schema_path: Path,
    last_message_path: Path,
    work_dir: Path,
    session_id: str | None,
) -> list[str]:
    cmd = [args.codex_bin]
    if args.approval_policy and not args.use_docker:
        cmd.extend(["--ask-for-approval", args.approval_policy])
    if args.codex_base_url:
        provider_key = toml_key_segment(args.codex_provider_name)
        cmd.extend(
            [
                "-c",
                f"model_providers.{provider_key}.base_url={json.dumps(args.codex_base_url)}",
            ]
        )
    if args.model_supports_reasoning_summaries is not None:
        value = str(args.model_supports_reasoning_summaries).lower()
        cmd.extend(["-c", f"model_supports_reasoning_summaries={value}"])
    if args.reasoning_effort:
        cmd.extend(
            ["-c", f"model_reasoning_effort={json.dumps(args.reasoning_effort)}"]
        )
    cmd.extend(["exec"])

    if session_id:
        cmd.extend(["resume", "--json", "--skip-git-repo-check"])
        if args.use_docker:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
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

    cmd.extend(["--json", "--skip-git-repo-check", "-C", str(work_dir)])
    if args.use_docker:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        cmd.extend(["--sandbox", args.sandbox])
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
    container_name: str | None = None,
) -> dict[str, str | None]:
    if not session_id:
        return {
            "manual_resume_note": None,
            "manual_resume_command": None,
        }

    if args.use_docker and container_name and not args.dry_run:
        return {
            "manual_resume_note": (
                "Run manual_resume_command while the per-task Docker container exists. "
                "Pass --keep-docker-container to keep it after the task finishes."
            ),
            "manual_resume_command": shlex.join(
                docker_exec_command(
                    args=args,
                    container_name=container_name,
                    inner_cmd=[args.codex_bin, "resume", session_id],
                    interactive=True,
                )
            ),
        }

    codex_home = run_dir / "codex_home"
    return {
        "manual_resume_note": (
            "Restore the selected config.toml and auth.json, when configured, to the saved "
            "Codex home before running manual_resume_command; copied credentials are removed "
            "after the task."
        ),
        "manual_resume_command": shlex.join(
            ["env", f"CODEX_HOME={codex_home}", args.codex_bin, "resume", session_id]
        ),
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


def format_codex_steps(
    stdout: str,
    *,
    prompt: str,
    turn_label: str,
    previous_thread_id: str | None,
) -> dict[str, Any]:
    """Convert raw Codex JSONL events into the compact step format shown in the terminal."""
    trace: dict[str, Any] = {
        "session": {},
        "turn": int(turn_label) if turn_label.isdigit() else turn_label,
        "user_request": prompt,
        "steps": [],
    }
    steps: dict[int, dict[str, Any]] = {}
    next_step = 0

    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", ""))
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            trace["session"] = {
                "thread_id": thread_id,
                "session_state": (
                    "new"
                    if previous_thread_id is None
                    else "same_as_previous_turn"
                    if thread_id == previous_thread_id
                    else "changed_from_previous_turn"
                ),
            }
            continue
        if event_type == "turn.completed":
            trace["usage"] = event.get("usage") or {}
            continue

        item = event.get("item")
        if not isinstance(item, dict) or item.get("status") == "in_progress":
            continue

        item_id = str(item.get("id") or "")
        match = re.fullmatch(r"item_(\d+)", item_id)
        step = int(match.group(1)) if match else next_step
        next_step = max(next_step, step + 1)
        item_type = item.get("type")
        formatted: dict[str, Any] = {"step": step}

        if item_type == "command_execution":
            if not event_type.endswith(".completed"):
                continue
            formatted["command"] = str(item.get("command") or "")
            formatted["output"] = str(item.get("aggregated_output") or "")
        elif item_type == "agent_message":
            text = str(item.get("text") or "")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                formatted["response"] = str(payload.get("answer") or "")
                formatted["reasoning_summary"] = str(payload.get("reasoning_summary") or "")
                if payload.get("files_used"):
                    formatted["files_used"] = payload["files_used"]
            else:
                formatted["message"] = text
        else:
            formatted["item_type"] = item_type
            formatted["item_data"] = {
                key: value
                for key, value in item.items()
                if key not in {"id", "type", "status"}
            }

        steps[step] = formatted

    trace["steps"] = [steps[step] for step in sorted(steps)]
    return trace


def normalize_turn_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
        return None

    reasoning_summary = payload.get("reasoning_summary", "")
    files_used = payload.get("files_used", [])
    if not isinstance(reasoning_summary, str):
        reasoning_summary = str(reasoning_summary)
    if not isinstance(files_used, list):
        files_used = []

    return {
        "answer": payload["answer"],
        "reasoning_summary": reasoning_summary,
        "files_used": [str(file_path) for file_path in files_used],
    }


def parse_embedded_json(text: str) -> dict[str, Any] | None:
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.append(text)
    decoder = json.JSONDecoder()

    for candidate in candidates:
        try:
            payload = json.loads(candidate.strip())
        except json.JSONDecodeError:
            payload = None
        normalized = normalize_turn_payload(payload)
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


def parse_last_message(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None, ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    normalized = normalize_turn_payload(payload)
    if normalized is not None:
        return normalized, text
    embedded = parse_embedded_json(text)
    if embedded is not None:
        return embedded, text
    return {
        "answer": text,
        "reasoning_summary": "",
        "files_used": [],
    }, text


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
    cwd: Path,
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
        cwd=str(cwd),
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
    codex_home: Path,
    session_id: str | None,
    container_name: str | None = None,
) -> tuple[str | None, dict[str, Any]]:
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    last_message_path = turn_dir / "last_message.json"
    container_last_message_path = (
        f"{CONTAINER_CODEX_LAST_MESSAGE_PREFIX}_{slugify(turn_label)}.json"
    )

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

    inner_cmd = codex_command(
        args=args,
        schema_path=Path(CONTAINER_CODEX_SCHEMA) if args.use_docker else schema_path,
        last_message_path=(
            Path(container_last_message_path) if args.use_docker else last_message_path
        ),
        work_dir=Path(CONTAINER_WORKSPACE) if args.use_docker else work_dir,
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
        process_env["CODEX_HOME"] = str(codex_home)

    start = time.time()
    returncode, stdout, stderr = run_command_streaming(
        cmd,
        cwd=work_dir,
        timeout=args.timeout,
        env=process_env,
        prompt=prompt,
        turn_label=turn_label,
        previous_thread_id=session_id,
    )
    elapsed = time.time() - start

    (turn_dir / "codex_stdout.jsonl").write_text(stdout, encoding="utf-8")
    (turn_dir / "codex_stderr.txt").write_text(stderr, encoding="utf-8")
    if args.use_docker and returncode == 0:
        copy_codex_last_message_from_container(
            args=args,
            container_name=container_name,
            container_last_message_path=container_last_message_path,
            last_message_path=last_message_path,
        )
    write_json(
        turn_dir / "formatted_steps.json",
        format_codex_steps(
            stdout,
            prompt=prompt,
            turn_label=turn_label,
            previous_thread_id=session_id,
        ),
    )
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
        raise RuntimeError(f"Codex did not produce a final answer; see {turn_dir}")
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

    model_slug = slugify(args.codex_model or "codex-default")
    workspace_dir = run_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    codex_home = run_dir / "codex_home"
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
    schema_path = run_dir / "codex_turn.schema.json"
    write_json(schema_path, TURN_SCHEMA)

    task_result: dict[str, Any] = {
        **dataset_metadata(args),
        "task_domain": domain,
        "dataset_name": dataset,
        "task_id": task_id,
        "run_name": run_name,
        "model_slug": model_slug,
        "codex_model": args.codex_model,
        "codex_model_source": args.codex_model_source,
        "codex_config_file": str(args.codex_config) if args.codex_config else None,
        "codex_auth_file": str(args.codex_auth) if args.codex_auth else None,
        "codex_provider": args.codex_provider_name,
        "codex_config_has_direct_token": args.codex_config_has_direct_token,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_effort_source": args.reasoning_effort_source,
        "execution_mode": "docker" if args.use_docker else "local",
        "docker_image": args.docker_image if args.use_docker else None,
        "container_workspace": CONTAINER_WORKSPACE if args.use_docker else None,
        "container_home": CONTAINER_HOME if args.use_docker else None,
        "container_codex_home": CONTAINER_CODEX_HOME if args.use_docker else None,
        "docker_container_name": container_name,
        "saved_codex_home_dir": str(codex_home),
        "local_data_dir": str(local_data_dir),
        "workspace_dir": str(workspace_dir),
        "run_dir": str(run_dir),
        "thread_id": None,
        **build_manual_resume_metadata(
            args=args,
            run_dir=run_dir,
            schema_path=schema_path,
            session_id=None,
            container_name=container_name,
        ),
        "turns": [],
    }
    write_json(run_dir / "task_metadata.json", task_result)

    container_started = False
    try:
        if not args.use_docker and not args.dry_run:
            prepare_local_codex_home(args=args, run_dir=run_dir)
        if args.use_docker and not args.dry_run:
            if container_name is None:
                raise RuntimeError("Docker mode requires a per-task container name.")
            print(paint(f"Starting Docker task container {container_name} ...", COLOR_DIM), flush=True)
            start_task_container(args=args, container_name=container_name)
            container_started = True
            task_result["docker_container_started"] = True
            write_json(run_dir / "task_metadata.json", task_result)
            copy_codex_config_to_container(args=args, container_name=container_name)
            copy_codex_auth_to_container(args=args, container_name=container_name)
            copy_task_data_to_container(
                args=args,
                container_name=container_name,
                source_data_dir=source_data_dir,
            )
            copy_codex_schema_to_container(
                args=args,
                container_name=container_name,
                schema_path=schema_path,
            )
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
            session_id, result = run_codex_turn(
                args=args,
                prompt=prompt,
                turn_label=str(turn_id),
                schema_path=schema_path,
                turn_dir=turn_dir,
                work_dir=workspace_dir,
                codex_home=codex_home,
                session_id=session_id,
                container_name=container_name,
            )
            task_result["thread_id"] = session_id
            task_result.update(
                build_manual_resume_metadata(
                    args=args,
                    run_dir=run_dir,
                    schema_path=schema_path,
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
            workspace_sync = sync_workspace_from_container(
                args=args,
                container_name=container_name,
                workspace_dir=workspace_dir,
            )
            task_result["workspace_sync"] = workspace_sync
            write_json(run_dir / "task_metadata.json", task_result)
            if workspace_sync["returncode"] != 0:
                raise RuntimeError(f"Failed to sync Docker workspace: {workspace_sync['stderr']}")

            codex_home_sync = sync_codex_home_from_container(
                args=args,
                container_name=container_name,
                codex_home=codex_home,
            )
            task_result["codex_home_sync"] = codex_home_sync
            write_json(run_dir / "task_metadata.json", task_result)
            if codex_home_sync["returncode"] != 0:
                raise RuntimeError(f"Failed to sync Codex home: {codex_home_sync['stderr']}")

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
            task_result["codex_home_sync"] = sync_codex_home_from_container(
                args=args,
                container_name=container_name,
                codex_home=codex_home,
            )
            write_json(run_dir / "task_metadata.json", task_result)
        raise
    finally:
        if not args.use_docker:
            scrub_codex_home(codex_home)
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
    judge_script = Path(__file__).resolve().parents[1] / "src" / "judge.py"
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
    except Exception as exc:
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
    print("Codex LongDS run configuration:", flush=True)
    print(f"  run_name: {run_name}", flush=True)
    print(f"  task_root: {args.task_root}", flush=True)
    print(f"  longds_version: {args.longds_version}", flush=True)
    print(f"  split: {args.split}", flush=True)
    print(f"  task_list_name: {args.task_list_name}", flush=True)
    print(f"  data_root: {args.data_root}", flush=True)
    print(f"  output_dir: {results_root}", flush=True)
    print(f"  codex_bin: {args.codex_bin}", flush=True)
    print(f"  codex_config: {args.codex_config or 'none'}", flush=True)
    print(f"  codex_auth: {args.codex_auth or 'none'}", flush=True)
    print(
        f"  codex_model: {args.codex_model or 'Codex default'} ({args.codex_model_source})",
        flush=True,
    )
    print(
        f"  reasoning_effort: {args.reasoning_effort or 'Codex default'} "
        f"({args.reasoning_effort_source})",
        flush=True,
    )
    print(f"  codex_base_url: {args.codex_base_url or 'config-default'}", flush=True)
    print(
        "  model_supports_reasoning_summaries: "
        f"{args.model_supports_reasoning_summaries if args.model_supports_reasoning_summaries is not None else 'config-default'}",
        flush=True,
    )
    print(f"  codex_provider: {args.codex_provider_name or 'Codex default'}", flush=True)
    print(
        f"  config_direct_bearer_token: {args.codex_config_has_direct_token}",
        flush=True,
    )
    print(f"  analysis_python: {args.analysis_python}", flush=True)
    print(f"  sandbox: {args.sandbox}", flush=True)
    print(f"  approval_policy: {args.approval_policy}", flush=True)
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
    resolve_dataset(args)
    args.output_dir = result_root(args)
    if args.codex_config is not None:
        args.codex_config = args.codex_config.expanduser().resolve()
        if not args.codex_config.is_file():
            raise FileNotFoundError(f"--codex-config does not exist: {args.codex_config}")
    if args.codex_auth is not None:
        args.codex_auth = args.codex_auth.expanduser().resolve()
        if not args.codex_auth.is_file():
            raise FileNotFoundError(f"--codex-auth does not exist: {args.codex_auth}")
    config = load_codex_config(args.codex_config)
    args.codex_provider_name, provider_config = codex_config_provider(config)
    args.codex_config_has_direct_token = bool(
        provider_config.get("experimental_bearer_token")
    )
    if args.codex_model is None and isinstance(config.get("model"), str):
        args.codex_model = config["model"]
        args.codex_model_source = "config"
    else:
        args.codex_model_source = "argument" if args.codex_model else "Codex default"
    if args.reasoning_effort is None and isinstance(config.get("model_reasoning_effort"), str):
        args.reasoning_effort = config["model_reasoning_effort"]
        args.reasoning_effort_source = "config"
    else:
        args.reasoning_effort_source = "argument" if args.reasoning_effort else "Codex default"
    args.docker_env_file = [path.resolve() for path in args.docker_env_file]
    for env_file in args.docker_env_file:
        if not env_file.is_file():
            raise FileNotFoundError(f"--docker-env-file does not exist: {env_file}")
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
    if args.codex_base_url and not args.codex_provider_name:
        raise ValueError(
            "--codex-base-url requires model_provider in --codex-config"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(args.codex_model or "default")
    run_name = args.run_name or f"codex_{model_slug}_{timestamp}"
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

    summary = RunSummary(args, "codex", args.codex_model, run_name, selected, results_root / run_name)
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
            "Finished Codex LongDS run: "
            f"completed_tasks={completed_tasks}, "
            f"failed_tasks={failed_tasks}, "
            f"skipped_tasks={skipped_tasks}, "
            f"selected_tasks={len(selected)}",
            flush=True,
        )

    failed = False
    if args.run_parallel == 1:
        for task_info in selected:
            succeeded = execute(task_info)
            if not succeeded:
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
# python run_codex_longds.py    --task-limit 1 --turn-limit 1
