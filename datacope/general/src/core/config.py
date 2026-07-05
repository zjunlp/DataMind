import argparse
import os
from pathlib import Path
from typing import Optional
import yaml
from src.runtimes.registry import AgentRegistry
import src.runtimes.agents  # triggers @register_agent decorators


def get_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find repository root")


REPO_ROOT = get_repo_root()
DATA_ROOT = REPO_ROOT / "data"
TASK_DIR = DATA_ROOT / "task"
RAW_DATA_DIR = DATA_ROOT / "data"

def get_task_path(dataset_name: str) -> Path:
    return TASK_DIR / dataset_name


def get_data_path(dataset_name: str) -> Path:
    return RAW_DATA_DIR / dataset_name

PATH_LIKE_CONFIG_KEYS = {"output_dir", "skills_base_dir"}
CONFIG_GROUP_KEYS = {"da_agent", "verifier", "skill_manager"}
STAGES = ("da_agent", "verifier", "skill_manager")


def build_parser(config_defaults: Optional[dict] = None) -> argparse.ArgumentParser:
    config_defaults = config_defaults or {}

    # general arguments
    parser = argparse.ArgumentParser(
        description="Run any registered task dataset with ReActDSAgent."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_defaults.get("config"),
        help="Path to the YAML config file.",
    )
    parser.add_argument("--task", default=config_defaults.get("task"))
    parser.add_argument("--dataset", default=config_defaults.get("dataset"))
    parser.add_argument("--backend", default=config_defaults.get("backend", os.getenv("BACKEND", None)))
    parser.add_argument(
        "--manager-url",
        default=config_defaults.get("manager_url", os.getenv("MANAGER_URL", None)),
    )
    parser.add_argument("--workspace-dir", type=Path, default=config_defaults.get("workspace_dir"))
    parser.add_argument("--category_list", nargs="*", default=config_defaults.get("category_list", []))
    parser.add_argument("--run-name", default=config_defaults.get("run_name"))
    parser.add_argument(
        "--iterations",
        type=int,
        default=config_defaults.get("iterations", 0),
        help=(
            "Number of skill-feedback iterations after the initial trajectory run. "
            "When > 0, the runner executes: da_agent -> verifier.init_run -> "
            "skill_manager.create, then repeats da_agent with generated skills -> "
            "verifier.iterate_run -> skill_manager.modify."
        ),
    )
    parser.add_argument(
        "--resume-from-iteration",
        type=int,
        default=config_defaults.get("resume_from_iteration", 0),
        help=(
            "Iteration number to resume from. "
            "When > 0, the runner resumes from the specified iteration instead of starting from the beginning."
        )
    )
    parser.add_argument(
        "--resume-from-stage",
        choices=STAGES,
        default=config_defaults.get("resume_from_stage", "da_agent"),
        help=(
            "Stage to resume from. "
            "When resuming, the runner skips all stages before this stage in the specified iteration."
        )
    )
    debug_default = bool(config_defaults.get("debug", False))
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable debugpy and wait for a debugger to attach before running.",
    )
    debug_group.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable debugpy even if config.yaml enables it.",
    )
    parser.set_defaults(debug=debug_default)

    # data-analysis agent arguments
    parser.add_argument(
        "--da-model",
        default=config_defaults.get("da_model", os.getenv("MODEL")),
        required=config_defaults.get("da_model", os.getenv("MODEL")) is None,
    )
    parser.add_argument("--da-agent-type", default=config_defaults.get("da_agent_type", "react"),
                        choices=AgentRegistry.list_agents(),
                        help="Agent type (registered agents).")
    
    parser.add_argument("--da-api-key", default=config_defaults.get("da_api_key", os.getenv("API_KEY")))
    parser.add_argument("--da-base-url", default=config_defaults.get("da_base_url", os.getenv("BASE_URL")))
    parser.add_argument("--da-max-turns", type=int, default=config_defaults.get("da_max_turns", 15))
    parser.add_argument("--da-max-workers", type=int, default=config_defaults.get("da_max_workers", 1))
    parser.add_argument("--da-limit", type=int, default=config_defaults.get("da_limit"))
    parser.add_argument("--da-start-index", type=int, default=config_defaults.get("da_start_index", 0))
    parser.add_argument("--da-temperature", type=float, default=config_defaults.get("da_temperature", 0.0))
    parser.add_argument("--da-top-p", type=float, default=config_defaults.get("da_top_p", 1.0))
    parser.add_argument("--da-max-tokens", type=int, default=config_defaults.get("da_max_tokens", 1524))
    parser.add_argument("--da-sample-nums", type=int, default=config_defaults.get("da_sample_nums", 1))
    
    # verifier arguments
    parser.add_argument("--verifier", default=config_defaults.get("verifier"),
                        help="Verifier name (e.g. agreement, supervised). Omit to skip verification.")
    
    # skill manager arguments
    parser.add_argument("--skill-manager-model", type=str, default=config_defaults.get("skill_manager_model"))
    parser.add_argument("--skill-manager-agent-type", type=str, default=config_defaults.get("skill_manager_agent_type"))
    parser.add_argument("--skill-manager-api-key", type=str, default=config_defaults.get("skill_manager_api_key"))
    parser.add_argument("--skill-manager-base-url", type=str, default=config_defaults.get("skill_manager_base_url"))
    parser.add_argument("--skill-manager-max-workers", type=int, default=config_defaults.get("skill_manager_max_workers"))

    return parser


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = {k: v for k, v in raw.items() if k not in CONFIG_GROUP_KEYS}

    da_agent_config = raw.get("da_agent") or {}
    if not isinstance(da_agent_config, dict):
        raise ValueError("config.yaml field 'da_agent' must be a mapping.")
    for key, value in da_agent_config.items():
        if key.startswith("da_"):
            config[key] = value
        else:
            config[f"da_{key}"] = value

    verifier_config = raw.get("verifier") or {}
    if isinstance(verifier_config, str):
        config["verifier"] = verifier_config
    elif isinstance(verifier_config, dict):
        if "name" in verifier_config:
            config["verifier"] = verifier_config["name"]
    else:
        raise ValueError("config.yaml field 'verifier' must be a mapping, string, or null.")

    skill_manager_config = raw.get("skill_manager") or {}
    if not isinstance(skill_manager_config, dict):
        raise ValueError("config.yaml field 'skill_manager' must be a mapping.")
    for key, value in skill_manager_config.items():
        if key.startswith("skill_manager_"):
            config[key] = value
        else:
            config[f"skill_manager_{key}"] = value

    for key in PATH_LIKE_CONFIG_KEYS:
        if key in config and config[key] is not None:
            config[key] = Path(config[key])
    return config
