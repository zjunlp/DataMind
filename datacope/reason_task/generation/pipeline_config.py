"""Configuration for the reason-task generation pipeline.

Edit `pipeline_config.yaml` for normal runs. This file only loads YAML,
applies defaults, and exposes `PipelineConfig` for `run_pipeline.py`.
"""

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "pipeline_config.yaml"


DEFAULT_CONFIG: dict[str, Any] = {
    "task": {
        "name": "dabstep",
        "target_categories": [
            "Applicable_Fee_IDs",
            "Average_Fee_Estimation",
            "Average_Transaction_Value_Stats",
            "Dataset_Metadata_and_Business_Rules",
            "Fee_Delta_and_Impact_Simulation",
            "Fraud_and_General_Macro_Analysis",
            "Highest_Cost_Scenario_Identification",
            "Routing_and_Cost_Optimization",
            "Total_Fees_Calculation",
        ],
    },
    "model": {
        "name": "Qwen3.5-397B-A17B",
        "api_key": "",
        "base_url": "",
        "manager_url": "http://localhost:5005",
    },
    "runtime": {
        "num_runs": 10,
        "max_workers": 30,
        "max_concurrent_skill_tasks": 3,
        "temperature": 1.0,
        "max_tokens": 16384,
        "no_skill_explore_output_name": "dabstep_qwen3_5_397b_nothiking_t1_explore_run_{run_idx}",
        "with_skill_explore_output_name": "dabstep_qwen3_5_397b_nothiking_t1_explore_{category}_run_{run_idx}",
    },
    "paths": {
        "results_run_name": "dabstep_dspk_explore_test",
        "category_file": "dataagent/DSGym/data/task/DABStep/all_query_category.json",
        "skill_save_dir": "skill_manager/skills",
        "verifier_traj_output_base_dir": "verifier",
        "skill_manager_workspace_dir": "skill_manager/workspace",
        "data_dir": "dataagent/DSGym/data/data",
        "evaluate_script_dir": "dataagent/DSGym/examples",
    },
}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return deepcopy(DEFAULT_CONFIG)

    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    if not isinstance(user_config, dict):
        raise ValueError(f"{CONFIG_FILE} must contain a YAML mapping at the top level")

    return _merge_dicts(DEFAULT_CONFIG, user_config)


def _path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else BASE_DIR / path)


def _env_or_value(env_name: str, value: str) -> str:
    return os.environ.get(env_name) or value


CONFIG = _load_config()
TASK = CONFIG["task"]
MODEL = CONFIG["model"]
RUNTIME = CONFIG["runtime"]
PATHS = CONFIG["paths"]


@dataclass
class PipelineConfig:
    """Runtime config consumed by `run_pipeline.py`."""

    # Task
    task_name: str = TASK["name"]
    target_categories: list[str] = field(
        default_factory=lambda: list(TASK["target_categories"])
    )

    # Model/API
    model: str = MODEL["name"]
    api_key: str = field(
        default_factory=lambda: _env_or_value("REASON_API_KEY", MODEL.get("api_key", ""))
    )
    base_url: str = field(
        default_factory=lambda: _env_or_value("REASON_BASE_URL", MODEL.get("base_url", ""))
    )
    manager_url: str = MODEL["manager_url"]

    # Runtime
    num_runs: int = int(RUNTIME["num_runs"])
    max_workers: int = int(RUNTIME["max_workers"])
    max_concurrent: int = int(RUNTIME["max_concurrent_skill_tasks"])
    explore_temperature: float = float(RUNTIME["temperature"])
    explore_max_tokens: int = int(RUNTIME["max_tokens"])
    no_skill_explore_output_name: str = RUNTIME["no_skill_explore_output_name"]
    with_skill_explore_output_name: str = RUNTIME["with_skill_explore_output_name"]

    # Paths from YAML
    base_dir: str = str(BASE_DIR)
    results_base_dir: str = field(
        default_factory=lambda: _path(
            PATHS.get(
                "results_base_dir",
                f"dataagent/DSGym/examples/results/{PATHS['results_run_name']}",
            )
        )
    )
    category_file: str = field(default_factory=lambda: _path(PATHS["category_file"]))
    skill_save_dir: str = field(default_factory=lambda: _path(PATHS["skill_save_dir"]))
    verifier_traj_output_base_dir: str = field(
        default_factory=lambda: _path(PATHS["verifier_traj_output_base_dir"])
    )
    skill_manager_workspace_dir: str = field(
        default_factory=lambda: _path(PATHS["skill_manager_workspace_dir"])
    )
    data_dir: str = field(default_factory=lambda: _path(PATHS["data_dir"]))
    evaluate_script_dir: str = field(
        default_factory=lambda: _path(PATHS["evaluate_script_dir"])
    )

    # Derived paths
    skill_output_dir: str = field(init=False)
    skill_manager_data_dir: str = field(init=False)

    def __post_init__(self) -> None:
        self.skill_output_dir = os.path.join(
            self.skill_manager_workspace_dir,
            "current_skills",
        )
        self.skill_manager_data_dir = os.path.join(
            self.skill_manager_workspace_dir,
            "data",
        )

    def explore_results_dir(self, iteration: int) -> str:
        if iteration == 0:
            return os.path.join(self.results_base_dir, "no_skill")
        return os.path.join(self.results_base_dir, str(iteration))

    def organized_traj_dir(self, iteration: int) -> str:
        """Directory for organized trajectories from organize_dabstep.py."""
        return os.path.join(
            self.verifier_traj_output_base_dir,
            self.task_name,
            f"{self.task_name}_iter{iteration}",
        )

    def paired_traj_dir(self, iter_prev: int, iter_curr: int) -> str:
        """Directory for paired trajectories from organize_dabstep_iter1_pair.py."""
        return os.path.join(
            self.verifier_traj_output_base_dir,
            self.task_name,
            f"{self.task_name}_sc1_iter{iter_prev}_vs_iter{iter_curr}",
        )

    def skill_manager_traj_input_dir(self) -> str:
        """Stable trajectory directory consumed by skill_manager scripts."""
        return os.path.join(self.skill_manager_workspace_dir, self.task_name)

    def skill_save_run_dir(self, run_id: str) -> str:
        """Root directory for skill snapshots created during one pipeline run."""
        return os.path.join(self.skill_save_dir, run_id)
