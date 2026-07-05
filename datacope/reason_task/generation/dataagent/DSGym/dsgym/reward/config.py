import os
from pathlib import Path


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

