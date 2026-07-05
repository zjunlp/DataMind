"""Utilities for the DataCOPE execution loop."""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from src.task_processor import DatasetRegistry
from src.task_processor.loaders.naive import NaiveDataset
from src.core.config import STAGES

def load_dataset(
    *,
    task_name: str,
    dataset_name: str,
    limit: Optional[int] = None,
    start_index: int = 0,
    use_skill: bool = False,
    skills_base_dir: Optional[str] = None,
    query_categories: Optional[list[str]] = None,
    virtual_data_root: Optional[str] = None,
) -> list[dict]:
    registered = DatasetRegistry.list_datasets()
    if task_name in registered:
        dataset = DatasetRegistry.load(task_name, virtual_data_root=virtual_data_root)
    else:
        dataset = NaiveDataset(task_name=task_name, virtual_data_root=virtual_data_root)
    return dataset.load(
        dataset_name=dataset_name,
        limit=limit,
        start_index=start_index,
        use_skill=use_skill,
        skills_base_dir=skills_base_dir,
        query_categories=query_categories,
    )

def save_iteration_results(iter_results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(iter_results, f, indent=2, ensure_ascii=False)

def validate_resume_args(
    *,
    iterations: int,
    resume_from_iteration: int,
    resume_from_stage: str,
) -> None:
    if resume_from_iteration < 0:
        raise ValueError("resume_from_iteration must be >= 0.")

    if resume_from_iteration > iterations:
        raise ValueError(
            f"resume_from_iteration={resume_from_iteration} exceeds iterations={iterations}."
        )
    
    if resume_from_stage not in STAGES:
        raise ValueError(
            f"resume_from_stage must be one of {STAGES}, got {resume_from_stage!r}."
        )

def stage_index(stage: str) -> int:
    return STAGES.index(stage)


def should_run_stage(
    *,
    iteration: int,
    stage: str,
    resume_from_iteration: int,
    resume_from_stage: str,
) -> bool:
    """
    True means execute this stage.
    False means skip this stage and load/check its existing output.
    """
    if resume_from_iteration == 0 and resume_from_stage == "da_agent":
        return True

    resume_stage = resume_from_stage

    if iteration < resume_from_iteration:
        return False

    if iteration > resume_from_iteration:
        return True

    return stage_index(stage) >= stage_index(resume_stage)

def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
