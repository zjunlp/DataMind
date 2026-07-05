#!/usr/bin/env python3
"""DataCOPE execution loop entry point."""

import json
import shutil
import argparse
from pathlib import Path
from typing import Optional
import os
from src.core.config import REPO_ROOT
from src.core.utils import (
    reset_output_dir,
    save_iteration_results,
    should_run_stage,
    validate_resume_args,
    load_dataset,
)
from src.da_agent.da_agent import DaAgent
from src.core.schema import EvaluationConfig
from src.skill_manager.skill_manager import SkillManager
from src.verifier import VerifierRegistry
from src.core.config import (
    build_parser,
    load_yaml_config,
    get_data_path,
    REPO_ROOT
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

def _format_skill_base_dir(base_dir: Optional[str], category: Optional[str]) -> Optional[str]:
    if base_dir is None:
        return None

    if category is None:
        return os.path.abspath(base_dir)
    else:
        tmp_base_dir = os.path.join(os.path.abspath(base_dir), category)
        
        if not os.path.isdir(tmp_base_dir):
            return None
        
        skill_path = os.path.join(tmp_base_dir, "SKILL.md")
        if os.path.isfile(skill_path):
            return tmp_base_dir

        for root, _, files in os.walk(tmp_base_dir):
            if "SKILL.md" in files:
                return root

        return None
    
def _build_agent_kwargs(
    *,
    manager_url: str,
    max_turns: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    working_dir: Optional[str],
    timeout: int,
    api_key: Optional[str],
    base_url: Optional[str],
) -> dict:
    agent_kwargs = {
        "manager_url": manager_url,
        "max_turns": max_turns,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "working_dir": working_dir or ".",
        "timeout": timeout,
    }
    if api_key:
        agent_kwargs["api_key"] = api_key
    if base_url:
        agent_kwargs["base_url"] = base_url
    return agent_kwargs

def run_da_stage(
    *,
    task_name: str,
    dataset_name: str,
    sample_nums: int,
    model: str,
    agent_type: str,
    api_key: str,
    base_url: str,
    da_output_dir: Path,
    da_run_name: str,
    use_skill: bool,
    skills_base_dir: Optional[str],
    category_list: Optional[list[str]],
    manager_url: Optional[str] = None,
    working_dir: Optional[str] = None,
    timeout: Optional[int] = None,
    backend: Optional[str] = None,
    max_turns: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    start_index: int = 0,
    limit: Optional[int] = None,
    max_tokens: int = 8192,
    max_workers: Optional[int] = None,
    virtual_data_root: Optional[str] = None,
) -> list[dict]:
    
    stage_results = []
    for i in range(sample_nums):
        agent_kwargs = _build_agent_kwargs(
            manager_url=manager_url,
            max_turns=max_turns,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            working_dir=working_dir,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
        )

        config = EvaluationConfig(
            model_name=model,
            backend_type=backend,
            dataset_name=f"{task_name}_{dataset_name}",
            run_name=f"{da_run_name}_sample{i}" if sample_nums > 1 else da_run_name,
            max_turns=max_turns,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_workers=max_workers,
            limit=limit,
            start_index=start_index,
        )

        if category_list:
            for category in category_list:
                category_skills_base_dir = _format_skill_base_dir(skills_base_dir, category)

                print(f"[{da_run_name} sample {i}] Running category: {category}")
                samples = load_dataset(
                    task_name=task_name,
                    dataset_name=dataset_name,
                    limit=limit,
                    start_index=start_index,
                    use_skill=use_skill,
                    skills_base_dir=category_skills_base_dir,
                    query_categories=[category],
                    virtual_data_root=virtual_data_root,
                )

                sample_output_dir = da_output_dir / str(i) / category if sample_nums > 1 else da_output_dir / category
                sample_output_dir.mkdir(parents=True, exist_ok=True)

                evaluator = DaAgent(
                    agent_type=agent_type,
                    backend=backend,
                    model=model,
                    output_dir=str(sample_output_dir),
                    parallel_workers=max_workers,
                    **agent_kwargs,
                )

                result = evaluator.generate(
                    tasks=samples,
                    config=config,
                )
                stage_results.append(result)
                print(f"[{da_run_name} sample {i}] Finished {len(result['results'])} tasks in {result['total_time']:.2f}s")

        else:
            samples = load_dataset(
                task_name=task_name,
                dataset_name=dataset_name,
                limit=limit,
                start_index=start_index,
                use_skill=use_skill,
                skills_base_dir=skills_base_dir,
                virtual_data_root=virtual_data_root,
            )

            sample_output_dir = da_output_dir / str(i) if sample_nums > 1 else da_output_dir
            sample_output_dir.mkdir(parents=True, exist_ok=True)


            evaluator = DaAgent(
                agent_type=agent_type,
                backend=backend,
                model=model,
                output_dir=str(sample_output_dir),
                parallel_workers=max_workers,
                **agent_kwargs,
            )
            result = evaluator.generate(
                tasks=samples,
                config=config,
            )
            stage_results.append(result)
            print(f"[{da_run_name} sample {i}] Finished {len(result['results'])} tasks in {result['total_time']:.2f}s")

    return stage_results

def run_verifier_stage(
    *,
    verifier: str,
    input_file_dirs: list[str],
    output_dir: Path,
    category_list: Optional[list[str]],
    is_iteration: bool,
) -> dict:
    print(f"\nRunning verifier: {verifier}")
    v = VerifierRegistry.load(
        verifier,
        input_file_dirs=input_file_dirs,
        output_dir=str(output_dir),
        category_list=category_list or [],
    )
    verifier_results = v.iterate_run() if is_iteration else v.init_run()
    print(f"Verification output: {output_dir}")
    return verifier_results or {}


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _prepare_skill_manager_working_dir(
    *,
    task_name: str,
    working_dir: Path,
) -> Path:
    working_dir.mkdir(parents=True, exist_ok=True)

    source_data_dir = get_data_path(task_name)
    if not source_data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {source_data_dir}")

    target_data_dir = working_dir / "data" / task_name
    target_data_dir.parent.mkdir(parents=True, exist_ok=True)
    _replace_path(target_data_dir)
    target_data_dir.symlink_to(source_data_dir, target_is_directory=True)

    source_skill_creator_dir = REPO_ROOT / "skills" / "skill-creator"
    if not source_skill_creator_dir.is_dir():
        raise FileNotFoundError(f"skill-creator skill not found: {source_skill_creator_dir}")

    target_skill_creator_dir = working_dir / "skills" / "skill-creator"
    target_skill_creator_dir.parent.mkdir(parents=True, exist_ok=True)
    _replace_path(target_skill_creator_dir)
    shutil.copytree(source_skill_creator_dir, target_skill_creator_dir)

    source_claude_code_config_dir = REPO_ROOT / ".claude"
    if source_claude_code_config_dir.is_dir():
        target_claude_code_config_dir = working_dir / ".claude"
        target_claude_code_config_dir.parent.mkdir(parents=True, exist_ok=True)
        _replace_path(target_claude_code_config_dir)
        shutil.copytree(source_claude_code_config_dir, target_claude_code_config_dir)
    
    source_codex_config_dir = REPO_ROOT / ".codex"
    if source_codex_config_dir.is_dir():
        target_codex_config_dir = working_dir / ".codex"
        target_codex_config_dir.parent.mkdir(parents=True, exist_ok=True)
        _replace_path(target_codex_config_dir)
        shutil.copytree(source_codex_config_dir, target_codex_config_dir)

    return target_data_dir


def run_skill_manager_stage(
    *,
    task_name: str,
    traj_dir: str,
    verifier_prompt: str,
    mode: str,
    skill_output_dir: Path,
    skill_manager_model: str,
    skill_manager_agent_type: str,
    skill_manager_api_key: str,
    skill_manager_base_url: str,
    skill_manager_backend: Optional[str],
    skill_manager_max_workers: Optional[int],
    skill_manager_use_category: bool,
    working_dir: Optional[str],
    category_list: Optional[list[str]],
) -> None:
    skill_output_dir.mkdir(parents=True, exist_ok=True)
    skill_manager_working_dir = Path(working_dir) if working_dir else skill_output_dir.parent
    data_dir = _prepare_skill_manager_working_dir(
        task_name=task_name,
        working_dir=skill_manager_working_dir,
    )

    skill_agent_kwargs = {
        "backend": skill_manager_backend,
        "api_key": skill_manager_api_key,
        "base_url": skill_manager_base_url,
        "working_dir": str(skill_manager_working_dir),
    }

    manager = SkillManager(
        model=skill_manager_model,
        agent_type=skill_manager_agent_type,
        task=task_name,
        data_dir=str(data_dir),
        current_skill_dir=str(skill_output_dir),
        category_list=category_list,
        parallel_workers=skill_manager_max_workers,
        **skill_agent_kwargs,
    )

    manager.run_batch(
        traj_base_dir=traj_dir,
        verifier_prompt=verifier_prompt,
        mode=mode,
        target_categories=category_list,
        use_category=skill_manager_use_category,
    )

    print(f"Skill manager output: {skill_output_dir}")


def run(
    *,
    task_name: str,
    dataset_name: str,
    workspace_dir: Optional[Path] = None,
    category_list: Optional[list[str]] = None,
    run_name: Optional[str] = None,
    iterations: int = 0,
    resume_from_iteration: int = 0,
    resume_from_stage: str = "da_agent",
    backend: Optional[str] = None,
    manager_url: Optional[str] = None,

    da_model: str,
    da_agent_type: str,
    da_api_key: str,
    da_base_url: str,
    da_sample_nums: int = 1,
    da_max_turns: Optional[int] = None,
    da_max_workers: int = 1,
    da_limit: Optional[int] = None,    
    da_start_index: int = 0,
    da_temperature: float = 0.0,
    da_top_p: float = 1.0,
    da_max_tokens: int = 8192,

    verifier: str,
    
    skill_manager_model: str,
    skill_manager_agent_type: str,
    skill_manager_api_key: str,
    skill_manager_base_url: str,
    skill_manager_max_workers: int = 1,
) -> dict:
    if workspace_dir is None:
        workspace_dir = REPO_ROOT / "outputs" / task_name / dataset_name

    agent_working_dir = str(REPO_ROOT)

    if iterations < 0:
        raise ValueError("skill_iterations must be >= 0")
    if iterations and not verifier:
        raise ValueError("skill_iterations requires --verifier.")
    validate_resume_args(
        iterations=iterations,
        resume_from_iteration=resume_from_iteration,
        resume_from_stage=resume_from_stage,
    )

    base_run_name = run_name

    da_stage_output_dir_prefix = workspace_dir / base_run_name / "da_agent"
    verifier_output_dir_prefix = workspace_dir / base_run_name / "verifier"
    skill_manager_output_dir_prefix = workspace_dir / base_run_name / "skill_manager"

    stage_success_status = {}
    stage_success_status_path = workspace_dir / base_run_name / "stage_success_status.json"
    if stage_success_status_path.exists():
        stage_success_status = json.load(open(stage_success_status_path, "r", encoding="utf-8"))

    iter_results = []
    trajectory_dirs: list[Path] = []

    for iteration in range(iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        da_stage_output_dir = da_stage_output_dir_prefix / f"iter_{iteration}"
        verify_output_dir = verifier_output_dir_prefix / f"iter_{iteration}"
        skill_manager_iter_dir = skill_manager_output_dir_prefix / f"iter_{iteration}"
        skill_manager_output_dir = skill_manager_iter_dir / "skills"

        previous_skill_output_dir = None
        if iteration > 0:
            previous_skill_output_dir = skill_manager_output_dir_prefix / f"iter_{iteration - 1}" / "skills"

        stage_use_skill = iteration > 0
        stage_skills_base_dir = (
            None if previous_skill_output_dir is None else str(previous_skill_output_dir)
        )

        run_da = should_run_stage(
            iteration=iteration,
            stage="da_agent",
            resume_from_iteration=resume_from_iteration,
            resume_from_stage=resume_from_stage,
        )

        if run_da:
            print(f"[run] DA stage: iter_{iteration}, agent_type={da_agent_type}, model={da_model}")
            reset_output_dir(da_stage_output_dir)

            run_da_stage(
                task_name=task_name,
                dataset_name=dataset_name,
                sample_nums=da_sample_nums,
                model=da_model,
                agent_type=da_agent_type,
                api_key=da_api_key,
                base_url=da_base_url,
                da_output_dir=da_stage_output_dir,
                da_run_name=base_run_name,
                use_skill=stage_use_skill,
                skills_base_dir=stage_skills_base_dir,
                category_list=category_list,
                manager_url=manager_url,
                backend=backend,
                working_dir=agent_working_dir,
                max_turns=da_max_turns,
                temperature=da_temperature,
                top_p=da_top_p,
                start_index=da_start_index,
                limit=da_limit,
                max_tokens=da_max_tokens,
                max_workers=da_max_workers,
            )

            stage_success_status[f"da_agent_iter_{iteration}"] = {
                "stage": "da_agent",
                "iteration": iteration,
                "success": True,
            }
            save_iteration_results(stage_success_status, stage_success_status_path)

        else:
            print(f"[skip] DA stage: iter_{iteration}")
            
        trajectory_dirs.append(da_stage_output_dir)

        run_verifier = should_run_stage(
            iteration=iteration,
            stage="verifier",
            resume_from_iteration=resume_from_iteration,
            resume_from_stage=resume_from_stage,
        )

        if run_verifier:
            print(f"[run] Verifier stage: iter_{iteration}")
            reset_output_dir(verify_output_dir)
            verifier_results = run_verifier_stage(
                verifier=verifier,
                input_file_dirs=[str(p) for p in trajectory_dirs],
                output_dir=verify_output_dir,
                category_list=category_list,
                is_iteration=iteration > 0,
            )

            stage_success_status[f"verifier_iter_{iteration}"] = {
                "stage": "verifier",
                "iteration": iteration,
                "success": True,
                "stage_results": verifier_results,
            }
            save_iteration_results(stage_success_status, stage_success_status_path)
        else:
            print(f"[skip] Verifier stage: iter_{iteration}")
            tmp_stage_success_status = json.load(open(stage_success_status_path, "r", encoding="utf-8"))
            verifier_results = tmp_stage_success_status.get(f"verifier_iter_{iteration}", {}).get("stage_results", {})

        traj_dir = verifier_results.get("traj_dir")
        verifier_prompt = verifier_results.get("prompt")
        if not traj_dir:
            raise ValueError("Verifier did not return a trajectory directory for skill manager.")

        run_skill_manager = should_run_stage(
            iteration=iteration,
            stage="skill_manager",
            resume_from_iteration=resume_from_iteration,
            resume_from_stage=resume_from_stage,
        )

        if run_skill_manager:
            print(f"[run] Skill Manager stage: iter_{iteration}, agent_type={skill_manager_agent_type}, model={skill_manager_model}")
            reset_output_dir(skill_manager_iter_dir)

            if iteration > 0:
                if skill_manager_output_dir.exists():
                    shutil.rmtree(skill_manager_output_dir)

                shutil.copytree(
                    previous_skill_output_dir,
                    skill_manager_output_dir,
                    ignore=shutil.ignore_patterns("prediction_*.json", "data", "task", "skills"),
                )

            run_skill_manager_stage(
                task_name=task_name,
                traj_dir=traj_dir,
                verifier_prompt=verifier_prompt,
                mode="create" if iteration == 0 else "modify",
                skill_output_dir=skill_manager_output_dir,
                skill_manager_model=skill_manager_model,
                skill_manager_agent_type=skill_manager_agent_type,
                skill_manager_api_key=skill_manager_api_key,
                skill_manager_base_url=skill_manager_base_url,
                skill_manager_backend=backend,
                skill_manager_max_workers=skill_manager_max_workers,
                skill_manager_use_category=bool(category_list),
                working_dir=str(skill_manager_iter_dir),
                category_list=category_list,
            )
            stage_success_status[f"skill_manager_iter_{iteration}"] = {
                "stage": "skill_manager",
                "iteration": iteration,
                "success": True,
                "stage_results": {
                    "skill_output_dir": str(skill_manager_output_dir),
                },
            }
            save_iteration_results(stage_success_status, stage_success_status_path)
        else:
            print(f"[skip] Skill Manager stage: iter_{iteration}")
            
    return iter_results


def main() -> int:
    config_arg_parser = argparse.ArgumentParser(add_help=False)
    config_arg_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_arg_parser.parse_known_args()

    config_defaults = (
        load_yaml_config(config_args.config) if config_args.config.exists() else {}
    )
    config_defaults["config"] = config_args.config

    args = build_parser(config_defaults).parse_args()
    if args.debug:
        import debugpy
        debugpy.listen(("localhost", 5680))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached. Continuing execution.")

    results = run(
        task_name=args.task,
        dataset_name=args.dataset,
        workspace_dir=args.workspace_dir,
        category_list=args.category_list,
        run_name=args.run_name,
        iterations=args.iterations,
        resume_from_iteration=args.resume_from_iteration,
        resume_from_stage=args.resume_from_stage,
        backend=args.backend,
        manager_url=args.manager_url,

        da_model=args.da_model,
        da_agent_type=args.da_agent_type,
        da_api_key=args.da_api_key,
        da_base_url=args.da_base_url,
        
        da_sample_nums=args.da_sample_nums,
        da_max_turns=args.da_max_turns,
        da_max_workers=args.da_max_workers,
        da_limit=args.da_limit,
        da_start_index=args.da_start_index,
        da_temperature=args.da_temperature,
        da_top_p=args.da_top_p,
        da_max_tokens=args.da_max_tokens,
        
        verifier=args.verifier,

        skill_manager_model=args.skill_manager_model,
        skill_manager_agent_type=args.skill_manager_agent_type,
        skill_manager_api_key=args.skill_manager_api_key,
        skill_manager_base_url=args.skill_manager_base_url,
        skill_manager_max_workers=args.skill_manager_max_workers,
    )

    if args.iterations:
        for iter_result in results:
            iteration = iter_result["iteration"]
            print(f"[Iteration {iteration}] Output: {iter_result['output_dir']}")
            for i, result in enumerate(iter_result["results"]):
                prefix = f"  [Sample {i}] " if len(iter_result["results"]) > 1 else "  "
                print(f"{prefix}Finished {len(result['results'])} tasks in {result['total_time']:.2f}s")
                if "file_paths" in result:
                    for file_type, file_path in result["file_paths"].items():
                        print(f"    {file_type}: {file_path}")
    else:
        for i, result in enumerate(results):
            prefix = f"[Sample {i}] " if len(results) > 1 else ""
            print(f"{prefix}Finished {len(result['results'])} tasks in {result['total_time']:.2f}s")
            if "file_paths" in result:
                for file_type, file_path in result["file_paths"].items():
                    print(f"  {file_type}: {file_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
