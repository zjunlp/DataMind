#!/usr/bin/env python3
"""
Reason-task generation pipeline orchestrator.

Pipeline flow:

  Iteration 1 (create skill):
    iter1_explore       -> explore without skills
    iter1_organize      -> organize trajectories
    iter1_create_skill  -> create skills from trajectories

  Iteration 2 (modify skill #1):
    iter2_explore       -> explore with skills
    iter2_organize      -> organize paired (iter0 vs iter1) trajectories
    iter2_modify_skill  -> refine skills from comparison

  Iteration 3 (modify skill #2):
    iter3_explore       -> explore with refined skills
    iter3_organize      -> organize paired (iter1 vs iter2) trajectories
    iter3_modify_skill  -> refine skills from comparison

Usage:
  python run_pipeline.py                                     # full run
  python run_pipeline.py --start-from iter2_explore          # resume
  python run_pipeline.py --dry-run                           # preview
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from pipeline_config import PipelineConfig

# ── Step definitions ─────────────────────────────────────────────────────────

STEPS = [
    # Iteration 1
    "iter1_explore",
    "iter1_organize",
    "iter1_create_skill",
    # Iteration 2
    "iter2_explore",
    "iter2_organize",
    "iter2_modify_skill",
    # Iteration 3
    "iter3_explore",
    "iter3_organize",
    "iter3_modify_skill",
]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _pipeline_run_id() -> str:
    return "run_" + datetime.now().strftime("%m%d%H")


def _run(cmd: list[str], cwd: str, dry_run: bool) -> None:
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  CWD: {cwd}")
    if dry_run:
        print("  [DRY RUN - skipped]")
        return
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n  FAILED with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def _sync_trajectory_dir(src: str, dst: str, dry_run: bool) -> None:
    print(f"  SYNC TRAJ: {src} -> {dst}")
    if dry_run:
        print("  [DRY RUN - skipped]")
        return
    if not os.path.isdir(src):
        print(f"\n  FAILED: trajectory source directory not found: {src}", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _sync_data_dirs(cfg: PipelineConfig, dry_run: bool) -> None:
    src = cfg.data_dir
    dst = cfg.skill_manager_data_dir
    print(f"  SYNC DATA: {src}/* -> {dst}/")
    if dry_run:
        print("  [DRY RUN - skipped]")
        return
    if not os.path.isdir(src):
        print(f"\n  FAILED: data source directory not found: {src}", file=sys.stderr)
        sys.exit(1)
    if os.path.abspath(src) == os.path.abspath(dst):
        print("  Source and destination are the same; data sync skipped.")
        return

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        src_item = os.path.join(src, name)
        dst_item = os.path.join(dst, name)
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)


def _reset_dir(path: str, dry_run: bool) -> None:
    print(f"  RESET DIR: {path}")
    if dry_run:
        print("  [DRY RUN - skipped]")
        return
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _prepare_workspace(cfg: PipelineConfig, reset: bool, dry_run: bool) -> None:
    if not reset:
        print("  Workspace reset skipped because --start-from is set.")
        return

    _reset_dir(cfg.skill_manager_data_dir, dry_run)
    _reset_dir(cfg.skill_output_dir, dry_run)
    _reset_dir(cfg.skill_manager_traj_input_dir(), dry_run)
    _sync_data_dirs(cfg, dry_run)


def _copy_dir_snapshot(src: str, dst: str, label: str, dry_run: bool) -> None:
    final_dst = dst
    suffix = 2
    while os.path.exists(final_dst):
        final_dst = f"{dst}_{suffix}"
        suffix += 1

    print(f"  {label}: {src} -> {final_dst}")
    if dry_run:
        print("  [DRY RUN - skipped]")
        return
    if not os.path.isdir(src):
        print(f"\n  FAILED: source directory not found: {src}", file=sys.stderr)
        sys.exit(1)

    shutil.copytree(src, final_dst)


def _save_skill_snapshot(cfg: PipelineConfig, step_name: str, run_id: str, dry_run: bool) -> None:
    src = cfg.skill_output_dir
    dst = os.path.join(cfg.skill_save_run_dir(run_id), step_name)
    _copy_dir_snapshot(src, dst, "SAVE SKILLS", dry_run)


# ── Step implementations ─────────────────────────────────────────────────────

def _run_explore(cmd_args: list[str], cfg: PipelineConfig, dry_run: bool) -> None:
    explore_py = os.path.join(cfg.evaluate_script_dir, "evaluate.py")
    _run(
        [sys.executable, explore_py, *cmd_args],
        cwd=cfg.evaluate_script_dir,
        dry_run=dry_run,
    )


def _base_explore_args(cfg: PipelineConfig) -> list[str]:
    return [
        "--dataset", cfg.task_name,
        "--backend", "openai",
        "--manager-url", cfg.manager_url,
        "--model", cfg.model,
        "--temperature", str(cfg.explore_temperature),
        "--max-tokens", str(cfg.explore_max_tokens),
        "--max-workers", str(cfg.max_workers),
        "--split", "explore",
    ]


def step_explore_no_skill(cfg: PipelineConfig, dry_run: bool) -> None:
    for run_idx in range(1, cfg.num_runs + 1):
        output_dir = os.path.join(
            cfg.explore_results_dir(0),
            cfg.no_skill_explore_output_name.format(run_idx=run_idx),
        )
        print(f"  EXPLORE NO SKILL: run {run_idx}/{cfg.num_runs} -> {output_dir}")
        _run_explore(
            _base_explore_args(cfg) + [
                "--output-dir", output_dir,
                "--api-key", cfg.api_key,
                "--base-url", cfg.base_url,
                "--no-skill",
            ],
            cfg=cfg,
            dry_run=dry_run,
        )


def step_organize_iter0(cfg: PipelineConfig, dry_run: bool) -> None:
    verifier_output_dir = cfg.organized_traj_dir(0)
    _run(
        [
            sys.executable, "-m", "verifier.organize_dabstep",
            "--results-dir", cfg.explore_results_dir(0),
            "--category-file", cfg.category_file,
            "--output-dir", verifier_output_dir,
        ],
        cwd=cfg.base_dir,
        dry_run=dry_run,
    )
    _sync_trajectory_dir(
        verifier_output_dir,
        cfg.skill_manager_traj_input_dir(),
        dry_run=dry_run,
    )


def _skill_generation_cmd(cfg: PipelineConfig, module_name: str) -> list[str]:
    return [
        sys.executable, "-m", module_name,
        "--task-name", cfg.task_name,
        "--traj-base-dir", cfg.skill_manager_traj_input_dir(),
        "--skill-output-dir", cfg.skill_output_dir,
        "--data-dir", cfg.skill_manager_data_dir,
        "--categories", *cfg.target_categories,
        "--max-concurrent", str(cfg.max_concurrent),
    ]


def step_create_skill(cfg: PipelineConfig, dry_run: bool, run_id: str | None = None) -> None:
    _run(
        _skill_generation_cmd(cfg, "skill_manager.create_skill"),
        cwd=cfg.base_dir,
        dry_run=dry_run,
    )
    if run_id:
        _save_skill_snapshot(cfg, "iter1_create_skill", run_id, dry_run)


def step_explore_with_skill(cfg: PipelineConfig, explore_round: int, dry_run: bool) -> None:
    """explore_round: which round of with-skill exploration (1, 2, ...)."""
    for category in cfg.target_categories:
        for run_idx in range(1, cfg.num_runs + 1):
            output_dir = os.path.join(
                cfg.results_base_dir,
                str(explore_round),
                cfg.with_skill_explore_output_name.format(category=category, run_idx=run_idx),
            )
            skill_dir = os.path.join(cfg.skill_output_dir, category)
            print(
                f"  EXPLORE WITH SKILL: iter {explore_round} | "
                f"{category} | run {run_idx}/{cfg.num_runs}"
            )
            _run_explore(
                _base_explore_args(cfg) + [
                    "--skills-base-dir", skill_dir,
                    "--query-categories", category,
                    "--query-category-file", cfg.category_file,
                    "--output-dir", output_dir,
                    "--api-key", cfg.api_key,
                    "--base-url", cfg.base_url,
                ],
                cfg=cfg,
                dry_run=dry_run,
            )


def step_organize_pair(cfg: PipelineConfig, iter_prev: int, iter_curr: int, dry_run: bool) -> None:
    verifier_output_dir = cfg.paired_traj_dir(iter_prev, iter_curr)
    _run(
        [
            sys.executable, "-m", "verifier.organize_dabstep_iter1_pair",
            "--iter-prev", str(iter_prev),
            "--iter-curr", str(iter_curr),
            "--results-dir-prev", cfg.explore_results_dir(iter_prev),
            "--results-dir-curr", cfg.explore_results_dir(iter_curr),
            "--category-file", cfg.category_file,
            "--output-dir", verifier_output_dir,
        ],
        cwd=cfg.base_dir,
        dry_run=dry_run,
    )
    _sync_trajectory_dir(
        verifier_output_dir,
        cfg.skill_manager_traj_input_dir(),
        dry_run=dry_run,
    )


def step_modify_skill(
    cfg: PipelineConfig,
    iter_prev: int,
    iter_curr: int,
    dry_run: bool,
    run_id: str | None = None,
) -> None:
    _run(
        _skill_generation_cmd(cfg, "skill_manager.modify_skill"),
        cwd=cfg.base_dir,
        dry_run=dry_run,
    )
    if run_id:
        _save_skill_snapshot(cfg, f"iter{iter_curr + 1}_modify_skill", run_id, dry_run)


# ── Step dispatcher ──────────────────────────────────────────────────────────
def run_step(step_name: str, cfg: PipelineConfig, dry_run: bool, run_id: str) -> None:
    handlers = {
        "iter1_explore": lambda: step_explore_no_skill(cfg, dry_run),
        "iter1_organize": lambda: step_organize_iter0(cfg, dry_run),
        "iter1_create_skill": lambda: step_create_skill(cfg, dry_run, run_id),
        "iter2_explore": lambda: step_explore_with_skill(cfg, 1, dry_run),
        "iter2_organize": lambda: step_organize_pair(cfg, 0, 1, dry_run),
        "iter2_modify_skill": lambda: step_modify_skill(cfg, 0, 1, dry_run, run_id),
        "iter3_explore": lambda: step_explore_with_skill(cfg, 2, dry_run),
        "iter3_organize": lambda: step_organize_pair(cfg, 1, 2, dry_run),
        "iter3_modify_skill": lambda: step_modify_skill(cfg, 1, 2, dry_run, run_id),
    }
    handlers[step_name]()


def _apply_cli_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> None:
    if args.api_key:
        cfg.api_key = args.api_key
    if args.base_url:
        cfg.base_url = args.base_url
    if args.model:
        cfg.model = args.model
    if args.num_runs:
        cfg.num_runs = args.num_runs
    if args.max_workers:
        cfg.max_workers = args.max_workers


def _validate_config(cfg: PipelineConfig, dry_run: bool) -> None:
    if dry_run:
        return
    if not cfg.api_key:
        print("ERROR: API key not set. Use --api-key or set REASON_API_KEY env var.", file=sys.stderr)
        sys.exit(1)
    if not cfg.base_url:
        print("ERROR: Base URL not set. Use --base-url or set REASON_BASE_URL env var.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reason-task generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available steps:\n  " + "\n  ".join(
            f"{i+1:2d}. {s}" for i, s in enumerate(STEPS)
        ),
    )
    parser.add_argument(
        "--start-from",
        choices=STEPS,
        default=STEPS[0],
        help="Resume pipeline from this step (inclusive)",
    )
    parser.add_argument(
        "--stop-after",
        choices=STEPS,
        default=STEPS[-1],
        help="Stop pipeline after this step (inclusive)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument("--api-key", default="", help="API key (or set REASON_API_KEY)")
    parser.add_argument("--base-url", default="", help="API base URL (or set REASON_BASE_URL)")
    parser.add_argument("--model", default="", help="Model name override")
    parser.add_argument("--num-runs", type=int, default=0, help="Number of explore runs per config")
    parser.add_argument("--max-workers", type=int, default=0, help="Max parallel workers for explore")
    args = parser.parse_args()

    cfg = PipelineConfig()
    _apply_cli_overrides(cfg, args)
    _validate_config(cfg, args.dry_run)

    start_idx = STEPS.index(args.start_from)
    stop_idx = STEPS.index(args.stop_after)

    if start_idx > stop_idx:
        print(f"ERROR: --start-from ({args.start_from}) is after --stop-after ({args.stop_after})",
              file=sys.stderr)
        sys.exit(1)

    active_steps = STEPS[start_idx:stop_idx + 1]
    run_id = _pipeline_run_id()

    print(f"{'='*60}")
    print(f"Reason-task generation pipeline")
    print(f"{'='*60}")
    print(f"  Mode:       {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Model:      {cfg.model}")
    print(f"  Runs/step:  {cfg.num_runs}")
    print(f"  Categories: {len(cfg.target_categories)}")
    print(f"  Steps:      {start_idx+1}-{stop_idx+1} / {len(STEPS)}")
    print(f"  Skill log:  {cfg.skill_save_run_dir(run_id)}")
    print(f"{'='*60}\n")

    _prepare_workspace(
        cfg,
        reset=args.start_from == STEPS[0],
        dry_run=args.dry_run,
    )

    for step_name in active_steps:
        global_idx = STEPS.index(step_name) + 1
        iter_num = step_name.split("_")[0]  # e.g. "iter1"
        print(f"\n[{_ts()}] === {iter_num} | Step {global_idx}/{len(STEPS)}: {step_name} ===")
        t0 = time.time()

        run_step(step_name, cfg, args.dry_run, run_id)

        elapsed = time.time() - t0
        print(f"[{_ts()}] === {step_name} done ({elapsed:.1f}s) ===\n")

    print(f"\n{'='*60}")
    print(f"[{_ts()}] Pipeline complete ({args.start_from} -> {args.stop_after})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
