#!/usr/bin/env python3
"""
Evaluation pipeline orchestrator for DABStep.

Pipeline flow:
  eval_run         -> Run evaluation with skills (per category)
  collect_results  -> Collect predictions into a single JSONL
  validate         -> Score predictions against ground truth

Usage:
  python run_eval.py --dry-run
  python run_eval.py --start-from collect_results   # resume
  python run_eval.py --skill-dir /path/to/skills    # specify skill source
  python run_eval.py --config eval.yaml             # specify config file
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_FILE = os.path.join(BASE_DIR, "eval.yaml")
API_KEY_ENV = "REASON_API_KEY"
BASE_URL_ENV = "REASON_BASE_URL"

TARGET_CATEGORIES = [
    "Applicable_Fee_IDs",
    "Average_Fee_Estimation",
    "Average_Transaction_Value_Stats",
    "Dataset_Metadata_and_Business_Rules",
    "Fee_Delta_and_Impact_Simulation",
    "Fraud_and_General_Macro_Analysis",
    "Highest_Cost_Scenario_Identification",
    "Routing_and_Cost_Optimization",
    "Total_Fees_Calculation",
]


@dataclass
class EvalConfig:
    base_dir: str = BASE_DIR
    results_dir: str = os.path.join(BASE_DIR, "DSGym", "examples", "results", "dabstep_test")
    skill_dir: str = os.path.join(BASE_DIR, "skill")
    category_file: str = os.path.join(BASE_DIR, "DSGym", "data", "task", "DABStep", "all_query_category.json")
    all_jsonl: str = os.path.join(BASE_DIR, "DSGym", "data", "task", "DABStep", "all.jsonl")
    gt_file: str = os.path.join(BASE_DIR, "DSGym", "data", "task", "DABStep", "test.jsonl")
    eval_script_dir: str = os.path.join(BASE_DIR, "DSGym", "examples")
    val_script_dir: str = os.path.join(BASE_DIR, "val", "dabstep_benchmark")
    output_jsonl: str = os.path.join(BASE_DIR, "results", "predictions.jsonl")

    model: str = "Qwen3.5-397B-A17B"
    api_key: str = ""
    base_url: str = ""
    manager_url: str = "http://localhost:5005"
    max_workers: int = 30

    target_categories: list[str] = field(default_factory=lambda: list(TARGET_CATEGORIES))

    def __post_init__(self):
        self.api_key = os.environ.get(API_KEY_ENV) or self.api_key
        self.base_url = os.environ.get(BASE_URL_ENV) or self.base_url


# ── Steps ────────────────────────────────────────────────────────────────────

STEPS = [
    "eval_run",
    "collect_results",
    "validate",
]

CONFIG_PATH_FIELDS = {
    "results_dir",
    "skill_dir",
    "category_file",
    "all_jsonl",
    "gt_file",
    "eval_script_dir",
    "val_script_dir",
    "output_jsonl",
}

CONFIG_VALUE_FIELDS = {
    "model",
    "api_key",
    "base_url",
    "manager_url",
    "max_workers",
    "target_categories",
}

CONFIG_RUN_FIELDS = {
    "start_from",
    "stop_after",
    "dry_run",
}


@dataclass
class RunOptions:
    start_from: str = STEPS[0]
    stop_after: str = STEPS[-1]
    dry_run: bool = False


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _resolve_path(path: str, root: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))


def _config_items(data: dict[str, Any]) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for section_name in ("paths", "pipeline", "model", "runtime", "run"):
        section = data.get(section_name)
        if section is not None:
            if not isinstance(section, dict):
                raise ValueError(f"Config section '{section_name}' must be a mapping.")
            items.update(section)
    items.update({
        key: value
        for key, value in data.items()
        if key not in {"paths", "pipeline", "model", "runtime", "run"}
    })
    return items


def load_config(config_path: str) -> tuple[EvalConfig, RunOptions, str | None]:
    cfg = EvalConfig()
    opts = RunOptions()

    if not config_path:
        config_path = DEFAULT_CONFIG_FILE

    config_path = os.path.abspath(config_path)
    loaded_path: str | None = None
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a YAML mapping at the top level.")

        config_root = os.path.dirname(config_path)
        for key, value in _config_items(data).items():
            if value is None:
                continue
            if key in CONFIG_PATH_FIELDS:
                setattr(cfg, key, _resolve_path(str(value), config_root))
            elif key in CONFIG_VALUE_FIELDS:
                setattr(cfg, key, value)
            elif key in CONFIG_RUN_FIELDS:
                setattr(opts, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
        loaded_path = config_path
    elif config_path != DEFAULT_CONFIG_FILE:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg.api_key = os.environ.get(API_KEY_ENV) or cfg.api_key
    cfg.base_url = os.environ.get(BASE_URL_ENV) or cfg.base_url

    return cfg, opts, loaded_path


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


def step_eval_run(cfg: EvalConfig, dry_run: bool) -> None:
    script = os.path.join(cfg.eval_script_dir, "evaluate.py")
    for category in cfg.target_categories:
        output_dir = os.path.join(cfg.results_dir, f"dabstep_test_{category}")
        skill_dir = os.path.join(cfg.skill_dir, category)
        print(f"=== Eval | {category} ===")
        _run(
            [
                sys.executable, script,
                "--dataset", "dabstep",
                "--backend", "openai",
                "--manager-url", cfg.manager_url,
                "--model", cfg.model,
                "--temperature", "0.0",
                "--max-tokens", "16384",
                "--max-workers", str(cfg.max_workers),
                "--split", "test",
                "--skills-base-dir", skill_dir,
                "--query-categories", category,
                "--query-category-file", cfg.category_file,
                "--output-dir", output_dir,
                "--api-key", cfg.api_key,
                "--base-url", cfg.base_url,
            ],
            cwd=cfg.eval_script_dir,
            dry_run=dry_run,
        )
    print("=== All categories evaluated ===")


def step_collect_results(cfg: EvalConfig, dry_run: bool) -> None:
    os.makedirs(os.path.dirname(cfg.output_jsonl), exist_ok=True)
    _run(
        [
            sys.executable, os.path.join(cfg.eval_script_dir, "get_result_jsonl.py"),
            "--results-dir", cfg.results_dir,
            "--all-jsonl", cfg.all_jsonl,
            "--output", cfg.output_jsonl,
            "--categories", *cfg.target_categories,
        ],
        cwd=cfg.eval_script_dir,
        dry_run=dry_run,
    )


def step_validate(cfg: EvalConfig, dry_run: bool) -> None:
    _run(
        [
            sys.executable, os.path.join(cfg.val_script_dir, "val.py"),
            "--pred-file", cfg.output_jsonl,
            "--gt-file", cfg.gt_file,
        ],
        cwd=cfg.val_script_dir,
        dry_run=dry_run,
    )


STEP_DISPATCH = {
    "eval_run":        step_eval_run,
    "collect_results": step_collect_results,
    "validate":        step_validate,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the DABStep evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available steps:\n  " + "\n  ".join(f"{i+1}. {s}" for i, s in enumerate(STEPS)),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_FILE, help="YAML config path")
    parser.add_argument("--start-from", choices=STEPS, default="", help="Resume from step")
    parser.add_argument("--stop-after", choices=STEPS, default="", help="Stop after step")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--api-key", default="", help=f"API key (or set {API_KEY_ENV})")
    parser.add_argument("--base-url", default="", help=f"API base URL (or set {BASE_URL_ENV})")
    parser.add_argument("--model", default="", help="Model name override")
    parser.add_argument("--skill-dir", default="", help="Skill directory override")
    parser.add_argument("--output-jsonl", default="", help="Output JSONL path override")
    parser.add_argument("--max-workers", type=int, default=0, help="Max parallel workers")
    args = parser.parse_args()

    try:
        cfg, opts, loaded_config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    if args.start_from:
        opts.start_from = args.start_from
    if args.stop_after:
        opts.stop_after = args.stop_after
    if args.dry_run:
        opts.dry_run = True
    if args.api_key:
        cfg.api_key = args.api_key
    if args.base_url:
        cfg.base_url = args.base_url
    if args.model:
        cfg.model = args.model
    if args.skill_dir:
        cfg.skill_dir = args.skill_dir
    if args.output_jsonl:
        cfg.output_jsonl = args.output_jsonl
    if args.max_workers:
        cfg.max_workers = args.max_workers

    if opts.start_from not in STEPS:
        print(f"ERROR: start_from must be one of {', '.join(STEPS)}", file=sys.stderr)
        sys.exit(1)
    if opts.stop_after not in STEPS:
        print(f"ERROR: stop_after must be one of {', '.join(STEPS)}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(cfg.target_categories, list) or not all(isinstance(c, str) for c in cfg.target_categories):
        print("ERROR: target_categories must be a list of strings.", file=sys.stderr)
        sys.exit(1)

    if not cfg.api_key and not opts.dry_run:
        print(f"ERROR: API key not set. Use --api-key or set {API_KEY_ENV} env var.", file=sys.stderr)
        sys.exit(1)
    if not cfg.base_url and not opts.dry_run:
        print(f"ERROR: Base URL not set. Use --base-url or set {BASE_URL_ENV} env var.", file=sys.stderr)
        sys.exit(1)

    start_idx = STEPS.index(opts.start_from)
    stop_idx = STEPS.index(opts.stop_after)

    if start_idx > stop_idx:
        print(f"ERROR: start_from ({opts.start_from}) is after stop_after ({opts.stop_after})",
              file=sys.stderr)
        sys.exit(1)

    active_steps = STEPS[start_idx:stop_idx + 1]

    print(f"{'='*60}")
    print(f"DABStep Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"  Mode:       {'DRY RUN' if opts.dry_run else 'LIVE'}")
    print(f"  Config:     {loaded_config or '(defaults only)'}")
    print(f"  Model:      {cfg.model}")
    print(f"  Skill dir:  {cfg.skill_dir}")
    print(f"  Output:     {cfg.output_jsonl}")
    print(f"  Steps:      {start_idx+1}-{stop_idx+1} / {len(STEPS)}")
    print(f"{'='*60}\n")

    for step_name in active_steps:
        global_idx = STEPS.index(step_name) + 1
        print(f"\n[{_ts()}] === Step {global_idx}/{len(STEPS)}: {step_name} ===")
        t0 = time.time()

        STEP_DISPATCH[step_name](cfg, opts.dry_run)

        elapsed = time.time() - t0
        print(f"[{_ts()}] === {step_name} done ({elapsed:.1f}s) ===\n")

    print(f"\n{'='*60}")
    print(f"[{_ts()}] Eval pipeline complete ({opts.start_from} -> {opts.stop_after})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
