#!/usr/bin/env python3
"""DDR Pipeline for iterative skill generation and evaluation."""
import argparse
import logging
import sys
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "DataCOPE_DDR.pipeline"

from .config_manager import ConfigManager
from .pipeline import Pipeline
from .utils import setup_logging


def load_yaml_config(path: Path) -> dict:
    """Load pipeline defaults from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    """Main entry point for the DDR skill-generation pipeline."""
    parser = argparse.ArgumentParser(description="DDR Pipeline")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--scenario", type=str, choices=["mimic", "10k", "globem"])
    parser.add_argument("--model", type=str)
    parser.add_argument("--eval-model", type=str)
    parser.add_argument("--eval-provider", type=str)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--provider", type=str)
    parser.add_argument("--ddr-bench-dir", type=str)
    parser.add_argument("--ddr-data-dir", type=str)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    defaults = {}
    config_file = Path(args.config).expanduser() if args.config else project_root / "config.yaml"
    if args.config and not config_file.exists():
        parser.error(f"Config file not found: {config_file}")
    if config_file.exists():
        defaults = load_yaml_config(config_file)

    def val(cli, key, default=None):
        return cli if cli is not None else defaults.get(key, default)

    def nested_val(cli, section, key, legacy_key=None, default=None):
        if cli is not None:
            return cli
        section_data = defaults.get(section)
        if isinstance(section_data, dict) and key in section_data:
            return section_data[key]
        if legacy_key and legacy_key in defaults and not isinstance(defaults[legacy_key], dict):
            return defaults[legacy_key]
        return default

    scenario = val(args.scenario, "scenario")
    model = nested_val(args.model, "provider", "default_model", "model")
    eval_model = nested_val(args.eval_model, "evaluation", "model", "eval_model", "gpt-5-mini")
    max_workers = nested_val(args.max_workers, "agent", "max_workers", "max_workers", 5)
    provider = nested_val(args.provider, "provider", "default_provider", "provider", "openai")
    eval_provider = nested_val(args.eval_provider, "evaluation", "provider", "eval_provider", "")

    if not scenario or not model:
        parser.error("--scenario and --model are required (via CLI or config)")
    if model == "MODEL_NAME":
        parser.error("provider.default_model is still MODEL_NAME; set a concrete agent model")
    if max_workers < 1:
        parser.error("--max-workers must be at least 1")

    ddr_bench_dir = Path(
        nested_val(args.ddr_bench_dir, "paths", "ddr_bench_dir", "ddr_bench_dir", "")
        or str(project_root.parent / "DDR_Bench")
    ).expanduser()
    ddr_data_dir = Path(
        nested_val(args.ddr_data_dir, "paths", "ddr_data_dir", "ddr_data_dir", "")
        or str(project_root.parent / "DDRBench_data")
    ).expanduser()

    if not ddr_bench_dir.exists():
        parser.error(f"DDR_Bench directory not found: {ddr_bench_dir}")
    if not ddr_data_dir.exists():
        parser.error(f"DDRBench_data directory not found: {ddr_data_dir}")

    cfg = ConfigManager(
        scenario=scenario,
        model=model,
        eval_model=eval_model,
        max_workers=max_workers,
        ddr_bench_dir=ddr_bench_dir,
        ddr_data_dir=ddr_data_dir,
        project_root=project_root,
        provider=provider,
        eval_provider=eval_provider,
    )
    cfg.setup(resume=args.resume)

    log_file = cfg.records_dir / "pipeline.log"
    setup_logging(log_file, verbose=args.verbose)
    logger = logging.getLogger("ddr_pipeline")

    logger.info("=" * 60)
    logger.info("DDR Pipeline")
    logger.info(f"  scenario:    {scenario}")
    logger.info(f"  model:       {model}")
    logger.info(f"  provider:    {provider}")
    logger.info(f"  eval_model:  {eval_model}")
    logger.info(f"  eval_provider: {eval_provider or '(from DDR_Bench config)'}")
    logger.info(f"  workers:     {max_workers}")
    logger.info(f"  DDR_Bench:   {ddr_bench_dir}")
    logger.info(f"  records:     {cfg.records_dir}")
    logger.info(f"  config:      {cfg.config_path}")
    logger.info(f"  resume:      {args.resume}")
    logger.info("=" * 60)

    pipeline = Pipeline(cfg=cfg, resume=args.resume)
    pipeline.run()


if __name__ == "__main__":
    main()
