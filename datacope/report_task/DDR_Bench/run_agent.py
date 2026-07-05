#!/usr/bin/env python3
"""
Unified Agent Runner for DDR_Bench.

Single entry point for running data analysis agents across all scenarios:
- MIMIC: Patient data analysis using MIMIC-IV database
- 10-K: Financial report analysis using SEC 10-K filings
- GLOBEM: Behavioral data analysis using GLOBEM dataset

See README.md for detailed usage instructions.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

try:
    from .config import get_config
    from .base_batch_analyzer import BaseBatchAnalyzer
except ImportError:
    from config import get_config
    from base_batch_analyzer import BaseBatchAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_cli_path(raw_path: Optional[str], original_cwd: Path, prefer_existing: bool = False) -> Optional[str]:
    """Resolve explicit CLI paths from either caller cwd or DDR_Bench root."""
    if not raw_path:
        return raw_path
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_candidate = original_cwd / path
    if prefer_existing and cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    project_candidate = PROJECT_ROOT / path
    if prefer_existing and project_candidate.exists():
        return str(project_candidate.resolve())

    return str(cwd_candidate.resolve())


def derive_split_artifact_path(base_path: str, split: str) -> str:
    """Derive a split-specific artifact path by appending _<split> to the stem."""
    path = Path(base_path)
    return str(path.with_name(f"{path.stem}_{split}{path.suffix}"))


def append_log_dir_suffix(log_dir: str, suffixes: List[str]) -> str:
    """Append normalized suffixes to the final log directory name."""
    clean_suffixes = [suffix.strip("_") for suffix in suffixes if suffix and suffix.strip("_")]
    if not clean_suffixes:
        return log_dir

    path = Path(log_dir)
    return str(path.with_name(f"{path.name}_{'_'.join(clean_suffixes)}"))


def build_log_dir(
    base_log_dir: str,
    agent_mode: str,
    split: Optional[str],
) -> str:
    """Build the final log directory name using checklist and split suffixes."""
    suffixes: List[str] = []
    if agent_mode == "checklist":
        suffixes.append("checklist")
    if split:
        suffixes.append(split)
    return append_log_dir_suffix(base_log_dir, suffixes)


def resolve_split_run_paths(
    scenario_config,
    split: Optional[str],
    agent_mode: str = "analysis",
) -> tuple[str, str, str]:
    """Resolve split-aware ID/log/data paths for all scenarios."""
    log_dir = build_log_dir(scenario_config.log_dir, agent_mode, split)
    id_file = scenario_config.id_file
    split_db_path = ""

    if not split:
        return log_dir, id_file, split_db_path

    id_file = derive_split_artifact_path(id_file, split)
    if getattr(scenario_config, "db_path", ""):
        candidate_split_db_path = derive_split_artifact_path(scenario_config.db_path, split)
        if Path(candidate_split_db_path).exists():
            split_db_path = candidate_split_db_path

    return log_dir, id_file, split_db_path


def get_entity_prefix_for_scenario(scenario_config, scenario: str) -> str:
    """Return the log subdirectory prefix for a scenario."""
    prefix = getattr(scenario_config, "identifier_prefix", "") or ""
    if prefix:
        return prefix
    return {
        "mimic": "patient",
        "10k": "company",
        "globem": "user",
    }.get(scenario, "entity")


def derive_split_data_dir(base_path: str, split: str) -> str:
    """Derive a split-specific data directory by appending _<split> to the name."""
    path = Path(base_path)
    return str(path.with_name(f"{path.name}_{split}"))


def resolve_qa_logs_dir(raw_path: str) -> Path:
    """Resolve a checklist-mode log directory, accepting names under ./logs."""
    path = Path(raw_path)
    if path.exists():
        return path
    return Path("logs") / raw_path


def load_id_order(path: Optional[str]) -> List[str]:
    """Load an optional JSON list of entity IDs used to preserve output order."""
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [str(item) for item in data]


def entity_id_from_dir(path: Path, entity_prefix: str) -> str:
    """Extract the entity ID from a log directory name."""
    prefix = f"{entity_prefix}_"
    name = path.name
    return name[len(prefix):] if name.startswith(prefix) else name


def ordered_entity_dirs(logs_dir: Path, id_order: List[str], entity_prefix: str) -> Iterable[Path]:
    """Yield entity log directories in ID-file order, then any remaining directories."""
    dirs_by_id: Dict[str, Path] = {
        entity_id_from_dir(path, entity_prefix): path
        for path in sorted(logs_dir.glob(f"{entity_prefix}_*"))
        if path.is_dir()
    }

    yielded = set()
    for entity_id in id_order:
        path = dirs_by_id.get(entity_id)
        if path is not None:
            yielded.add(entity_id)
            yield path

    for entity_id, path in sorted(dirs_by_id.items()):
        if entity_id not in yielded:
            yield path


def qa_submission_files(entity_dir: Path) -> List[Path]:
    """Return the latest QA submission file for an entity log directory."""
    files = sorted(entity_dir.glob("qa_submissions_*.jsonl"), key=lambda p: (p.stat().st_mtime, p.name))
    return files[-1:] if files else []


def load_qa_submission_rows(path: Path) -> List[Dict[str, str]]:
    """Load QA submissions from one JSONL file."""
    qa_pairs: List[Dict[str, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            question = str(row.get("q", "")).strip()
            answer = str(row.get("a", "")).strip()
            if not question or not answer:
                raise ValueError("missing non-empty q/a")
        except Exception as exc:
            print(f"WARNING: Malformed row in {path}:{line_no}: {exc}")
            continue

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "source_text": "",
        })
    return qa_pairs


def convert_qa_submissions_to_qa(
    logs_dir: str,
    output: str,
    id_file: Optional[str],
    entity_prefix: str,
    split: Optional[str] = None,
) -> None:
    """Convert checklist-mode submission logs into a standard DDR_Bench qa.json file."""
    logs_path = resolve_qa_logs_dir(logs_dir)
    if not logs_path.exists():
        raise FileNotFoundError(f"Log directory not found: {logs_path}")

    normalized_prefix = entity_prefix.strip().strip("_")
    if not normalized_prefix:
        raise ValueError("entity_prefix must be non-empty")

    id_order = load_id_order(id_file)
    results = []
    total_qa_pairs = 0

    for entity_dir in ordered_entity_dirs(logs_path, id_order, normalized_prefix):
        qa_pairs = []
        for path in qa_submission_files(entity_dir):
            qa_pairs.extend(load_qa_submission_rows(path))
        if not qa_pairs:
            continue

        total_qa_pairs += len(qa_pairs)
        results.append({
            "entity_id": entity_id_from_dir(entity_dir, normalized_prefix),
            "qa_pairs": qa_pairs,
        })

    payload = {
        "metadata": {
            "source": "qa_submissions",
            "source_log_dir": str(logs_path),
            "entity_count": len(results),
            "qa_pair_count": total_qa_pairs,
        },
        "results": results,
    }
    if split:
        payload["metadata"]["split"] = split

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}: entities={len(results)}, qa_pairs={total_qa_pairs}")


class PatientBatchAnalyzer(BaseBatchAnalyzer):
    """Batch analyzer for MIMIC patient data."""
    
    def __init__(self, base_log_dir: str, target_ids: Optional[Set[str]] = None, overwrite: bool = False):
        super().__init__(base_log_dir, target_ids, overwrite)

    def extract_identifiers(self, source_file: Path) -> List[Dict[str, Any]]:
        """Extract patient identifiers from pre-defined ID list file."""
        try:
            print(f"Reading patient IDs from: {source_file}")
            with open(source_file, 'r', encoding='utf-8') as f:
                patient_ids = json.load(f)
            
            if not isinstance(patient_ids, list):
                print(f"   Error: Expected a list of patient IDs")
                return []
            
            patients_list = []
            for pid in patient_ids:
                subject_id = str(pid)
                patients_list.append({
                    "patient_id": f"patient_{subject_id}",
                    "subject_id": subject_id,
                    "identifier": subject_id,
                    "data": {}
                })
            
            print(f"   Found {len(patients_list)} patients")
            return patients_list
            
        except Exception as e:
            print(f"   Error reading file: {e}")
            return []
    
    def _prepare_analysis_command(self, identifier_info: Dict[str, Any], source_file: Path,
                                  subdir_name: str, **kwargs) -> tuple:
        """Prepare the command for patient analysis."""
        subject_id = identifier_info["subject_id"]
        
        patient_log_dir = self.base_log_dir / subdir_name
        patient_log_dir.mkdir(parents=True, exist_ok=True)
        
        task = f"Analyze patient {subject_id}"
        
        cmd = [
            sys.executable,
            "agent/data_agent.py",
            "--task", task,
            "--log-dir", str(patient_log_dir)
        ]
        
        if not kwargs.get("auto_finish", True):
            cmd.append("--no-auto-finish")
        
        # Pass config info
        if kwargs.get("config_path"):
            cmd.extend(["--config", kwargs.get("config_path")])
        if kwargs.get("scenario"):
            cmd.extend(["--scenario", kwargs.get("scenario")])
        if kwargs.get("skill_dir"):
            cmd.extend(["--skill-dir", kwargs.get("skill_dir")])
        if kwargs.get("agent_mode"):
            cmd.extend(["--agent-mode", kwargs.get("agent_mode")])

        # MCP arguments: Only pass server script, agent will load config for DB path
        cmd.extend(["--sql-server", "tool_server/sqlite_mcp.py"])

        task_db_path = kwargs.get("split_db_path")
        if task_db_path:
            cmd.extend(["--data-path", task_db_path])
        
        # Pass max_turns if provided
        if kwargs.get("max_turns"):
            cmd.extend(["--max-turns", str(kwargs.get("max_turns"))])
            
        env = os.environ.copy()
        env['CUSTOM_LOG_DIR'] = str(patient_log_dir)
        
        return cmd, env, f"Patient {subject_id}"
    
    def get_subdir_name(self, identifier: str) -> str:
        return f"patient_{identifier}"
    
    def _create_identifier_from_logs(self, identifier: str, dirname: str) -> Optional[Dict[str, Any]]:
        return {
            "patient_id": f"patient_{identifier}",
            "subject_id": identifier,
            "identifier": identifier,
            "data": {}
        }


class CompanyBatchAnalyzer(BaseBatchAnalyzer):
    """Batch analyzer for 10-K company data."""
    
    def __init__(self, base_log_dir: str, target_ids: Optional[Set[str]] = None, overwrite: bool = False):
        super().__init__(base_log_dir, target_ids, overwrite)

    def extract_identifiers(self, source_file: Path) -> List[Dict[str, Any]]:
        """Extract company identifiers (CIKs) from pre-defined ID list file."""
        companies = []
        try:
            print(f"Reading company CIKs from: {source_file}")
            with open(source_file, 'r', encoding='utf-8') as f:
                ciks = json.load(f)
            
            if not isinstance(ciks, list):
                print(f"   Error: Expected a list of company CIKs")
                return []
            
            for cik in ciks:
                companies.append({
                    "cik": str(cik),
                    "identifier": str(cik)
                })
            
            print(f"Found {len(companies)} companies")
            
        except Exception as e:
            print(f"Error reading ID file: {e}")
        
        return companies
    
    def _prepare_analysis_command(self, identifier_info: Dict[str, Any], source_file: Path,
                                  subdir_name: str, **kwargs) -> tuple:
        """Prepare the command for company analysis."""
        cik = identifier_info["cik"]
        
        company_log_dir = self.base_log_dir / subdir_name
        company_log_dir.mkdir(parents=True, exist_ok=True)
        
        task = f"Analyze company with CIK {cik}"
        
        cmd = [
            sys.executable,
            "agent/data_agent.py",
            "--task", task,
            "--log-dir", str(company_log_dir)
        ]
        
        if not kwargs.get("auto_finish", True):
            cmd.append("--no-auto-finish")
        
        if kwargs.get("config_path"):
            cmd.extend(["--config", kwargs.get("config_path")])
        if kwargs.get("scenario"):
            cmd.extend(["--scenario", kwargs.get("scenario")])
        if kwargs.get("skill_dir"):
            cmd.extend(["--skill-dir", kwargs.get("skill_dir")])
        if kwargs.get("agent_mode"):
            cmd.extend(["--agent-mode", kwargs.get("agent_mode")])

        # Setup MCP arguments
        cmd.extend(["--sql-server", "tool_server/sqlite_mcp.py"])

        task_db_path = kwargs.get("split_db_path")
        if task_db_path:
            cmd.extend(["--data-path", task_db_path])
        
        # Pass max_turns if provided
        if kwargs.get("max_turns"):
            cmd.extend(["--max-turns", str(kwargs.get("max_turns"))])
        
        env = os.environ.copy()
        env['CUSTOM_LOG_DIR'] = str(company_log_dir)
        
        return cmd, env, f"Company CIK {cik}"
    
    def get_subdir_name(self, identifier: str) -> str:
        return f"company_{identifier}"
    
    def _create_identifier_from_logs(self, identifier: str, dirname: str) -> Optional[Dict[str, Any]]:
        return {
            "cik": identifier,
            "identifier": identifier
        }


class UserBatchAnalyzer(BaseBatchAnalyzer):
    """Batch analyzer for GLOBEM user data."""
    
    def __init__(self, base_log_dir: str, target_ids: Optional[Set[str]] = None, overwrite: bool = False):
        super().__init__(base_log_dir, target_ids, overwrite)

    def extract_identifiers(self, source_file: Path) -> List[Dict[str, Any]]:
        """Extract user identifiers from pre-defined ID list file."""
        users = []
        try:
            print(f"Reading user IDs from: {source_file}")
            with open(source_file, 'r', encoding='utf-8') as f:
                user_ids = json.load(f)
            
            if not isinstance(user_ids, list):
                print(f"   Error: Expected a list of user IDs")
                return []
            
            for pid in user_ids:
                users.append({
                    "pid": str(pid),
                    "identifier": str(pid)
                })
            
            print(f"Found {len(users)} users")
            
        except Exception as e:
            print(f"Error reading ID file: {e}")
        
        return users
    
    def _prepare_analysis_command(self, identifier_info: Dict[str, Any], source_file: Path,
                                  subdir_name: str, **kwargs) -> tuple:
        """Prepare the command for user analysis."""
        pid = identifier_info["pid"]
        
        user_log_dir = self.base_log_dir / subdir_name
        user_log_dir.mkdir(parents=True, exist_ok=True)
        
        task = f"Analyze user {pid}"
        
        cmd = [
            sys.executable,
            "agent/data_agent.py",
            "--task", task,
            "--log-dir", str(user_log_dir)
        ]
        
        if not kwargs.get("auto_finish", True):
            cmd.append("--no-auto-finish")
        
        if kwargs.get("config_path"):
            cmd.extend(["--config", kwargs.get("config_path")])
        if kwargs.get("scenario"):
            cmd.extend(["--scenario", kwargs.get("scenario")])
        if kwargs.get("skill_dir"):
            cmd.extend(["--skill-dir", kwargs.get("skill_dir")])
        if kwargs.get("agent_mode"):
            cmd.extend(["--agent-mode", kwargs.get("agent_mode")])

        # Setup MCP arguments
        cmd.extend(["--code-server", "tool_server/code_mcp.py"])

        task_data_path = kwargs.get("split_data_path")
        if task_data_path:
            cmd.extend(["--data-path", task_data_path])
        
        # Pass max_turns if provided
        if kwargs.get("max_turns"):
            cmd.extend(["--max-turns", str(kwargs.get("max_turns"))])
        
        env = os.environ.copy()
        env['CUSTOM_LOG_DIR'] = str(user_log_dir)
        
        return cmd, env, f"User {pid}"
    
    def get_subdir_name(self, identifier: str) -> str:
        return f"user_{identifier}"
    
    def _create_identifier_from_logs(self, identifier: str, dirname: str) -> Optional[Dict[str, Any]]:
        return {
            "pid": identifier,
            "identifier": identifier
        }


def main():
    """Main entry point for running the agent."""
    parser = argparse.ArgumentParser(
        description="DDR_Bench Unified Agent Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MIMIC patient analysis (configured via config.yaml)
  python run_agent.py --scenario mimic

  # Run 10-K company analysis
  python run_agent.py --scenario 10k

  # Run GLOBEM user analysis
  python run_agent.py --scenario globem
        """
    )
    
    # Required arguments
    parser.add_argument("--scenario", required=True, choices=["mimic", "10k", "globem"],
                        help="Analysis scenario to run")
    
    # Configuration file
    parser.add_argument("--config", help="Path to config.yaml file")
    
    # Execution options
    parser.add_argument("--target-ids", help="Comma-separated list of specific IDs to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--retry-only", action="store_true", help="Only retry failed analyses")
    parser.add_argument("--split", choices=["explore", "test"], help="Explore/test split to run")
    parser.add_argument("--log-dir", help="Override the output log directory (skips all auto-derived suffixes)")
    parser.add_argument("--skill-dir", help="Directory containing SKILL.md to inject as global prompt context")
    parser.add_argument(
        "--agent-mode",
        choices=["analysis", "checklist"],
        default="analysis",
        help="Agent operating mode: analysis uses insight summaries; checklist uses the checklist prompt and submission tool",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of identifiers to analyze in parallel (default: 1)")
    args = parser.parse_args()

    original_cwd = Path.cwd()
    if args.config:
        args.config = resolve_cli_path(args.config, original_cwd, prefer_existing=True)
    if args.log_dir:
        args.log_dir = resolve_cli_path(args.log_dir, original_cwd)
    if args.skill_dir:
        args.skill_dir = resolve_cli_path(args.skill_dir, original_cwd, prefer_existing=True)
    os.chdir(PROJECT_ROOT)

    # Resolve config path
    config_path = args.config
    if not config_path and Path("config.yaml").exists():
        config_path = str(Path("config.yaml").resolve())
    
    # Load configuration
    config = get_config(args.config) # get_config handles default loading too
    scenario_config = config.get_scenario(args.scenario)

    # Get settings from config
    log_dir, id_file, split_db_path = resolve_split_run_paths(
        scenario_config,
        args.split,
        args.agent_mode,
    )
    if args.log_dir:
        log_dir = str(Path(args.log_dir).expanduser().resolve())
    split_data_path = ""
    if args.split and args.scenario == "globem" and getattr(scenario_config, "data_path", ""):
        split_data_path = derive_split_data_dir(scenario_config.data_path, args.split)

    auto_finish = config.agent.auto_finish if hasattr(config.agent, 'auto_finish') else True
    max_retries = config.agent.max_retries or 2
    max_turns = config.agent.max_turns or 100
    log_level = config.agent.log_level or "INFO"
    
    # Set log level for subprocesses and current process
    os.environ["DDR_LOG_LEVEL"] = log_level
    
    # Process target IDs
    target_ids = None
    if args.target_ids:
        target_ids = set(id.strip() for id in args.target_ids.split(',') if id.strip())
        print(f"Target IDs: {sorted(target_ids)}")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    if args.skill_dir:
        skill_dir_path = Path(args.skill_dir).expanduser().resolve()
        skill_file_path = skill_dir_path / "SKILL.md"
        if not skill_file_path.exists() or not skill_file_path.is_file():
            parser.error(f"SKILL.md not found under --skill-dir: {skill_file_path}")
    
    # Validate id_file exists
    if not id_file or not Path(id_file).exists():
        parser.error(f"ID file not found: {id_file}. Please check config.yaml.")

    if args.scenario == "10k" and args.split and not split_db_path:
        parser.error(
            f"10-K split '{args.split}' requires a shared split DB. "
            f"Expected {derive_split_artifact_path(scenario_config.db_path, args.split)}"
        )

    # Validate scenario paths are configured (just valid check, not passed via args)
    if args.scenario == "mimic" and not scenario_config.db_path:
        parser.error("db_path for mimic not found in config.yaml")
    if args.scenario == "10k" and not scenario_config.db_path:
        parser.error("db_path for 10k not found in config.yaml")
    if args.scenario == "globem" and not scenario_config.data_path:
        parser.error("data_path for globem not found in config.yaml")
    if args.scenario == "mimic" and args.split and not split_db_path:
        parser.error(
            f"MIMIC split '{args.split}' requires a split DB. "
            f"Expected {derive_split_artifact_path(scenario_config.db_path, args.split)}"
        )
    if args.scenario == "globem" and args.split:
        if not split_data_path or not Path(split_data_path).exists():
            parser.error(
                f"GLOBEM split '{args.split}' requires a split data directory. "
                f"Expected {derive_split_data_dir(scenario_config.data_path, args.split)}"
            )

    # Create analyzer based on scenario
    if args.scenario == "mimic":
        analyzer = PatientBatchAnalyzer(log_dir, target_ids, args.overwrite)
    elif args.scenario == "10k":
        analyzer = CompanyBatchAnalyzer(log_dir, target_ids, args.overwrite)
    elif args.scenario == "globem":
        analyzer = UserBatchAnalyzer(log_dir, target_ids, args.overwrite)
    
    source_file = Path(id_file)
    run_kwargs = {
        "max_turns": max_turns, 
        "auto_finish": auto_finish
    }
    
    # Run analysis
    print(f"\n{'='*60}")
    print(f"DDR_Bench Agent Runner")
    print(f"Scenario: {args.scenario}")
    if args.split:
        print(f"Split: {args.split}")
    if split_db_path:
        print(f"Split Database: {split_db_path}")
    if split_data_path:
        print(f"Split Data Path: {split_data_path}")
    print(f"Provider: {config.provider.default_provider}")
    print(f"Model: {config.provider.default_model}")
    print(f"Log Directory: {log_dir}")
    print(f"Agent Mode: {args.agent_mode}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Config File: {config_path}")
    print(f"{'='*60}\n")
    
    # Add config_path and scenario to run_kwargs
    if config_path:
        run_kwargs["config_path"] = config_path
    run_kwargs["scenario"] = args.scenario
    run_kwargs["agent_mode"] = args.agent_mode
    run_kwargs["max_workers"] = args.max_workers
    if args.skill_dir:
        run_kwargs["skill_dir"] = str(Path(args.skill_dir).expanduser().resolve())
    if split_db_path:
        run_kwargs["split_db_path"] = split_db_path
    if split_data_path:
        run_kwargs["split_data_path"] = split_data_path

    if args.retry_only:
        analyzer.retry_failed_analyses(max_retries=max_retries, **run_kwargs)
    else:
        analyzer.run_batch_analysis(source_file, max_retries=max_retries, **run_kwargs)

    if args.agent_mode == "checklist":
        qa_output = str(Path("data") / args.scenario / f"{Path(log_dir).name}.json")
        convert_qa_submissions_to_qa(
            logs_dir=log_dir,
            output=qa_output,
            id_file=id_file,
            entity_prefix=get_entity_prefix_for_scenario(scenario_config, args.scenario),
            split=args.split,
        )


if __name__ == "__main__":
    main()
