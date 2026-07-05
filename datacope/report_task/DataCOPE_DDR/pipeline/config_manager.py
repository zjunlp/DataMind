import re
import shutil
from pathlib import Path


WORKSPACE_DATA_MAP = {
    "mimic": "DDRBench_mimic/mimic_iv_explore.db",
    "10k": "DDRBench_10K/raw/10k_financial_data_explore.db",
    "globem": "DDRBench_globem_explore",
}


class ConfigManager:
    """Manage generated DDR_Bench config files and pipeline workspace paths."""

    def __init__(
        self,
        scenario: str,
        model: str,
        eval_model: str,
        max_workers: int,
        ddr_bench_dir: Path,
        ddr_data_dir: Path,
        project_root: Path,
        provider: str = "openai",
        eval_provider: str = "",
    ):
        self.scenario = scenario
        self.model = model
        self.eval_model = eval_model
        self.max_workers = max_workers
        self.ddr_bench_dir = ddr_bench_dir
        self.ddr_data_dir = ddr_data_dir
        self.project_root = project_root
        self.provider = provider
        self.eval_provider = eval_provider
        self.run_name = f"{scenario}-{model}"

        self.workspace = project_root / "workspace"
        self.records_dir = project_root / "records" / self.run_name
        self.config_path = ddr_bench_dir / f"config_{self.run_name}_iteration.yaml"
        self.eval_results_dir = ddr_bench_dir / "evaluate" / f"results-{self.run_name}-iteration"
        self.log_base = f"./logs-{self.run_name}-iteration/{scenario}"
        self.qa_dir_rel = f"data/{scenario}/{self.run_name}-iteration"
        self.qa_dir_abs = ddr_bench_dir / self.qa_dir_rel

    def setup(self, resume: bool = False) -> None:
        """Create directories and the run-specific DDR_Bench config."""
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)
        self.qa_dir_abs.mkdir(parents=True, exist_ok=True)
        (self.workspace / "trajectory").mkdir(parents=True, exist_ok=True)
        (self.workspace / "data").mkdir(parents=True, exist_ok=True)
        if not resume or not self.config_path.exists():
            self._create_config()
        self._setup_workspace_data()

    def qa_file_path(self, log_name: str) -> str:
        """Return the DDR_Bench-relative QA file path for an iteration log name."""
        return f"{self.qa_dir_rel}/{log_name}.json"

    def qa_file_abs(self, log_name: str) -> Path:
        return self.qa_dir_abs / f"{log_name}.json"

    def _create_config(self) -> None:
        src = self.ddr_bench_dir / "config.yaml"
        if not src.exists():
            raise FileNotFoundError(f"DDR_Bench config not found: {src}")

        text = src.read_text(encoding="utf-8")

        text = re.sub(
            r'(default_provider:\s*)"[^"]*"',
            rf'\1"{self.provider}"',
            text,
        )
        text = re.sub(
            r'(default_model:\s*)"[^"]*"',
            rf'\1"{self.model}"',
            text,
        )
        text = re.sub(
            r'(evaluation:\s*\n(?:\s+\w+:.*\n)*?\s+model:\s*)"[^"]*"',
            rf'\1"{self.eval_model}"',
            text,
        )
        if self.eval_provider:
            text = re.sub(
                r'(evaluation:\s*\n(?:\s+\w+:.*\n)*?\s+provider:\s*)"[^"]*"',
                rf'\1"{self.eval_provider}"',
                text,
            )
        text = re.sub(
            r'(max_turns:\s*)\d+',
            r'\g<1>50',
            text,
            count=1,
        )

        lines = text.splitlines(keepends=True)
        in_scenario = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == f"{self.scenario}:":
                in_scenario = True
                continue
            if in_scenario and line.startswith("    ") and stripped.startswith("log_dir:"):
                indent = line[: len(line) - len(line.lstrip())]
                nl = "\n" if line.endswith("\n") else ""
                lines[i] = f'{indent}log_dir: "{self.log_base}"{nl}'
                break
        text = "".join(lines)

        self.config_path.write_text(text, encoding="utf-8")

    def _setup_workspace_data(self) -> None:
        rel = WORKSPACE_DATA_MAP.get(self.scenario)
        if not rel:
            return
        src = self.ddr_data_dir / rel
        if not src.exists():
            return
        dst = self.workspace / "data" / src.name
        if dst.exists():
            return
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def update_max_turns(self, turns: int) -> None:
        text = self.config_path.read_text(encoding="utf-8")
        text = re.sub(r'(max_turns:\s*)\d+', rf'\g<1>{turns}', text, count=1)
        self.config_path.write_text(text, encoding="utf-8")

    def update_qa_file(self, qa_file_path: str) -> None:
        """Update scenarios.<scenario>.qa_file while preserving inline comments."""
        lines = self.config_path.read_text(encoding="utf-8").splitlines(keepends=True)
        in_scenarios = False
        in_target = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "scenarios:":
                in_scenarios = True
                in_target = False
                continue
            if in_scenarios and line and not line.startswith(" "):
                break
            if in_scenarios and line.startswith("  ") and not line.startswith("    "):
                in_target = stripped == f"{self.scenario}:"
                continue
            if in_target and line.startswith("    ") and stripped.startswith("qa_file:"):
                indent = line[: len(line) - len(line.lstrip())]
                nl = "\n" if line.endswith("\n") else ""
                body = line[:-1] if nl else line
                comment = ""
                if "#" in body:
                    comment = "  #" + body.split("#", 1)[1]
                lines[i] = f'{indent}qa_file: "{qa_file_path}"{comment}{nl}'
                self.config_path.write_text("".join(lines), encoding="utf-8")
                return
        raise RuntimeError(f"Could not find scenarios.{self.scenario}.qa_file in {self.config_path}")

    def get_qa_file_from_config(self) -> str:
        """Return scenarios.<scenario>.qa_file from the generated config."""
        lines = self.config_path.read_text(encoding="utf-8").splitlines()
        in_scenarios = False
        in_target = False
        for line in lines:
            stripped = line.strip()
            if stripped == "scenarios:":
                in_scenarios = True
                in_target = False
                continue
            if in_scenarios and line and not line.startswith(" "):
                break
            if in_scenarios and line.startswith("  ") and not line.startswith("    "):
                in_target = stripped == f"{self.scenario}:"
                continue
            if in_target and line.startswith("    ") and stripped.startswith("qa_file:"):
                val = stripped.split(":", 1)[1].strip()
                if "#" in val:
                    val = val.split("#", 1)[0].strip()
                return val.strip('"').strip("'")
        return ""
