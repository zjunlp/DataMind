import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config_manager import ConfigManager
from .skill_generator import SkillGeneratorAgent, get_prompt, save_skill_artifacts
from .trajectory_ops import copy_trajectories_by_score, copy_trajectories_top_bottom
from .utils import (
    extract_avg_score,
    format_metric_line,
    build_eval_filename,
    find_skill_dirs_in,
)

logger = logging.getLogger("ddr_pipeline")

MAX_ROUNDS = 3
MAX_SUB_ITERATIONS = 10
TOP_N_TRAJECTORIES = 5


@dataclass
class PipelineState:
    """State persisted to JSON so interrupted runs can resume."""

    completed_steps: list = field(default_factory=list)

    round0_analysis_logs: str = ""
    round0_checklist_logs: str = ""
    baseline_eval_file: str = ""
    baseline_score: float = 0.0

    analysis_origin_skill_dir: str = ""
    checklist_origin_skill_dir: str = ""

    analysis_skill_dirs: dict = field(default_factory=dict)
    checklist_skill_dirs: dict = field(default_factory=dict)
    analysis_logs_map: dict = field(default_factory=dict)
    checklist_logs_map: dict = field(default_factory=dict)
    analysis_eval_map: dict = field(default_factory=dict)
    checklist_eval_map: dict = field(default_factory=dict)
    analysis_score_map: dict = field(default_factory=dict)
    checklist_score_map: dict = field(default_factory=dict)

    last_valid_analysis: dict = field(default_factory=dict)
    last_valid_checklist: dict = field(default_factory=dict)

    qa_files: dict = field(default_factory=dict)
    completed: bool = False
    exit_reason: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        data = json.loads(path.read_text(encoding="utf-8"))
        s = cls()
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s

    def done(self, step_id: str) -> None:
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)

    def is_done(self, step_id: str) -> bool:
        return step_id in self.completed_steps


def _load_dotenv(env_file: Path) -> dict:
    """Load a simple KEY=VALUE .env file into a subprocess environment."""
    env = os.environ.copy()
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val:
                env[key] = val
    return env


def _run_cmd(cmd: list[str], cwd: Path, label: str) -> tuple[int, str]:
    logger.info(f"[CMD] {label}")
    logger.debug(f"  cwd={cwd}")
    logger.debug(f"  {' '.join(cmd)}")
    env = _load_dotenv(cwd / ".env")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc.stdin.write("y\ny\n")
    proc.stdin.close()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _read_stream(stream, lines, tag):
        for line in stream:
            line = line.rstrip("\n")
            lines.append(line)
            logger.debug(f"  [{tag}] {line}")

    t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines, "stdout"), daemon=True)
    t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines, "stderr"), daemon=True)
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()

    stdout_text = "\n".join(stdout_lines)
    if proc.returncode != 0:
        stderr_tail = "\n".join(stderr_lines[-20:])
        logger.error(f"Command failed (rc={proc.returncode}): {label}")
        logger.error(f"  stderr: {stderr_tail}")
    return proc.returncode, stdout_text


def _check_agent_success(stdout: str) -> bool:
    m = re.search(r"Success rate:\s*([\d.]+)%", stdout)
    if m:
        if float(m.group(1)) == 0.0:
            return False

    success = re.search(r"Success:\s*(\d+)", stdout)
    failed = re.search(r"Failed:\s*(\d+)", stdout)
    if success and failed and int(success.group(1)) == 0 and int(failed.group(1)) > 0:
        return False

    return True


def _verify_logs_complete(log_dir: str) -> None:
    """Remove entity log directories that lack session statistics before retrying."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    removed = []
    for entity_dir in sorted(log_path.iterdir()):
        if not entity_dir.is_dir():
            continue
        has_stats = list(entity_dir.glob("session_stats_*.json"))
        if not has_stats:
            shutil.rmtree(entity_dir)
            removed.append(entity_dir.name)
    if removed:
        logger.warning(f"Removed {len(removed)} incomplete entity dirs from {log_path.name}: {removed[:5]}...")


def _run_agent(
    cfg: ConfigManager,
    split: str = "explore",
    agent_mode: str = "analysis",
    skill_dir: str | None = None,
    log_dir: str | None = None,
) -> None:
    if agent_mode == "checklist":
        cfg.update_max_turns(100)
    else:
        cfg.update_max_turns(50)

    if log_dir:
        _verify_logs_complete(log_dir)

    cmd = [
        sys.executable, "run_agent.py",
        "--scenario", cfg.scenario,
        "--max-workers", str(cfg.max_workers),
        "--split", split,
        "--config", str(cfg.config_path),
        "--agent-mode", agent_mode,
    ]
    if skill_dir:
        cmd.extend(["--skill-dir", skill_dir])
    if log_dir:
        cmd.extend(["--log-dir", log_dir])

    label = f"run_agent scenario={cfg.scenario} mode={agent_mode} split={split}"
    if skill_dir:
        sp = Path(skill_dir)
        label += f" skill={sp.parent.name}/{sp.name}"
    rc, stdout = _run_cmd(cmd, cfg.ddr_bench_dir, label)
    if rc != 0:
        raise RuntimeError(f"run_agent failed: {label}")
    if not _check_agent_success(stdout):
        raise RuntimeError(f"run_agent all entities failed: {label}")

    if agent_mode == "checklist" and log_dir:
        # DDR_Bench writes QA output next to the scenario data; keep generated
        # iteration artifacts in a run-specific subdirectory before evaluation.
        log_name = Path(log_dir).name
        default_qa_rel = f"data/{cfg.scenario}/{log_name}.json"
        default_qa_abs = cfg.ddr_bench_dir / default_qa_rel
        target_qa_rel = cfg.qa_file_path(log_name)
        target_qa_abs = cfg.qa_file_abs(log_name)
        if default_qa_abs.exists():
            target_qa_abs.parent.mkdir(parents=True, exist_ok=True)
            if target_qa_abs.exists():
                target_qa_abs.unlink()
            shutil.move(str(default_qa_abs), str(target_qa_abs))
            logger.info(f"Relocated qa file: {default_qa_rel} -> {target_qa_rel}")
        elif not target_qa_abs.exists():
            raise RuntimeError(f"qa file missing after run_agent checklist mode: {default_qa_abs} / {target_qa_abs}")
        current = cfg.get_qa_file_from_config()
        if current != target_qa_rel:
            cfg.update_qa_file(target_qa_rel)
            logger.info(f"Updated qa_file in config: {current} -> {target_qa_rel}")


def _run_evaluation(cfg: ConfigManager, logs_dir: str, output_file: str) -> str:
    logs_path = Path(logs_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "run_evaluation.py",
        "--scenario", cfg.scenario,
        "--max-workers", str(cfg.max_workers),
        "--logs", str(logs_path),
        "--output", str(out_path),
        "--config", str(cfg.config_path),
    ]
    label = f"run_evaluation logs={logs_path.name} qa_file={cfg.get_qa_file_from_config()}"
    rc, _ = _run_cmd(cmd, cfg.ddr_bench_dir, label)
    if not out_path.exists():
        raise RuntimeError(f"run_evaluation failed (no output file): {label}")
    if rc != 0:
        logger.warning(f"run_evaluation exited with rc={rc} but output file exists, continuing")

    logger.info(f"Evaluation: {format_metric_line(out_path)}")
    return str(out_path)


class Pipeline:
    """Coordinate DDR_Bench runs, evaluations, and skill refinement rounds."""

    def __init__(self, cfg: ConfigManager, resume: bool = False):
        self.cfg = cfg
        self.state_file = cfg.records_dir / "pipeline_state.json"
        self.logs_root = cfg.ddr_bench_dir / f"logs-{cfg.run_name}-iteration"

        if resume and self.state_file.exists():
            self.st = PipelineState.load(self.state_file)
            logger.info(f"Resumed pipeline (completed steps: {len(self.st.completed_steps)})")
        else:
            self.st = PipelineState()

    def _save(self) -> None:
        self.st.save(self.state_file)

    def _verify_skill_created(self) -> None:
        skills_dir = self.cfg.workspace / ".claude" / "skills"
        found = [
            p for p in skills_dir.iterdir()
            if p.is_dir() and p.name != "skill-creator" and (p / "SKILL.md").exists()
        ] if skills_dir.exists() else []
        if not found:
            raise RuntimeError(
                "Skill generation completed but no skill was written to "
                f"{skills_dir}. Check skill generator logs for permission errors."
            )
        logger.info(f"Skill verified: {[p.name for p in found]}")

    def _log_dir(self, name: str) -> str:
        return str(self.logs_root / name)

    def _eval_path(self, name: str) -> str:
        return str(self.cfg.eval_results_dir / name)

    def _build_log_name(self, skill_name: str, split: str, agent_mode: str = "analysis") -> str:
        parts = [self.cfg.scenario, skill_name, "skill"]
        if agent_mode == "checklist":
            parts.append("checklist")
        parts.append(split)
        return "_".join(parts)

    def _checklist_log_name(self, base_name: str, split: str) -> str:
        return "_".join([self.cfg.scenario, base_name, split])

    def _find_skill_md(self, skill_dir: Path) -> str:
        dirs = find_skill_dirs_in(skill_dir)
        if dirs:
            return str(dirs[0])
        if (skill_dir / "SKILL.md").exists():
            return str(skill_dir)
        for sub in sorted(skill_dir.rglob("SKILL.md")):
            return str(sub.parent)
        return str(skill_dir)

    def _analysis_iteration_paths(self, rnd: int, iteration: int) -> tuple[str, str, str]:
        folder = f"analysis-iteration-{rnd}-{iteration}"
        log_name = self._build_log_name(f"analysis_{rnd}_{iteration}", "explore")
        log_dir = self.st.analysis_logs_map.get(str(rnd), {}).get(str(iteration)) or self._log_dir(log_name)
        return folder, log_name, log_dir

    def _checklist_iteration_paths(self, rnd: int, iteration: int) -> tuple[str, str, str, str]:
        folder = f"checklist-iteration-{rnd}-{iteration}"
        log_name = self._build_log_name(f"checklist_{rnd}_{iteration}", "explore", agent_mode="checklist")
        log_dir = self.st.checklist_logs_map.get(str(rnd), {}).get(str(iteration)) or self._log_dir(log_name)
        qa_file = self.cfg.qa_file_path(Path(log_dir).name)
        return folder, log_name, log_dir, qa_file

    def _load_checklist_iter0(
        self,
        rk: str,
        fallback_skill_dir: str,
        fallback_log_name: str,
        fallback_log_dir: str,
        fallback_qa_file: str,
    ) -> tuple[str, str, str, str, str, float]:
        """Restore checklist iteration 0 details from state when resuming."""
        record = self.st.last_valid_checklist.get(rk)
        if record:
            skill_dir = record.get("skill_dir", fallback_skill_dir)
            log_dir = record.get("logs", fallback_log_dir)
            log_name = Path(log_dir).name
            qa_file = record.get("qa_file") or self.cfg.qa_file_path(log_name)
            eval_file = record.get("eval_file", "")
            score = record.get("score", 0.0)
            return skill_dir, log_name, log_dir, qa_file, eval_file, score

        log_dir = self.st.checklist_logs_map.get(rk, {}).get("0", fallback_log_dir)
        log_name = Path(log_dir).name if log_dir else fallback_log_name
        qa_file = self.st.qa_files.get(rk, fallback_qa_file)
        eval_file = self.st.checklist_eval_map.get(rk, {}).get("0", "")
        score = self.st.checklist_score_map.get(rk, {}).get("0", 0.0)
        return fallback_skill_dir, log_name, log_dir, qa_file, eval_file, score

    def _record_valid_analysis(
        self,
        rk: str,
        iteration: int,
        score: float,
        eval_file: str,
        logs: str,
        skill_dir: str,
        folder: str,
    ) -> None:
        self.st.last_valid_analysis[rk] = {
            "iter": iteration,
            "score": score,
            "eval_file": eval_file,
            "logs": logs,
            "skill_dir": skill_dir,
            "skill_folder": folder,
        }

    def _record_valid_checklist(
        self,
        rk: str,
        iteration: int,
        score: float,
        eval_file: str,
        logs: str,
        skill_dir: str,
        folder: str,
        qa_file: str,
    ) -> None:
        self.st.last_valid_checklist[rk] = {
            "iter": iteration,
            "score": score,
            "eval_file": eval_file,
            "logs": logs,
            "skill_dir": skill_dir,
            "skill_folder": folder,
            "qa_file": qa_file,
        }
        self.st.qa_files[rk] = qa_file

    def run(self) -> None:
        if self.st.completed:
            logger.info(f"Pipeline already completed: {self.st.exit_reason}")
            return

        self._round_zero()
        if self.st.completed:
            return

        for rnd in range(1, MAX_ROUNDS + 1):
            self._round_n(rnd)
            if self.st.completed:
                return

        self._exit(f"All {MAX_ROUNDS} rounds completed.")

    def _exit(self, reason: str) -> None:
        self._export_final_analysis_skill()
        self.st.completed = True
        self.st.exit_reason = reason
        self._save()
        logger.info(f"[Pipeline] Exit: {reason}")

    def _export_final_analysis_skill(self) -> None:
        """Copy the last valid analysis skill to a stable final folder."""
        if not self.st.last_valid_analysis:
            logger.warning("[Pipeline] No valid analysis skill to export.")
            return

        def _round_key(value: str) -> int:
            try:
                return int(value)
            except ValueError:
                return -1

        key = max(self.st.last_valid_analysis, key=_round_key)
        record = self.st.last_valid_analysis.get(key) or {}
        src = Path(record.get("skill_dir", ""))
        skill_dirs = find_skill_dirs_in(src)
        if not skill_dirs and (src / "SKILL.md").exists():
            skill_dirs = [src]
        if not skill_dirs:
            logger.warning(f"[Pipeline] No SKILL.md found for final analysis skill: {src}")
            return

        final_dir = self.cfg.records_dir / "analysis-final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(skill_dirs[0], final_dir)
        logger.info(f"[Pipeline] Exported final analysis skill: {skill_dirs[0]} -> {final_dir}")

    def _round_zero(self) -> None:
        """Run the baseline and seed the first analysis skill."""
        logger.info("=" * 60)
        logger.info("[Round 0] Starting baseline runs")
        logger.info("=" * 60)

        sid = "r0_analysis"
        if not self.st.is_done(sid):
            logger.info("[Round 0] Step 1/12: Running analysis agent (explore)...")
            log_name = f"{self.cfg.scenario}_explore"
            log_dir = self._log_dir(log_name)
            _run_agent(self.cfg, split="explore", agent_mode="analysis", log_dir=log_dir)
            self.st.round0_analysis_logs = log_dir
            self.st.done(sid)
            self._save()
            logger.info(f"[Round 0] Step 1/12: Done. logs={log_name}")

        sid = "r0_checklist"
        if not self.st.is_done(sid):
            logger.info("[Round 0] Step 2/12: Running checklist agent (explore)...")
            log_name = self._checklist_log_name("checklist", "explore")
            log_dir = self._log_dir(log_name)
            _run_agent(self.cfg, split="explore", agent_mode="checklist", log_dir=log_dir)
            self.st.round0_checklist_logs = log_dir
            self.st.done(sid)
            self._save()
            logger.info(f"[Round 0] Step 2/12: Done. logs={log_name}")

        expected_qa = self.cfg.qa_file_path(Path(self.st.round0_checklist_logs).name)
        current_qa = self.cfg.get_qa_file_from_config()
        if current_qa != expected_qa:
            self.cfg.update_qa_file(expected_qa)
            logger.info(f"[Round 0] Updated qa_file: {current_qa} -> {expected_qa}")

        sid = "r0_eval"
        if not self.st.is_done(sid):
            logger.info("[Round 0] Step 3/12: Evaluating baseline...")
            a_name = Path(self.st.round0_analysis_logs).name
            checklist_name = Path(self.st.round0_checklist_logs).name
            eval_fname = build_eval_filename(a_name, checklist_name, self.cfg.scenario, self.cfg.eval_model)
            eval_file = _run_evaluation(self.cfg, self.st.round0_analysis_logs, self._eval_path(eval_fname))
            self.st.baseline_eval_file = eval_file
            self.st.baseline_score = extract_avg_score(Path(eval_file))
            self.st.done(sid)
            self._save()
            logger.info(f"[Round 0] Step 3/12: Baseline score={self.st.baseline_score:.4f}")

        sid = "r0_copy_fwd"
        if not self.st.is_done(sid):
            logger.info("[Round 0] Step 4/12: Copying trajectories (forward)...")
            copy_trajectories_by_score(
                eval_file=Path(self.st.baseline_eval_file),
                logs_dir=Path(self.st.round0_analysis_logs),
                direction="forward",
                output_dir=self.cfg.workspace / "trajectory",
                scenario=self.cfg.scenario,
            )
            self.st.done(sid)
            self._save()

        sid = "r0_skill_analysis"
        if not self.st.is_done(sid):
            logger.info("[Round 0] Step 5-6/12: Generating & saving analysis origin skill...")
            prompt = get_prompt("origin", self.cfg.workspace)
            SkillGeneratorAgent(workspace=self.cfg.workspace).run(prompt)
            self._verify_skill_created()
            save_skill_artifacts(self.cfg.workspace, self.cfg.records_dir, "analysis-origin")
            self.st.analysis_origin_skill_dir = str(self.cfg.records_dir / "analysis-origin")
            self.st.done(sid)
            self._save()

        sid = "r0_copy_rev"
        if not self.st.is_done(sid):
            logger.info("[Round 0] Step 7/12: Copying trajectories (reverse)...")
            copy_trajectories_by_score(
                eval_file=Path(self.st.baseline_eval_file),
                logs_dir=Path(self.st.round0_checklist_logs),
                direction="reverse",
                output_dir=self.cfg.workspace / "trajectory",
                scenario=self.cfg.scenario,
            )
            self.st.done(sid)
            self._save()

        sid = "r0_skill_checklist"
        if not self.st.is_done(sid):
            logger.info("[Round 0] Step 8-9/12: Generating & saving checklist origin skill...")
            prompt = get_prompt("origin", self.cfg.workspace)
            SkillGeneratorAgent(workspace=self.cfg.workspace).run(prompt)
            self._verify_skill_created()
            save_skill_artifacts(self.cfg.workspace, self.cfg.records_dir, "checklist-origin")
            self.st.checklist_origin_skill_dir = str(self.cfg.records_dir / "checklist-origin")
            self.st.done(sid)
            self._save()

        sid = "r0_origin_analysis_run"
        origin_a_skill_name = "analysis_origin"
        origin_a_log_name = self._build_log_name(origin_a_skill_name, "explore")
        origin_a_log_dir = self._log_dir(origin_a_log_name)
        if not self.st.is_done(sid):
            logger.info("[Round 0] Step 10/12: Running analysis agent with origin skill...")
            skill_md = self._find_skill_md(Path(self.st.analysis_origin_skill_dir))
            _run_agent(self.cfg, split="explore", agent_mode="analysis", skill_dir=skill_md, log_dir=origin_a_log_dir)
            self.st.done(sid)
            self._save()

        sid = "r0_origin_eval"
        if not self.st.is_done(sid):
            logger.info("[Round 0] Step 11/12: Evaluating origin analysis with baseline checklist...")
            baseline_checklist_name = Path(self.st.round0_checklist_logs).name
            eval_fname = build_eval_filename(origin_a_log_name, baseline_checklist_name, self.cfg.scenario, self.cfg.eval_model)
            origin_eval = _run_evaluation(self.cfg, origin_a_log_dir, self._eval_path(eval_fname))
            origin_score = extract_avg_score(Path(origin_eval))

            logger.info(f"[Round 0] Origin analysis score={origin_score:.4f} vs baseline={self.st.baseline_score:.4f}")
            if origin_score <= self.st.baseline_score:
                self._exit(f"Round 0 origin analysis ({origin_score:.4f}) <= baseline ({self.st.baseline_score:.4f})")
                return

            self.st.last_valid_analysis["0"] = {
                "iter": 0, "score": origin_score, "eval_file": origin_eval,
                "logs": origin_a_log_dir, "skill_dir": self.st.analysis_origin_skill_dir,
                "skill_folder": "analysis-origin",
            }
            self.st.done(sid)
            self._save()
            logger.info(f"[Round 0] Step 12/12: Seeded last_valid_analysis['0']")

        logger.info("[Round 0] Complete.")

    def _round_n(self, rnd: int) -> None:
        """Run one analysis/checklist refinement round."""
        logger.info("=" * 60)
        logger.info(f"[Round {rnd}] Starting")
        logger.info("=" * 60)
        rk = str(rnd)
        prev_rk = str(rnd - 1)

        prev_a = self.st.last_valid_analysis[prev_rk]
        prev_cl = self.st.last_valid_checklist.get(prev_rk)

        if prev_cl:
            checklist_ref_logs = prev_cl["logs"]
            iter0_eval_file = prev_cl["eval_file"]
            iter0_score = prev_cl["score"]
        else:
            checklist_ref_logs = self.st.round0_checklist_logs
            iter0_eval_file = prev_a["eval_file"]
            iter0_score = prev_a["score"]
        checklist_ref_name = Path(checklist_ref_logs).name

        origin_log_dir = prev_a["logs"]
        origin_skill_dir = prev_a["skill_dir"]
        self.st.analysis_logs_map.setdefault(rk, {})["0"] = origin_log_dir
        self.st.analysis_eval_map.setdefault(rk, {})["0"] = iter0_eval_file
        self.st.analysis_score_map.setdefault(rk, {})["0"] = iter0_score
        self._save()
        logger.info(f"[Round {rnd}] Reusing Round {rnd-1} best analysis: score={iter0_score:.4f}")

        prev_eval_file = iter0_eval_file
        prev_score = iter0_score

        for x in range(1, MAX_SUB_ITERATIONS + 1):
            sid_run = f"r{rnd}_analysis_{x}_run"
            sid_eval = f"r{rnd}_analysis_{x}_eval"
            if self.st.is_done(sid_eval):
                stored_score = self.st.analysis_score_map.get(rk, {}).get(str(x))
                if stored_score is not None and stored_score > prev_score:
                    folder, _, iter_log_dir = self._analysis_iteration_paths(rnd, x)
                    iter_eval = self.st.analysis_eval_map.get(rk, {}).get(str(x), prev_eval_file)
                    skill_dir = self.st.analysis_skill_dirs.get(rk, {}).get(
                        str(x),
                        str(self.cfg.records_dir / folder),
                    )
                    self._record_valid_analysis(rk, x, stored_score, iter_eval, iter_log_dir, skill_dir, folder)
                    self._save()
                    prev_score = stored_score
                    prev_eval_file = iter_eval
                else:
                    break
                continue

            logger.info(f"[Round {rnd}] Analysis iteration {x}")

            if self.st.is_done(sid_run):
                folder, iter_log_name, iter_log_dir = self._analysis_iteration_paths(rnd, x)
                logger.info(f"[Round {rnd}] Analysis iter {x}: skill+run already done, resuming at eval")
            else:
                prev_analysis_logs = self._get_prev_logs(self.st.analysis_logs_map, rk, x)
                prev_skill = self._get_prev_skill(self.st.analysis_skill_dirs, self.st.last_valid_analysis, rk, x, origin_skill_dir)
                self._do_copy_top_bottom(prev_eval_file, "insight", Path(prev_analysis_logs), Path(checklist_ref_logs), prev_skill)

                prompt = get_prompt("analysis_iteration", self.cfg.workspace)
                SkillGeneratorAgent(workspace=self.cfg.workspace).run(prompt)
                self._verify_skill_created()
                folder = f"analysis-iteration-{rnd}-{x}"
                save_skill_artifacts(self.cfg.workspace, self.cfg.records_dir, folder)
                self.st.analysis_skill_dirs.setdefault(rk, {})[str(x)] = str(self.cfg.records_dir / folder)
                self._save()

                iter_skill_md = self._find_skill_md(self.cfg.records_dir / folder)
                iter_log_name = self._build_log_name(f"analysis_{rnd}_{x}", "explore")
                iter_log_dir = self._log_dir(iter_log_name)
                _run_agent(self.cfg, split="explore", agent_mode="analysis", skill_dir=iter_skill_md, log_dir=iter_log_dir)
                self.st.analysis_logs_map.setdefault(rk, {})[str(x)] = iter_log_dir
                self.st.done(sid_run)
                self._save()

            eval_fname = build_eval_filename(iter_log_name, checklist_ref_name, self.cfg.scenario, self.cfg.eval_model)
            iter_eval = _run_evaluation(self.cfg, iter_log_dir, self._eval_path(eval_fname))
            iter_score = extract_avg_score(Path(iter_eval))
            self.st.analysis_eval_map.setdefault(rk, {})[str(x)] = iter_eval
            self.st.analysis_score_map.setdefault(rk, {})[str(x)] = iter_score
            self.st.done(sid_eval)
            self._save()

            logger.info(f"[Round {rnd}] Analysis iter {x}: {iter_score:.4f} vs {prev_score:.4f}")
            if iter_score > prev_score:
                logger.info(f"[Round {rnd}] Analysis iter {x}: valid")
                self._record_valid_analysis(
                    rk,
                    x,
                    iter_score,
                    iter_eval,
                    iter_log_dir,
                    str(self.cfg.records_dir / folder),
                    folder,
                )
                self._save()
                prev_score = iter_score
                prev_eval_file = iter_eval
            else:
                logger.info(f"[Round {rnd}] Analysis iter {x}: no improvement")
                break

        valid_a = self.st.last_valid_analysis.get(rk)
        if not valid_a:
            if rnd > 1:
                self._exit(f"Round {rnd}: no valid analysis iterations")
                return
            self.st.last_valid_analysis[rk] = prev_a.copy()
            valid_a = self.st.last_valid_analysis[rk]
            self._save()
        final_a_score = valid_a["score"]
        final_a_logs = valid_a["logs"]

        if prev_cl:
            cl_origin_skill_dir = prev_cl["skill_dir"]
            cl_origin_log_dir = prev_cl["logs"]
            cl_origin_log_name = Path(cl_origin_log_dir).name
            cl_origin_qa_file = prev_cl["qa_file"]
            checklist_eval_file = valid_a["eval_file"]
            checklist_eval_score = valid_a["score"]
            self.st.checklist_logs_map.setdefault(rk, {})["0"] = cl_origin_log_dir
            self.st.checklist_eval_map.setdefault(rk, {})["0"] = checklist_eval_file
            self.st.checklist_score_map.setdefault(rk, {})["0"] = checklist_eval_score
            logger.info(
                f"[Round {rnd}] Reusing Round {rnd-1} best checklist: {cl_origin_log_name}, "
                f"score={checklist_eval_score:.4f}"
            )
        else:
            cl_origin_skill_dir = self.st.checklist_origin_skill_dir
            cl_origin_log_name = self._build_log_name("checklist_origin", "explore", agent_mode="checklist")
            cl_origin_log_dir = self._log_dir(cl_origin_log_name)
            cl_origin_qa_file = self.cfg.qa_file_path(cl_origin_log_name)
            retry_folder = f"checklist-origin-retry-{rnd}"
            retry_log_name = self._build_log_name(
                f"checklist_origin_retry_{rnd}", "explore", agent_mode="checklist",
            )
            retry_log_dir = self._log_dir(retry_log_name)
            retry_qa_file = self.cfg.qa_file_path(retry_log_name)

            sid = f"r{rnd}_checklist_run"
            if not self.st.is_done(sid):
                logger.info(f"[Round {rnd}] Running checklist agent with origin checklist skill...")
                checklist_md = self._find_skill_md(Path(cl_origin_skill_dir))
                _run_agent(
                    self.cfg,
                    split="explore",
                    agent_mode="checklist",
                    skill_dir=checklist_md,
                    log_dir=cl_origin_log_dir,
                )
                self.st.done(sid)
                self._save()
            self.st.checklist_logs_map.setdefault(rk, {})["0"] = cl_origin_log_dir

            current_qa = self.cfg.get_qa_file_from_config()
            if current_qa != cl_origin_qa_file:
                self.cfg.update_qa_file(cl_origin_qa_file)
                logger.info(f"[Round {rnd}] Updated qa_file: {current_qa} -> {cl_origin_qa_file}")

            sid = f"r{rnd}_checklist_eval"
            if not self.st.is_done(sid):
                a_name = Path(final_a_logs).name
                eval_fname = build_eval_filename(a_name, cl_origin_log_name, self.cfg.scenario, self.cfg.eval_model)
                checklist_eval_file = _run_evaluation(self.cfg, final_a_logs, self._eval_path(eval_fname))
                checklist_eval_score = extract_avg_score(Path(checklist_eval_file))
                self.st.checklist_eval_map.setdefault(rk, {})["0"] = checklist_eval_file
                self.st.checklist_score_map.setdefault(rk, {})["0"] = checklist_eval_score

                logger.info(f"[Round {rnd}] Origin checklist eval={checklist_eval_score:.4f} vs analysis={final_a_score:.4f}")
                if checklist_eval_score >= final_a_score:
                    logger.info(
                        f"[Round {rnd}] Origin checklist invalid, regenerating checklist skill "
                        f"from best analysis trajectories..."
                    )

                    sid_retry = f"r{rnd}_checklist_retry_run"
                    if not self.st.is_done(sid_retry):
                        copy_trajectories_by_score(
                            eval_file=Path(valid_a["eval_file"]),
                            logs_dir=Path(self.st.round0_checklist_logs),
                            direction="reverse",
                            output_dir=self.cfg.workspace / "trajectory",
                            scenario=self.cfg.scenario,
                        )
                        prompt = get_prompt("origin", self.cfg.workspace)
                        SkillGeneratorAgent(workspace=self.cfg.workspace).run(prompt)
                        self._verify_skill_created()
                        save_skill_artifacts(
                            self.cfg.workspace, self.cfg.records_dir, retry_folder,
                        )
                        retry_skill_md = self._find_skill_md(
                            self.cfg.records_dir / retry_folder,
                        )
                        _run_agent(
                            self.cfg, split="explore", agent_mode="checklist",
                            skill_dir=retry_skill_md, log_dir=retry_log_dir,
                        )
                        self.st.done(sid_retry)
                        self._save()

                    sid_retry_eval = f"r{rnd}_checklist_retry_eval"
                    if not self.st.is_done(sid_retry_eval):
                        _cur_qa = self.cfg.get_qa_file_from_config()
                        if _cur_qa != retry_qa_file:
                            self.cfg.update_qa_file(retry_qa_file)
                        retry_eval_fname = build_eval_filename(
                            Path(final_a_logs).name, retry_log_name,
                            self.cfg.scenario, self.cfg.eval_model,
                        )
                        checklist_eval_file = _run_evaluation(
                            self.cfg, final_a_logs,
                            self._eval_path(retry_eval_fname),
                        )
                        checklist_eval_score = extract_avg_score(Path(checklist_eval_file))
                        logger.info(
                            f"[Round {rnd}] Retry checklist eval={checklist_eval_score:.4f} "
                            f"vs analysis={final_a_score:.4f}"
                        )
                        if checklist_eval_score >= final_a_score:
                            self._exit(
                                f"Round {rnd} retry checklist ({checklist_eval_score:.4f}) "
                                f">= analysis ({final_a_score:.4f})"
                            )
                            return
                        self.st.checklist_eval_map.setdefault(rk, {})["0"] = checklist_eval_file
                        self.st.checklist_score_map.setdefault(rk, {})["0"] = checklist_eval_score
                        self.st.checklist_logs_map.setdefault(rk, {})["0"] = retry_log_dir
                        self.st.done(sid_retry_eval)
                        self._save()
                    else:
                        checklist_eval_file = self.st.checklist_eval_map.get(rk, {}).get("0", "")
                        checklist_eval_score = self.st.checklist_score_map.get(rk, {}).get("0", 0.0)

                    cl_origin_skill_dir = str(self.cfg.records_dir / retry_folder)
                    cl_origin_log_name = retry_log_name
                    cl_origin_log_dir = retry_log_dir
                    cl_origin_qa_file = retry_qa_file

                self.st.done(sid)
                self._save()
            else:
                if self.st.is_done(f"r{rnd}_checklist_retry_eval"):
                    cl_origin_skill_dir = str(self.cfg.records_dir / retry_folder)
                    cl_origin_log_name = retry_log_name
                    cl_origin_log_dir = retry_log_dir
                    cl_origin_qa_file = retry_qa_file
                    self.st.checklist_logs_map.setdefault(rk, {})["0"] = cl_origin_log_dir

                (
                    cl_origin_skill_dir,
                    cl_origin_log_name,
                    cl_origin_log_dir,
                    cl_origin_qa_file,
                    checklist_eval_file,
                    checklist_eval_score,
                ) = self._load_checklist_iter0(
                    rk,
                    cl_origin_skill_dir,
                    cl_origin_log_name,
                    cl_origin_log_dir,
                    cl_origin_qa_file,
                )

        current_qa = self.cfg.get_qa_file_from_config()
        if current_qa != cl_origin_qa_file:
            self.cfg.update_qa_file(cl_origin_qa_file)

        if rk not in self.st.last_valid_checklist:
            self.st.last_valid_checklist[rk] = {
                "iter": 0, "score": checklist_eval_score, "eval_file": checklist_eval_file,
                "logs": cl_origin_log_dir, "skill_dir": cl_origin_skill_dir,
                "skill_folder": Path(cl_origin_skill_dir).name, "qa_file": cl_origin_qa_file,
            }
            self.st.qa_files[rk] = cl_origin_qa_file
            self._save()

        prev_cl_eval = checklist_eval_file
        prev_cl_score = checklist_eval_score

        for x in range(1, MAX_SUB_ITERATIONS + 1):
            sid_run = f"r{rnd}_checklist_{x}_run"
            sid_eval = f"r{rnd}_checklist_{x}_eval"
            if self.st.is_done(sid_eval):
                stored = self.st.checklist_score_map.get(rk, {}).get(str(x))
                if stored is not None and stored < prev_cl_score:
                    folder, _, cl_log_dir, qa_file = self._checklist_iteration_paths(rnd, x)
                    cl_eval = self.st.checklist_eval_map.get(rk, {}).get(str(x), prev_cl_eval)
                    skill_dir = self.st.checklist_skill_dirs.get(rk, {}).get(
                        str(x),
                        str(self.cfg.records_dir / folder),
                    )
                    self._record_valid_checklist(rk, x, stored, cl_eval, cl_log_dir, skill_dir, folder, qa_file)
                    self._save()
                    prev_cl_score = stored
                    prev_cl_eval = cl_eval
                else:
                    break
                continue

            logger.info(f"[Round {rnd}] Checklist iteration {x}")

            if self.st.is_done(sid_run):
                folder, cl_log_name, cl_log_dir, _ = self._checklist_iteration_paths(rnd, x)
                logger.info(f"[Round {rnd}] Checklist iter {x}: skill+run already done, resuming at eval")
            else:
                prev_cl_logs = self._get_prev_logs(self.st.checklist_logs_map, rk, x)
                prev_cl_skill = self._get_prev_skill(self.st.checklist_skill_dirs, self.st.last_valid_checklist, rk, x, cl_origin_skill_dir)
                self._do_copy_top_bottom(prev_cl_eval, "checklist", Path(final_a_logs), Path(prev_cl_logs), prev_cl_skill)

                prompt = get_prompt("checklist_iteration", self.cfg.workspace)
                SkillGeneratorAgent(workspace=self.cfg.workspace).run(prompt)
                self._verify_skill_created()
                folder = f"checklist-iteration-{rnd}-{x}"
                save_skill_artifacts(self.cfg.workspace, self.cfg.records_dir, folder)
                self.st.checklist_skill_dirs.setdefault(rk, {})[str(x)] = str(self.cfg.records_dir / folder)
                self._save()

                cl_skill_md = self._find_skill_md(self.cfg.records_dir / folder)
                cl_log_name = self._build_log_name(f"checklist_{rnd}_{x}", "explore", agent_mode="checklist")
                cl_log_dir = self._log_dir(cl_log_name)
                _run_agent(self.cfg, split="explore", agent_mode="checklist", skill_dir=cl_skill_md, log_dir=cl_log_dir)
                self.st.checklist_logs_map.setdefault(rk, {})[str(x)] = cl_log_dir
                self.st.done(sid_run)
                self._save()

            expected_qa = self.cfg.qa_file_path(Path(cl_log_dir).name)
            current_qa = self.cfg.get_qa_file_from_config()
            if current_qa != expected_qa:
                self.cfg.update_qa_file(expected_qa)
                logger.info(f"[Round {rnd}] Checklist iter {x}: Updated qa_file -> {expected_qa}")
            self._save()

            a_name = Path(final_a_logs).name
            checklist_name = Path(cl_log_dir).name
            eval_fname = build_eval_filename(a_name, checklist_name, self.cfg.scenario, self.cfg.eval_model)
            cl_eval = _run_evaluation(self.cfg, final_a_logs, self._eval_path(eval_fname))
            cl_score = extract_avg_score(Path(cl_eval))
            self.st.checklist_eval_map.setdefault(rk, {})[str(x)] = cl_eval
            self.st.checklist_score_map.setdefault(rk, {})[str(x)] = cl_score
            self.st.done(sid_eval)
            self._save()

            logger.info(f"[Round {rnd}] Checklist iter {x}: {cl_score:.4f} vs {prev_cl_score:.4f}")
            if cl_score < prev_cl_score:
                logger.info(f"[Round {rnd}] Checklist iter {x}: valid (lower is better)")
                qa_file_now = self.cfg.get_qa_file_from_config()
                self._record_valid_checklist(
                    rk,
                    x,
                    cl_score,
                    cl_eval,
                    cl_log_dir,
                    str(self.cfg.records_dir / folder),
                    folder,
                    qa_file_now,
                )
                self._save()
                prev_cl_score = cl_score
                prev_cl_eval = cl_eval
            else:
                logger.info(f"[Round {rnd}] Checklist iter {x}: no improvement")
                break

        best_cl = self.st.last_valid_checklist[rk]
        self.cfg.update_qa_file(best_cl["qa_file"])

        if rnd == MAX_ROUNDS:
            self._exit(f"Completed all {MAX_ROUNDS} rounds.")
            return

        logger.info(f"[Round {rnd}] Complete. Proceeding to round {rnd + 1}.")

    def _get_prev_logs(self, logs_map: dict, rk: str, x: int) -> str:
        m = logs_map.get(rk, {})
        return m.get(str(x - 1), m.get("0", ""))

    def _get_prev_skill(self, skill_map: dict, valid_map: dict, rk: str, x: int, fallback: str) -> str:
        if x > 1:
            prev = skill_map.get(rk, {}).get(str(x - 1))
            if prev:
                return prev
        v = valid_map.get(rk)
        if v and v.get("skill_dir"):
            return v["skill_dir"]
        prev_v = valid_map.get(str(int(rk) - 1))
        if prev_v and prev_v.get("skill_dir"):
            return prev_v["skill_dir"]
        return fallback

    def _do_copy_top_bottom(
        self, eval_file: str, mode: str, insight_logs: Path, checklist_logs: Path, skill_src: str
    ):
        logger.info(f"Preparing trajectories: eval_file={Path(eval_file).name} mode={mode}")
        copy_trajectories_top_bottom(
            eval_file=Path(eval_file),
            mode=mode,
            insight_logs=insight_logs,
            checklist_logs=checklist_logs,
            output_dir=self.cfg.workspace / "trajectory",
            scenario=self.cfg.scenario,
            skill_source_dir=Path(skill_src) if skill_src else None,
            skill_dest_dir=self.cfg.workspace / ".claude" / "skills",
            n=TOP_N_TRAJECTORIES,
        )
