import json
import logging
import sys
from pathlib import Path


def setup_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """Configure file and console logging for a pipeline run."""
    logger = logging.getLogger("ddr_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt_detail = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_detail)
    fh.stream.reconfigure(line_buffering=True)
    logger.addHandler(fh)

    info_file = log_file.with_name("pipeline_info.log")
    fh_info = logging.FileHandler(info_file, encoding="utf-8")
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(fmt_detail)
    fh_info.stream.reconfigure(line_buffering=True)
    logger.addHandler(fh_info)

    fmt_brief = logging.Formatter("[%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt_brief)
    logger.addHandler(ch)

    return logger


SCENARIO_ENTITY_PREFIX = {
    "mimic": "patient",
    "10k": "company",
    "globem": "user",
}

SCENARIO_LOG_PREFIXES = {
    "mimic": "mimic_",
    "globem": "globem_",
}


def entity_dir_name(entity_prefix: str, eid: str) -> str:
    if entity_prefix == "user" and str(eid).startswith("INS-W_"):
        return f"user_{eid}"
    return f"{entity_prefix}_{eid}"


def compute_score(entity_result: dict) -> float:
    s = entity_result["summary"]
    return (
        float(s["message_wise_context"]["correct_percentage"])
        + float(s["chat_wise_context"]["correct_percentage"])
    ) / 2


def _safe_stat(context: dict, key: str) -> float:
    return float(context.get(key, 0.0)) if context.get("total_questions", 0) > 0 else 0.0


def extract_avg_score(eval_file: Path) -> float:
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    stats = data["overall_statistics"]
    msg = stats["message_wise_context"]
    chat = stats["chat_wise_context"]
    values = [
        _safe_stat(msg, "average_correct_percentage"),
        _safe_stat(chat, "average_correct_percentage"),
        _safe_stat(msg, "overall_correct_percentage"),
        _safe_stat(chat, "overall_correct_percentage"),
    ]
    return sum(values) / len(values)


def format_metric_line(eval_file: Path) -> str:
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    stats = data["overall_statistics"]
    msg = stats["message_wise_context"]
    chat = stats["chat_wise_context"]
    metric_rows = [
        ("sample_message", _safe_stat(msg, "average_correct_percentage")),
        ("sample_trajectory", _safe_stat(chat, "average_correct_percentage")),
        ("item_message", _safe_stat(msg, "overall_correct_percentage")),
        ("item_trajectory", _safe_stat(chat, "overall_correct_percentage")),
    ]
    avg = sum(value for _, value in metric_rows) / len(metric_rows)
    metric_rows.append(("average", avg))
    return " ".join(f"{label}={value:.2f}" for label, value in metric_rows)


def eval_output_suffix(eval_model: str) -> str:
    if eval_model == "deepseek-v4-flash":
        return "dpsk_high"
    return ""


def build_eval_filename(
    analysis_logs_name: str,
    checklist_logs_name: str,
    scenario: str,
    eval_model: str,
) -> str:
    prefix = SCENARIO_LOG_PREFIXES.get(scenario, "")
    stripped = checklist_logs_name
    if prefix and stripped.startswith(prefix):
        stripped = stripped[len(prefix):]

    suffix = eval_output_suffix(eval_model)
    model_part = f"_{suffix}" if suffix else ""
    return f"{analysis_logs_name}_{stripped}{model_part}_evaluation_result.json"


def workspace_cache_dir_name(workspace_path: Path) -> str:
    resolved = str(workspace_path.resolve())
    return resolved.replace("/", "-").replace("_", "-")


def find_skill_dirs_in(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    result = [p for p in sorted(folder.iterdir()) if p.is_dir() and (p / "SKILL.md").exists()]
    if not result:
        result = [p.parent for p in sorted(folder.rglob("SKILL.md")) if p.is_file()]
    return result
