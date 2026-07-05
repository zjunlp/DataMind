import csv
import json
import logging
import shutil
from pathlib import Path

from .utils import compute_score, entity_dir_name, SCENARIO_ENTITY_PREFIX

logger = logging.getLogger("ddr_pipeline")

DROP_PATTERNS = ("chat_messages_*", "sqlite_mcp_server_*", "session_stats_*", "message_stats_*")


def _clean(dst: Path):
    for pat in DROP_PATTERNS:
        for p in dst.glob(pat):
            shutil.rmtree(p) if p.is_dir() else p.unlink()


def _append_final_report(src: Path, dst_file: Path) -> int:
    chat_files = sorted(src.glob("chat_messages_*"))
    if not chat_files:
        return 0
    with chat_files[-1].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "FINISH" not in rows[-1].get("content", ""):
        return 0
    before, after = rows[-1]["content"].split("FINISH", 1)
    after = after.lstrip(": \n")
    with dst_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "assistant_message", "user_message", "insight"]
        )
        writer.writerow({
            "timestamp": rows[-1].get("timestamp", ""),
            "assistant_message": "",
            "user_message": before.strip(),
            "insight": after.strip(),
        })
    return 1


def copy_trajectories_by_score(
    eval_file: Path,
    logs_dir: Path,
    direction: str,
    output_dir: Path,
    scenario: str,
):
    entity_prefix = SCENARIO_ENTITY_PREFIX[scenario]
    pos = output_dir / "positive"
    neg = output_dir / "negative"

    data = json.loads(eval_file.read_text(encoding="utf-8"))
    rows = data["entity_results"]
    vals = [(str(r["entity_id"]), compute_score(r)) for r in rows]
    mean = sum(v for _, v in vals) / len(vals)

    for d in (pos, neg):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob(f"{entity_prefix}_*"):
            shutil.rmtree(old)

    pc = nc = eq = 0
    for eid, v in vals:
        if v == mean:
            eq += 1
            continue
        high = v > mean
        dst_dir = pos if (high == (direction == "forward")) else neg
        dirname = entity_dir_name(entity_prefix, eid)
        src = logs_dir / dirname
        if src.exists():
            shutil.copytree(src, dst_dir / dirname)
        pc += dst_dir == pos
        nc += dst_dir == neg

    logger.info(
        f"copy_by_score: direction={direction} mean={mean:.4f} positive={pc} negative={nc} equal={eq}"
    )


def _copy_refs(src: Path, dst: Path, pattern: str, final_report: bool = False):
    if not src.exists():
        logger.warning(f"Missing reference folder: {src}")
        return 0
    files = sorted(src.glob(pattern))
    if not files:
        logger.warning(f"No reference files matching {pattern} in {src}")
        return 0
    copied = 0
    for p in files:
        name = f"reference_{p.name}"
        if final_report and p.name.startswith("insights_"):
            name = f"reference_insights_and_final_reports_{p.name[len('insights_'):]}"
        out = dst / name
        shutil.copy2(p, out)
        if final_report:
            _append_final_report(src, out)
        copied += 1
    return copied


def copy_trajectories_top_bottom(
    eval_file: Path,
    mode: str,
    insight_logs: Path,
    checklist_logs: Path,
    output_dir: Path,
    scenario: str,
    skill_source_dir: Path | None,
    skill_dest_dir: Path,
    n: int = 5,
):
    entity_prefix = SCENARIO_ENTITY_PREFIX[scenario]
    pos = output_dir / "positive"
    neg = output_dir / "negative"

    data = json.loads(eval_file.read_text(encoding="utf-8"))
    rows = data["entity_results"]
    vals = sorted(
        [(str(r["entity_id"]), compute_score(r)) for r in rows],
        key=lambda x: (x[1], x[0]),
    )
    worst = [eid for eid, _ in vals[:n]]
    best = [eid for eid, _ in sorted(vals, key=lambda x: (-x[1], x[0]))[:n]]

    for d in (pos, neg):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob(f"{entity_prefix}_*"):
            shutil.rmtree(old)

    if mode == "insight":
        main_logs, ref_logs, ref_pattern = insight_logs, checklist_logs, "qa_submissions_*"
        groups = ((pos, best), (neg, worst))
        final_report = False
    else:
        main_logs, ref_logs, ref_pattern = checklist_logs, insight_logs, "insights_*"
        groups = ((neg, best), (pos, worst))
        final_report = True

    ref_count = 0
    for dst_root, eids in groups:
        for eid in eids:
            dirname = entity_dir_name(entity_prefix, eid)
            src = main_logs / dirname
            dst = dst_root / dirname
            if not src.exists():
                logger.warning(f"Missing source folder: {src}")
                continue
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            _clean(dst)
            ref_count += _copy_refs(ref_logs / dirname, dst, ref_pattern, final_report)

    if skill_source_dir and skill_source_dir.exists():
        _copy_skills_to_workspace(skill_source_dir, skill_dest_dir)

    logger.info(
        f"copy_top_bottom: mode={mode} best={best} worst={worst} refs={ref_count}"
    )


def _copy_skills_to_workspace(src_dir: Path, dest_dir: Path):
    from .utils import find_skill_dirs_in

    skill_dirs = find_skill_dirs_in(src_dir)
    if not skill_dirs:
        logger.warning(f"No skills found in {src_dir}")
        return 0

    copied = 0
    for sd in skill_dirs:
        if sd.name == "skill-creator":
            continue
        dst = dest_dir / sd.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.copytree(sd, dst)
        copied += 1
    logger.info(f"Copied {copied} skills from {src_dir} to {dest_dir}")
    return copied
