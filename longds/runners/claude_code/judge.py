#!/usr/bin/env python3
"""LLM judge for Claude Code LongDS runs.

This script reads Claude Code runner outputs, scores each turn with the same
JUDGE_PROMPT used by the DSGym LongDS runner, and writes results_eval.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def load_judge_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "DSGym" / "scripts" / "prompt.py"
    spec = importlib.util.spec_from_file_location("longds_dsgym_prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load DSGym prompt from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.JUDGE_PROMPT


JUDGE_PROMPT = load_judge_prompt()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def format_for_prompt(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def combine_context_question(context: str, question: str) -> str:
    context = (context or "").strip()
    question = (question or "").strip()
    if context and question:
        return f"Context:\n{context}\n\nQuestion:\n{question}"
    return question or context


def infer_metadata(run_dir: Path) -> dict[str, str]:
    return {
        "task_domain": run_dir.parent.parent.parent.name if len(run_dir.parents) >= 4 else "",
        "dataset_name": run_dir.parent.parent.name if len(run_dir.parents) >= 3 else "",
        "task_id": run_dir.parent.name if len(run_dir.parents) >= 2 else "",
        "run_name": run_dir.name,
    }


def load_run_metadata(run_dir: Path) -> dict[str, str]:
    inferred = infer_metadata(run_dir)
    metadata_path = run_dir / "task_metadata.json"
    if not metadata_path.is_file():
        return inferred
    metadata = read_json(metadata_path)
    return {
        "task_domain": str(metadata.get("task_domain") or inferred["task_domain"]),
        "dataset_name": str(metadata.get("dataset_name") or inferred["dataset_name"]),
        "task_id": str(metadata.get("task_id") or inferred["task_id"]),
        "run_name": str(metadata.get("run_name") or inferred["run_name"]),
    }


def load_run_turns(run_dir: Path) -> list[dict[str, Any]]:
    result_path = run_dir / "results_with_ground_truth.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing {result_path}")

    metadata = load_run_metadata(run_dir)
    task_key = (
        f"{metadata['task_domain']}/"
        f"{metadata['dataset_name']}/"
        f"{metadata['task_id']}"
    )
    run_key = f"{task_key}/{metadata['run_name']}"

    turns: list[dict[str, Any]] = []
    for idx, source_turn in enumerate(read_json(result_path), start=1):
        context = source_turn.get("context", "")
        question = source_turn.get("question", "")
        solution = source_turn.get("solution", source_turn.get("answer", ""))
        turn = {
            **metadata,
            "task_key": task_key,
            "run_key": run_key,
            "run_dir": str(run_dir),
            "turn_id": source_turn.get("turn_id", idx),
            "context": context,
            "question": question,
            "judge_question": combine_context_question(context, question),
            "solution": solution,
            "ground_truth": source_turn.get("ground_truth"),
            "reasoning_summary": source_turn.get("reasoning_summary", ""),
            "files_used": source_turn.get("files_used", []),
            "success": source_turn.get("success"),
            "elapsed_seconds": source_turn.get("elapsed_seconds"),
            "turn_dir": source_turn.get("turn_dir"),
        }
        turns.append(turn)
    return turns


def discover_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted({path.parent for path in results_root.rglob("results_with_ground_truth.json")})


def parse_judge_response(text: str) -> dict[str, Any] | None:
    score_match = re.search(r"<score>\s*(\d)\s*</score>", text)
    if not score_match:
        return None

    score = int(score_match.group(1))
    if score not in (0, 1):
        return None

    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    error_match = re.search(r"<error>(.*?)</error>", text, re.DOTALL)
    return {
        "score": score,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
        "error_detail": error_match.group(1).strip() if error_match else "",
        "judge_response": text,
        "error": None,
    }


def judge_one(client: Any, judge_model: str, retries: int, turn: dict[str, Any]) -> dict[str, Any]:
    solution = format_for_prompt(turn.get("solution")).strip()
    ground_truth = turn.get("ground_truth")
    if ground_truth is None or ground_truth == "" or not solution:
        return {
            "score": 0,
            "reasoning": "",
            "error_detail": "Empty solution or ground truth",
            "judge_response": "",
            "error": None,
        }

    prompt = JUDGE_PROMPT.format(
        question=turn["judge_question"],
        ground_truth=format_for_prompt(ground_truth),
        solution=solution,
    )
    last_response = ""
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            last_response = response.choices[0].message.content or ""
            parsed = parse_judge_response(last_response)
            if parsed is not None:
                parsed["attempts"] = attempt
                return parsed
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                return {
                    "score": None,
                    "reasoning": "",
                    "error_detail": "",
                    "judge_response": last_response,
                    "error": str(exc),
                    "attempts": attempt,
                }

    return {
        "score": None,
        "reasoning": "",
        "error_detail": "",
        "judge_response": last_response,
        "error": f"Could not parse <score> after {retries} attempts",
        "attempts": retries,
    }


def average(scores: list[int | float]) -> float:
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def summarize(turns: list[dict[str, Any]]) -> dict[str, Any]:
    scored: list[int | float] = []
    by_domain: dict[str, list[int | float]] = {}
    by_task: dict[str, list[int | float]] = {}
    by_run: dict[str, list[int | float]] = {}

    for turn in turns:
        score = (turn.get("judge") or {}).get("score")
        if score is None:
            continue
        scored.append(score)
        by_domain.setdefault(turn["task_domain"], []).append(score)
        by_task.setdefault(turn["task_key"], []).append(score)
        by_run.setdefault(turn["run_key"], []).append(score)

    return {
        "overall_accuracy": average(scored),
        "turns_judged": len(scored),
        "turns_total": len(turns),
        "by_domain": {
            key: {"accuracy": average(values), "turns": len(values)}
            for key, values in sorted(by_domain.items())
        },
        "by_task": {
            key: {"accuracy": average(values), "turns": len(values)}
            for key, values in sorted(by_task.items())
        },
        "by_run": {
            key: {"accuracy": average(values), "turns": len(values)}
            for key, values in sorted(by_run.items())
        },
    }


def write_run_outputs(turns: list[dict[str, Any]]) -> list[Path]:
    written: list[Path] = []
    by_run_dir: dict[str, list[dict[str, Any]]] = {}
    for turn in turns:
        by_run_dir.setdefault(turn["run_dir"], []).append(turn)

    for run_dir_text, run_turns in sorted(by_run_dir.items()):
        run_dir = Path(run_dir_text)
        payload = {
            "task": {
                "task_domain": run_turns[0]["task_domain"],
                "dataset_name": run_turns[0]["dataset_name"],
                "task_id": run_turns[0]["task_id"],
                "run_name": run_turns[0]["run_name"],
                "run_dir": run_dir_text,
            },
            "turns": run_turns,
            "summary": summarize(run_turns),
        }
        path = run_dir / "results_eval.json"
        write_json(path, payload)
        written.append(path)
    return written


def load_existing_eval_turns(run_dir: Path) -> list[dict[str, Any]]:
    eval_path = run_dir / "results_eval.json"
    payload = read_json(eval_path)
    turns = payload.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"Invalid results_eval.json format: {eval_path}")

    metadata = load_run_metadata(run_dir)
    task_key = (
        f"{metadata['task_domain']}/"
        f"{metadata['dataset_name']}/"
        f"{metadata['task_id']}"
    )
    run_key = f"{task_key}/{metadata['run_name']}"
    normalized_turns = []
    for turn in turns:
        normalized = dict(turn)
        normalized.setdefault("task_domain", metadata["task_domain"])
        normalized.setdefault("dataset_name", metadata["dataset_name"])
        normalized.setdefault("task_id", metadata["task_id"])
        normalized.setdefault("run_name", metadata["run_name"])
        normalized.setdefault("task_key", task_key)
        normalized.setdefault("run_key", run_key)
        normalized.setdefault("run_dir", str(run_dir))
        normalized_turns.append(normalized)
    return normalized_turns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM judge for Claude Code LongDS runs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Evaluate one run directory containing results_with_ground_truth.json.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory to scan when --run-dir is not set. Default: results",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional aggregate output path. By default, all-runs mode only writes per-run results_eval.json.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", "deepseek-v4-pro"),
        help="Judge model name. Default: JUDGE_MODEL env or deepseek-v4-pro.",
    )
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY"))
    parser.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and report turns without calling the judge API or writing results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rejudge runs even when results_eval.json already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.run_dir is not None:
        run_dirs = [args.run_dir]
        out_path = args.out or args.run_dir / "results_eval.json"
    else:
        run_dirs = discover_run_dirs(args.results_root)
        out_path = args.out

    if not run_dirs:
        print("ERROR: no Claude Code run directories found.", file=sys.stderr)
        return 1

    existing_turns: list[dict[str, Any]] = []
    run_dirs_to_judge: list[Path] = []
    skipped_run_dirs: list[Path] = []
    for run_dir in run_dirs:
        eval_path = run_dir / "results_eval.json"
        if eval_path.is_file() and not args.overwrite:
            skipped_run_dirs.append(run_dir)
            existing_turns.extend(load_existing_eval_turns(run_dir))
        else:
            run_dirs_to_judge.append(run_dir)

    turns_to_judge: list[dict[str, Any]] = []
    for run_dir in run_dirs_to_judge:
        turns_to_judge.extend(load_run_turns(run_dir))

    if not turns_to_judge and not existing_turns:
        print("ERROR: no turns found in Claude Code results.", file=sys.stderr)
        return 1

    print(f"Run directories: {len(run_dirs)}")
    print(f"Runs skipped with existing results_eval.json: {len(skipped_run_dirs)}")
    print(f"Runs to judge: {len(run_dirs_to_judge)}")
    print(f"Existing judged turns: {len(existing_turns)}")
    print(f"Turns to judge: {len(turns_to_judge)}")
    print(f"Judge model: {args.judge_model}")
    print(f"Aggregate output: {out_path if out_path is not None else 'disabled'}")

    if args.dry_run:
        print("Dry run only; no judge API call and no files written.")
        return 0

    if not turns_to_judge:
        print("All discovered runs already have results_eval.json; nothing to judge.")
        if out_path is not None:
            payload = {"turns": existing_turns, "summary": summarize(existing_turns)}
            write_json(out_path, payload)
            print(f"aggregate_saved: {out_path}")
        return 0

    if not args.judge_api_key or not args.judge_base_url:
        print(
            "ERROR: set JUDGE_API_KEY and JUDGE_BASE_URL, or pass "
            "--judge-api-key and --judge-base-url.",
            file=sys.stderr,
        )
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: missing dependency: pip install openai", file=sys.stderr)
        return 1

    client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    judged_turns = [dict(turn) for turn in turns_to_judge]
    results: list[dict[str, Any] | None] = [None] * len(judged_turns)

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {
            pool.submit(judge_one, client, args.judge_model, args.retries, turn): idx
            for idx, turn in enumerate(judged_turns)
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            turn = judged_turns[idx]
            print(f"{turn['task_key']} turn {turn['turn_id']}: score={result.get('score')}")

    for turn, result in zip(judged_turns, results):
        turn["judge"] = result

    all_turns = existing_turns + judged_turns
    written = write_run_outputs(judged_turns)
    payload = {"turns": all_turns, "summary": summarize(all_turns)}
    if out_path is not None:
        write_json(out_path, payload)

    summary = payload["summary"]
    print("")
    print("LLM judge finished")
    print(f"overall_accuracy: {summary['overall_accuracy']:.4f}")
    print(f"turns_judged: {summary['turns_judged']}")
    print(f"turns_total: {summary['turns_total']}")
    if out_path is not None:
        print(f"aggregate_saved: {out_path}")
    else:
        print("aggregate_saved: disabled")
    print(f"run_files_saved: {len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
