#!/usr/bin/env python3
"""Prepare an ALREADY-DOWNLOADED LongDS dataset for agent self-evaluation.

This does NOT download anything (the full dataset is ~19.5 GB; download it
yourself first — see README.md). It splits each task into:

  <out>/manifest/<key>.json   <- the AGENT sees this: turn_id / context / question / data_dir (NO answers)
  <out>/gold/<key>.json        <- JUDGE only (held out): question + ground_truth
  <out>/answers/               <- the agent WRITES its answers here, one <key>.json per task
  <out>/index.json             <- task list with turn counts and data_dir_exists

LongDS task.json carries the reference `answer` (ground truth) and `code` (gold
solution) inline. With --strip-source, the answer-bearing files (task.json,
task.py, task.ipynb) are deleted from the download tree after extraction, so the
only copy of the answers lives in <out>/gold (judge-only). The raw `data/` files
the agent analyzes are always kept.

Usage:
  python prepare_dataset.py --dataset-root <hf_download_dir> --out-dir <work_dir> \
      [--start-index N] [--task-limit N] [--turn-limit N] [--domain D ...] [--strip-source]
"""
import argparse, json, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Split a downloaded LongDS dataset into agent manifest + held-out gold.")
    ap.add_argument("--dataset-root", required=True,
                    help="Where you ran `hf download zjunlp/LongDS --local-dir ...`; must contain task/longds and data/longds")
    ap.add_argument("--out-dir", required=True, help="Output workspace (manifest/, gold/, answers/, index.json)")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--task-limit", type=int, default=None)
    ap.add_argument("--turn-limit", type=int, default=None)
    ap.add_argument("--domain", action="append", default=None, help="Only include these domains (repeatable)")
    ap.add_argument("--strip-source", action="store_true",
                    help="After extraction, delete answer-bearing files (task.json/task.py/task.ipynb) from the download tree.")
    args = ap.parse_args()

    root = Path(args.dataset_root).resolve()
    task_root = root / "task" / "longds"
    data_root = root / "data" / "longds"
    task_list_path = task_root / "task_list.json"
    if not task_list_path.is_file():
        print(f"ERROR: {task_list_path} not found. Download the dataset first:\n"
              f"  hf download zjunlp/LongDS --repo-type dataset --local-dir {root}", file=sys.stderr)
        return 1

    task_list = json.loads(task_list_path.read_text(encoding="utf-8"))
    if args.domain:
        task_list = [t for t in task_list if t["task_domain"] in set(args.domain)]
    task_list = task_list[args.start_index:]
    if args.task_limit is not None:
        task_list = task_list[: args.task_limit]

    out = Path(args.out_dir).resolve()
    for sub in ("manifest", "gold", "answers"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    index, total_turns, stripped = [], 0, 0
    for ti in task_list:
        domain, ds, tid = ti["task_domain"], ti["dataset_name"], ti["task_id"]
        key = f"{domain}__{ds}__{tid}"
        task_dir = task_root / domain / ds / tid
        tjson = task_dir / "task.json"
        if not tjson.is_file():
            print(f"WARN: missing {tjson}, skipping {key}", file=sys.stderr)
            continue
        turns = json.loads(tjson.read_text(encoding="utf-8"))
        if args.turn_limit is not None:
            turns = turns[: args.turn_limit]

        data_dir = data_root / domain / ds / tid / "data"
        (out / "manifest" / f"{key}.json").write_text(json.dumps({
            "key": key, "domain": domain, "dataset": ds, "task_id": tid, "data_dir": str(data_dir),
            "turns": [{"turn_id": t["turn_id"], "context": t["context"], "question": t["question"]} for t in turns],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "gold" / f"{key}.json").write_text(json.dumps({
            "key": key, "domain": domain, "dataset": ds, "task_id": tid,
            "turns": [{"turn_id": t["turn_id"], "question": f"{t['context']}\nQuestion: {t['question']}",
                       "ground_truth": t.get("answer")} for t in turns],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append({"key": key, "domain": domain, "dataset": ds, "task_id": tid,
                      "turns": len(turns), "data_dir": str(data_dir), "data_dir_exists": data_dir.is_dir()})
        total_turns += len(turns)

        if args.strip_source:
            for name in ("task.json", "task.py", "task.ipynb"):
                f = task_dir / name
                if f.is_file():
                    f.unlink(); stripped += 1

    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(index)} tasks, {total_turns} turns -> {out}")
    if args.strip_source:
        print(f"Stripped {stripped} answer-bearing files from {task_root} (gold kept in {out/'gold'}).")
    missing = [i["key"] for i in index if not i["data_dir_exists"]]
    if missing:
        print(f"WARN: {len(missing)} tasks have no data_dir (partial download?): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)
    print(f"Agent reads:  {out/'manifest'}  (+ each task's data_dir)\n"
          f"Judge reads:  {out/'gold'}  (held out)\n"
          f"Agent writes: {out/'answers'}/<key>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
