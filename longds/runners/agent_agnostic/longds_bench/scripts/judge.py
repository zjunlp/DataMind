#!/usr/bin/env python3
"""LongDS LLM-as-judge scorer (faithful to DataMind longds/DSGym/examples).

Joins the agent's answers with the held-out gold by turn_id and scores each turn
0/1 with the shared LongDS JUDGE_PROMPT. Average over all judged
turns is the accuracy, reported overall, per domain, and per task.

Inputs:
  --answers DIR   answers/<key>.json = {"key","domain","dataset","task_id",
                                        "answers":[{"turn_id":N,"answer":"..."}]}
  --gold DIR      gold/<key>.json from prepare_dataset.py (held out from the agent)
Judge endpoint (OpenAI-compatible) from env JUDGE_API_KEY / JUDGE_BASE_URL or flags.

  python judge.py --answers <work>/answers --gold <work>/gold \
      --out <work>/results_eval.json [--judge-model NAME] [--max-workers 8]
"""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.judge_prompt import JUDGE_PROMPT


def build_turns(answers_dir: Path, gold_dir: Path):
    turns = []
    for gf in sorted(gold_dir.glob("*.json")):
        gold = json.loads(gf.read_text(encoding="utf-8"))
        af = answers_dir / gf.name
        ans_by_turn = {}
        if af.is_file():
            adoc = json.loads(af.read_text(encoding="utf-8"))
            ans_by_turn = {a["turn_id"]: a.get("answer", "") for a in adoc.get("answers", [])}
        for t in gold["turns"]:
            turns.append({
                "key": gold["key"], "domain": gold["domain"], "turn_id": t["turn_id"],
                "question": t["question"], "ground_truth": t["ground_truth"],
                "solution": ans_by_turn.get(t["turn_id"], ""),
            })
    return turns


def judge_one(client, judge_model, t):
    if t["ground_truth"] in (None, "") or not t["solution"]:
        return {"score": 0, "reasoning": "", "error_detail": "Empty solution or ground truth", "error": None}
    gt = json.dumps(t["ground_truth"], ensure_ascii=False) if isinstance(t["ground_truth"], (dict, list)) else str(t["ground_truth"])
    prompt = JUDGE_PROMPT.format(question=t["question"], ground_truth=gt, solution=t["solution"])
    try:
        for attempt in range(3):
            resp = client.chat.completions.create(
                model=judge_model, temperature=0.0,
                messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content
            m = re.search(r"<score>\s*(\d)\s*</score>", text)
            if not m:
                continue
            rm = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
            em = re.search(r"<error>(.*?)</error>", text, re.DOTALL)
            return {"score": int(m.group(1)),
                    "reasoning": rm.group(1).strip() if rm else "",
                    "error_detail": em.group(1).strip() if em else "", "error": None}
        return {"score": None, "error": "could not parse <score> after 3 tries"}
    except Exception as e:  # noqa: BLE001
        return {"score": None, "error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser(description="LongDS LLM-as-judge scorer.")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY"))
    ap.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"))
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    if not args.judge_api_key or not args.judge_base_url:
        print("ERROR: set JUDGE_API_KEY and JUDGE_BASE_URL (or pass --judge-api-key/--judge-base-url).",
              file=sys.stderr)
        return 1
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        return 1
    client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)

    turns = build_turns(Path(args.answers), Path(args.gold))
    if not turns:
        print("ERROR: no gold turns found.", file=sys.stderr)
        return 1

    results = [None] * len(turns)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(judge_one, client, args.judge_model, t): i for i, t in enumerate(turns)}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            print(f"  {turns[i]['key']} turn {turns[i]['turn_id']}: score={results[i]['score']}")

    by_domain, by_task = {}, {}
    scored = []
    for t, r in zip(turns, results):
        t["judge"] = r
        if r["score"] is not None:
            scored.append(r["score"])
            by_domain.setdefault(t["domain"], []).append(r["score"])
            by_task.setdefault(t["key"], []).append(r["score"])

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    summary = {
        "overall_accuracy": avg(scored),
        "turns_judged": len(scored), "turns_total": len(turns),
        "by_domain": {d: {"accuracy": avg(s), "turns": len(s)} for d, s in sorted(by_domain.items())},
        "by_task": {k: {"accuracy": avg(s), "turns": len(s)} for k, s in sorted(by_task.items())},
    }
    Path(args.out).write_text(
        json.dumps({"turns": turns, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== LongDS-Bench (self-run) ===")
    print(f"Overall accuracy: {summary['overall_accuracy']:.4f}  "
          f"({summary['turns_judged']}/{summary['turns_total']} turns judged)")
    for d, v in summary["by_domain"].items():
        print(f"  {d:<12} {v['accuracy']:.4f}  ({v['turns']} turns)")
    print(f"\nReference (paper, official DSGym runtime): best ~48.45 (Gemini-3.1-Pro), "
          f"GPT-5.4 43.50, Claude-4.6-Sonnet 41.56. Self-run numbers are indicative, not officially comparable.")
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
