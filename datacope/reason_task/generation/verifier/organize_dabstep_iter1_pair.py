import argparse
import json
import glob
import os
import re
from collections import defaultdict

from .scorer import question_scorer


def extract_answer(data: dict) -> str | None:
    """Return prediction answer, or None if turns==15 with no <answer> tag."""
    turns = data.get("turns", 0)
    if turns == 15:
        last_assistant = ""
        for msg in reversed(data.get("conversation", [])):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break
        if "<answer>" not in last_assistant:
            return None
    return data.get("prediction")


_ERROR_PATTERN = re.compile(
    r"Traceback \(most recent call|"
    r"\b(?:TypeError|AttributeError|ValueError|KeyError|IndexError|"
    r"NameError|ImportError|RuntimeError|OSError|FileNotFoundError|"
    r"ZeroDivisionError|AssertionError|StopIteration|RecursionError|"
    r"MemoryError|OverflowError|SyntaxError|IndentationError|"
    r"UnboundLocalError|NotImplementedError)\s*:"
)


def has_errors(data: dict) -> bool:
    for msg in data.get("conversation", []):
        if msg.get("role") != "user":
            continue
        for block in re.findall(
            r"<information>(.*?)</information>",
            msg.get("content", ""),
            re.DOTALL,
        ):
            if _ERROR_PATTERN.search(block):
                return True
    return False


def best_file(pred_list: list[dict]) -> dict:
    """Pick representative: no-error first, then fewest turns, then path order."""
    no_err = [p for p in pred_list if not p["has_error"]]
    candidates = no_err if no_err else pred_list
    return min(candidates, key=lambda p: (p["turns"], p["file_path"]))


def group_answers(preds: list[dict]) -> list[list[dict]]:
    """Cluster predictions by equivalent answers; groups sorted by size desc."""
    groups: list[list[dict]] = []
    repr_answers: list[str | None] = []

    for pred in preds:
        answer = pred["answer"]
        placed = False
        for i, rep in enumerate(repr_answers):
            if answer is None and rep is None:
                groups[i].append(pred)
                placed = True
                break
            if (
                answer is not None
                and rep is not None
                and question_scorer(str(answer), str(rep))
            ):
                groups[i].append(pred)
                placed = True
                break
        if not placed:
            groups.append([pred])
            repr_answers.append(answer)

    groups.sort(key=lambda g: -len(g))
    return groups


def find_run_dirs(results_dir: str) -> list[str]:
    """Recursively find *_run_N dirs (N=1..10) under results_dir."""
    run_dirs = []
    for root, dirs, _ in os.walk(results_dir):
        for d in dirs:
            m = re.search(r"_run_(\d+)$", d)
            if m and 1 <= int(m.group(1)) <= 10:
                run_dirs.append(os.path.join(root, d))
    return sorted(run_dirs)


def load_predictions(results_dir: str, iter_label: str) -> dict[str, list[dict]]:
    run_dirs = find_run_dirs(results_dir)
    print(f"  [{iter_label}] found {len(run_dirs)} run dirs (run1-run10)")
    predictions_by_query: dict[str, list[dict]] = {}

    for run_dir in run_dirs:
        raw = os.path.basename(run_dir.rstrip("/"))
        parts = raw.split("_")
        run_name = "_".join([parts[0], parts[-2], parts[-1]])
        for pred_file in sorted(glob.glob(os.path.join(run_dir, "prediction_*.json"))):
            with open(pred_file, encoding="utf-8") as f:
                data = json.load(f)
            query_id = str(data["extra_info"]["query_id"])
            predictions_by_query.setdefault(query_id, []).append({
                "answer":    extract_answer(data),
                "has_error": has_errors(data),
                "turns":     data.get("turns", 0),
                "file_path": pred_file,
                "run_name":  run_name,
            })

    print(f"  [{iter_label}] collected predictions for "
          f"{len(predictions_by_query)} unique query IDs")
    return predictions_by_query


def write_iter_dir(
    preds: list[dict],
    groups: list[list[dict]],
    iter_dir: str,
    query_id: str,
    category: str,
    iter_n: int,
) -> None:
    """Write Answer_i sub-dirs and summary.json under iter_dir."""
    os.makedirs(iter_dir, exist_ok=True)
    total = len(preds)
    summary_answers = []

    for rank, group in enumerate(groups, start=1):
        answer_dir = os.path.join(iter_dir, f"Answer{rank}")
        os.makedirs(answer_dir, exist_ok=True)

        chosen = best_file(group)
        src = chosen["file_path"]
        stem, ext = os.path.splitext(os.path.basename(src))
        dst = os.path.join(answer_dir, f"{stem}_{chosen['run_name']}{ext}")

        with open(src, encoding="utf-8") as f:
            pred_data = json.load(f)
        pred_data.pop("ground_truth", None)
        pred_data.pop("extra_info", None)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(pred_data, f, indent=2, ensure_ascii=False)

        summary_answers.append({
            "answer_dir":       f"Answer{rank}",
            "answer":           chosen["answer"],
            "count":            len(group),
            "self_consistency": round(len(group) / total, 4),
        })

    summary = {
        "query_id":      query_id,
        "category":      category,
        "iter":          iter_n,
        "total_samples": total,
        "answers":       summary_answers,
    }
    with open(os.path.join(iter_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run(
    iter_prev: int,
    iter_curr: int,
    results_dir_prev: str,
    results_dir_curr: str,
    category_file: str,
    output_dir: str,
) -> None:
    with open(category_file) as f:
        cat_data = json.load(f)

    query_id_to_cat: dict[str, str] = {}
    for cat, items in cat_data.items():
        for item in items:
            query_id_to_cat[str(item["query_id"])] = cat

    print(f"Loaded {len(query_id_to_cat)} query-id -> category mappings, "
          f"{len(cat_data)} categories")

    print(f"\nLoading iter{iter_prev} predictions from:\n  {results_dir_prev}")
    prev_by_query = load_predictions(results_dir_prev, f"iter{iter_prev}")

    print(f"\nLoading iter{iter_curr} predictions from:\n  {results_dir_curr}")
    curr_by_query = load_predictions(results_dir_curr, f"iter{iter_curr}")

    all_query_ids = set(prev_by_query) & set(curr_by_query)
    print(f"\nQuery IDs present in both iters: {len(all_query_ids)}\n")

    os.makedirs(output_dir, exist_ok=True)

    skipped_unknown_cat = 0
    skipped_sc1_both = 0
    kept = 0

    for query_id in sorted(all_query_ids):
        category = query_id_to_cat.get(query_id)
        if category is None:
            skipped_unknown_cat += 1
            category = "Unknown"

        prev_preds = prev_by_query[query_id]
        curr_preds = curr_by_query[query_id]

        prev_groups = group_answers(prev_preds)
        curr_groups = group_answers(curr_preds)

        prev_sc = len(prev_groups[0]) / len(prev_preds)
        curr_sc = len(curr_groups[0]) / len(curr_preds)
        prev_answer = prev_groups[0][0]["answer"]
        curr_answer = curr_groups[0][0]["answer"]

        if prev_sc == 1.0 and curr_sc == 1.0 and prev_answer is not None and curr_answer is not None:
            skipped_sc1_both += 1
            continue

        kept += 1

        task_dir = os.path.join(output_dir, category, query_id)

        write_iter_dir(
            prev_preds, prev_groups,
            os.path.join(task_dir, f"iter{iter_prev}"),
            query_id, category, iter_prev,
        )

        write_iter_dir(
            curr_preds, curr_groups,
            os.path.join(task_dir, f"iter{iter_curr}"),
            query_id, category, iter_curr,
        )

    print(f"\nDone. Output written to: {output_dir}")
    print(f"  Kept:                                                        {kept}")
    print(f"  Skipped (SC==1.0 both iters, both answers not None):        {skipped_sc1_both}")
    if skipped_unknown_cat:
        print(f"  WARNING: {skipped_unknown_cat} query IDs not found in category file "
              f"-> placed in 'Unknown/'")

    cat_counts: dict[str, int] = defaultdict(int)
    for qid in all_query_ids:
        cat_counts[query_id_to_cat.get(qid, "Unknown")] += 1

    print("\nQuery IDs per category (present in both iters, before filtering):")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize DABStep paired iteration results"
    )
    parser.add_argument("--iter-prev", type=int, required=True, help="Previous iteration number")
    parser.add_argument("--iter-curr", type=int, required=True, help="Current iteration number")
    parser.add_argument("--results-dir-prev", required=True, help="Results dir for previous iteration")
    parser.add_argument("--results-dir-curr", required=True, help="Results dir for current iteration")
    parser.add_argument("--category-file", required=True, help="Path to all_query_category.json")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    args = parser.parse_args()
    run(
        args.iter_prev, args.iter_curr,
        args.results_dir_prev, args.results_dir_curr,
        args.category_file, args.output_dir,
    )


if __name__ == "__main__":
    main()
