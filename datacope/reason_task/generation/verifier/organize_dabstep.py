import argparse
import json
import glob
import os
import re
from collections import defaultdict

from .scorer import question_scorer


def extract_answer(data: dict) -> str | None:
    """
    Return the prediction answer string, or None when:
      - turns == 15  AND
      - the last assistant message contains no <answer> tag
    """
    turns = data.get("turns", 0)
    if turns == 15:
        conv = data.get("conversation", [])
        last_assistant_content = ""
        for msg in reversed(conv):
            if msg.get("role") == "assistant":
                last_assistant_content = msg.get("content", "")
                break
        if "<answer>" not in last_assistant_content:
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
    """Return True if any information block in user messages contains an error."""
    for msg in data.get("conversation", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        for block in re.findall(r"<information>(.*?)</information>", content, re.DOTALL):
            if _ERROR_PATTERN.search(block):
                return True
    return False


def best_file(pred_list: list[dict]) -> dict:
    """
    Choose the representative prediction from a group sharing the same answer.
    Priority: no errors first, then fewest turns, then earliest in list (file_path).
    """
    no_err = [p for p in pred_list if not p["has_error"]]
    candidates = no_err if no_err else pred_list
    return min(candidates, key=lambda p: (p["turns"], p["file_path"]))


def group_answers(preds: list[dict]) -> list[list[dict]]:
    """
    Cluster predictions by equivalent answers using question_scorer.
    Returns groups sorted by size descending.
    """
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
            if answer is not None and rep is not None and question_scorer(str(answer), str(rep)):
                groups[i].append(pred)
                placed = True
                break
        if not placed:
            groups.append([pred])
            repr_answers.append(answer)

    groups.sort(key=lambda g: -len(g))
    return groups


def run(results_dir: str, category_file: str, output_dir: str) -> None:
    with open(category_file) as f:
        cat_data = json.load(f)

    query_id_to_cat: dict[str, str] = {}
    for cat, items in cat_data.items():
        for item in items:
            query_id_to_cat[str(item["query_id"])] = cat

    print(f"Loaded {len(query_id_to_cat)} query-id -> category mappings, "
          f"{len(cat_data)} categories")

    run_dirs = sorted(glob.glob(os.path.join(results_dir, "*/")))
    print(f"Found {len(run_dirs)} run directories\n")

    predictions_by_query: dict[str, list[dict]] = {}

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir.rstrip('/'))
        run_name = "_".join([run_name.split("_")[0], run_name.split("_")[-2], run_name.split("_")[-1]])
        pred_files = sorted(glob.glob(os.path.join(run_dir, "prediction_*.json")))
        print(f"  {run_name}: {len(pred_files)} prediction files")
        for pred_file in pred_files:
            with open(pred_file) as f:
                data = json.load(f)
            query_id = str(data["extra_info"]["query_id"])
            entry = {
                "answer":    extract_answer(data),
                "has_error": has_errors(data),
                "turns":     data.get("turns", 0),
                "file_path": pred_file,
                "run_name":  run_name,
            }
            predictions_by_query.setdefault(query_id, []).append(entry)

    print(f"\nCollected predictions for {len(predictions_by_query)} unique query IDs\n")

    os.makedirs(output_dir, exist_ok=True)

    skipped_unknown_cat = 0

    for query_id, preds in sorted(predictions_by_query.items()):
        category = query_id_to_cat.get(query_id)
        if category is None:
            skipped_unknown_cat += 1
            category = "Unknown"

        total = len(preds)
        groups = group_answers(preds)

        query_dir = os.path.join(output_dir, category, query_id)
        os.makedirs(query_dir, exist_ok=True)

        summary_answers = []

        for rank, group in enumerate(groups, start=1):
            answer_dir_name = f"Answer{rank}"
            answer_dir = os.path.join(query_dir, answer_dir_name)
            os.makedirs(answer_dir, exist_ok=True)

            chosen = best_file(group)
            src = chosen["file_path"]
            stem, ext = os.path.splitext(os.path.basename(src))
            dst_name = f"{stem}_{chosen['run_name']}{ext}"
            dst = os.path.join(answer_dir, dst_name)

            with open(src, encoding="utf-8") as f:
                pred_data = json.load(f)

            pred_data.pop("ground_truth", None)
            pred_data.pop("extra_info", None)
            pred_data.pop("token_usage", None)
            
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(pred_data, f, indent=2, ensure_ascii=False)

            sc = len(group) / total

            summary_answers.append({
                "answer_dir":        answer_dir_name,
                "answer":            chosen["answer"],
                "count":             len(group),
                "self_consistency":  round(sc, 4),
            })

        summary = {
            "query_id":      query_id,
            "category":      category,
            "total_samples": total,
            "answers":       summary_answers,
        }
        with open(os.path.join(query_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Done. Output written to: {output_dir}")
    if skipped_unknown_cat:
        print(f"  WARNING: {skipped_unknown_cat} query IDs not found in category file -> placed in 'Unknown/'")

    cat_counts: dict[str, int] = defaultdict(int)
    for qid in predictions_by_query:
        cat_counts[query_id_to_cat.get(qid, "Unknown")] += 1

    print("\nQuery IDs per category:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="Organize DABStep exploration results")
    parser.add_argument("--results-dir", required=True, help="Path to explore results directory")
    parser.add_argument("--category-file", required=True, help="Path to all_query_category.json")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    args = parser.parse_args()
    run(args.results_dir, args.category_file, args.output_dir)


if __name__ == "__main__":
    main()
