import argparse
import json
import glob
import re
import os


def load_predictions(directory):
    predictions = {}
    for path in glob.glob(os.path.join(directory, "prediction_*.json")):
        m = re.search(r"prediction_(\d+)\.json", path)
        if m:
            with open(path) as f:
                data = json.load(f)
            idx = data["extra_info"]["query_id"]
            predictions[idx] = data.get("prediction")
    return predictions


def run(results_dir: str, all_jsonl: str, output_file: str, categories: list[str]) -> None:
    dirs = [
        os.path.join(results_dir, f"dabstep_test_{category}")
        for category in categories
    ]

    with open(all_jsonl) as f:
        tasks = [json.loads(line) for line in f]

    preds = {}
    for d in dirs:
        if os.path.isdir(d):
            preds.update(load_predictions(d))
        else:
            print(f"Warning: directory not found, skipping: {d}")

    with open(output_file, "w") as out:
        for task in tasks:
            task_id = task["task_id"]
            answer = preds.get(task_id, "PlaceHolder, None Answer")
            if not isinstance(answer, str):
                print(f"Warning: answer for task {task_id} is not a string, converting.")
                answer = str(answer)
            if not isinstance(task_id, str):
                task_id = str(task_id)
            out.write(json.dumps({"task_id": task_id, "agent_answer": answer}, ensure_ascii=False) + "\n")

    covered = sum(1 for t in tasks if t["task_id"] in preds)
    print(f"Written {len(tasks)} entries to {output_file}")
    print(f"Covered: {covered}, Missing: {len(tasks) - covered}")


def main():
    parser = argparse.ArgumentParser(description="Collect prediction results into a JSONL file")
    parser.add_argument("--results-dir", required=True, help="Base directory containing per-category result dirs")
    parser.add_argument("--all-jsonl", required=True, help="Path to all.jsonl (task list)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--categories", nargs="*", default=[
        "Applicable_Fee_IDs", "Average_Fee_Estimation", "Average_Transaction_Value_Stats",
        "Dataset_Metadata_and_Business_Rules", "Fee_Delta_and_Impact_Simulation",
        "Fraud_and_General_Macro_Analysis", "Highest_Cost_Scenario_Identification",
        "Routing_and_Cost_Optimization", "Total_Fees_Calculation",
    ], help="Category list")
    args = parser.parse_args()
    run(args.results_dir, args.all_jsonl, args.output, args.categories)


if __name__ == "__main__":
    main()
