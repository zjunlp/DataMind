import argparse
import json
import pandas as pd
from utils import evaluate


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def run(pred_file: str, gt_file: str) -> None:
    pred_data = read_jsonl(pred_file)
    gt_data = read_jsonl(gt_file)

    scores = evaluate(agent_answers=pd.DataFrame(pred_data), tasks_with_gt=pd.DataFrame(gt_data))

    print(f"Total tasks evaluated: {len(scores)}")

    easy_scores = [e["score"] for e in scores if e["level"] == "easy"]
    hard_scores = [e["score"] for e in scores if e["level"] == "hard"]

    print(f"Total easy tasks: {len(easy_scores)}")
    print(f"Total hard tasks: {len(hard_scores)}")

    accuracy = sum(e["score"] for e in scores) / len(scores)
    easy_accuracy = sum(easy_scores) / len(easy_scores) if easy_scores else 0.0
    hard_accuracy = sum(hard_scores) / len(hard_scores) if hard_scores else 0.0

    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Easy Level Accuracy: {easy_accuracy:.4f}")
    print(f"Hard Level Accuracy: {hard_accuracy:.4f}")

    correct = [e["task_id"] for e in scores if e["score"]]
    wrong = [e["task_id"] for e in scores if not e["score"]]
    print(f"Correct: {len(correct)}, Wrong: {len(wrong)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DABStep predictions")
    parser.add_argument("--pred-file", required=True, help="Prediction JSONL file")
    parser.add_argument("--gt-file", required=True, help="Ground truth JSONL file (test.jsonl)")
    args = parser.parse_args()
    run(args.pred_file, args.gt_file)


if __name__ == "__main__":
    main()
