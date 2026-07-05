import json
import glob
import os
import re
from collections import defaultdict

from src.verifier.base import BaseVerifier
from src.verifier.registry import register_verifier
from src.verifier.instances.utils import question_scorer

VERIFIER_INIT_PROMPT = """
# Terminology for Trajectory Analysis:
The trajectories are grouped by changes in Answer.
1. For different tasks, trajectories with different answers are collected in the folders named Answer1, Answer2, Answer 3 and so on. Each answer folder contains one trajectory which is the most representative trajectory among all trajectories with the same answer.
2. Each task has summary.json file which contains the self-consistency of each answer. You should analyze the self-consistency of different answers and compare the trajectories with different answers to find out what leads to the change of answer.

# Important:
1. Before writing the Skill, you should actively explore the relevant dataset files available under the dataset directory and referenced by the exploration trajectories.
2. Answers and Self-Consistency do not guarantee correctness! You should analyze every trajectory, corresponding self-consistency and related files to find out the effective solution strategies.
3. Ensure that the final Skill is consistent with the conditions in the problem and the information in the data file, rather than originating from conjecture or assumption.

# Note:
1. Self-Consistency does not correlate with correctness.
2. You should analyze every trajectory and the corresponding self-consistency to find out the effective solution strategies.
3. Summarize the effective solution strategies into a comprehensive Skill guide.
"""

VERIFIER_ITERATE_PROMPT = """
# Terminology for Trajectory Analysis:
The trajectories are grouped by changes in Answer.
1. For different tasks, trajectories with different answers are collected in the folders named Answer1, Answer2, Answer 3 and so on. Each answer folder contains one trajectory which is the most representative trajectory among all trajectories with the same answer.
2. Each task has summary.json file which contains the self-consistency of each answer. You should analyze the self-consistency of different answers and compare the trajectories with different answers to find out what leads to the change of answer.
3. We provide trajectory information before and after the iteration, including their answer clustering and self-consistency information. Changes in answer clustering and self-consistency may occur either because correct information has been found or because a wrong pattern has been fitted. You need to carefully analyze and discover reusable successful experiences or patterns.
4. The trajectories currently provided are those that have undergone significant changes after adding the current Skill. They may be moving in a positive direction or a negative direction. You need to carefully diagnose the current Skill and the trajectories to ensure that the information in the Skill is consistent with the data file and can provide reasonable solutions.
5. If the given folder does not provide trajectory information, it indicates that the self-consistency of all trajectories is 1.0 and the answers are consistent before and after the iteration. In this case, no modification to the skill is required.

# Important:
1. Before refining the Skill, you should actively explore the relevant dataset files available under the dataset directory and referenced by the exploration trajectories.
2. Answers and Self-Consistency do not guarantee correctness! You should analyze every trajectory, corresponding self-consistency and related files to find out the effective solution strategies.
3. Ensure that the final Skill is consistent with the conditions in the problem and the information in the data file, rather than originating from conjecture or assumption.
4. The provided SKILL is not completely correct. If there is any conflict with the current data or trajectories, please carefully analyze and make corrections.

# Note:
1. Self-Consistency does not necessarily correlate with correctness, but it can provide insights into the reliability of the trajectories.
2. You should analyze every trajectory and the corresponding self-consistency to find out the effective solution strategies. Your goal is to maximize the performance of the refined Skill.
3. Summarize the effective solution strategies, and modify the existing Skill guide based on your analysis.
"""

@register_verifier("agreement")
class AgreementVerifier(BaseVerifier):

    def __init__(self, input_file_dirs: list, output_dir: str, category_list: list, **kwargs):
        super().__init__(input_file_dirs, output_dir, category_list, **kwargs)

    # ------------------------------------------------------------------
    # category_list / output-path helpers
    # ------------------------------------------------------------------

    def _should_process(self, category: str) -> bool:
        """Return True if this query should be processed.

        When self.category_list is non-empty, only the listed categories pass.
        When self.category_list is empty, all queries pass.
        """
        return (not self.category_list) or (category in self.category_list)

    def _query_dir(self, query_id: str, category: str) -> str:
        """Build the output directory for a query.

        With categories specified : output_dir/{category}/{query_id}/
        Without categories        : output_dir/{query_id}/
        """
        if self.category_list:
            return os.path.join(self.output_dir, category, query_id)
        return os.path.join(self.output_dir, query_id)

    # ------------------------------------------------------------------
    # Directory / file discovery
    # ------------------------------------------------------------------

    def _find_sample_dirs(self, iter_dir: str) -> list[str]:
        """Return leaf dirs under iter_dir that contain prediction_*.json.

        Each such dir is treated as one sample run.  If iter_dir itself
        holds prediction files directly (no subdirs), it is returned as-is.
        """
        sample_dirs: list[str] = []
        for root, dirs, _ in os.walk(iter_dir):
            for d in sorted(dirs):
                full = os.path.join(root, d)
                if glob.glob(os.path.join(full, "prediction_*.json")):
                    sample_dirs.append(full)
        if not sample_dirs and glob.glob(os.path.join(iter_dir, "prediction_*.json")):
            return [iter_dir]
        return sorted(sample_dirs)

    @staticmethod
    def _make_run_name(sample_dir: str) -> str:
        raw = os.path.basename(sample_dir.rstrip("/"))
        parts = raw.split("_")
        if len(parts) >= 3:
            return "_".join([parts[0], parts[-2], parts[-1]])
        return raw

    # ------------------------------------------------------------------
    # Prediction loading
    # ------------------------------------------------------------------

    def _load_predictions(self, iter_dir: str, label: str = "") -> dict[str, list[dict]]:
        """Load all predictions from iter_dir.

        iter_dir contains one or more sample subdirs (each = one sample run).
        All samples are merged per query_id for agreement computation.
        Category is read from each prediction's extra_info.
        """
        sample_dirs = self._find_sample_dirs(iter_dir)
        print(f"  [{label}] found {len(sample_dirs)} sample dir(s) in {iter_dir}")
        predictions_by_query: dict[str, list[dict]] = {}

        for index, sample_dir in enumerate(sample_dirs):
            run_name = f"{index}_{self._make_run_name(sample_dir)}"
            for pred_file in sorted(glob.glob(os.path.join(sample_dir, "prediction_*.json"))):
                with open(pred_file, encoding="utf-8") as f:
                    data = json.load(f)
                extra_info = data.get("extra_info", {})
                query_id = str(extra_info["query_id"])
                predictions_by_query.setdefault(query_id, []).append({
                    "answer":    data.get("prediction"),
                    "category":  extra_info.get("category", ""),
                    "turns":     data.get("turns", 0),
                    "file_path": pred_file,
                    "run_name":  run_name,
                })

        print(f"  [{label}] collected predictions for "
              f"{len(predictions_by_query)} unique query IDs")
        return predictions_by_query

    # ------------------------------------------------------------------
    # Agreement grouping
    # ------------------------------------------------------------------

    def _group_answers(self, preds: list[dict]) -> list[list[dict]]:
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

    @staticmethod
    def _best_file(pred_list: list[dict]) -> dict:
        return min(pred_list, key=lambda p: (p["turns"], p["file_path"]))

    def _write_answer_dirs(
        self,
        preds: list[dict],
        groups: list[list[dict]],
        base_dir: str,
        query_id: str,
        category: str,
        iter_n: int | None = None,
    ) -> None:
        os.makedirs(base_dir, exist_ok=True)
        total = len(preds)
        summary_answers = []

        for rank, group in enumerate(groups, start=1):
            answer_dir = os.path.join(base_dir, f"Answer{rank}")
            os.makedirs(answer_dir, exist_ok=True)

            chosen = self._best_file(group)
            src = chosen["file_path"]
            stem, ext = os.path.splitext(os.path.basename(src))
            dst = os.path.join(answer_dir, f"{stem}_{chosen['run_name']}{ext}")

            with open(src, encoding="utf-8") as f:
                pred_data = json.load(f)

            pred_data.pop("ground_truth", None)
            pred_data.pop("extra_info", None)
            pred_data.pop("token_usage", None)

            with open(dst, "w", encoding="utf-8") as f:
                json.dump(pred_data, f, indent=2, ensure_ascii=False)

            summary_answers.append({
                "answer_dir":       f"Answer{rank}",
                "answer":           chosen["answer"],
                "count":            len(group),
                "self_consistency": round(len(group) / total, 4),
            })

        if self.category_list:
            summary: dict = {
                "query_id":      query_id,
                "category":      category,
                "total_samples": total,
                "answers":       summary_answers,
            }
        else:
            summary: dict = {
                "query_id":      query_id,
                "total_samples": total,
                "answers":       summary_answers,
            }
            
        if iter_n is not None:
            summary["iter"] = iter_n

        with open(os.path.join(base_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # init_run  (single iteration: input_file_dirs[-1])
    # ------------------------------------------------------------------

    def init_run(self):
        """Process the current iteration directory (input_file_dirs[-1]).

        Subdirs within it are treated as independent samples; agreement
        is computed across all samples for each query_id.
        """
        iter_dir = self.input_file_dirs[-1]
        predictions_by_query = self._load_predictions(iter_dir, label="iter")

        print(f"\nCollected predictions for {len(predictions_by_query)} unique query IDs\n")
        os.makedirs(self.output_dir, exist_ok=True)

        for query_id, preds in sorted(predictions_by_query.items()):
            category = preds[0]["category"]
            if not self._should_process(category):
                continue

            groups = self._group_answers(preds)
            query_dir = self._query_dir(query_id, category)
            self._write_answer_dirs(preds, groups, query_dir, query_id, category)

        return {
            "traj_dir": self.output_dir,
            "prompt": VERIFIER_INIT_PROMPT
        }

    # ------------------------------------------------------------------
    # iterate_run  (two iterations: input_file_dirs[-2] vs [-1])
    # ------------------------------------------------------------------

    def iterate_run(self) -> None:
        """Compare the previous and current iteration directories.

        Uses input_file_dirs[-2] as prev and input_file_dirs[-1] as curr.
        Query IDs where the top-agreement answer has SC==1.0 in both iters
        are skipped.
        """
        prev_dir = self.input_file_dirs[-2]
        curr_dir = self.input_file_dirs[-1]

        print(f"\nLoading prev predictions (input_file_dirs[-2]):")
        prev_by_query = self._load_predictions(prev_dir, label="prev")

        print(f"\nLoading curr predictions (input_file_dirs[-1]):")
        curr_by_query = self._load_predictions(curr_dir, label="curr")

        all_query_ids = set(prev_by_query) & set(curr_by_query)
        print(f"\nQuery IDs present in both iters: {len(all_query_ids)}\n")

        os.makedirs(self.output_dir, exist_ok=True)

        skipped_sc1_both = 0
        kept = 0

        for query_id in sorted(all_query_ids):
            prev_preds = prev_by_query[query_id]
            curr_preds = curr_by_query[query_id]
            category = curr_preds[0]["category"]

            if not self._should_process(category):
                continue

            prev_groups = self._group_answers(prev_preds)
            curr_groups = self._group_answers(curr_preds)

            prev_sc = len(prev_groups[0]) / len(prev_preds)
            curr_sc = len(curr_groups[0]) / len(curr_preds)
            prev_answer = prev_groups[0][0]["answer"]
            curr_answer = curr_groups[0][0]["answer"]

            if (
                prev_sc == 1.0
                and curr_sc == 1.0
                and prev_answer is not None
                and curr_answer is not None
            ):
                skipped_sc1_both += 1
                continue

            kept += 1
            task_dir = self._query_dir(query_id, category)

            self._write_answer_dirs(
                prev_preds, prev_groups,
                os.path.join(task_dir, "iter_prev"),
                query_id, category, iter_n=len(self.input_file_dirs) - 2,
            )
            self._write_answer_dirs(
                curr_preds, curr_groups,
                os.path.join(task_dir, "iter_curr"),
                query_id, category, iter_n=len(self.input_file_dirs) - 1,
            )

        return {
            "traj_dir": self.output_dir,
            "prompt": VERIFIER_ITERATE_PROMPT
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _print_cat_counts(self, predictions_by_query: dict[str, list[dict]]) -> None:
        cat_counts: dict[str, int] = defaultdict(int)
        for preds in predictions_by_query.values():
            cat_counts[preds[0]["category"]] += 1
        print("\nQuery IDs per category:")
        for cat, cnt in sorted(cat_counts.items()):
            print(f"  {cat if cat else '(no category)'}: {cnt}")
