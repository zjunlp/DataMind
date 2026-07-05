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
Trajectories are grouped by success and failure.
1. For different tasks, trajectories that succeeded and those that failed are collected in two folders named `success` and `fail`, respectively.
2. The `success` folder contains one trajectory, which is the most representative successful trajectory.
3. The `fail` folder contains the shortest failed trajectory and, when different, the longest failed trajectory.

# Important:
1. Before writing the Skill guide, you should actively explore the relevant dataset files available under the dataset directory and those referenced by the exploration trajectories.
2. Ensure that the final Skill guide is consistent with the conditions in the problem and the information in the data files, rather than originating from conjecture or assumption.
3. The Skill guide should capture reusable analysis procedures, verification checks, and generalizable lessons. Do not hard-code task-specific details, such as exact task answers, specific row/record values, dataset-specific output numbers, or any information that would only apply to a particular exploration task. When referring to observed failures or successes, abstract them into transferable principles rather than copying concrete task data.
"""

VERIFIER_ITERATE_PROMPT = """
# Terminology for Trajectory Analysis:
Trajectories are grouped by success and failure.
1. For different tasks, trajectories that succeeded and those that failed are collected in two folders named `success` and `fail`, respectively.
2. The `success` folder contains one trajectory, which is the most representative successful trajectory.
3. The `fail` folder contains the shortest failed trajectory and, when different, the longest failed trajectory.

# Important:
1. Before writing the Skill guide, you should actively explore the relevant dataset files available under the dataset directory and those referenced by the exploration trajectories.
2. Ensure that the final Skill guide is consistent with the conditions in the problem and the information in the data files, rather than originating from conjecture or assumption.
3. The Skill guide should capture reusable analysis procedures, verification checks, and generalizable lessons. Do not hard-code task-specific details, such as exact task answers, specific row/record values, dataset-specific output numbers, or any information that would only apply to a particular exploration task. When referring to observed failures or successes, abstract them into transferable principles rather than copying concrete task data.
"""

@register_verifier("supervised")
class SupervisedVerifier(BaseVerifier):

    def __init__(self, input_file_dirs: list, output_dir: str, category_list: list, **kwargs):
        super().__init__(input_file_dirs, output_dir, category_list, **kwargs)

    # ------------------------------------------------------------------
    # Public dispatch
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
        All samples are merged per query_id.
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
                    "answer":       data.get("prediction"),
                    "ground_truth": data.get("ground_truth"),
                    "category":     extra_info.get("category", ""),
                    "turns":        data.get("turns", 0),
                    "file_path":    pred_file,
                    "run_name":     run_name,
                })

        print(f"  [{label}] collected predictions for "
              f"{len(predictions_by_query)} unique query IDs")
        return predictions_by_query

    # ------------------------------------------------------------------
    # Classification and output helpers
    # ------------------------------------------------------------------

    def _classify(self, preds: list[dict]) -> tuple[list[dict], list[dict]]:
        success: list[dict] = []
        fail: list[dict] = []
        for pred in preds:
            answer = pred["answer"]
            gt = pred["ground_truth"]
            if gt == None:
                raise ValueError(f"Ground truth is None for prediction file {pred['file_path']}")
            if (
                answer is not None
                and gt is not None
                and question_scorer(str(answer), str(gt))
            ):
                success.append(pred)
            else:
                fail.append(pred)
        return success, fail

    @staticmethod
    def _best_file(pred_list: list[dict]) -> dict:
        return min(pred_list, key=lambda p: (p["turns"], p["file_path"]))

    @staticmethod
    def _max_turns_file(pred_list: list[dict]) -> dict:
        return max(pred_list, key=lambda p: (p["turns"], p["file_path"]))

    def _write_prediction_file(self, pred: dict, group_dir: str) -> None:
        os.makedirs(group_dir, exist_ok=True)
        src = pred["file_path"]
        stem, ext = os.path.splitext(os.path.basename(src))
        dst = os.path.join(group_dir, f"{stem}_{pred['run_name']}{ext}")

        with open(src, encoding="utf-8") as f:
            pred_data = json.load(f)
            
        pred_data.pop("extra_info", None)
        pred_data.pop("token_usage", None)

        with open(dst, "w", encoding="utf-8") as f:
            json.dump(pred_data, f, indent=2, ensure_ascii=False)

    def _write_sample(self, group: list[dict], group_dir: str) -> None:
        self._write_prediction_file(self._best_file(group), group_dir)

    def _write_fail_samples(self, group: list[dict], group_dir: str) -> None:
        chosen: list[dict] = []
        seen_paths: set[str] = set()
        for pred in (self._best_file(group), self._max_turns_file(group)):
            if pred["file_path"] in seen_paths:
                continue
            chosen.append(pred)
            seen_paths.add(pred["file_path"])

        for pred in chosen:
            self._write_prediction_file(pred, group_dir)

    def _write_query_dir(
        self,
        preds: list[dict],
        query_dir: str,
        query_id: str,
        category: str,
    ) -> None:
        """Write success/ and fail/ samples plus summary.json for one query."""
        os.makedirs(query_dir, exist_ok=True)
        success, fail = self._classify(preds)
        total = len(preds)

        if success:
            self._write_sample(success, os.path.join(query_dir, "success"))
        if fail:
            self._write_fail_samples(fail, os.path.join(query_dir, "fail"))

        summary: dict = {
            "query_id":      query_id,
            "total_samples": total,
            "success_count": len(success),
            "fail_count":    len(fail),
            "accuracy":      round(len(success) / total, 4) if total else 0,
        }
        if self.category_list:
            summary["category"] = category
        with open(os.path.join(query_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # init_run / iterate_run  (both use input_file_dirs[-1])
    # ------------------------------------------------------------------

    def init_run(self) -> None:
        """Process the current iteration directory (input_file_dirs[-1]).

        Subdirs within it are treated as independent samples; each sample
        prediction is classified as success or fail against ground truth.
        """
        iter_dir = self.input_file_dirs[-1]
        predictions_by_query = self._load_predictions(iter_dir, label="iter")

        print(f"\nCollected predictions for {len(predictions_by_query)} unique query IDs\n")
        os.makedirs(self.output_dir, exist_ok=True)

        for query_id, preds in sorted(predictions_by_query.items()):
            category = preds[0]["category"]
            if not self._should_process(category):
                continue

            query_dir = self._query_dir(query_id, category)
            self._write_query_dir(preds, query_dir, query_id, category)

        return {
            "traj_dir": self.output_dir,
            "prompt": VERIFIER_INIT_PROMPT,
        }

    def iterate_run(self) -> None:
        """Process the current iteration directory (input_file_dirs[-1]).

        No cross-iteration comparison; delegates to init_run.
        """
        results = self.init_run()
        return {
            "traj_dir": results["traj_dir"],
            "prompt": VERIFIER_ITERATE_PROMPT,
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
