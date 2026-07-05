"""
MLE-Bench submission metric for DSGym.

This metric validates and grades a produced `submission.csv` against the
offline MLE-Bench leaderboard and grading utilities bundled in
`examples/MLE_Bench_Eval/mle-bench`.

It mirrors the Kaggle metric shape but calls `mlebench.grade.grade_csv`
and enriches the result with medal/threshold information. The agent is
expected to return the path to the produced submission (prediction).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from ..base import BaseMetric, MetricResult


class MLEBenchSubmissionMetric(BaseMetric):
    """Metric that grades an offline MLE-Bench submission.csv using local graders."""

    def __init__(self, private_data_root: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        # Optional override for the MLE-Bench private data root
        self.private_data_root = private_data_root

    @property
    def name(self) -> str:
        return "mlebench_submission"

    @property
    def requires_ground_truth(self) -> bool:
        # Ground truth is not part of each sample; grading uses local datasets
        return False

    def _import_mlebench(self):
        """Ensure mlebench helpers are importable and return registry, grade_csv, aggregate_reports."""
        # Add embedded mle-bench to path lazily to avoid global side effects
        repo_root = Path(__file__).resolve().parents[4]
        mle_repo = repo_root / "examples" / "MLE_Bench_Eval" / "mle-bench"
        if str(mle_repo) not in sys.path:
            sys.path.insert(0, str(mle_repo))

        from mlebench.registry import registry as mle_registry  # type: ignore
        from mlebench.grade import grade_csv, aggregate_reports  # type: ignore
        return mle_registry, grade_csv, aggregate_reports

    def evaluate(
        self,
        prediction: str,
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> MetricResult:
        """
        Evaluate an MLE-Bench submission by running the local grader.

        Expected context in kwargs (forwarded by Evaluator):
          - extra_info.challenge_name: competition ID
          - extra_info.*: passthrough metadata
        """
        extra_info: Dict[str, Any] = kwargs.get("extra_info", {}) or {}
        competition_id = extra_info.get("challenge_name")
        submission_path = (prediction or "").strip()

        # Basic validation
        if not submission_path or not os.path.exists(submission_path):
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={
                    "reason": "submission file not found",
                    "competition_id": competition_id,
                    "prediction": prediction,
                },
            )


        try:
            mle_registry, grade_csv, _ = self._import_mlebench()

            # Bind registry to provided data root if configured
            if self.private_data_root:
                from pathlib import Path as _P
                mle_registry = mle_registry.set_data_dir(_P(self.private_data_root))

            competition = mle_registry.get_competition(str(competition_id))
            report = grade_csv(Path(submission_path), competition)

            # Use the numeric competition score as the primary score (may be None)
            score = report.score if report.score is not None else None
            details = {
                "competition_id": report.competition_id,
                "score": report.score,
                "gold_threshold": report.gold_threshold,
                "silver_threshold": report.silver_threshold,
                "bronze_threshold": report.bronze_threshold,
                "median_threshold": report.median_threshold,
                "any_medal": report.any_medal,
                "gold_medal": report.gold_medal,
                "silver_medal": report.silver_medal,
                "bronze_medal": report.bronze_medal,
                "above_median": report.above_median,
                "submission_exists": report.submission_exists,
                "valid_submission": report.valid_submission,
                "is_lower_better": report.is_lower_better,
                "created_at": report.created_at.isoformat(),
                "submission_path": report.submission_path,
            }

            return MetricResult(
                metric_name=self.name,
                score=score,
                details=details,
            )
        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                score=None,
                error=str(e),
                details={
                    "competition_id": competition_id,
                    "submission": submission_path,
                },
            )
