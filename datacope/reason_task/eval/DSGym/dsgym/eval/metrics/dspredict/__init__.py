"""
DSPredict evaluation utilities and metrics.

This subpackage provides a metric that submits a submission.csv to DSPredict,
waits for scoring, and enriches results with leaderboard statistics.
"""

from .dspredict_metric import KaggleSubmissionMetric

__all__ = [
    "KaggleSubmissionMetric",
]

