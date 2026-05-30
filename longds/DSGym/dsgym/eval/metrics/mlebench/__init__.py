"""
MLE-Bench metric package.

Exposes MLEBenchSubmissionMetric for offline Kaggle-style grading using
the MLE-Bench helpers embedded under examples/MLE_Bench_Eval/mle-bench.
"""

from .mlebench_metric import MLEBenchSubmissionMetric

__all__ = [
    "MLEBenchSubmissionMetric",
]

