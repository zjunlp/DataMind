"""
MLE-Bench dataset loader (offline Kaggle-style challenges).

Builds prompts for each competition ID listed in the provided split file
(default: examples/MLE_Bench_Eval/mle-bench/experiments/splits/low.txt),
using the competition description and container dataset path (/data/<id>).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from ..base import BaseDataset
from ..registry import register_dataset
from ..utils import apply_limit_and_start, validate_file_exists, create_standard_task
from ..prompts.system_prompts import SYSTEM_PROMPT_MLEBENCH_TEMPLATE

# ----------------------
# Prompt construction
# ----------------------

MLEBENCH_INSTRUCTIONS = """**INSTRUCTIONS:**
1. Load and explore the datasets provided at the path above. Use text-based summaries and statistics (avoid plotting).
2. Preprocess the data (cleaning, encoding, feature engineering) and perform exploratory analysis to understand key variables and relationships.
3. Train and validate models with proper strategies to avoid overfitting. Try multiple approaches if helpful (model selection, cross-validation, tuning, ensembling).
4. When satisfied, generate predictions for the test data and save a CSV submission to /submission/submission.csv.
5. If a submission validator is available, verify the file format before finalization.
6. Work step-by-step, validating intermediate results and refining your approach until confident.

When you have produced the final submission:
1. Ensure predictions are written to /submission/submission.csv in the required format.
2. Provide a concise summary of your approach in the format: <answer>your final summary</answer>
"""


def create_mlebench_prompt(
    challenge_name: str,
    description: str,
    dataset_path: str,
) -> str:
    """Create user-facing MLE-Bench content (competition + dataset path only)."""
    return f"""**CHALLENGE NAME: {challenge_name}**

{description}

**DATASET LOCATIONS (use full paths):**
{dataset_path}

{MLEBENCH_INSTRUCTIONS}
"""


def build_system_prompt(
    challenge_name: str,
    dataset_path: str,
    submission_path: str,
    validator_url: Optional[str],
) -> str:
    """Fill the MLE-Bench system prompt template with dataset-specific details.

    This function formats the placeholders in SYSTEM_PROMPT_MLEBENCH_TEMPLATE.
    Expected placeholders in the template are: competition_name, submission_path,
    SERVER_URL, and SUBMISSION_FILE. The template itself must not be modified.
    """
    # Server URL for the validator endpoint (falls back to localhost if not provided)
    server_url = validator_url

    # Use the concrete submission path for both the explicit path and example variable
    submission_file = submission_path

    # Fill the imported template exactly with the required placeholders
    return SYSTEM_PROMPT_MLEBENCH_TEMPLATE.format(
        competition_name=challenge_name,
        submission_path=submission_path,
        SERVER_URL=server_url,
        SUBMISSION_FILE=submission_file,
    )


def _repo_root() -> Path:
    # This file lives at dsgym/datasets/loaders/mlebench.py
    # Go up three levels to reach repo root
    return Path(__file__).resolve().parents[3]


@register_dataset("mlebench")
class MLEBenchDataset(BaseDataset):
    """MLE-Bench dataset loader."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _split_file(self, split: str) -> Path:
        # Normalize input (strip extension and lowercase)
        name = split.lower().removesuffix(".txt")
        # Map common aliases
        alias_map = {
            "lite": "low",
            "easy": "low",
        }
        name = alias_map.get(name, name)

        splits_dir = _repo_root() / "examples/MLE_Bench_Eval/mle-bench/experiments/splits"

        # If exact match exists, use it
        candidate = splits_dir / f"{name}.txt"
        if candidate.exists():
            return candidate

        # Otherwise, build helpful error listing available options
        available = sorted(p.stem for p in splits_dir.glob("*.txt"))
        raise FileNotFoundError(
            f"Unknown MLE-Bench split '{split}'. Available: {available}"
        )

    def _competitions_dir(self) -> Path:
        return _repo_root() / "examples/MLE_Bench_Eval/mle-bench/mlebench/competitions"

    def _read_description(self, comp_id: str) -> str:
        comp_dir = self._competitions_dir() / comp_id
        primary = comp_dir / "description.md"
        fallback = comp_dir / "description_obfuscated.md"
        if primary.exists():
            return primary.read_text(encoding="utf-8")
        if fallback.exists():
            return fallback.read_text(encoding="utf-8")
        return f"Offline Kaggle-style competition: {comp_id}. Use data under /data/{comp_id}."


    def load(
        self,
        limit: Optional[int] = None,
        split: str = "low",
        start_index: int = 0,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Load MLE-Bench dataset using competition IDs from a split file.

        Args:
            limit: Maximum number of samples
            split: Dataset split (maps to a split file; defaults to low)
            start_index: Starting index for selection

        Returns:
            List of standardized samples
        """
        split_file = self._split_file(split)
        validate_file_exists(str(split_file), f"MLE-Bench {split} split file")

        with open(split_file, "r", encoding="utf-8") as f:
            comp_ids = [line.strip() for line in f.readlines() if line.strip()]

        comp_ids = apply_limit_and_start(
            comp_ids,
            limit,
            start_index,
            random_sample=False,
            random_seed=self.config.get("random_seed", 42),
        )

        samples: List[Dict[str, Any]] = []


        for idx, comp_id in enumerate(comp_ids):
            description = self._read_description(comp_id)
            dataset_path = f"/data/{comp_id}"

            # Keep the user prompt focused on competition instructions and dataset path
            user_content = create_mlebench_prompt(
                challenge_name=comp_id,
                description=description,
                dataset_path=dataset_path,
            )
            system_prompt = build_system_prompt(
                challenge_name=comp_id,
                dataset_path=dataset_path,
                submission_path="/submission/submission.csv",
                validator_url="http://localhost:5000/validate",
            )

            extra_info = {
                "challenge_name": comp_id,
                "docker_challenge_path": dataset_path,
                "question": user_content,
                "index": start_index + idx,
                "source": "mlebench",
                "metadata_id": comp_id,
                "query_id": comp_id,
                "id": comp_id,
            }

            standard_sample = create_standard_task(
                prompt_content=user_content,
                ground_truth="",
                extra_info=extra_info,
                system_prompt=system_prompt,
            )
            samples.append(standard_sample)

        self._samples = samples
        return samples
        
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index."""
        if self._samples is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        if index < 0 or index >= len(self._samples):
            raise IndexError(f"Sample index {index} out of range [0, {len(self._samples)})")
        
        return self._samples[index]
    def get_metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            splits_dir = _repo_root() / "examples/MLE_Bench_Eval/mle-bench/experiments/splits"
            available_splits = sorted(p.stem for p in splits_dir.glob("*.txt"))
            self._metadata = {
                "name": "MLE-Bench",
                "description": "Offline Kaggle-style challenges from MLE-Bench",
                "format": "text",
                "splits": available_splits,
                "fields": ["challenge_name", "description", "docker_challenge_path"],
                "source": "mlebench",
            }
        return self._metadata

    def get_metrics(self) -> list[str]:
        # Prefer the MLEBench submission metric by default
        return ["mlebench_submission"]

    def get_metric_configs(self) -> dict:
        # No dataset-wide default configs; script can pass private_data_root via metric_configs
        return {}
