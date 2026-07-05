"""
DABStep dataset loader.
"""

import os
from typing import List, Dict, Any, Optional

from src.task_processor.base import BaseDataset
from src.task_processor.registry import register_dataset
from src.task_processor.utils import load_json, apply_limit_and_start, validate_file_exists, create_standard_task, construct_data_paths
from src.task_processor.skill_utils import build_skill_content
from src.core.config import get_task_path, RAW_DATA_DIR

DABSTEP_INSTRUCTIONS = """1. You MUST thoroughly read and internalize the manual.md and payments-readme.md files COMPLETELY before proceeding. (You may print them out to read them.)
2. You can use these python libraries: pandas, numpy, scipy, scikit-learn, statsmodels, linearmodels, etc.
3. For categorical answers, use exact terminology from the manual/data
4. IF AND ONLY IF you have exhausted all possibles solution plans you can come up with and still can not find a valid answer, then provide "Not Applicable" as a final answer.
"""

DABSTEP_DATA_FILES_RELATIVE = [
    "acquirer_countries.csv",
    "fees.json",
    "manual.md",
    "merchant_category_codes.csv",
    "merchant_data.json",
    "payments-readme.md",
    "payments.csv"
]


def create_dabstep_query(
    task: Dict[str, Any],
    data_paths: Dict[str, List[str]],
    use_skill: bool = True,
    skill_content: Optional[str] = None,
) -> str:
    question = task["query"]

    files_list = "\n".join(data_paths['virtual'])

    skill_section = ""
    if use_skill:
        content = skill_content if skill_content is not None else ""
        skill_section = f"\n\nSKILL:\n{content}\n"

    query = f"""QUESTION: {question}

DATASET INFORMATION:
Dataset 1: acquirer_countries.csv
This dataset contains the country_code of the acquirer.

Dataset 2: fees.json
This dataset contains the Payment Processing Fees. For fee calculations, confirm you're applying the right fee rules and formulas

Dataset 3: merchant_category_codes.csv
This dataset contains the mcc and descriptions.

Dataset 4: merchant_data.json
This dataset contains the data for the merchants. Each merchant has merchant_category_code, account_type, capture_delay and a list of acquirers.

Dataset 5: payments.csv
This dataset contains payment transactions processed by the Payments Processor.

Dataset 6: manual.md
This file contains domain-specific definitions that are ESSENTIAL for correct interpretation.

Dataset 7: payments-readme.md
This file contains important information about the payments.csv dataset and relevant terminology.

For JSON files, examine the schema by looking at a small sample (first few entries).
For CSV files, check the column headers first to understand the data structure

DATASET LOCATIONS (use full paths):
{files_list}

INSTRUCTIONS:
{DABSTEP_INSTRUCTIONS}{skill_section}"""

    return query


@register_dataset("dabstep")
class DABStepDataset(BaseDataset):
    """DABStep dataset loader."""

    def __init__(self, task_name: str, tasks_dir: Optional[str] = None, virtual_data_root: Optional[str] = None, **kwargs):
        """
        Args:
            task_name: Name of the task to load
            tasks_dir: Directory containing DABStep task files
            virtual_data_root: Root path for virtual/docker paths (default: "/data")
            **kwargs: Additional configuration
        """
        self.task_name = task_name
        if tasks_dir is None:
            tasks_dir = str(get_task_path(task_name))
        super().__init__(data_dir=tasks_dir, virtual_data_root=virtual_data_root, **kwargs)
        self.tasks_dir = tasks_dir

    def load(
        self,
        dataset_name: str = None,
        limit: Optional[int] = None,
        level_filter: Optional[str] = None,
        start_index: int = 0,
        dataset_type: str = "original",
        filtered_dataset_path: Optional[str] = None,
        use_skill: bool = True,
        query_categories: Optional[List[str]] = None,
        skills_base_dir: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load DABStep dataset (original, synthetic, or filtered).

        Args:
            dataset_name: dataset_name
            limit: Maximum number of samples to load
            level_filter: Filter by difficulty level (easy/hard)
            start_index: Starting index for data selection
            dataset_type: Type of dataset (original/synthetic/filtered)
            filtered_dataset_path: Path to filtered dataset JSONL file
            use_skill: Whether to include SKILL section in the prompt
            query_categories: Filter tasks to these query categories as defined in
                query_category_file (e.g. ["Total_Fees_Calculation", "Applicable_Fee_IDs"]).
                When None, all tasks are included.
            skills_base_dir: Root directory containing per-category skill subdirectories.
                Defaults to the module-level SKILLS_BASE_DIR constant.
            query_category_file: Path to the JSON category index file.
                Defaults to the module-level QUERY_CATEGORY_FILE constant.
            **kwargs: Additional loading parameters

        Returns:
            List of dataset samples
        """
        _skills_dir = skills_base_dir

        if use_skill == True and _skills_dir == None:
            raise ValueError(f"No skill dir provided.")
        
        if dataset_name is None:
            dataset_name = self.task_name

        # Load tasks based on dataset type
        if dataset_type == "filtered":
            if not filtered_dataset_path:
                raise ValueError("filtered_dataset_path is required when dataset_type='filtered'")
            validate_file_exists(filtered_dataset_path, "Filtered DABStep dataset")
            tasks = load_json(filtered_dataset_path)
        else:  # dataset_type == "original"
            task_file = os.path.join(self.tasks_dir, f"{dataset_name}.json")
            validate_file_exists(task_file, f"DABStep {dataset_name} task file")
            tasks = load_json(task_file)

        # Filter by level if specified
        if level_filter:
            task_0 = tasks[0] if tasks else None
            if task_0 is None or "metadata" not in task_0 or "level" not in task_0["metadata"]:
                raise ValueError("Tasks do not contain 'metadata.level' field for filtering")
            tasks = [task for task in tasks if task.get("metadata").get("level") == level_filter]

        # Filter by query categories if specified
        if query_categories:
            task_0 = tasks[0] if tasks else None
            if task_0 is None or "metadata" not in task_0 or "category" not in task_0["metadata"]:
                raise ValueError("Tasks do not contain 'metadata.category' field for filtering")
            tasks = [task for task in tasks if task.get("metadata").get("category") in query_categories]

        # Apply start_index and limit
        tasks = apply_limit_and_start(
            tasks, limit, start_index,
            random_sample=False,
            random_seed=self.config.get('random_seed', 42)
        )

        # Build skill content once for all samples
        skill_content = build_skill_content(_skills_dir) if use_skill else None

        # Construct data paths for DABStep files
        data_paths = construct_data_paths(
            relative_paths=DABSTEP_DATA_FILES_RELATIVE,
            dataset_name='dabstep',
            data_root=RAW_DATA_DIR,
            virtual_data_root=self.virtual_data_root
        )

        # Convert to evaluation format
        samples = []
        for i, task in enumerate(tasks):
            original_idx = start_index + i

            extra_info = {
                "query": task.get("query", ""),
                "ground_truth": task.get("ground_truth", ""),
                "source": f"{self.task_name}_{dataset_type}",
                "metadata_id": task.get("task_id", ""),
                "query_id": task.get("task_id", ""),
                "index": original_idx,
                "data_files": data_paths,
                "category": task.get("metadata", {}).get("category", ""),
            }

            standard_sample = create_standard_task(
                prompt_content=create_dabstep_query(
                    task, data_paths,
                    use_skill=use_skill,
                    skill_content=skill_content,
                ),
                ground_truth=str(task.get("ground_truth", "")),
                extra_info=extra_info
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
        """Get DABStep dataset metadata."""
        if self._metadata is None:
            self._metadata = {
                'name': 'DABStep',
                'description': 'Dataset for evaluating data analysis capabilities step by step',
                'tasks_dir': self.tasks_dir,
                'format': 'jsonl',
                'splits': ['dev', 'all'],
                'levels': ['easy', 'hard'],
                'fields': ['question', 'answer', 'guidelines'],
                'source': 'dabstep'
            }

        return self._metadata

    def get_metrics(self) -> List[str]:
        """
        Get metrics for DABStep dataset.

        Note: DABStep metric will return None scores when no ground truth is available,
        but this allows the evaluation framework to still process predictions.
        """
        return ["dabstep"]

    def get_metric_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metric configurations for DABStep dataset.
        """
        return {
            "dabstep": {}
        }