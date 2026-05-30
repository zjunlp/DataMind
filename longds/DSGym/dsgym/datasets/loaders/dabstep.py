"""
DABStep dataset loader.
"""

import os
import json
from typing import List, Dict, Any, Optional

from ..base import BaseDataset
from ..registry import register_dataset
from ..utils import load_jsonl, apply_limit_and_start, validate_file_exists, create_standard_task, construct_data_paths
from ..prompts import SYSTEM_PROMPT
from dsgym.datasets.config import get_task_path, RAW_DATA_DIR


DABSTEP_ANALYSIS_PROCESS = """
ANALYSIS PROCESS:
1) You MUST thoroughly read and internalize the manual.md and payments-readme.md files COMPLETELY before proceeding.
   - The manual contains domain-specific definitions that are ESSENTIAL for correct interpretation
   - Terms like "fees", "transactions", and other concepts have specialized meanings in this context
   - Misunderstanding these definitions will GUARANTEE incorrect answers
   - Create a mental model of how all concepts interrelate based on the manual's definitions
   - Pay special attention to any hierarchical relationships, exclusions, or conditional statements
   - If manual.md file does not contain enough information. You should also read the payments-readme.md file. The payments-readme.md file contains important information about the payments.csv dataset and relevant terminology.
   - You can use file.read() to load the md files and print them out to read them.

2) When reading the question, map it back to the exact terminology and definitions from the manual
   - Do NOT rely on your general knowledge about these terms
   - The correct answer depends on using the EXACT definitions from the manual
   - Identify which specific section of the manual is most relevant to the question

3) FOR COMPLEX MULTI-STEP QUESTIONS: Break down the question into logical sub-components
   - Identify all the specific filters needed (merchant names, time periods, fee IDs, etc.)
   - Determine the sequence of operations required (filter → calculate → aggregate → compare)
   - For hypothetical scenarios (e.g., "what if fee changed to X"), clearly identify:
       * Current state calculation
       * Hypothetical state calculation  
       * Delta/difference calculation
   - For time-based questions, ensure you understand the exact date ranges and formatting
   - For merchant-specific questions, verify exact merchant name matching (case-sensitive)
   - For fee-related questions, distinguish between fee applicability vs. fee amounts vs. fee calculations

4) Next, read the payments-readme.md file to understand the payment data structure and relevant terminology.

5) For each additional file you need to access:
   - For CSV files: Check the column headers first to understand the data structure
   - For JSON files: Examine the schema by looking at a small sample (first few entries)
   - For text/markdown files: Read through the entire content for critical information

6) When working with large files, start by understanding their structure before attempting to process all the data.

7) Data validation and quality checks:
   - Check for missing values, duplicates, or data inconsistencies
   - Verify data types match expectations (strings, numbers, dates, etc.)
   - Look for outliers or anomalies that might affect your analysis
   - Cross-reference data between files to ensure consistency

8) VERIFICATION STEP: Before finalizing your answer, always:
   - Re-read the relevant sections of the manual to confirm your interpretation
   - Double-check your calculations and data aggregations
   - For multi-step calculations, verify each intermediate result makes sense
   - For time-based filtering, confirm you're using the correct date format and range
   - For merchant-specific queries, verify exact name matches
   - For fee calculations, confirm you're applying the right fee rules and formulas
   - Verify your answer makes logical sense given the context
   - Ensure you're answering the exact question asked (not a related but different question)

Note:
- Be precise with numerical answers (include appropriate decimal places, units, etc.)
- If calculations are involved, show your work clearly step-by-step
- For complex multi-step problems, show all intermediate calculations
- If the answer requires aggregation, explicitly state what you're aggregating
- For categorical answers, use exact terminology from the manual/data
- If data is missing or incomplete, state this clearly rather than guessing
- For hypothetical scenarios, clearly distinguish current vs. hypothetical calculations
- STRICTLY ADHERE TO THE GUIDELINES in the query for formatting your output
"""

DABSTEP_INSTRUCTIONS = """INSTRUCTIONS:
1. You MUST thoroughly read and internalize the manual.md and payments-readme.md files COMPLETELY before proceeding. (You may print them out to read them.)
2. You can use these python libraries: pandas, numpy, scipy, scikit-learn, statsmodels, linearmodels, etc.
3. For categorical answers, use exact terminology from the manual/data
4. IF AND ONLY IF you have exhausted all possibles solution plans you can come up with and still can not find a valid answer, then provide "Not Applicable" as a final answer.
5. Your final answer should be specific and directly address the question. Do not include any other text. e.g., <answer>\n0.23\n</answer>
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


def create_dabstep_query(task: Dict[str, Any], data_paths: Dict[str, List[str]]) -> str:
    question = task["question"]
    guidelines = task.get("guidelines", "")
    
    files_list = "\n".join(data_paths['virtual'])
    
    query = f"""QUESTION: {question} {guidelines}

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
{DABSTEP_INSTRUCTIONS}
"""

    return query


@register_dataset("dabstep")
class DABStepDataset(BaseDataset):
    """DABStep dataset loader."""
    
    def __init__(self, tasks_dir: Optional[str] = None, virtual_data_root: Optional[str] = None, **kwargs):
        """
        Initialize DABStep dataset.
        
        Args:
            tasks_dir: Directory containing DABStep task files
            virtual_data_root: Root path for virtual/docker paths (default: "/data")
            **kwargs: Additional configuration
        """
        if tasks_dir is None:
            tasks_dir = str(get_task_path("DABStep"))
        super().__init__(data_dir=tasks_dir, virtual_data_root=virtual_data_root, **kwargs)
        self.tasks_dir = tasks_dir
    
    
    def load(
        self, 
        split: str = "dev", 
        limit: Optional[int] = None,
        level_filter: Optional[str] = None,
        start_index: int = 0,
        dataset_type: str = "original",
        synthetic_dataset_path: Optional[str] = None,
        filtered_dataset_path: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load DABStep dataset (original, synthetic, or filtered).
        
        Args:
            split: Dataset split (dev/all)
            limit: Maximum number of samples to load
            level_filter: Filter by difficulty level (easy/hard)
            start_index: Starting index for data selection
            dataset_type: Type of dataset (original/synthetic/filtered)
            synthetic_dataset_path: Path to synthetic dataset JSONL file
            filtered_dataset_path: Path to filtered dataset JSONL file
            **kwargs: Additional loading parameters
            
        Returns:
            List of dataset samples
        """
        # Load tasks based on dataset type
        if dataset_type == "synthetic":
            if not synthetic_dataset_path:
                raise ValueError("synthetic_dataset_path is required when dataset_type='synthetic'")
            validate_file_exists(synthetic_dataset_path, "Synthetic DABStep dataset")
            tasks = load_jsonl(synthetic_dataset_path)
        elif dataset_type == "filtered":
            if not filtered_dataset_path:
                raise ValueError("filtered_dataset_path is required when dataset_type='filtered'")
            validate_file_exists(filtered_dataset_path, "Filtered DABStep dataset")
            tasks = load_jsonl(filtered_dataset_path)
        else:  # dataset_type == "original"
            task_file = os.path.join(self.tasks_dir, f"{split}.jsonl")
            validate_file_exists(task_file, f"DABStep {split} task file")
            tasks = load_jsonl(task_file)
        
        # Filter by level if specified
        if level_filter:
            tasks = [task for task in tasks if task.get("level") == level_filter]
        
        # Apply start_index and limit
        tasks = apply_limit_and_start(
            tasks, limit, start_index, 
            random_sample=False, 
            random_seed=self.config.get('random_seed', 42)
        )
        
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
            
            # Create extra_info with appropriate source information
            extra_info = {
                "question": task.get("question", ""),
                "source": f"dabstep_{dataset_type}",
                "metadata_id": task.get("task_id", ""),
                "query_id": task.get("task_id", ""),
                "index": original_idx,
                "data_files": data_paths,
            }
            
            # Add additional information for synthetic and filtered datasets
            if dataset_type in ["synthetic", "filtered"]:
                extra_info.update({
                    "original_task_id": task.get("original_task_id", ""),
                    "original_question": task.get("original_question", ""),
                    "original_answer": task.get("original_answer", ""),
                    "generation_method": task.get("generation_method", ""),
                })
            
            # Create standardized sample
            standard_sample = create_standard_task(
                prompt_content=create_dabstep_query(task, data_paths),
                ground_truth=str(task.get("answer", "")),
                extra_info=extra_info,
                system_prompt=SYSTEM_PROMPT
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