"""
DAEval dataset loader.
"""

import os
import json
from typing import List, Dict, Any, Optional

from ..base import BaseDataset
from ..registry import register_dataset
from ..utils import apply_limit_and_start, validate_file_exists, create_standard_task, construct_data_paths
from ..prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_DEEPANALYZE
from dsgym.datasets.config import get_task_path, RAW_DATA_DIR


DAEVAL_DEMONS = r"""Format:
@shapiro_wilk_statistic[test_statistic]
@shapiro_wilk_p_value[p_value]
where "test_statistic" is a number between 0 and 1 representing the Shapiro-Wilk test statistic. Rounding off the answer to two decimal places.
where "p_value" is a number between 0 and 1 representing the p-value from the Shapiro-Wilk test. Rounding off the answer to four decimal places.

Final Answer:
@shapiro_wilk_statistic[0.56]
@shapiro_wilk_p_value[0.0002]   


Format:
@total_votes_outliers_num[outlier_num]
where "outlier_num" is an integer representing the number of values considered outliers in the 'total_votes' column.

Final Answer:
@total_votes_outliers[10]   
"""

DAEVAL_REFORMAT_TEMPLATE = """Your final answer should strictly follow the output requirements in the Format part. 
Your answer should contain all the \"@answer_name[answer]\" in the order mentioned, each \"answer\" should be in the range of value as required. 
Here're some examples: 
{demons}. 
The format requirements of this question is:
{format}."""


def create_daeval_prompt(question: str, context: str, data_paths: Dict[str, List[str]], metadata: Dict[str, Any]) -> str:
    virtual_paths = data_paths['virtual']
    dataset_locations = '\n'.join(virtual_paths) if virtual_paths else 'No dataset provided'
    
    keywords = metadata.get('keywords', [])
    constraints = metadata.get('constraints', '')
    output_format = metadata.get('format', '')
    daeval_instructions = DAEVAL_REFORMAT_TEMPLATE.format(demons=DAEVAL_DEMONS, format=output_format)
    
    metadata_section = f"""Keywords: {', '.join(keywords)}"""
    
    prompt = f"""QUESTION: {question}\n{constraints}\n

{metadata_section}

DATASET INFORMATION:
{context}

DATASET LOCATIONS (use full paths):
{dataset_locations}

INSTRUCTIONS:
{daeval_instructions}
"""
    return prompt


@register_dataset("daeval")
class DAEvalDataset(BaseDataset):
    """DAEval dataset loader."""
    
    def __init__(self, daeval_path: Optional[str] = None, virtual_data_root: Optional[str] = None, **kwargs):
        """
        Initialize DAEval dataset.
        
        Args:
            daeval_path: Path to DAEval JSON file
            virtual_data_root: Root path for virtual/docker paths (default: "/data")
            **kwargs: Additional configuration
        """
        if daeval_path is None:
            daeval_path = str(get_task_path("daeval") / "daeval.json")
        super().__init__(virtual_data_root=virtual_data_root, **kwargs)
        self.daeval_path = daeval_path
    
    def load(
        self, 
        limit: Optional[int] = None, 
        start_index: int = 0,
        random_sample: bool = False,
        dataset_type: str = "original",
        synthetic_dataset_path: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load DAEval dataset. Only validation dataset is available online. (original or synthetic).
        Source: https://github.com/InfiAgent/InfiAgent/tree/main
        
        Args:
            limit: Maximum number of samples to load
            start_index: Starting index for data selection
            random_sample: If True, randomly sample from dataset
            dataset_type: Type of dataset (original/synthetic)
            synthetic_dataset_path: Path to synthetic dataset JSON file
            **kwargs: Additional loading parameters
            
        Returns:
            List of dataset samples
        """
        # Load data based on dataset type
        if dataset_type == "synthetic":
            if not synthetic_dataset_path:
                raise ValueError("synthetic_dataset_path is required when dataset_type='synthetic'")
            validate_file_exists(synthetic_dataset_path, "Synthetic DAEval dataset")
            
            # Load synthetic dataset
            with open(synthetic_dataset_path, 'r', encoding='utf-8') as f:
                synthetic_data = json.load(f)

            # Load original DAEval for metadata lookup (if needed)
            validate_file_exists(self.daeval_path, "Original DAEval file")
            with open(self.daeval_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Apply start_index and limit to synthetic data
            raw_data = apply_limit_and_start(
                synthetic_data, limit, start_index, random_sample,
                self.config.get('random_seed', 42)
            )
            
        else:  # original dataset
            validate_file_exists(self.daeval_path, "DAEval file")
            
            with open(self.daeval_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Apply start_index and limit
            raw_data = apply_limit_and_start(
                original_data, limit, start_index, random_sample,
                self.config.get('random_seed', 42)
            )
        
        samples = []
        for idx, item in enumerate(raw_data):
            original_index = start_index + idx

            if dataset_type == "synthetic":
                # For synthetic data, extract information from the different structure
                question = item['query']
                
                # Extract data, keywords, and format from extra_info
                extra_info_data = item['extra_info']
                raw_paths = extra_info_data.get('data', [])
                keywords = extra_info_data.get('keywords', [])
                format_str = extra_info_data.get('format', '')
                context = extra_info_data.get('context', '')
                level = extra_info_data.get('level', 'unknown')
                constraints = extra_info_data.get('constraints', '')
                
                relative_paths = []
                for path in raw_paths:
                    if path.startswith('/data/daeval/'):
                        relative_paths.append(path.replace('/data/daeval/', ''))
                    else:
                        relative_paths.append(path)
                
                data_paths = construct_data_paths(
                    relative_paths=relative_paths,
                    dataset_name='daeval',
                    data_root=RAW_DATA_DIR,
                    virtual_data_root=self.virtual_data_root
                )

                # Create metadata dict for synthetic data
                metadata = {
                    'keywords': keywords,
                    'format': format_str,
                    'constraints': constraints,
                    'original_id': extra_info_data.get('synthetic_id', f'syn_{original_index}'),
                    'level': level
                }
                
                # Create prompt using centralized function
                prompt_content = create_daeval_prompt(
                    question, 
                    context, 
                    data_paths, 
                    metadata
                )
                
                ground_truth = item['ground_truth']

                # Create extra_info for synthetic data
                extra_info = {
                    'question': question,
                    'answer': ground_truth,
                    'data_files': data_paths,
                    'synthetic_id': extra_info_data.get('synthetic_id', ''),
                    'original_question': extra_info_data.get('original_question', ''),
                    'original_answer': extra_info_data.get('original_answer', ''),
                    'generation_method': extra_info_data.get('generation_method', ''),
                    'constraints': constraints,
                    'format': format_str,
                    'level': level,
                    'keywords': keywords,
                    'context': context,
                    'index': original_index,
                    'source': 'daeval_synthetic',
                    'dataset_source': extra_info_data.get('dataset_source', 'daeval'),
                    # Add fields needed for sample_id generation
                    'metadata_id': extra_info_data.get('synthetic_id', f'syn_{original_index}'),
                    'query_id': extra_info_data.get('synthetic_id', f'syn_{original_index}'),
                    'id': extra_info_data.get('synthetic_id', f'syn_{original_index}')
                }
                
            else:  # original dataset
                # Extract information from item
                data_description = item['context']
                question = item['question']
                raw_paths = item['data']
                
                relative_paths = []
                for path in raw_paths:
                    if path.startswith('/data/daeval/'):
                        relative_paths.append(path.replace('/data/daeval/', ''))
                    else:
                        relative_paths.append(path)
                
                data_paths = construct_data_paths(
                    relative_paths=relative_paths,
                    dataset_name='daeval',
                    data_root=RAW_DATA_DIR,
                    virtual_data_root=self.virtual_data_root
                )

                # Create prompt using centralized function
                prompt_content = create_daeval_prompt(
                    question, 
                    data_description, 
                    data_paths, 
                    item['metadata']
                )
                
                ground_truth = str(item['answer'])

                # Create extra_info
                extra_info = {
                    'question': item['question'],
                    'answer': item['answer'],
                    'data_files': data_paths,
                    'original_id': item['metadata']['original_id'],
                    'constraints': item['metadata']['constraints'],
                    'format': item['metadata']['format'],
                    'level': item['metadata']['level'],
                    'keywords': item.get('metadata', {}).get('keywords', []),
                    'index': original_index,
                    'source': 'daeval_original',
                    # Add fields needed for sample_id generation
                    'metadata_id': str(item['metadata']['original_id']),
                    'query_id': str(item['metadata']['original_id']),
                    'id': str(item['metadata']['original_id'])
                }
            
            # Create standardized sample
            standard_sample = create_standard_task(
                prompt_content=prompt_content,
                ground_truth=ground_truth,
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
        """Get DAEval dataset metadata."""
        if self._metadata is None:
            self._metadata = {
                'name': 'DAEval',
                'description': 'Dataset for data analysis evaluation with specific output format requirements',
                'daeval_path': self.daeval_path,
                'format': 'json',
                'splits': ['original', 'synthetic'],
                'fields': ['question', 'context', 'data', 'answer', 'metadata'],
                'source': 'daeval',
                'reference': 'https://github.com/InfiAgent/InfiAgent/tree/main'
            }
        
        return self._metadata
    
    def get_metrics(self) -> List[str]:
        """Get metrics for DAEval dataset."""
        return ["list_match"]
    
    def get_metric_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get metric configurations for DAEval dataset."""
        return {
            "list_match": {}
        }