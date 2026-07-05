"""
DiscoveryBench dataset loader.
"""

import os
from typing import List, Dict, Any, Optional
import json

from ..base import BaseDataset
from ..registry import register_dataset
from ..utils import apply_limit_and_start, validate_file_exists, create_standard_task, construct_data_paths
from ..prompts import SYSTEM_PROMPT
from dsgym.datasets.config import get_task_path, RAW_DATA_DIR


DISCOVERY_INSTRUCTIONS = """
1. Load and explore the provided datasets using Python.
2. Understand the domain-specific context and workflow requirements if provided.
3. If multiple datasets are provided, analyze each one to find out what is relevant and consider how they relate to each other.
4. If the workflow tags are provided (e.g., regression, etc.), use them to guide your analysis.
5. Generate scientific hypotheses derived from the provided dataset that addresses the research question in the format: <answer>your final hypothesis</answer>. Your final hypothesis should be one or several concise but complete sentences.
6. In your final hypothesis, you MUST clearly state the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any) including any statistical significance. 
Definitions of contexts, variables, and relations:
E.g., A final hypothesis could be "From 1995 to 2009, the number of sandhill cranes around the tundra (Indigilka River) surged by an astounding ~10X":
    * Contexts refer to stratification of the data under which the given hypothesis is True. E.g., "For all women", "From 1995 to 2009".
    * Variables refer to the set of variables (either dependent or independent) that are mentioned in the hypothesis. E.g., number of sandhill cranes, location.
    * Relations refer to the form of relation between the variables. E.g., "surged by ~10x".
"""


def create_discovery_prompt(question: str, context: str, data_paths: Dict[str, List[str]], metadata: Dict[str, Any]) -> str:
    virtual_paths = data_paths['virtual']
    dataset_locations = '\n'.join(virtual_paths) if virtual_paths else 'No dataset provided'
    
    domain_knowledge = metadata['domain_knowledge']
    workflow_tags = metadata['workflow_tags']
    columns_info = metadata['columns_info']
    
    metadata_lines = []
    
    metadata_lines.append("COLUMNS:")
    metadata_lines.append(columns_info)
    
    if domain_knowledge:
        metadata_lines.append(f"DOMAIN KNOWLEDGE: {domain_knowledge}")
    if workflow_tags:
        metadata_lines.append(f"WORKFLOW TAGS: {workflow_tags}")
    
    if 'workflow' in metadata:
        workflow = metadata['workflow']
        metadata_lines.append("WORKFLOW:")
        metadata_lines.append(workflow)
    
    metadata_section = '\n\n'.join(metadata_lines)
    
    prompt = f"""QUESTION: {question}

{metadata_section}

DATASET INFORMATION:
{context}

DATASET LOCATIONS (use full paths):
You do not have any access to the dataset for this question.

INSTRUCTIONS:
{DISCOVERY_INSTRUCTIONS}
"""
    return prompt


@register_dataset("discoverybench")
class DiscoveryBenchDataset(BaseDataset):
    """DiscoveryBench dataset loader."""
    
    def __init__(self, data_dir: Optional[str] = None, virtual_data_root: Optional[str] = None, **kwargs):
        """
        Initialize DiscoveryBench dataset.
        
        Args:
            data_dir: Directory containing DiscoveryBench parquet files
            virtual_data_root: Root path for virtual/docker paths (default: "/data")
            **kwargs: Additional configuration
        """
        if data_dir is None:
            data_dir = str(get_task_path("discovery"))
        super().__init__(data_dir=data_dir, virtual_data_root=virtual_data_root, **kwargs)
    
    def load(
        self, 
        split: str = "test", 
        limit: Optional[int] = None,
        start_index: int = 0,
        random_sample: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load DiscoveryBench dataset.
        
        Args:
            split: Dataset split (train/validation/test)
            limit: Maximum number of samples to load
            start_index: Starting index for data selection
            random_sample: If True, randomly sample from dataset
            **kwargs: Additional loading parameters
            
        Returns:
            List of dataset samples
        """
        # Determine file path
        if split == "test":
            file_path = os.path.join(self.data_dir, "discoverybench_test.json")
        elif split == "train":
            file_path = os.path.join(self.data_dir, f"discoverybench.json")
        else:
            raise ValueError(f"Invalid split: {split}")

        # Validate file exists
        validate_file_exists(file_path, f"DiscoveryBench {split} dataset")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # Apply start_index and limit
        raw_data = apply_limit_and_start(
            original_data, limit, start_index, random_sample,
            self.config.get('random_seed', 42)
        )

        samples = []
        
        # Convert to standard format
        samples = []
        for idx, item in enumerate(raw_data):
            original_index = start_index + idx
            
            data_description = item['context']
            question = item['question']
            raw_paths = item['data']
            
            relative_paths = []
            for path in raw_paths:
                if path.startswith('/data/discoverybench/'):
                    relative_paths.append(path.replace('/data/discoverybench/', ''))
                else:
                    relative_paths.append(path)
            
            data_paths = construct_data_paths(
                relative_paths=relative_paths,
                dataset_name='discoverybench',
                data_root=RAW_DATA_DIR,
                virtual_data_root=self.virtual_data_root
            )

            # Create prompt using new format
            prompt_content = create_discovery_prompt(
                question, 
                data_description, 
                data_paths, 
                item['metadata']
            )

            ground_truth = str(item['answer'])
            
            extra_info = {
                'question': question,
                'answer': ground_truth,
                'data_files': data_paths,
                'index': original_index,
                'source': 'discoverybench',
                'metadata_id': str(original_index),
                'query_id': str(original_index),
                'id': str(original_index)
            }
            
            # Create standardized task
            standard_task = create_standard_task(
                prompt_content=prompt_content,
                ground_truth=ground_truth,
                extra_info=extra_info,
                system_prompt=SYSTEM_PROMPT
            )
            
            samples.append(standard_task)
        
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
        """Get DiscoveryBench dataset metadata."""
        if self._metadata is None:
            self._metadata = {
                'name': 'DiscoveryBench',
                'description': 'Dataset for evaluating discovery capabilities in data science',
                'data_dir': self.data_dir,
                'format': 'json',
                'splits': ['train', 'test'],
                'fields': ['query', 'ground_truth'],
                'source': 'discovery_bench'
            }
        
        return self._metadata
    
    def get_metrics(self) -> List[str]:
        """Get metrics for DiscoveryBench dataset."""
        return ["llm_score", "hms_score"]
    
    def get_metric_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get metric configurations for DiscoveryBench dataset."""
        return {
            "llm_score": {
                "model": "gpt-4o"
            },
            "hms_score": {
                "model": "gpt-4o"
            }
        }