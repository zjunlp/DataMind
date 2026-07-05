"""
Common utilities for dataset handling.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries from JSONL
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def apply_limit_and_start(
    data: List[Any], 
    limit: Optional[int], 
    start_index: int = 0, 
    random_sample: bool = False,
    random_seed: int = 42
) -> List[Any]:
    """
    Apply start_index and limit to data list.
    
    Args:
        data: Input data list
        limit: Maximum number of samples to return
        start_index: Starting index for data selection
        random_sample: If True, use random sampling
        random_seed: Random seed for reproducible sampling
        
    Returns:
        Limited data list
    """
    # Apply start_index first
    if start_index > 0:
        if start_index >= len(data):
            return []
        data = data[start_index:]
    
    # Then apply limit
    if limit is None:
        return data
    
    if limit >= len(data):
        return data
    
    if random_sample:
        np.random.seed(random_seed)
        indices = np.random.choice(len(data), size=limit, replace=False)
        return [data[i] for i in sorted(indices)]
    else:
        return data[:limit]


def validate_file_exists(file_path: str, description: str = "File"):
    """
    Validate that a file exists, raise informative error if not.
    
    Args:
        file_path: Path to check
        description: Description for error message
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found: {file_path}")


def construct_data_paths(
    relative_paths: List[str],
    dataset_name: str,
    data_root: Path,
    virtual_data_root: str = ""
) -> Dict[str, List[str]]:
    """
    Construct three-path structure for dataset files to support different execution environments.
    
    This function generates three variants of file paths for each data file:
    - relative: Path relative to dataset root (preserves subdirectories)
    - absolute: Full local filesystem path (for finding files locally)
    - virtual: Configurable path for agent prompts (varies by execution environment)
    
    Args:
        relative_paths: List of relative paths from dataset root. Subdirectories are preserved.
                       Example: ['file.csv', 'real/test/data.csv']
        dataset_name: Name of the dataset (e.g., 'bio', 'discoverybench', 'qrdata/data')
        data_root: Absolute path to data root directory (typically RAW_DATA_DIR)
        virtual_data_root: Root path for virtual/agent paths. Controls what the agent sees:
                          - "" or None: Use absolute local paths (default, for local execution)
                          - "/data": Docker-style paths (e.g., /data/bio/file.csv)
                          - "/workspace": Custom mount point (e.g., /workspace/bio/file.csv)
        
    Returns:
        Dictionary with three keys:
        - 'relative': Original relative paths (unchanged)
        - 'absolute': Full local paths (e.g., /Users/.../DSGym/data/data/bio/file.csv)
        - 'virtual': Paths for agent prompts (format depends on virtual_data_root)
        
    Examples:
        >>> construct_data_paths(['file.csv'], 'bio', Path('/data'), '')
        {
            'relative': ['file.csv'],
            'absolute': ['/data/bio/file.csv'],
            'virtual': ['/data/bio/file.csv']
        }
        
        >>> construct_data_paths(['file.csv'], 'bio', Path('/data'), '/docker')
        {
            'relative': ['file.csv'],
            'absolute': ['/data/bio/file.csv'],
            'virtual': ['/docker/bio/file.csv']
        }
    """
    absolute_paths = []
    virtual_paths = []
    
    for rel_path in relative_paths:
        abs_path = str(data_root / dataset_name / rel_path)
        absolute_paths.append(abs_path)
        
        if virtual_data_root:
            if dataset_name:
                virtual_path = f"{virtual_data_root}/{dataset_name}/{rel_path}"
            else:
                virtual_path = f"{virtual_data_root}/{rel_path}"
        else:
            virtual_path = abs_path
        virtual_paths.append(virtual_path)
    
    return {
        'relative': relative_paths,
        'absolute': absolute_paths,
        'virtual': virtual_paths
    }


def create_standard_task(
    prompt_content: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
    system_prompt: str
) -> Dict[str, Any]:
    """
    Create a standard task format for DSGym.
    
    Args:
        prompt_content: Main prompt content
        ground_truth: Expected answer
        extra_info: Additional metadata
        system_prompt: System prompt to use
        
    Returns:
        Standardized task dictionary
    """
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content}
        ],
        "ground_truth": ground_truth,
        "reward_spec": {
            "ground_truth": ground_truth,
        },
        "original_index": extra_info.get("index", 0),
        "extra_info": extra_info,
    }


def create_custom_task(
    query: str,
    data_files: Optional[List[str]] = None,
    context: Optional[str] = None,
    ground_truth: Optional[str] = None,
    **metadata
) -> Dict[str, Any]:
    """
    Create a custom task using default template.
    
    Args:
        query: Question/query to analyze
        data_files: List of data file paths
        context: Background information
        ground_truth: Expected answer (optional)
        **metadata: Additional metadata
        
    Returns:
        Standardized task dictionary
        
    Example:
        task = create_custom_task(
            query="Analyze sales trends",
            data_files=["/path/to/sales.csv"],
            context="E-commerce data from 2023"
        )
    """
    import time
    from .prompts import SYSTEM_PROMPT
    
    # Generate task ID
    task_id = f"custom_{int(time.time())}"
    
    # Build prompt using default template
    data_paths_str = ""
    if data_files:
        data_paths_str = "DATASET LOCATIONS:\n" + "\n".join(data_files)
    
    context_str = f"CONTEXT:\n{context}\n\n" if context else ""
    
    # Default instruction template
    prompt_content = f"""QUESTION: {query}

{context_str}{data_paths_str}

INSTRUCTIONS:
1. Analyze the provided data to answer the question.
2. Use Python for all computations and analysis.
3. Provide your final answer in the format: <answer>your answer</answer>
4. Show your reasoning process step by step."""
    
    # Build extra_info
    extra_info = {
        'question': query,
        'context': context,
        'data_files': data_files or [],
        'index': 0,
        'source': 'custom',
        'metadata_id': task_id,
        'query_id': task_id,
        'id': task_id,
        **metadata
    }
    
    return create_standard_task(
        prompt_content=prompt_content,
        ground_truth=ground_truth,
        extra_info=extra_info,
        system_prompt=SYSTEM_PROMPT
    )


def load_tasks_from_dataset(
    dataset_name: str,
    indices: Optional[List[int]] = None,
    limit: Optional[int] = None,
    start_index: int = 0
) -> List[Dict[str, Any]]:
    """
    Load tasks from a dataset.
    
    Args:
        dataset_name: Name of the dataset
        indices: Specific indices to load (takes precedence over limit/start_index)
        limit: Maximum number of tasks to load
        start_index: Starting index
        
    Returns:
        List of task dictionaries
        
    Examples:
        # Load specific tasks
        tasks = load_tasks_from_dataset("discoverybench", indices=[0, 5, 10])
        
        # Load range of tasks  
        tasks = load_tasks_from_dataset("discoverybench", start_index=10, limit=5)
        
        # Load single task
        tasks = load_tasks_from_dataset("discoverybench", indices=[0])
    """
    from .registry import DatasetRegistry
    
    dataset = DatasetRegistry.load(dataset_name)
    
    if indices is not None:
        # Load specific indices
        all_tasks = dataset.load()
        selected_tasks = []
        for idx in indices:
            if 0 <= idx < len(all_tasks):
                selected_tasks.append(all_tasks[idx])
            else:
                raise IndexError(f"Task index {idx} out of range [0, {len(all_tasks)})")
        return selected_tasks
    else:
        # Load range of tasks
        return dataset.load(limit=limit, start_index=start_index)