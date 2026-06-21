"""
QRData dataset loader.
"""

import os
import json
from typing import List, Dict, Any, Optional

from ..base import BaseDataset
from ..registry import register_dataset
from ..utils import load_jsonl, apply_limit_and_start, validate_file_exists, create_standard_task, construct_data_paths
from ..prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_DEEPANALYZE
from dsgym.datasets.config import get_task_path, RAW_DATA_DIR


QRDATA_INSTRUCTIONS = """
1. Load and explore the provided datasets using Python.
2. Consider useful python libraries such as pandas, numpy, scipy, scikit-learn, statsmodels, dowhy, econml, causalml, linearmodels, networkx, etc.
3. Apply appropriate statistical methods or analysis techniques to answer the research question.
4. Generate a final answer that directly addresses the question.
5. Your final answer should be specific and directly address the question. Do not include any other text. e.g., <answer>0.23</answer>
"""


def create_qrdata_prompt(question: str, context: str, data_paths: Dict[str, List[str]], metadata: Dict[str, Any]) -> str:
    virtual_paths = data_paths['virtual']
    dataset_locations = '\n'.join(virtual_paths) if virtual_paths else 'No dataset provided'
    
    keywords = metadata.get('keywords', [])
    question_type = metadata.get('question_type', 'unknown')
    reference = metadata.get('reference', 'N/A')
    
    metadata_section = f"""Keywords: {', '.join(keywords)}
Question Type: {question_type}
Reference: {reference}"""
    
    prompt = f"""QUESTION: {question}

{metadata_section}

DATASET INFORMATION:
{context}

DATASET LOCATIONS (use full paths):
{dataset_locations}

INSTRUCTIONS:
{QRDATA_INSTRUCTIONS}
"""
    return prompt


@register_dataset("qrdata")
class QRDataDataset(BaseDataset):
    """QRData dataset loader."""
    
    def __init__(self, qrdata_path: Optional[str] = None, virtual_data_root: Optional[str] = None, **kwargs):
        """
        Initialize QRData dataset.
        
        Args:
            qrdata_path: Path to QRData JSON file
            virtual_data_root: Root path for virtual/docker paths (default: "/data")
            **kwargs: Additional configuration
        """
        if qrdata_path is None:
            qrdata_path = str(get_task_path("QRData") / "qrdata.json")
        super().__init__(virtual_data_root=virtual_data_root, **kwargs)
        self.qrdata_path = qrdata_path
    
    def load(
        self, 
        limit: Optional[int] = None, 
        random_sample: bool = False,
        dataset_type: str = "original",
        synthetic_dataset_path: Optional[str] = None,
        start_index: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load QRData dataset (original or synthetic).
        
        Args:
            limit: Maximum number of samples to load
            random_sample: If True, randomly sample from dataset
            dataset_type: Type of dataset (original/synthetic)
            synthetic_dataset_path: Path to synthetic dataset JSONL file
            start_index: Starting index for data selection
            **kwargs: Additional loading parameters
            
        Returns:
            List of dataset samples
        """
        # Load data based on dataset type
        if dataset_type == "synthetic":
            if not synthetic_dataset_path:
                raise ValueError("synthetic_dataset_path is required when dataset_type='synthetic'")
            validate_file_exists(synthetic_dataset_path, "Synthetic QRData dataset")
            
            # Load synthetic dataset
            with open(synthetic_dataset_path, 'r', encoding='utf-8') as f:
                synthetic_data = json.load(f)

            # Load original QRData for metadata lookup
            validate_file_exists(self.qrdata_path, "Original QRData file")
            with open(self.qrdata_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Create mapping of original questions to metadata
            original_metadata_map = {}
            for item in original_data:
                raw_paths = item['data']
                relative_paths = []
                for path in raw_paths:
                    if path.startswith('/data/qrdata/data/'):
                        relative_paths.append(path.replace('/data/qrdata/data/', ''))
                    else:
                        relative_paths.append(path)
                
                data_paths = construct_data_paths(
                    relative_paths=relative_paths,
                    dataset_name='qrdata/data',
                    data_root=RAW_DATA_DIR,
                    virtual_data_root=self.virtual_data_root
                )
                
                original_metadata_map[item['question']] = {
                    'context': item['context'],
                    'data': data_paths,
                    'metadata': item['metadata'],
                    'answer': item['answer']
                }
            
            # Apply start_index and limit to synthetic data
            raw_data = apply_limit_and_start(
                synthetic_data, limit, start_index, random_sample,
                self.config.get('random_seed', 42)
            )
            
        else:  # original dataset
            validate_file_exists(self.qrdata_path, "QRData file")
            
            with open(self.qrdata_path, 'r', encoding='utf-8') as f:
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
                # For synthetic data, use original question to get metadata
                question = item['question']
                original_question = item['extra_info'].get('original_question', '')
                
                # Look up original metadata
                if original_question in original_metadata_map:
                    original_info = original_metadata_map[original_question]
                    context = original_info['context']
                    data_paths = original_info['data']
                    metadata = original_info['metadata']
                    original_answer = original_info['answer']
                    
                    # Create prompt using new format
                    prompt_content = create_qrdata_prompt(question, context, data_paths, metadata)
                    
                    # For synthetic tasks, answer is empty (to be evaluated)
                    reference_answer = item['answer']
                    
                    ground_truth = reference_answer
                    extra_info = {
                        'question': item['question'],
                        'synthetic_id': item['extra_info'].get('synthetic_id', ''),
                        'original_question': original_question,
                        "original_answer": original_answer,
                        'generation_method': item.get('generation_method', ''),
                        'reference_answer': reference_answer,
                        'context': context,
                        'data_files': data_paths,
                        'question_type': metadata.get('question_type', 'unknown'),
                        'keywords': metadata.get('keywords', []),
                        'index': original_index,
                        'source': 'qrdata_synthetic',
                        'dataset_name': 'qrdata_synthetic',
                        'metadata_id': item.get('task_id', f"qrdata_syn_{original_index}"),
                        'query_id': item.get('task_id', f"q_syn_{original_index}"),
                    }
                else:
                    continue  # Skip if no original metadata found
                
            else:  # original dataset
                data_description = item['context']
                question = item['question']
                raw_paths = item['data']
                
                relative_paths = []
                for path in raw_paths:
                    if path.startswith('/data/qrdata/data/'):
                        relative_paths.append(path.replace('/data/qrdata/data/', ''))
                    else:
                        relative_paths.append(path)
                
                data_paths = construct_data_paths(
                    relative_paths=relative_paths,
                    dataset_name='qrdata/data',
                    data_root=RAW_DATA_DIR,
                    virtual_data_root=self.virtual_data_root
                )
                
                # Create prompt using new format
                prompt_content = create_qrdata_prompt(
                    question, 
                    data_description, 
                    data_paths, 
                    item['metadata']
                )
                
                ground_truth = str(item['answer'])
                
                extra_info = {
                    'question': item['question'],
                    'answer': item['answer'],
                    'context': data_description,
                    'data_files': data_paths,
                    'question_type': item.get('metadata', {}).get('question_type', 'unknown'),
                    'keywords': item.get('metadata', {}).get('keywords', []),
                    'reference': item.get('metadata', {}).get('reference', ''),
                    'index': original_index,
                    'source': 'qrdata_original',
                    'dataset_name': 'qrdata_original',
                    'metadata_id': f"qrdata_{original_index}",
                    'query_id': f"{original_index}"
                }
            
            # Create standardized sample
            standard_sample = create_standard_task(
                prompt_content=prompt_content,
                ground_truth=ground_truth,
                extra_info=extra_info,
                system_prompt=SYSTEM_PROMPT_DEEPANALYZE
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
        """Get QRData dataset metadata."""
        if self._metadata is None:
            self._metadata = {
                'name': 'QRData',
                'description': 'Question-reasoning dataset for data analysis',
                'qrdata_path': self.qrdata_path,
                'format': 'json',
                'splits': ['original', 'synthetic'],
                'fields': ['question', 'context', 'data', 'answer', 'metadata'],
                'source': 'qrdata'
            }
        
        return self._metadata
    
    def get_metrics(self) -> List[str]:
        """Get metrics for QRData dataset."""
        return ["exact_match"]
    
    def get_metric_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get metric configurations for QRData dataset."""
        return {
            "exact_match": {
                "numeric_tolerance": 0.03
            }
        }