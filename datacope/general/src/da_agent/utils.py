import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field

from src.core.schema import EvaluationConfig, EvaluationResult

class ModelOutput(BaseModel):
    """The schema exposed to the model through StructuredOutput."""
    reasoning: str
    answer: str

def extract_sample_info(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract standardized information from a sample.
    Handles cases where ground truth might be missing or empty.
    
    Args:
        sample: Sample data dictionary
        
    Returns:
        Dictionary with standardized fields
    """
    query = ""
    ground_truth = None  # Start with None
    dataset_name = ""
    metadata_id = ""
    query_id = ""
    
    # Extract ground truth from various possible locations
    if "reward_spec" in sample and "ground_truth" in sample["reward_spec"]:
        gt = sample["reward_spec"]["ground_truth"]
        ground_truth = gt if gt and str(gt).strip() else None
    else:
        gt = sample.get("ground_truth", sample.get("answer", ""))
        ground_truth = gt if gt and str(gt).strip() else None
    
    if "extra_info" in sample:
        extra_info = sample["extra_info"]
        query = extra_info.get("query", extra_info.get("question", ""))
        dataset_name = extra_info.get("source", extra_info.get("dataset", extra_info.get("dataset_name", "")))
        metadata_id = str(extra_info.get("metadata_id", ""))
        query_id = str(extra_info.get("query_id", extra_info.get("id", "")))
    
    if not query and "prompt" in sample and isinstance(sample["prompt"], list):
        for msg in sample["prompt"]:
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = msg.get("content", "")
                break
    
    if not query:
        query = sample.get("query", "")
    if not dataset_name:
        dataset_name = sample.get("dataset", sample.get("dataset_name", ""))
    if not metadata_id:
        metadata_id = str(sample.get("metadata_id", ""))
    if not query_id:
        query_id = str(sample.get("query_id", sample.get("id", "")))
        
    return {
        "query": query, 
        "ground_truth": ground_truth,  # Can be None
        "dataset_name": dataset_name, 
        "metadata_id": metadata_id, 
        "query_id": query_id
    }

def save_evaluation_results(
    results: List[EvaluationResult],
    config: EvaluationConfig,
    output_dir: Union[str, Path],
    run_name: str,
    additional_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Save evaluation results to files.
    
    Args:
        results: List of evaluation results
        config: Evaluation configuration
        output_dir: Output directory
        run_name: Name for this evaluation run
        additional_metrics: Additional metrics to include
        
    Returns:
        Dictionary mapping file types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Save detailed results (with raw_response and trajectory)
    results_file = output_dir / f"{run_name}_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
    file_paths["results"] = str(results_file)
    
    # Save summary results (without raw_response and trajectory for readability)
    summary_file = output_dir / f"{run_name}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_summary_dict() for r in results], f, indent=2, ensure_ascii=False)
    file_paths["summary"] = str(summary_file)
    
    # Save configuration
    config_file = output_dir / f"{run_name}_config.json"
    config_dict = config.to_dict()
    config_dict["timestamp"] = datetime.now().isoformat()
    config_dict["total_samples"] = len(results)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    file_paths["config"] = str(config_file)
    
    # Save summary CSV for easy analysis
    import pandas as pd
    summary_data = []
    for result in results:
        summary_row = {
            "sample_id": result.sample_id,
            "dataset_name": result.dataset_name,
            "query": result.query[:100] + "..." if len(result.query) > 100 else result.query,
            "ground_truth": result.ground_truth if result.has_ground_truth else "N/A",
            "prediction": result.prediction,
            "success": result.success,
            "execution_time": result.execution_time,
            "total_turns": result.total_turns,
            "has_ground_truth": result.has_ground_truth,
        }

        
        summary_data.append(summary_row)
    
    summary_file = output_dir / f"{run_name}_summary.csv"
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    file_paths["summary"] = str(summary_file)
    
    return file_paths
