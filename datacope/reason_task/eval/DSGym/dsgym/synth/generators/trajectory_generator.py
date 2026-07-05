"""
Trajectory generator for creating multiple agent trajectories per sample.

This module provides functionality to generate k trajectories for each sample
with configurable temperature and model settings, including optional metric
computation and pass@k evaluation.
"""

import json
import os
import time
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import concurrent.futures
from dataclasses import dataclass, asdict
from tqdm import tqdm

from ...datasets import DatasetRegistry
from ...agents import ReActDSAgent
from ...eval import Evaluator
from ...eval.utils import EvaluationResult, save_evaluation_results


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""
    model: str
    backend: str = "litellm"
    temperature: float = 0.8
    k: int = 8
    max_workers: int = 24
    max_turns: int = 15
    manager_url: str = "http://localhost:5000"
    api_key: Optional[str] = None
    dataset_name: str = "daeval"
    synthetic_path: Optional[str] = None
    compute_metrics: bool = True
    output_dir: str = "./trajectory_results"
    run_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class TrajectoryGenerator:
    """
    Generator for creating multiple trajectories per sample with pass@k evaluation.
    """
    
    def __init__(self, config: TrajectoryConfig):
        """
        Initialize trajectory generator.
        
        Args:
            config: Configuration for trajectory generation
        """
        self.config = config
        
        # Automatically set max_workers=1 for single-instance vLLM and SGLang backends
        # These backends manage their own parallelism through tensor_parallel_size
        # But multi-vllm backend can handle multiple workers since it uses multiple instances
        if self.config.backend in ["vllm", "sglang"] and self.config.max_workers > 1:
            print(f"⚠️  Setting max_workers=1 for {self.config.backend} backend (was {self.config.max_workers})")
            print(f"   {self.config.backend} manages parallelism internally via tensor_parallel_size")
            self.config.max_workers = 1
        elif self.config.backend == "multi-vllm":
            print(f"✅ Using multi-vllm backend with max_workers={self.config.max_workers}")
            print(f"   multi-vllm backend supports parallel workers across GPU instances")
        
        self.dataset = None
        self.agent = None
        self.evaluator = None
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def _initialize_components(self):
        """Initialize dataset, agent, and evaluator components."""
        # Load dataset
        dataset_config = {
            "virtual_data_root": "/data"
        }
        if self.config.synthetic_path:
            dataset_config["dataset_type"] = "synthetic"
            dataset_config["synthetic_dataset_path"] = self.config.synthetic_path
            
        self.dataset = DatasetRegistry.load(self.config.dataset_name, **dataset_config)
        
        # Initialize agent with temperature
        agent_config = {
            "manager_url": self.config.manager_url,
            "max_turns": self.config.max_turns,
            "temperature": self.config.temperature,
            "output_dir": self.config.output_dir,
        }
        
        if self.config.backend == "litellm" and self.config.api_key:
            agent_config["api_key"] = self.config.api_key
            
        self.agent = ReActDSAgent(
            backend=self.config.backend,
            model=self.config.model,
            **agent_config
        )
        
        # Create evaluator if metrics computation is needed
        if self.config.compute_metrics:
            self.evaluator = Evaluator(
                protocol="multi_turn",
                dataset=self.dataset,
                parallel_workers=1  # Sequential for individual trajectory evaluation
            )
    
    def generate_trajectory(self, sample: Dict[str, Any], trajectory_id: int) -> EvaluationResult:
        """
        Generate a single trajectory for a sample.
        
        Args:
            sample: Sample to evaluate
            trajectory_id: ID of this trajectory (0-indexed)
            
        Returns:
            EvaluationResult for this trajectory
        """
        # Add trajectory_id to sample's extra_info so agent can access it
        # Use deep copy to ensure each trajectory gets its own independent copy of the sample
        sample_with_traj_id = copy.deepcopy(sample)
        if "extra_info" not in sample_with_traj_id:
            sample_with_traj_id["extra_info"] = {}
        sample_with_traj_id["extra_info"]["trajectory_id"] = trajectory_id
        
        if self.config.compute_metrics and self.evaluator:
            result = self.evaluator.evaluate_single(self.agent, sample_with_traj_id)
        else:
            # Just run the agent without metrics - create a minimal evaluator
            from ...eval import Evaluator
            temp_evaluator = Evaluator(metrics=[])
            result = temp_evaluator._evaluate_single_sample(self.agent, sample_with_traj_id)
        
        # Add trajectory ID to the result
        result.trajectory_id = trajectory_id
        
        return result
    
    def generate_trajectories_for_sample(
        self, 
        sample: Dict[str, Any], 
        sample_idx: int
    ) -> List[EvaluationResult]:
        """
        Generate k trajectories for a single sample.
        
        Args:
            sample: Sample to evaluate
            sample_idx: Index of the sample in the dataset
            
        Returns:
            List of EvaluationResult objects, one per trajectory
        """
        trajectories = []
        
        for traj_id in range(self.config.k):
            try:
                result = self.generate_trajectory(sample, traj_id)
                result.sample_id = f"{sample_idx}_traj_{traj_id}"
                trajectories.append(result)
            except Exception as e:
                # Create error result for failed trajectory
                from ...eval.utils import extract_sample_info
                sample_info = extract_sample_info(sample)
                
                error_result = EvaluationResult(
                    sample_id=f"{sample_idx}_traj_{traj_id}",
                    dataset_name=sample_info.get("dataset_name", self.config.dataset_name),
                    query=sample_info.get("query", ""),
                    ground_truth=sample_info.get("ground_truth"),
                    success=False,
                    error_info={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "error_category": "TRAJECTORY_GENERATION_ERROR"
                    }
                )
                error_result.trajectory_id = traj_id
                trajectories.append(error_result)
        
        return trajectories
    
    def _compute_sample_metrics(
        self, 
        trajectories: List[EvaluationResult]
    ) -> Dict[str, float]:
        """
        Compute both pass@k and average metrics for a single sample's trajectories.
        
        Args:
            trajectories: List of trajectory results for the sample
            
        Returns:
            Dictionary with pass@k scores and average scores for each metric
        """
        if not self.config.compute_metrics:
            return {}
        
        # Get all metric names from successful trajectories
        metric_names = set()
        for traj in trajectories:
            if traj.success and traj.metrics:
                metric_names.update(traj.metrics.keys())
        
        sample_metrics = {}
        
        for metric_name in metric_names:
            scores = []
            for traj in trajectories:
                if traj.success and metric_name in traj.metrics:
                    # Extract score from metric result
                    metric_result = traj.metrics[metric_name]
                    if isinstance(metric_result, dict) and "score" in metric_result:
                        scores.append(metric_result["score"])
                    elif isinstance(metric_result, (int, float)):
                        scores.append(metric_result)
            
            if scores:
                # Pass@k is the maximum score across all trajectories
                sample_metrics[f"{metric_name}_pass_at_{self.config.k}"] = max(scores)
                # Average is the mean score across all trajectories
                sample_metrics[f"{metric_name}_avg"] = sum(scores) / len(scores)
            else:
                sample_metrics[f"{metric_name}_pass_at_{self.config.k}"] = 0.0
                sample_metrics[f"{metric_name}_avg"] = 0.0
        
        return sample_metrics
    
    def _evaluate_single_sample_wrapper(self, args):
        """Wrapper for parallel sample evaluation."""
        sample_idx, sample = args
        return sample_idx, self.generate_trajectories_for_sample(sample, sample_idx)
    
    def generate(
        self, 
        samples: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
        start_index: int = 0,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Generate trajectories for all samples.
        
        Args:
            samples: List of samples to evaluate (loads from dataset if None)
            limit: Number of samples to process (optional)
            start_index: Starting index for sample processing (default: 0)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary containing all results and summary statistics
        """
        start_time = time.time()
        
        # Initialize components
        print("🔧 Initializing components...")
        self._initialize_components()
        
        # Load samples if not provided
        if samples is None:
            load_config = {"limit": limit} if limit else {}
            # Add synthetic dataset configuration if provided
            if self.config.synthetic_path:
                load_config["dataset_type"] = "synthetic"
                load_config["synthetic_dataset_path"] = self.config.synthetic_path
            samples = self.dataset.load(**load_config)
        
        # Apply start_index and limit
        if start_index > 0:
            samples = samples[start_index:]
        if limit:
            samples = samples[:limit]
        
        start_msg = f" (starting from index {start_index})" if start_index > 0 else ""
        print(f"📊 Generating {self.config.k} trajectories for {len(samples)} samples{start_msg}...")
        print(f"🌡️ Temperature: {self.config.temperature}")
        print(f"🤖 Model: {self.config.model}")
        print(f"⚙️ Backend: {self.config.backend}")
        print(f"👷 Max workers: {self.config.max_workers}")
        print(f"📏 Compute metrics: {self.config.compute_metrics}")
        
        # Generate trajectories
        all_trajectories = []
        sample_metrics = []
        
        if self.config.max_workers > 1:
            # Parallel processing of samples
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(self._evaluate_single_sample_wrapper, (idx + start_index, sample))
                    for idx, sample in enumerate(samples)
                ]
                
                iterator = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Generating trajectories (parallel)"
                ) if show_progress else concurrent.futures.as_completed(futures)
                
                # Collect results maintaining order
                results_dict = {}
                for future in iterator:
                    sample_idx, trajectories = future.result()
                    results_dict[sample_idx] = trajectories
                
                # Process in order
                for sample_idx in range(len(samples)):
                    actual_sample_idx = sample_idx + start_index
                    trajectories = results_dict[actual_sample_idx]
                    all_trajectories.extend(trajectories)
                    
                    # Compute metrics for this sample
                    if self.config.compute_metrics:
                        metrics = self._compute_sample_metrics(trajectories)
                        metrics["sample_id"] = sample_idx
                        sample_metrics.append(metrics)
        else:
            # Sequential processing
            iterator = tqdm(enumerate(samples), total=len(samples), desc="Generating trajectories") if show_progress else enumerate(samples)
            
            for sample_idx, sample in iterator:
                actual_sample_idx = sample_idx + start_index
                trajectories = self.generate_trajectories_for_sample(sample, actual_sample_idx)
                all_trajectories.extend(trajectories)
                
                # Compute metrics for this sample
                if self.config.compute_metrics:
                    metrics = self._compute_sample_metrics(trajectories)
                    metrics["sample_id"] = actual_sample_idx
                    sample_metrics.append(metrics)
        
        total_time = time.time() - start_time
        
        # Compute overall statistics
        overall_metrics = {}
        if self.config.compute_metrics and sample_metrics:
            # Get all metric names
            metric_names = set()
            for sample_stats in sample_metrics:
                metric_names.update(k for k in sample_stats.keys() if k != "sample_id")
            
            # Compute mean across all samples for both pass@k and average metrics
            for metric_name in metric_names:
                scores = [
                    sample_stats.get(metric_name, 0.0) 
                    for sample_stats in sample_metrics
                ]
                overall_metrics[metric_name] = sum(scores) / len(scores) if scores else 0.0
        
        # Prepare results
        results = {
            "trajectories": all_trajectories,
            "sample_metrics": sample_metrics,  # Now contains both pass@k and avg
            "overall_metrics": overall_metrics,  # Overall averages of both pass@k and avg
            "config": self.config.to_dict(),
            "total_time": total_time,
            "total_samples": len(samples),
            "total_trajectories": len(all_trajectories),
            "k": self.config.k
        }
        
        # Save results
        self._save_results(results)
        
        print(f"✅ Generated {len(all_trajectories)} trajectories in {total_time:.2f}s")
        if overall_metrics:
            print("📈 Overall Results:")
            # Separate pass@k and average results for better display
            pass_at_k_results = {k: v for k, v in overall_metrics.items() if "pass_at_" in k}
            avg_results = {k: v for k, v in overall_metrics.items() if "_avg" in k}
            
            if pass_at_k_results:
                print("  Pass@K Metrics:")
                for metric_name, score in pass_at_k_results.items():
                    print(f"    {metric_name}: {score:.3f}")
            
            if avg_results:
                print("  Average Metrics:")
                for metric_name, score in avg_results.items():
                    print(f"    {metric_name}: {score:.3f}")
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save trajectory generation results to files."""
        # Create run name if not provided
        run_name = self.config.run_name or f"trajectory_{self.config.dataset_name}_{self.config.backend}_{self.config.model.replace('/', '_')}_k{self.config.k}_temp{self.config.temperature}"
        
        # Note: Individual trajectory predictions are now automatically saved as 
        # prediction_i_traj_j.json by the agent during execution.
        # We only need to save the combined analysis file here.
        
        # Save all trajectories in one combined file for analysis
        predictions_dir = Path(self.config.output_dir) / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        all_predictions_file = predictions_dir / f"{run_name}_all.json"
        all_trajectory_data = []
        for traj in results["trajectories"]:
            traj_dict = {
                "sample_id": traj.sample_id,
                "trajectory_id": getattr(traj, "trajectory_id", 0),
                "dataset_name": traj.dataset_name,
                "query": traj.query,
                "prediction": traj.prediction,
                "ground_truth": traj.ground_truth,
                "success": traj.success,
                "metrics": traj.metrics if hasattr(traj, "metrics") else {},
                "error_info": traj.error_info if hasattr(traj, "error_info") else None,
                "execution_time": getattr(traj, "execution_time", None),
                "num_turns": getattr(traj, "num_turns", None),
                "conversation": getattr(traj, "conversation", [])
            }
            all_trajectory_data.append(traj_dict)
        
        with open(all_predictions_file, "w", encoding="utf-8") as f:
            json.dump(all_trajectory_data, f, indent=2, ensure_ascii=False)
        
        # Save metrics results (both pass@k and average)
        metrics_file = Path(self.config.output_dir) / "metrics" / f"{run_name}.json"
        metrics_file.parent.mkdir(exist_ok=True)
        
        metrics_data = {
            "config": results["config"],
            "overall_metrics": results["overall_metrics"],  # Contains both pass@k and avg
            "sample_metrics": results["sample_metrics"],    # Contains both pass@k and avg per sample
            "summary": {
                "total_samples": results["total_samples"],
                "total_trajectories": results["total_trajectories"],
                "k": results["k"],
                "total_time": results["total_time"]
            }
        }
        
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Results saved:")
        print(f"  Individual predictions: prediction_i_traj_j.json files in {self.config.output_dir}")
        print(f"  Combined predictions: {all_predictions_file}")
        print(f"  Metrics: {metrics_file}")


def create_trajectory_generator(
    model: str,
    dataset_name: str = "daeval",
    backend: str = "litellm",
    temperature: float = 0.8,
    k: int = 8,
    max_workers: int = 24,
    output_dir: str = "./trajectory_results",
    **kwargs
) -> TrajectoryGenerator:
    """
    Convenience function to create a trajectory generator.
    
    Args:
        model: Model name
        dataset_name: Name of dataset to use
        backend: Backend type (default: litellm)
        temperature: Sampling temperature (default: 0.8)
        k: Number of trajectories per sample (default: 8)
        max_workers: Max parallel workers (default: 24)
        output_dir: Output directory (default: ./trajectory_results)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured TrajectoryGenerator instance
    """
    config = TrajectoryConfig(
        model=model,
        dataset_name=dataset_name,
        backend=backend,
        temperature=temperature,
        k=k,
        max_workers=max_workers,
        output_dir=output_dir,
        **kwargs
    )
    
    return TrajectoryGenerator(config)