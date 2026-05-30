"""
Quality filter for synthetic query-trajectory pairs based on LLM evaluation.

This module provides functionality to filter synthetic queries and trajectories based on
quality assessment using LLM judges. Supports two modes:
1. Trajectory-only filtering: Filter poor trajectories while keeping queries fixed
2. Query-trajectory pair filtering: Filter both poor queries and poor trajectories
"""

import json
import os
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

# Import LiteLLM for judge functionality
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityFilterConfig:
    """Configuration for quality filtering."""
    input_dir: str
    output_dir: str
    mode: str = "trajectory_only"  # "trajectory_only" or "query_trajectory_pair"
    threshold: float = 0.6
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    max_workers: int = 16
    preserve_structure: bool = True
    overwrite: bool = False
    timeout: int = 300  # Default timeout in seconds
    num_judgments: int = 1  # Number of times to judge each sample (k-fold judgment)
    
    # Quality criteria weights
    trajectory_quality_weight: float = 0.4
    answer_quality_weight: float = 0.3
    query_trajectory_alignment_weight: float = 0.2
    query_quality_weight: float = 0.4  # Only used in query_trajectory_pair mode
    force_reeval: bool = False  # Force re-evaluation, ignore cached results


class SimpleJudge:
    """Simple LLM judge for quality evaluation."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, timeout: int = 300):
        """Initialize the judge with LiteLLM."""
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for LLM judge. Please install: pip install litellm")
        
        self.model = model
        self.api_key = api_key
        
        # Set timeout based on model - GPT-5 and reasoning models are very slow
        if model.startswith("gpt-5") or model.startswith("o1-") or model.startswith("o3-") or model.startswith("o4-"):
            self.timeout = max(timeout, 600)  # At least 10 minutes for reasoning models
            print(f"⏰ Using extended timeout of {self.timeout}s for reasoning model: {model}")
        else:
            self.timeout = timeout
        
        # Set up API key
        if api_key:
            if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("gpt-5"):
                os.environ["OPENAI_API_KEY"] = api_key
            elif model.startswith("together_ai/"):
                os.environ["TOGETHER_API_KEY"] = api_key
            elif model.startswith("claude-"):
                os.environ["ANTHROPIC_API_KEY"] = api_key
    
    def evaluate(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Evaluate using LLM and parse JSON response."""
        messages = [
            {"role": "system", "content": "You are an expert AI judge for data science tasks. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(max_retries):
            try:
                # Use appropriate parameters based on model type
                completion_params = {
                    "model": self.model,
                    "messages": messages,
                    "timeout": self.timeout
                }
                
                # Reasoning models don't support temperature/max_tokens
                if not (self.model.startswith("gpt-5") or self.model.startswith("o1-") or 
                        self.model.startswith("o3-") or self.model.startswith("o4-")):
                    completion_params.update({
                        "temperature": 0.1,
                        "max_tokens": 2048
                    })
                else:
                    # For reasoning models, use max_completion_tokens
                    completion_params["max_completion_tokens"] = 2048
                
                response = litellm.completion(**completion_params)
                
                response_text = response.choices[0].message.content
                
                # Try to parse JSON response
                try:
                    result = json.loads(response_text)
                    return result
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                        return result
                    else:
                        raise ValueError(f"Could not parse JSON from response: {response_text}")
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Judge evaluation failed after {max_retries} attempts: {e}")
                    return {
                        "error": str(e),
                        "final_score": 0.0,
                        "recommendation": "DISCARD"
                    }
                else:
                    logger.warning(f"Judge evaluation attempt {attempt + 1} failed: {e}, retrying...")
                    continue
        
        return {
            "error": "All attempts failed",
            "final_score": 0.0,
            "recommendation": "DISCARD"
        }


class QualityFilter:
    """
    Filter for evaluating synthetic query-trajectory pairs using LLM judge.
    
    Supports two filtering modes:
    1. trajectory_only: Query is given, filter out poor trajectories
    2. query_trajectory_pair: Both query and trajectory are synthetic, filter poor pairs
    """
    
    def __init__(self, config: QualityFilterConfig):
        """
        Initialize quality filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config
        
        # Validate configuration
        if self.config.mode not in ["trajectory_only", "query_trajectory_pair"]:
            raise ValueError(f"Invalid mode: {self.config.mode}. Must be 'trajectory_only' or 'query_trajectory_pair'")
        
        # Validate input directory
        if not os.path.exists(self.config.input_dir):
            raise ValueError(f"Input directory does not exist: {self.config.input_dir}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Check if output directory is empty (unless overwrite is True)
        if not self.config.overwrite and os.listdir(self.config.output_dir):
            raise ValueError(f"Output directory is not empty: {self.config.output_dir}. Use --overwrite to continue.")
        
        # Initialize LLM judge
        self.judge = SimpleJudge(
            model=self.config.model,
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
    
    def parse_filename(self, filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse filename to extract query index and trajectory index.
        
        Expected format: prediction_i_traj_j.json
        
        Returns:
            Tuple of (query_index, trajectory_index) or None if invalid format
        """
        match = re.match(r'prediction_(\d+)_traj_(\d+)\.json', filename)
        if match:
            query_idx = int(match.group(1))
            traj_idx = int(match.group(2))
            return query_idx, traj_idx
        return None
    
    def load_trajectory_file(self, file_path: str) -> Dict[str, Any]:
        """Load trajectory data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading trajectory file {file_path}: {e}")
            raise
    
    def _get_cache_key(self, eval_type: str) -> str:
        """Generate cache key for evaluation results."""
        return f"quality_evaluation_{eval_type}_{self.config.model}_{self.config.num_judgments}"
    
    def _get_cached_evaluation(self, data: Dict[str, Any], eval_type: str) -> Optional[Dict[str, Any]]:
        """Check if evaluation result is already cached in the data."""
        # Skip cache if force_reeval is enabled
        if self.config.force_reeval:
            logger.info(f"🔄 Skipping cache due to force_reeval for {eval_type} evaluation")
            return None
            
        cache_key = self._get_cache_key(eval_type)
        
        # Check if metadata exists and contains cached evaluation
        metadata = data.get("metadata", {})
        if cache_key in metadata:
            cached_result = metadata[cache_key]
            logger.info(f"💾 Using cached {eval_type} evaluation")
            return cached_result
        
        return None
    
    def _cache_evaluation_result(self, file_path: str, data: Dict[str, Any], eval_type: str, result: Dict[str, Any]):
        """Cache evaluation result in the original file."""
        try:
            cache_key = self._get_cache_key(eval_type)
            
            # Ensure metadata exists
            if "metadata" not in data:
                data["metadata"] = {}
            
            # Store the evaluation result
            data["metadata"][cache_key] = result
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Cached {eval_type} evaluation result in {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache evaluation result in {file_path}: {e}")
    
    def create_trajectory_quality_prompt(self, data: Dict[str, Any]) -> str:
        """Create evaluation prompt for trajectory quality assessment."""
        
        # Extract trajectory information
        query = data.get("query", "")
        prediction = data.get("prediction", "")
        ground_truth = data.get("ground_truth", "")
        conversation = data.get("conversation", [])
        turns = data.get("turns", 0)
        
        # Convert conversation to readable format
        trajectory_text = ""
        if conversation:
            for i, turn in enumerate(conversation):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                trajectory_text += f"\n--- Turn {i+1} ({role}) ---\n{content}\n"
        else:
            trajectory_text = "No conversation data available"
        
        prompt = f"""
You are an expert AI judge evaluating the quality of a data science trajectory. Your task is to assess how well an agent executed a given query.

EVALUATION CRITERIA:

1. Trajectory Executability (40%): Can the agent successfully execute the task?
- Does the trajectory run without critical errors?
- Can the agent access and process the required data?
- Are there any blocking failures that prevent task completion?
- Does the agent make reasonable progress toward answering the query?

2. Answer Quality (30%): Is the final answer reasonable and well-supported?
- Is the final answer logically derived from the analysis?
- Does the answer make sense given the data and query?
- Is there sufficient evidence supporting the conclusion?
- Are any major analytical flaws present?
- Does the code run correctly without bugs?

3. Query-Trajectory Alignment (20%): Does the trajectory appropriately address the query?
- Does the agent understand what the query is asking for?
- Does the analysis approach match the query requirements?
- Is the agent working on the right problem?
- Does the trajectory stay focused on the query objectives?

4. Technical Quality (10%): Is the technical execution sound?
- Are the analytical methods appropriate?
- Is the code well-structured and efficient?
- Are there any technical issues or inconsistencies?

QUERY:
{query}

Reference Answer (if available):
{ground_truth}
(Note: This answer is generated by a teacher model, so it's possible that it's not the correct answer)

AGENT TRAJECTORY ({turns} turns):
{trajectory_text}

FINAL PREDICTION:
{prediction}

INSTRUCTIONS:
1. Evaluate each criterion on a scale of 0.0 to 1.0
2. Provide specific justification for each score
3. Calculate the weighted final score: (executability × 0.4) + (answer_quality × 0.3) + (alignment × 0.2) + (technical × 0.1)
4. Recommend KEEP if final_score >= 0.6, otherwise DISCARD
5. RESPOND ONLY WITH THE JSON OBJECT - NO OTHER TEXT

Please respond in the following JSON format ONLY:
{{
    "trajectory_executability": {{
        "score": 0.0,
        "justification": "explanation"
    }},
    "answer_quality": {{
        "score": 0.0,
        "justification": "explanation"
    }},
    "query_trajectory_alignment": {{
        "score": 0.0,
        "justification": "explanation"
    }},
    "technical_quality": {{
        "score": 0.0,
        "justification": "explanation"
    }},
    "final_score": 0.0,
    "overall_justification": "explanation of final score calculation",
    "recommendation": "KEEP|DISCARD"
}}
"""
        return prompt.strip()
    
    def create_query_quality_prompt(self, data: Dict[str, Any], original_query: Optional[str] = None) -> str:
        """Create evaluation prompt for query quality assessment."""
        
        query = data.get("query", "")
        
        prompt = f"""
You are an expert AI judge evaluating the quality of a synthetic data science query. Your task is to assess whether this query is well-formed and valuable for training.

EVALUATION CRITERIA:

1. Query Clarity (30%): Is the query clearly stated and unambiguous?
- Is the question well-formed and specific?
- Are the requirements clearly defined?
- Is it easy to understand what is being asked?
- Does the query specify a clear answer format (numerical, list, etc.)?
- Is the expected answer format closed-form and verifiable?

2. Feasibility (25%): Can this query realistically be answered?
- Is the query achievable with typical data science methods?
- Are the requirements reasonable and not impossible?
- Does it require appropriate level of complexity?
- Does the query make logical sense in the given context?

3. Educational Value (30%): Does the query have learning value and sufficient complexity?
- Does it require meaningful data analysis skills and techniques?
- Is it sufficiently challenging (not too trivial or straightforward)?
- Would solving this help someone learn important data science concepts?
- Does it involve interesting analytical challenges that require thought?

4. Similarity to Original (15%): If this is synthetic, does it maintain appropriate similarity?
- Does it use similar methodology as the original query?
- Is the complexity level appropriate?
- Does it target similar concepts or metrics?

SYNTHETIC QUERY TO EVALUATE:
{query}
"""
        
        if original_query:
            prompt += f"""
ORIGINAL QUERY (for comparison):
{original_query}
"""
        
        prompt += """
INSTRUCTIONS:
1. Evaluate each criterion on a scale of 0.0 to 1.0
2. Provide specific justification for each score
3. Calculate the weighted final score: (clarity × 0.30) + (feasibility × 0.25) + (educational × 0.30) + (similarity × 0.15)
4. Recommend KEEP if final_score >= 0.6, otherwise DISCARD
5. RESPOND ONLY WITH THE JSON OBJECT - NO OTHER TEXT

Please respond in the following JSON format ONLY:
{
    "query_clarity": {
        "score": 0.0,
        "justification": "explanation"
    },
    "feasibility": {
        "score": 0.0,
        "justification": "explanation"
    },
    "educational_value": {
        "score": 0.0,
        "justification": "explanation"
    },
    "similarity_to_original": {
        "score": 0.0,
        "justification": "explanation"
    },
    "final_score": 0.0,
    "overall_justification": "explanation of final score calculation",
    "recommendation": "KEEP|DISCARD"
}
"""
        return prompt.strip()
    
    def evaluate_trajectory_quality(self, data: Dict[str, Any], file_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate trajectory quality using LLM judge with k-fold judgment."""
        # Check for cached evaluation result
        cached_result = self._get_cached_evaluation(data, "trajectory")
        if cached_result is not None:
            return cached_result
            
        # Perform evaluation
        if self.config.num_judgments == 1:
            # Single judgment - original behavior
            try:
                prompt = self.create_trajectory_quality_prompt(data)
                result = self.judge.evaluate(prompt)
                
            except Exception as e:
                logger.error(f"Error evaluating trajectory quality: {e}")
                result = {
                    "error": str(e),
                    "final_score": 0.0,
                    "recommendation": "DISCARD"
                }
        else:
            # Multiple judgments - average the scores
            result = self._evaluate_with_k_judgments(data, "trajectory")
        
        # Cache the result if file_path is provided
        if file_path:
            self._cache_evaluation_result(file_path, data, "trajectory", result)
            
        return result
    
    def evaluate_query_quality(self, data: Dict[str, Any], original_query: Optional[str] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate query quality using LLM judge with k-fold judgment."""
        # Check for cached evaluation result
        cached_result = self._get_cached_evaluation(data, "query")
        if cached_result is not None:
            return cached_result
            
        # Perform evaluation
        if self.config.num_judgments == 1:
            # Single judgment - original behavior
            try:
                prompt = self.create_query_quality_prompt(data, original_query)
                result = self.judge.evaluate(prompt)
                
            except Exception as e:
                logger.error(f"Error evaluating query quality: {e}")
                result = {
                    "error": str(e),
                    "final_score": 0.0,
                    "recommendation": "DISCARD"
                }
        else:
            # Multiple judgments - average the scores
            result = self._evaluate_with_k_judgments(data, "query", original_query)
        
        # Cache the result if file_path is provided
        if file_path:
            self._cache_evaluation_result(file_path, data, "query", result)
            
        return result
    
    def _evaluate_with_k_judgments(self, data: Dict[str, Any], eval_type: str, original_query: Optional[str] = None) -> Dict[str, Any]:
        """Perform k judgments and average the scores."""
        all_results = []
        all_scores = []
        
        for i in range(self.config.num_judgments):
            try:
                if eval_type == "trajectory":
                    prompt = self.create_trajectory_quality_prompt(data)
                elif eval_type == "query":
                    prompt = self.create_query_quality_prompt(data, original_query)
                else:
                    raise ValueError(f"Unknown eval_type: {eval_type}")
                
                result = self.judge.evaluate(prompt)
                all_results.append(result)
                
                if "final_score" in result and result["final_score"] is not None:
                    all_scores.append(result["final_score"])
                    
            except Exception as e:
                logger.warning(f"Error in judgment {i+1}/{self.config.num_judgments} for {eval_type}: {e}")
                # Add a failed result
                all_results.append({
                    "error": str(e),
                    "final_score": 0.0,
                    "recommendation": "DISCARD"
                })
        
        # Calculate average score
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            # Calculate score standard deviation for reliability measure
            score_variance = sum((s - avg_score) ** 2 for s in all_scores) / len(all_scores)
            score_std = score_variance ** 0.5
        else:
            avg_score = 0.0
            score_std = 0.0
        
        # Determine recommendation based on average score
        recommendation = "KEEP" if avg_score >= self.config.threshold else "DISCARD"
        
        # Create aggregated result
        aggregated_result = {
            "final_score": avg_score,
            "recommendation": recommendation,
            "k_judgment_stats": {
                "num_judgments": self.config.num_judgments,
                "successful_judgments": len(all_scores),
                "individual_scores": all_scores,
                "score_std": score_std,
                "score_min": min(all_scores) if all_scores else 0.0,
                "score_max": max(all_scores) if all_scores else 0.0,
            },
            "individual_results": all_results
        }
        
        # If we have successful judgments, try to aggregate the detailed scores
        if all_scores and eval_type == "trajectory":
            # Aggregate trajectory-specific scores
            self._aggregate_trajectory_scores(aggregated_result, all_results)
        elif all_scores and eval_type == "query":
            # Aggregate query-specific scores
            self._aggregate_query_scores(aggregated_result, all_results)
        
        return aggregated_result
    
    def _aggregate_trajectory_scores(self, aggregated_result: Dict[str, Any], all_results: List[Dict[str, Any]]):
        """Aggregate trajectory-specific detailed scores."""
        score_keys = ["trajectory_executability", "answer_quality", "query_trajectory_alignment", "technical_quality"]
        
        for key in score_keys:
            scores = []
            justifications = []
            
            for result in all_results:
                if key in result and isinstance(result[key], dict):
                    if "score" in result[key] and result[key]["score"] is not None:
                        scores.append(result[key]["score"])
                    if "justification" in result[key]:
                        justifications.append(result[key]["justification"])
            
            if scores:
                aggregated_result[key] = {
                    "score": sum(scores) / len(scores),
                    "justification": f"Average of {len(scores)} judgments. Individual justifications: " + " | ".join(justifications),
                    "individual_scores": scores
                }
    
    def _aggregate_query_scores(self, aggregated_result: Dict[str, Any], all_results: List[Dict[str, Any]]):
        """Aggregate query-specific detailed scores."""
        score_keys = ["query_clarity", "feasibility", "educational_value", "similarity_to_original"]
        
        for key in score_keys:
            scores = []
            justifications = []
            
            for result in all_results:
                if key in result and isinstance(result[key], dict):
                    if "score" in result[key] and result[key]["score"] is not None:
                        scores.append(result[key]["score"])
                    if "justification" in result[key]:
                        justifications.append(result[key]["justification"])
            
            if scores:
                aggregated_result[key] = {
                    "score": sum(scores) / len(scores),
                    "justification": f"Average of {len(scores)} judgments. Individual justifications: " + " | ".join(justifications),
                    "individual_scores": scores
                }
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single trajectory file for quality assessment."""
        filename = os.path.basename(file_path)
        
        try:
            # Parse filename to get indices
            parsed = self.parse_filename(filename)
            if not parsed:
                return {
                    "file_path": file_path,
                    "error": f"Invalid filename format: {filename}",
                    "success": False
                }
            
            query_idx, traj_idx = parsed
            
            # Load trajectory data
            data = self.load_trajectory_file(file_path)
            
            # Evaluate based on mode
            if self.config.mode == "trajectory_only":
                # Only evaluate trajectory quality
                evaluation = self.evaluate_trajectory_quality(data, file_path)
                
                result = {
                    "file_path": file_path,
                    "query_index": query_idx,
                    "trajectory_index": traj_idx,
                    "evaluation": evaluation,
                    "mode": "trajectory_only",
                    "success": True
                }
                
            elif self.config.mode == "query_trajectory_pair":
                # Evaluate both query and trajectory quality
                trajectory_eval = self.evaluate_trajectory_quality(data, file_path)
                
                # Get original query if available from extra_info
                original_query = None
                extra_info = data.get("extra_info", {})
                if "original_sample" in extra_info:
                    original_sample = extra_info["original_sample"]
                    if isinstance(original_sample, dict):
                        original_query = original_sample.get("question", "")
                
                query_eval = self.evaluate_query_quality(data, original_query, file_path)
                
                # Combine scores with weights
                trajectory_score = trajectory_eval.get("final_score", 0.0)
                query_score = query_eval.get("final_score", 0.0)
                
                combined_score = (
                    trajectory_score * (1 - self.config.query_quality_weight) +
                    query_score * self.config.query_quality_weight
                )
                
                recommendation = "KEEP" if combined_score >= self.config.threshold else "DISCARD"
                
                result = {
                    "file_path": file_path,
                    "query_index": query_idx,
                    "trajectory_index": traj_idx,
                    "trajectory_evaluation": trajectory_eval,
                    "query_evaluation": query_eval,
                    "combined_score": combined_score,
                    "recommendation": recommendation,
                    "mode": "query_trajectory_pair",
                    "success": True
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "success": False
            }
    
    def find_trajectory_files(self) -> List[str]:
        """Find all trajectory files in the input directory."""
        input_path = Path(self.config.input_dir)
        
        # Find all prediction_i_traj_j.json files
        files = list(input_path.glob("prediction_*_traj_*.json"))
        
        if not files:
            raise ValueError(f"No trajectory files found in {self.config.input_dir}")
        
        return [str(f) for f in files]
    
    def group_files_by_query(self, files: List[str]) -> Dict[int, List[str]]:
        """Group trajectory files by query index."""
        groups = {}
        
        for file_path in files:
            filename = os.path.basename(file_path)
            parsed = self.parse_filename(filename)
            
            if parsed:
                query_idx, _ = parsed
                if query_idx not in groups:
                    groups[query_idx] = []
                groups[query_idx].append(file_path)
        
        return groups
    
    def filter_trajectories(self) -> Dict[str, Any]:
        """Filter trajectories based on quality assessment."""
        print(f"🔍 Starting quality filtering with mode: {self.config.mode}")
        print(f"📂 Input directory: {self.config.input_dir}")
        print(f"📁 Output directory: {self.config.output_dir}")
        print(f"🎯 Threshold: {self.config.threshold}")
        print(f"🤖 Model: {self.config.model}")
        print(f"🔄 Judgments per sample: {self.config.num_judgments}")
        print(f"💾 Cache enabled: evaluation results will be saved to files")
        
        # Find all trajectory files
        files = self.find_trajectory_files()
        print(f"📊 Found {len(files)} trajectory files to process")
        
        # Group files by query if in trajectory_only mode
        if self.config.mode == "trajectory_only":
            query_groups = self.group_files_by_query(files)
            print(f"📝 Found {len(query_groups)} unique queries")
        
        # Process files in parallel
        results = []
        kept_files = []
        processing_errors = 0
        
        print(f"⚙️ Processing with {self.config.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path 
                for file_path in files
            }
            
            # Process completed tasks with progress bar
            progress_bar = tqdm(total=len(files), desc="Evaluating trajectories")
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["success"]:
                        # Determine if file should be kept
                        should_keep = False
                        
                        if self.config.mode == "trajectory_only":
                            score = result["evaluation"].get("final_score", 0.0)
                            recommendation = result["evaluation"].get("recommendation", "DISCARD")
                            should_keep = score >= self.config.threshold and recommendation == "KEEP"
                            progress_bar.set_postfix(score=f"{score:.3f}", rec=recommendation)
                            
                        elif self.config.mode == "query_trajectory_pair":
                            score = result["combined_score"]
                            recommendation = result["recommendation"]
                            should_keep = recommendation == "KEEP"
                            progress_bar.set_postfix(score=f"{score:.3f}", rec=recommendation)
                        
                        if should_keep:
                            kept_files.append(file_path)
                    else:
                        processing_errors += 1
                        progress_bar.set_postfix(status="ERROR")
                        
                except Exception as e:
                    logger.error(f"Error processing future for {file_path}: {e}")
                    processing_errors += 1
                
                progress_bar.update(1)
            
            progress_bar.close()
        
        # Copy kept files to output directory
        print(f"📋 Copying {len(kept_files)} files to output directory...")
        
        for file_path in tqdm(kept_files, desc="Copying files"):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.config.output_dir, filename)
            
            if self.config.preserve_structure:
                # Load and potentially modify the file
                data = self.load_trajectory_file(file_path)
                
                # Add quality evaluation to metadata
                for result in results:
                    if result["file_path"] == file_path and result["success"]:
                        if "metadata" not in data:
                            data["metadata"] = {}
                        data["metadata"]["quality_evaluation"] = result
                        break
                
                # Save modified file
                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                # Simple copy
                import shutil
                shutil.copy2(file_path, dest_path)
        
        # Calculate statistics
        total_files = len(files)
        successful_evaluations = len([r for r in results if r["success"]])
        kept_count = len(kept_files)
        discarded_count = successful_evaluations - kept_count
        keep_rate = kept_count / successful_evaluations if successful_evaluations > 0 else 0.0
        
        # Calculate average scores
        if self.config.mode == "trajectory_only":
            avg_score = sum(r["evaluation"].get("final_score", 0.0) for r in results if r["success"]) / successful_evaluations if successful_evaluations > 0 else 0.0
        else:
            avg_score = sum(r["combined_score"] for r in results if r["success"]) / successful_evaluations if successful_evaluations > 0 else 0.0
        
        stats = {
            "mode": self.config.mode,
            "config": {
                "threshold": self.config.threshold,
                "model": self.config.model,
                "input_dir": self.config.input_dir,
                "output_dir": self.config.output_dir,
            },
            "statistics": {
                "total_files": total_files,
                "successful_evaluations": successful_evaluations,
                "kept_files": kept_count,
                "discarded_files": discarded_count,
                "processing_errors": processing_errors,
                "keep_rate": keep_rate,
                "average_score": avg_score,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save detailed results
        results_file = os.path.join(self.config.output_dir, "quality_filter_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save statistics
        stats_file = os.path.join(self.config.output_dir, "quality_filter_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n📈 Quality Filtering Summary:")
        print(f"   Total files processed: {total_files}")
        print(f"   Successful evaluations: {successful_evaluations}")
        print(f"   Files kept: {kept_count}")
        print(f"   Files discarded: {discarded_count}")
        print(f"   Processing errors: {processing_errors}")
        print(f"   Keep rate: {keep_rate:.1%}")
        print(f"   Average score: {avg_score:.3f}")
        print(f"📁 Results saved to: {self.config.output_dir}")
        print(f"📋 Detailed results: {results_file}")
        print(f"📊 Statistics: {stats_file}")
        
        return stats


def create_quality_filter(
    input_dir: str,
    output_dir: str,
    mode: str = "trajectory_only",
    threshold: float = 0.6,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_workers: int = 16,
    preserve_structure: bool = True,
    overwrite: bool = False,
    timeout: int = 300,
    num_judgments: int = 1,
    force_reeval: bool = False,
    **kwargs
) -> QualityFilter:
    """
    Convenience function to create a quality filter.
    
    Args:
        input_dir: Input directory containing trajectory files
        output_dir: Output directory for filtered files
        mode: Filtering mode ("trajectory_only" or "query_trajectory_pair")
        threshold: Quality threshold for keeping files
        model: Model name for LLM judge
        api_key: API key for the model provider
        max_workers: Maximum number of parallel workers
        preserve_structure: Whether to preserve original JSON structure
        overwrite: Whether to overwrite existing output directory
        timeout: Timeout in seconds for LLM calls (default: 300, auto-extended for reasoning models)
        num_judgments: Number of judgments per sample for averaging (default: 1)
        force_reeval: Force re-evaluation, ignore cached results (default: False)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured QualityFilter instance
    """
    config = QualityFilterConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        mode=mode,
        threshold=threshold,
        model=model,
        api_key=api_key,
        max_workers=max_workers,
        preserve_structure=preserve_structure,
        overwrite=overwrite,
        timeout=timeout,
        num_judgments=num_judgments,
        force_reeval=force_reeval,
        **kwargs
    )
    
    return QualityFilter(config)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Filter synthetic query trajectories by quality")
    
    # Required arguments
    parser.add_argument("input_dir", type=str,
                       help="Input directory containing prediction_i_traj_j.json files")
    parser.add_argument("output_dir", type=str,
                       help="Output directory for filtered files")
    
    # Filtering mode
    parser.add_argument("--mode", type=str, default="trajectory_only",
                       choices=["trajectory_only", "query_trajectory_pair"],
                       help="Filtering mode (default: trajectory_only)")
    
    # Quality parameters
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Quality threshold for keeping files (default: 0.6)")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="Model name for LLM judge (default: gpt-4o)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for the model provider")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in seconds for LLM calls (default: 300, auto-extended for reasoning models)")
    
    # Performance options
    parser.add_argument("--max-workers", type=int, default=16,
                       help="Maximum number of parallel workers (default: 16)")
    
    # General options
    parser.add_argument("--preserve-structure", action="store_true", default=True,
                       help="Preserve original JSON structure (default: True)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output directory")
    
    args = parser.parse_args()
    
    print("🚀 Starting Quality Filter for Synthetic Trajectories")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    
    # Create filter
    try:
        filter_instance = create_quality_filter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            threshold=args.threshold,
            model=args.model,
            api_key=args.api_key,
            max_workers=args.max_workers,
            preserve_structure=args.preserve_structure,
            overwrite=args.overwrite,
            timeout=args.timeout
        )
        
        # Apply filtering
        results = filter_instance.filter_trajectories()
        
        print(f"\n✅ Quality filtering completed successfully!")
        print(f"📊 Kept {results['statistics']['kept_files']} out of {results['statistics']['total_files']} files ({results['statistics']['keep_rate']:.1%})")
        
    except Exception as e:
        print(f"❌ Error during filtering: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())