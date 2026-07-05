"""
Query generator for creating synthetic queries through dataset exploration.

This module provides functionality to generate synthetic queries by having an agent
explore datasets and understand their structure, then create similar queries based
on original samples.
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
from ...eval.utils import EvaluationResult
from ..prompts import get_system_prompt


@dataclass
class QueryGeneratorConfig:
    """Configuration for query generation."""
    model: str
    backend: str = "litellm"
    temperature: float = 0.2
    max_workers: int = 16
    max_turns: int = 15
    max_tokens: int = 1024
    num_queries_per_sample: int = 5
    manager_url: str = "http://localhost:5000"
    api_key: Optional[str] = None
    dataset_name: str = "daeval"
    output_dir: str = "./synthetic_queries"
    run_name: Optional[str] = None
    start_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class QueryGenerator:
    """
    Generator for creating synthetic queries through dataset exploration.
    
    The generator uses an agent to explore the dataset structure and generate
    synthetic queries similar to the original ones.
    """
    
    
    def __init__(self, config: QueryGeneratorConfig):
        """
        Initialize query generator.
        
        Args:
            config: Configuration for query generation
        """
        self.config = config
        
        # Automatically set max_workers=1 for single-instance backends
        if self.config.backend in ["vllm", "sglang"] and self.config.max_workers > 1:
            print(f"⚠️  Setting max_workers=1 for {self.config.backend} backend (was {self.config.max_workers})")
            print(f"   {self.config.backend} manages parallelism internally via tensor_parallel_size")
            self.config.max_workers = 1
        
        self.dataset = None
        self.agent = None
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def _initialize_components(self):
        """Initialize dataset and agent components."""
        # Load dataset
        self.dataset = DatasetRegistry.load(self.config.dataset_name)
        
        # Initialize agent
        agent_config = {
            "manager_url": self.config.manager_url,
            "max_turns": self.config.max_turns,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "output_dir": self.config.output_dir,
        }
        
        if self.config.backend == "litellm" and self.config.api_key:
            agent_config["api_key"] = self.config.api_key
            
        self.agent = ReActDSAgent(
            backend=self.config.backend,
            model=self.config.model,
            **agent_config
        )
    
    def _get_system_prompt(self, dataset_name: str) -> str:
        """Get system prompt based on dataset type."""
        return get_system_prompt(dataset_name, self.config.num_queries_per_sample)
    
    def _create_generation_query(self, original_sample: Dict[str, Any]) -> str:
        """Create generation query based on original sample and dataset."""
        # Extract information from sample
        # Get extra info if available
        extra_info = original_sample.get("extra_info", {})

        query = extra_info.get("question", "")
        ground_truth = str(extra_info.get("answer", ""))
        context = extra_info.get("context", "")
        data_files = extra_info.get("data_files", [])

        guidelines = extra_info.get("guidelines", "")
        level = extra_info.get("level", "")
        keywords = extra_info.get("keywords", [])
        
        if self.config.dataset_name.lower() == "qrdata":
            return self._create_qrdata_query(query, ground_truth, context, data_files, keywords)
        elif self.config.dataset_name.lower() == "daeval":
            answer_format = extra_info.get("format", "")
            constraints = extra_info.get("constraints", "")
            return self._create_daeval_query(query, ground_truth, context, data_files, answer_format, constraints)
        else:
            return self._create_default_query(query, ground_truth, guidelines, level, data_files)
    
    def _create_qrdata_query(self, query: str, answer: str, context: str, data_files: List[str], keywords: List[str]) -> str:
        """Create QRData-specific generation query."""
        files_list = "\n".join(data_files) if data_files else "No specific data files provided"
        
        keywords_list = "\n".join(keywords) if keywords else "No specific keywords provided"
        
        generation_instructions = """
ANALYSIS PROCESS:
1) Explore the dataset structure by examining the data files:
   - For CSV files: Check column headers and sample data to understand the structure
   - For other file types: Examine the data format and content
   - Understand the variables and their relationships

2) Analyze the original query to understand:
   - What type of statistical analysis, causal analysis or calculation it requires
   - Which data columns and variables are relevant
   - What statistical concepts or metrics it involves
   - The complexity level and approach needed

3) Generate {num_queries} synthetic query-answer pairs that:
   - Use similar statistical approaches but with different specifics
   - Explore different aspects of the same dataset
   - Maintain similar complexity level to the original
   - Can be answered using the available data
   - Focus on realistic statistical questions for the given domain
   - Include accurate, detailed answers for each query

IMPORTANT REQUIREMENTS:
- Each synthetic query should be answerable using the provided dataset
- Provide comprehensive answers that show the calculation/analysis process
- Maintain the same difficulty level as the original query
- Ensure the {num_queries} query-answer pairs are distinct from each other and the original
- Focus on realistic statistical or causal analysis questions
- Use proper statistical terminology and concepts
""".format(num_queries=self.config.num_queries_per_sample)

        return f"""Now you are given an original statistical query, dataset information, and data files. You need to generate {self.config.num_queries_per_sample} synthetic query-answer pairs similar to the original query by exploring and understanding the dataset structure.

ORIGINAL QUERY: {query}
DATASET DESCRIPTION: {context}
EXPECTED ANSWER (after changing the required @key[value] format to list of @key[value] pairs): {answer}

DATASET FILES:
{files_list}

KEYWORDS:
{keywords_list}

{generation_instructions}
"""
    
    def _create_daeval_query(self, query: str, answer: str, context: str, data_files: List[str], answer_format: str, constraints: str) -> str:
        """Create DAEval-specific generation query with answer format and constraints requirements."""
        files_list = "\n".join(data_files) if data_files else "Dataset files will be available in the environment"
        
        generation_instructions = """
ANALYSIS PROCESS:
1) Explore the dataset structure by examining the available data files:
   - Check file types and formats
   - Understand data schema and relationships
   - Examine sample data to understand the domain

2) Analyze the original query, constraints, and answer format to understand:
   - What type of analysis or calculation it requires
   - Which data sources and columns are relevant
   - The specific constraints and requirements for data processing
   - The specific answer format requirement (@key[value] structure)
   - What business concepts or metrics it involves
   - The complexity level and approach needed

3) Generate {num_queries} synthetic query-answer pairs that:
   - Use similar analytical approaches but with different specifics
   - Explore different aspects of the same business domain
   - Maintain similar complexity level to the original
   - Can be answered using the available data
   - Each include a query, constraints, AND a specific answer format requirement
   - Follow the @key[value] format structure in answers
   - Include comprehensive answers in the specified format
   - Have constraints that are relevant to the generated query

IMPORTANT REQUIREMENTS:
- Each synthetic query should be answerable using the provided dataset
- Each query MUST include "Constraints" that specify data processing requirements, model parameters, significance levels, etc.
- Each query MUST include an "Answer Format" specification that follows the @key[value] pattern
- The constraints should be similar in style and specificity to the original constraints
- The answer format should be consistent with the type of analysis being requested
- Provide detailed answers that demonstrate the solution process and follow the exact format
- Maintain the same difficulty level as the original
- Ensure the {num_queries} query-answer pairs are distinct from each other and the original
- Focus on realistic business questions relevant to this domain
- Generate constraints that are specific, actionable, and relevant to each synthetic query
""".format(num_queries=self.config.num_queries_per_sample)

        return f"""Now you are given an original query with specific constraints and answer format requirements, and dataset information. You need to generate {self.config.num_queries_per_sample} synthetic query-answer pairs similar to the original query by exploring and understanding the dataset structure. Each synthetic query must include the query itself, specific constraints, AND a specific answer format requirement.

ORIGINAL QUERY: {query}
ORIGINAL CONSTRAINTS: {constraints}
DATASET DESCRIPTION: {context}
ORIGINAL ANSWER FORMAT: {answer_format}
EXPECTED ANSWER: {answer}

AVAILABLE DATA:
{files_list}

{generation_instructions}
"""
    
    def _create_default_query(self, query: str, answer: str, guidelines: str, level: str, data_files: List[str]) -> str:
        """Create default generation query for other datasets."""
        files_list = "\n".join(data_files) if data_files else "Dataset files will be available in the environment"
        
        generation_instructions = """
ANALYSIS PROCESS:
1) Explore the dataset structure by examining the available data files:
   - Check file types and formats
   - Understand data schema and relationships
   - Examine sample data to understand the domain

2) Analyze the original query to understand:
   - What type of analysis or calculation it requires
   - Which data sources and columns are relevant
   - What business concepts or metrics it involves
   - The complexity level and approach needed

3) Generate {num_queries} synthetic query-answer pairs that:
   - Use similar analytical approaches but with different specifics
   - Explore different aspects of the same business domain
   - Maintain similar complexity level to the original
   - Can be answered using the available data
   - Follow the same format and guidelines as the original
   - Include comprehensive answers for each query

IMPORTANT REQUIREMENTS:
- Each synthetic query should be answerable using the provided dataset
- Provide detailed answers that demonstrate the solution process
- Maintain the same difficulty level as the original
- Use the same guidelines format as the original query
- Ensure the {num_queries} query-answer pairs are distinct from each other and the original
- Focus on realistic business questions relevant to this domain
""".format(num_queries=self.config.num_queries_per_sample)

        return f"""Now you are given an original query and dataset information. You need to generate {self.config.num_queries_per_sample} synthetic query-answer pairs similar to the original query by exploring and understanding the dataset structure.

ORIGINAL QUERY: {query}
GUIDELINES: {guidelines}
DIFFICULTY LEVEL: {level}
EXPECTED ANSWER: {answer}

AVAILABLE DATA:
{files_list}

{generation_instructions}
"""
    
    def generate_queries_for_sample(
        self, 
        sample: Dict[str, Any], 
        sample_idx: int
    ) -> Dict[str, Any]:
        """
        Generate synthetic queries for a single sample.
        
        Args:
            sample: Original sample to generate queries from
            sample_idx: Index of the sample
            
        Returns:
            Dictionary containing generation results
        """
        try:
            # Create generation query
            generation_query = self._create_generation_query(sample)
            system_prompt = self._get_system_prompt(self.config.dataset_name)
            
            # Create prompt for the agent
            generation_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": generation_query}
            ]
            
            # Create sample for the agent
            generation_sample = {
                "prompt": generation_prompt,
                "ground_truth": "",  # Not applicable for generation
                "extra_info": {
                    "question": f"Generate synthetic queries for sample {sample_idx}",
                    "source": f"{self.config.dataset_name}_generation",
                    "original_sample": sample,
                    "sample_index": sample_idx,
                    "index": sample_idx,  # This is what the environment expects for filename generation
                    "original_question": sample.get("question", ""),
                    "original_answer": sample.get("answer", ""),
                }
            }
            
            # Run generation using agent's method for single query
            if hasattr(self.agent, 'run_single_query_with_sample'):
                eval_result = self.agent.run_single_query_with_sample(generation_sample)
            else:
                # Fallback to evaluate method if available
                from ...eval import Evaluator
                temp_evaluator = Evaluator(metrics=[])
                eval_result = temp_evaluator._evaluate_single_sample(self.agent, generation_sample)
            
            # Extract generated queries from the trajectory
            generated_queries = self._extract_queries_from_prediction(eval_result.prediction)

            return {
                "original_sample": sample,
                "generated_queries": generated_queries,
                "generation_trajectory": eval_result.trajectory,
                "success": len(generated_queries) > 0,
                "sample_index": sample_idx,
            }
            
        except Exception as e:
            print(f"Error generating queries for sample {sample_idx}: {str(e)}")
            return {
                "original_sample": sample,
                "generated_queries": [],
                "error": str(e),
                "success": False,
                "sample_index": sample_idx,
            }
    
    def _extract_queries_from_prediction(self, prediction: str) -> List[Dict[str, str]]:
        """Extract generated query-answer pairs from agent trajectory."""
        query_answer_pairs = []
        
        answer_content = prediction
        
        # Parse numbered query-answer pairs
        lines = answer_content.split('\n')
        current_query = ""
        current_answer = ""
        current_answer_format = ""
        current_constraints = ""
        current_number = None
        parsing_query = False
        parsing_answer = False
        parsing_answer_format = False
        parsing_constraints = False
        
        for line in lines:
            line = line.strip()
            
            # Check if this line starts a new numbered item
            starts_new_item = False
            query_number = None
            
            for num in [str(i) for i in range(1, self.config.num_queries_per_sample + 1)]:
                if line.startswith(f"{num}.") or line.startswith(f"{num})"):
                    starts_new_item = True
                    query_number = num
                    # Remove the number prefix
                    if line.startswith(f"{num}."):
                        line = line[2:].strip()
                    else:  # starts with num)
                        line = line[2:].strip()
                    break
            
            if starts_new_item:
                # Save previous query-answer pair if it exists
                if current_query.strip() and current_number:
                    query_data = {
                        "query": current_query.strip(),
                        "answer": current_answer.strip()
                    }
                    # For DAEval, also include answer format and constraints if present
                    if current_answer_format.strip():
                        query_data["answer_format"] = current_answer_format.strip()
                    if current_constraints.strip():
                        query_data["constraints"] = current_constraints.strip()
                    query_answer_pairs.append(query_data)
                
                # Start new item
                current_number = query_number
                current_query = ""
                current_answer = ""
                current_answer_format = ""
                current_constraints = ""
                parsing_query = False
                parsing_answer = False
                parsing_answer_format = False
                parsing_constraints = False
                
                # Check if this line contains "Query:" prefix
                if line.lower().startswith("query:"):
                    current_query = line[6:].strip()
                    parsing_query = True
                else:
                    current_query = line
                    parsing_query = True
            else:
                # Check for Constraints: prefix (for DAEval)
                if line.lower().startswith("constraints:"):
                    current_constraints = line[12:].strip()
                    parsing_query = False
                    parsing_answer = False
                    parsing_answer_format = False
                    parsing_constraints = True
                # Check for Answer Format: prefix (for DAEval)
                elif line.lower().startswith("answer format:"):
                    current_answer_format = line[14:].strip()
                    parsing_query = False
                    parsing_answer = False
                    parsing_answer_format = True
                    parsing_constraints = False
                # Check for Answer: prefix
                elif line.lower().startswith("answer:"):
                    current_answer = line[7:].strip()
                    parsing_query = False
                    parsing_answer = True
                    parsing_answer_format = False
                    parsing_constraints = False
                elif parsing_query and line:
                    current_query += "\n" + line
                elif parsing_answer and line:
                    current_answer += "\n" + line
                elif parsing_answer_format and line:
                    current_answer_format += "\n" + line
                elif parsing_constraints and line:
                    current_constraints += "\n" + line
                elif current_number and line and not parsing_answer and not parsing_answer_format and not parsing_constraints:
                    # If we haven't seen Answer:, Answer Format:, or Constraints: yet, assume it's still part of the query
                    current_query += "\n" + line
        
        # Don't forget the last query-answer pair
        if current_query.strip() and current_number:
            query_data = {
                "query": current_query.strip(),
                "answer": current_answer.strip()
            }
            # For DAEval, also include answer format and constraints if present
            if current_answer_format.strip():
                query_data["answer_format"] = current_answer_format.strip()
            if current_constraints.strip():
                query_data["constraints"] = current_constraints.strip()
            query_answer_pairs.append(query_data)
        
        return query_answer_pairs[:self.config.num_queries_per_sample]
    
    def _evaluate_single_sample_wrapper(self, args):
        """Wrapper for parallel sample evaluation."""
        sample_idx, sample = args
        return sample_idx, self.generate_queries_for_sample(sample, sample_idx)
    
    def generate(
        self, 
        samples: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Generate synthetic queries for all samples.
        
        Args:
            samples: List of samples to generate from (loads from dataset if None)
            limit: Number of samples to process (optional)
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
            all_samples = self.dataset.load(**load_config)
            
            # Apply start_index and limit
            start_idx = self.config.start_index
            if start_idx > 0:
                if start_idx >= len(all_samples):
                    raise ValueError(f"start_index ({start_idx}) is greater than or equal to dataset size ({len(all_samples)})")
                all_samples = all_samples[start_idx:]
                print(f"📍 Starting from sample index {start_idx}")
            
            if limit:
                all_samples = all_samples[:limit]
                
            samples = all_samples
        elif limit:
            # Apply start_index and limit to provided samples
            start_idx = self.config.start_index
            if start_idx > 0:
                if start_idx >= len(samples):
                    raise ValueError(f"start_index ({start_idx}) is greater than or equal to provided samples size ({len(samples)})")
                samples = samples[start_idx:]
                print(f"📍 Starting from sample index {start_idx}")
            
            samples = samples[:limit]
        
        print(f"📊 Generating {self.config.num_queries_per_sample} synthetic queries for {len(samples)} samples...")
        print(f"🌡️ Temperature: {self.config.temperature}")
        print(f"🤖 Model: {self.config.model}")
        print(f"⚙️ Backend: {self.config.backend}")
        print(f"👷 Max workers: {self.config.max_workers}")
        
        # Generate queries
        all_results = []
        
        if self.config.max_workers > 1:
            # Parallel processing of samples
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(self._evaluate_single_sample_wrapper, (self.config.start_index + idx, sample))
                    for idx, sample in enumerate(samples)
                ]
                
                iterator = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Generating queries (parallel)"
                ) if show_progress else concurrent.futures.as_completed(futures)
                
                # Collect results maintaining order
                results_dict = {}
                for future in iterator:
                    sample_idx, result = future.result()
                    results_dict[sample_idx] = result
                
                # Process in order
                for idx in range(len(samples)):
                    actual_sample_idx = self.config.start_index + idx
                    all_results.append(results_dict[actual_sample_idx])
        else:
            # Sequential processing
            iterator = tqdm(enumerate(samples), total=len(samples), desc="Generating queries") if show_progress else enumerate(samples)
            
            for idx, sample in iterator:
                actual_sample_idx = self.config.start_index + idx
                result = self.generate_queries_for_sample(sample, actual_sample_idx)
                all_results.append(result)
        
        total_time = time.time() - start_time
        
        # Compute statistics
        successful_generations = sum(1 for r in all_results if r["success"])
        total_synthetic_queries = sum(len(r["generated_queries"]) for r in all_results if r["success"])
        
        # Create synthetic dataset
        synthetic_dataset = self._create_synthetic_dataset(all_results)
        
        # Prepare results
        results = {
            "generation_results": all_results,
            "synthetic_dataset": synthetic_dataset,
            "config": self.config.to_dict(),
            "total_time": total_time,
            "total_samples": len(samples),
            "successful_generations": successful_generations,
            "total_synthetic_queries": total_synthetic_queries,
            "success_rate": successful_generations / len(samples) if samples else 0.0,
            "avg_queries_per_sample": total_synthetic_queries / successful_generations if successful_generations > 0 else 0.0
        }
        
        # Save results
        self._save_results(results)
        
        print(f"✅ Generated {total_synthetic_queries} synthetic queries from {len(samples)} samples in {total_time:.2f}s")
        print(f"📈 Success rate: {results['success_rate']:.2%}")
        print(f"📊 Average queries per successful sample: {results['avg_queries_per_sample']:.1f}")
        
        return results
    
    def _create_synthetic_dataset(self, generation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create synthetic dataset from generation results."""
        synthetic_samples = []
        
        for result in generation_results:
            if result["success"] and result["generated_queries"]:
                extra_info = result["original_sample"]["extra_info"]
                original_sample = result["original_sample"]
                
                for i, query_answer_pair in enumerate(result["generated_queries"]):
                    # Extract query and answer from the pair
                    if isinstance(query_answer_pair, dict):
                        synthetic_query = query_answer_pair.get("query", "")
                        synthetic_answer = query_answer_pair.get("answer", "")
                        synthetic_answer_format = query_answer_pair.get("answer_format", "")
                        synthetic_constraints = query_answer_pair.get("constraints", "")
                    else:
                        # Fallback for backward compatibility (if still strings)
                        synthetic_query = str(query_answer_pair)
                        synthetic_answer = ""
                        synthetic_answer_format = ""
                        synthetic_constraints = ""
                    
                    # Create synthetic sample following the original format
                    synthetic_sample = {
                        "query": synthetic_query,
                        "ground_truth": synthetic_answer,  # Now filled with generated answer
                        "extra_info": {
                            "synthetic_id": f"sample_{result['sample_index']}_query_{i+1}",
                            "original_sample_index": result["sample_index"],
                            "original_question": extra_info.get("question", ""),
                            "original_answer": extra_info.get("answer", ""),
                            "generation_method": "agent_exploration",
                            "dataset_source": self.config.dataset_name,
                        }
                    }
                    
                    # Copy over relevant fields from original sample's extra_info
                    if "extra_info" in original_sample:
                        original_extra = original_sample["extra_info"]
                        synthetic_sample["extra_info"].update({
                            "guidelines": original_extra.get("guidelines", ""),
                            "level": original_extra.get("level", ""),
                            "context": original_extra.get("context", ""),
                            "data": original_extra.get("data_files", []),
                            'reference': original_extra.get("reference", ""),
                            'question_type': original_extra.get("question_type", ""),
                            'keywords': original_extra.get("keywords", [])
                        })
                        
                        # For DAEval, add answer format and constraints if present
                        if synthetic_answer_format and self.config.dataset_name.lower() == "daeval":
                            synthetic_sample["extra_info"]["format"] = synthetic_answer_format
                        if synthetic_constraints and self.config.dataset_name.lower() == "daeval":
                            synthetic_sample["extra_info"]["constraints"] = synthetic_constraints
                    
                    synthetic_samples.append(synthetic_sample)
        
        return synthetic_samples
    
    def _save_results(self, results: Dict[str, Any]):
        """Save query generation results to files."""
        # Create run name if not provided
        run_name = self.config.run_name or f"query_gen_{self.config.dataset_name}_{self.config.backend}_{self.config.model.replace('/', '_')}"
        run_name = run_name.replace('/', '_').replace(':', '_')
        
        # Save synthetic dataset as JSON
        synthetic_file = Path(self.config.output_dir) / f"{run_name}_synthetic.json"
        with open(synthetic_file, "w", encoding="utf-8") as f:
            json.dump(results["synthetic_dataset"], f, indent=2, ensure_ascii=False)
        
        # Save detailed results
        detailed_file = Path(self.config.output_dir) / f"{run_name}_detailed.json"
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump({
                "generation_results": results["generation_results"],
                "config": results["config"],
                "summary": {
                    "total_samples": results["total_samples"],
                    "successful_generations": results["successful_generations"],
                    "total_synthetic_queries": results["total_synthetic_queries"],
                    "success_rate": results["success_rate"],
                    "avg_queries_per_sample": results["avg_queries_per_sample"],
                    "total_time": results["total_time"]
                }
            }, f, indent=2, ensure_ascii=False, default=str)
        
        # Save generation statistics
        stats_file = Path(self.config.output_dir) / f"{run_name}_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_samples": results["total_samples"],
                "successful_generations": results["successful_generations"],
                "total_synthetic_queries": results["total_synthetic_queries"],
                "success_rate": results["success_rate"],
                "avg_queries_per_sample": results["avg_queries_per_sample"],
                "total_time": results["total_time"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": results["config"]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Results saved:")
        print(f"  Synthetic dataset: {synthetic_file}")
        print(f"  Detailed results: {detailed_file}")
        print(f"  Statistics: {stats_file}")


def create_query_generator(
    model: str,
    dataset_name: str = "daeval",
    backend: str = "litellm",
    temperature: float = 0.7,
    num_queries_per_sample: int = 5,
    max_workers: int = 16,
    output_dir: str = "./synthetic_queries",
    start_index: int = 0,
    **kwargs
) -> QueryGenerator:
    """
    Convenience function to create a query generator.
    
    Args:
        model: Model name
        dataset_name: Name of dataset to use
        backend: Backend type (default: litellm)
        temperature: Sampling temperature (default: 0.7)
        num_queries_per_sample: Number of queries to generate per sample (default: 5)
        max_workers: Max parallel workers (default: 16)
        output_dir: Output directory (default: ./synthetic_queries)
        start_index: Starting index for sample processing (default: 0)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured QueryGenerator instance
    """
    config = QueryGeneratorConfig(
        model=model,
        dataset_name=dataset_name,
        backend=backend,
        temperature=temperature,
        num_queries_per_sample=num_queries_per_sample,
        max_workers=max_workers,
        output_dir=output_dir,
        start_index=start_index,
        **kwargs
    )
    
    return QueryGenerator(config)