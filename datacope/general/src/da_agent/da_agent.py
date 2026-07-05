

import json
import os
import re
import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.da_agent.utils import (
    ModelOutput,
    save_evaluation_results,
    extract_sample_info
)
from src.da_agent.base import BaseDaAgent
from src.da_agent.prompts.react_system_prompt import REACT_SYSTEM_PROMPT
from src.da_agent.prompts.codex_system_prompt import CODEX_SYSTEM_PROMPT
from src.da_agent.prompts.claude_code_system_prompt import CLAUDE_CODE_SYSTEM_PROMPT
from src.core.schema import AgentResult, EvaluationResult, EvaluationConfig
from src.runtimes.registry import AgentRegistry
import src.runtimes.agents  # triggers @register_agent decorators

output_schema = ModelOutput.model_json_schema()

class DaAgent(BaseDaAgent):
    def __init__(
        self,
        agent_type: str,
        model: str,
        backend: str = "",
        output_dir: Optional[str] = None,
        parallel_workers: int = 1,
        **agent_kwargs,
    ):
        super().__init__(agent_type, model, backend, output_dir, parallel_workers, **agent_kwargs)
        self.agent = AgentRegistry.load(agent_type, backend=backend, model=model, output_schema=output_schema, **agent_kwargs)
        os.makedirs(self.output_dir, exist_ok=True) if self.output_dir else None

    def generate(
        self,
        tasks: List[Dict[str, Any]],
        config: Optional[EvaluationConfig] = None,
        save_results: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        if not tasks:
            raise ValueError("No tasks provided for evaluation")

        start_time = time.time()

        if self.parallel_workers and self.parallel_workers > 1:
            evaluation_results = self._evaluate_parallel(tasks, show_progress)
        else:
            evaluation_results = self._evaluate_sequential(tasks, show_progress)
        
        total_time = time.time() - start_time
        
        # Prepare return data
        result_data = {
            "results": evaluation_results,
            "config": config.to_dict() if config else {},
            "total_time": total_time,
        }
        
        # Save results if requested
        if save_results and config:
            file_paths = save_evaluation_results(
                evaluation_results, 
                config, 
                self.output_dir, 
                config.run_name or "evaluation"
            )
            result_data["file_paths"] = file_paths
        
        return result_data
    
    def _evaluate_sequential(
        self,
        tasks: List[Dict[str, Any]],
        show_progress: bool
    ) -> List[EvaluationResult]:
        results = []

        iterator = tqdm(tasks, desc="Evaluating") if show_progress else tasks

        for task in iterator:
            try:
                result = self._evaluate_single_sample(task)
                results.append(result)
                
            except Exception as e:
                # Create error result
                task_info = extract_sample_info(task)
                
                error_result = EvaluationResult(
                    sample_id=f"error_{len(results)}",
                    dataset_name=task_info.get("dataset_name", "unknown"),
                    query=task_info.get("query", ""),
                    ground_truth=task_info.get("ground_truth"),
                    success=False,
                    error_info={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "error_category": "EVALUATION_ERROR"
                    }
                )
                results.append(error_result)
        
        return results
    
    def _evaluate_parallel(
        self,
        tasks: List[Dict[str, Any]],
        show_progress: bool
    ) -> List[EvaluationResult]:
        results = [None] * len(tasks)

        def evaluate_single(index_task):
            index, task = index_task
            try:
                result = self._evaluate_single_sample(task)
                return index, result
            except Exception as e:
                # Create error result
                task_info = extract_sample_info(task)
                
                error_result = EvaluationResult(
                    sample_id=f"error_{index}",
                    dataset_name=task_info.get("dataset_name", "unknown"),
                    query=task_info.get("query", ""),
                    ground_truth=task_info.get("ground_truth"),
                    success=False,
                    error_info={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "error_category": "EVALUATION_ERROR"
                    }
                )
                return index, error_result
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = [
                executor.submit(evaluate_single, (i, task)) 
                for i, task in enumerate(tasks)
            ]
            
            # Collect results with progress bar
            if show_progress:
                futures = tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures),
                    desc="Evaluating (parallel)"
                )
            else:
                futures = concurrent.futures.as_completed(futures)
            
            for future in futures:
                index, result = future.result()
                results[index] = result
        
        return results
    
    def _evaluate_single_sample(
        self,
        sample: Dict[str, Any]
    ) -> EvaluationResult:
        start_time = time.time()
        agent = self.agent

        prompt = sample.get("prompt")
        if not prompt:
            raise ValueError("Sample must contain a 'prompt' key")
        system_prompt = self._get_system_prompt()
        sample_info = extract_sample_info(sample)
        query = sample_info["query"]
        ground_truth = sample_info["ground_truth"]
        dataset_name = sample_info["dataset_name"]
        sample_id = f"{dataset_name}_{sample_info['metadata_id']}_{sample_info['query_id']}"
        
        try:
            # Use agent's solve_task method (unified interface)
            if hasattr(agent, 'solve_task'):
                # Use standard agent evaluation method
                if agent.get_agent_type() == "react":
                    agent_result: AgentResult = agent.solve_task(prompt, system=system_prompt, extra_info=sample.get("extra_info", {}))
                else:
                    agent_result: AgentResult = agent.solve_task(prompt, system=system_prompt)
                    
                execution_time = time.time() - start_time
                total_turns = agent_result.turns
                
                # Extract info from agent result
                prediction = self._extract_answer(agent_result.response, agent.get_agent_type())
                raw_response = agent_result.raw_response if agent_result.raw_response else prediction
                conversation = agent_result.conversation if agent_result.conversation else []
                turns = agent_result.turns if agent_result.turns else 0

                token_usage = agent_result.metadata.get("token_usage", {}) if agent_result.metadata else {}
                self._save_prediction(prediction, sample.get("extra_info", {}), turns, conversation=conversation, token_usage=token_usage)
                         
                if agent_result.error:
                    error_info = agent_result.error
                    success = False
                else:
                    error_info = None
                    success = bool(prediction)
                
                result = EvaluationResult(
                    sample_id=sample_id,
                    dataset_name=dataset_name,
                    query=query,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    raw_response=raw_response,
                    execution_time=execution_time,
                    total_turns=total_turns,
                    success=success,
                    error_info=error_info,
                    extra_info={
                        **sample_info,
                        "done": bool(prediction)
                    }
                )
                
            else:
                raise ValueError(f"Agent {type(agent)} does not implement solve_task() method. Please inherit from BaseAgent.")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create error result
            result = EvaluationResult(
                sample_id=sample_id,
                dataset_name=dataset_name,
                query=query,
                ground_truth=ground_truth,
                execution_time=execution_time,
                total_turns=0,
                success=False,
                error_info={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_category": "AGENT_ERROR"
                },
                extra_info=sample_info
            )
            
            return result

    def _get_system_prompt(self) -> str:
        if self.agent.get_agent_type() == "react":
            return REACT_SYSTEM_PROMPT
        elif self.agent.get_agent_type() == "codex":
            return CODEX_SYSTEM_PROMPT
        elif self.agent.get_agent_type() == "claude_code":
            return CLAUDE_CODE_SYSTEM_PROMPT
        else:
            return CODEX_SYSTEM_PROMPT

    def _extract_answer(self, response: dict, agent_type: str) -> str:
        """
        Extract the final answer from the agent's response text.

        Args:
            response: The raw response from the agent
            agent_type: The type of agent (e.g., "react", "codex", etc.)

        Returns:
            Extracted answer string
        """
        if not response:
            return None
        return response["answer"]
    
    def _save_prediction(self, prediction: str, extra_info: dict, turns: int, filename_prefix: str = "prediction", conversation: Optional[List] = None, token_usage: Optional[Dict] = None):
        """Save prediction to output directory."""
        index = extra_info.get("index", 0)
        trajectory_id = extra_info.get("trajectory_id")
        if trajectory_id is not None:
            # Include trajectory ID in filename: prediction_i_traj_j.json
            filename = f"{filename_prefix}_{index}_traj_{trajectory_id}.json"
        else:
            # Original format: prediction_i.json
            filename = f"{filename_prefix}_{index}.json"

        prediction_data = {
            "prediction": prediction,
            "ground_truth": extra_info.get("ground_truth", ""),
            "query": extra_info.get("query", ""),
            "turns": turns,
            "conversation": conversation,
            "extra_info": extra_info,
            "token_usage": token_usage,
        }

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, indent=2, ensure_ascii=False) 