"""
ReActDSAgent implementation - main data science agent using ReAct pattern.

This agent provides a self-contained implementation using the new DSGym
backend and environment components.
"""

import time
import traceback
from typing import Dict, Any, List

from .base_agent import BaseAgent
from .backends import get_backend
from .environment import AllocatedCodeEnv

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
class ReActDSAgent(BaseAgent):
    """ReAct pattern data science agent with integrated backends and environment."""
    
    def __init__(self, backend: str, model: str, **kwargs):
        """
        Initialize ReActDSAgent.
        
        Args:
            backend: Backend type (litellm, vllm, sglang)
            model: Model name/identifier
            **kwargs: Additional configuration
        """
        super().__init__(backend, model, **kwargs)
        
        # Initialize backend
        try:
            self.backend_instance = get_backend(backend, model, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {backend} backend: {e}")
        
        # Environment configuration
        self.manager_url = kwargs.get('manager_url', 'http://localhost:5000')
        self.max_turns = kwargs.get('max_turns', 40)
        self.output_dir = kwargs.get('output_dir', './outputs')
    
    def solve_task(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve a given task using the ReAct pattern.
        
        Args:
            sample: Sample dictionary with prompt, ground_truth, extra_info, etc.
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary containing solution and metadata
        """
        start_time = time.time()
        
        try:
            # Extract conversation and metadata from sample
            conversation = sample.get("prompt", [])

            if not conversation:
                raise ValueError("Sample must contain 'prompt' field with conversation")
            
            extras = {
                "reward_spec": sample.get("reward_spec", {"ground_truth": ""}),
                "extra_info": sample.get("extra_info", {}),
                "max_turns": self.max_turns
            }
            
            # Create environment for this task (local to this call)
            env = AllocatedCodeEnv(
                manager_url=self.manager_url,
                max_turns=self.max_turns,
                output_dir=self.output_dir
            )
            
            # Initialize environment
            conversation, _ = env.init(conversation, **extras)
            
            # Run multi-turn interaction
            total_tokens = 0
            final_answer = ""
            actual_turns = 0
            
            for turn in range(self.max_turns):
                try:
                    # Generate response
                    response = self.backend_instance.generate(conversation)
                    
                    # Count tokens (approximate)
                    total_tokens += len(response.split())
                    
                    # Step environment
                    step_output = env.step(response)
                    actual_turns = turn + 1
                    
                    # Update conversation - add assistant response and new observations
                    conversation.append({"role": "assistant", "content": step_output.get('postprocessed_action', response)})
                    if step_output['observations']:
                        conversation.extend(step_output['observations'])
                    else:
                        print(f"Step output: {step_output}")
                    # Check if task is complete
                    if step_output['done']:
                        final_answer = step_output['metadata'].get('final_answer', response)
                        break
                        
                except Exception as e:
                    error_msg = f"Turn {turn + 1} failed: {e}"
                    print(f"⚠️ {error_msg}")
                    
                    # Add error to conversation for recovery
                    conversation.append({"role": "user", "content": f"Error: {error_msg}. Please try a different approach."})
                    continue
            

            # Check if this is part of trajectory generation
            trajectory_id = sample.get("extra_info", {}).get("trajectory_id")
            env.save_prediction(final_answer, trajectory_id=trajectory_id)
            
            execution_time = time.time() - start_time
            
            return {
                'solution': final_answer,
                'success': bool(final_answer),
                'turns': actual_turns,
                'error': None,
                'metadata': {
                    'model': self.model,
                    'backend': self.backend,
                    'max_turns': self.max_turns,
                    'total_tokens': total_tokens,
                    'execution_time': execution_time,
                    'conversation_length': len(conversation)
                },
                'conversation': conversation,
                'raw_result': {
                    'prediction': final_answer,
                    'turns': actual_turns,
                    'total_tokens': total_tokens
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            
            return {
                'solution': '',
                'success': False,
                'turns': actual_turns if 'actual_turns' in locals() else 0,
                'error': str(e),
                'metadata': {
                    'model': self.model,
                    'backend': self.backend,
                    'max_turns': self.max_turns,
                    'execution_time': execution_time,
                    'error_trace': error_trace
                },
                'conversation': [],
                'raw_result': None
            }
        finally:
            # Clean up environment
            if 'env' in locals():
                env.close()
    
    def solve_multi_task(self, sample_list: List[Dict[str, Any]], bak_path: str, reset_env_times: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Solve multiple tasks sequentially, sharing conversation context across tasks.

        After completing each task, the next task's prompt is appended to the
        accumulated conversation trajectory and execution continues.

        Args:
            sample_list: List of sample dictionaries, each with prompt, ground_truth, extra_info, etc.
            bak_path: Path to save the backup of the results
            reset_env_times: Number of times to reset the environment
            **kwargs: Additional task-specific parameters

        Returns:
            Dictionary containing all solutions and metadata for the multi-task session
        """
        start_time = time.time()
        print(f"{color.RED}🚀 Starting multi-task solving for {len(sample_list)} tasks with model {self.model} Reset Env Times: {reset_env_times}{color.END}")
        if not sample_list:
            return {
                'solutions': [],
                'success': False,
                'error': 'Empty sample_list provided',
                'metadata': {},
                'conversation': [],
            }

        all_results = []
        accumulated_conversation = []
        total_tokens = 0
        total_turns = 0
        env = None
        num_tasks = len(sample_list)
        if reset_env_times > 0:
            reset_points = [reset_env_times]
        else:
            reset_points = []
        print(f"{color.RED}🔧 Environment will be reset at task indices: {reset_points}{color.END}")
        try:
            # Create a single environment for all tasks
            env = AllocatedCodeEnv(
                manager_url=self.manager_url,
                max_turns=self.max_turns,  # Allow more turns for multi-task
                output_dir=self.output_dir
            )

            for task_idx, sample in enumerate(sample_list):
                extras = {
                    "reward_spec": sample.get("reward_spec", {"ground_truth": ""}),
                    "extra_info": sample.get("extra_info", {}),
                    "max_turns": self.max_turns
                }
                if task_idx+1 in reset_points:
                    print(f"{color.YELLOW}🔄 Resetting environment at task {task_idx+1}...{color.END}")
                    env.close()
                    env.init(accumulated_conversation, **extras)
                env.reset_turns()  # Reset turn count for each new task
                task_start_time = time.time()
                print(f"📋 {color.PURPLE}Starting task {task_idx + 1}/{len(sample_list)}{color.END}")

                # Extract prompt from current sample
                current_prompt = sample.get("prompt", [])
                if not current_prompt:
                    print(f"⚠️ {color.YELLOW}Task {task_idx + 1}: Empty prompt, skipping{color.END}")
                    all_results.append({
                        'solution': '',
                        'success': False,
                        'turns': 0,
                        'error': 'Empty prompt'
                    })
                    continue


                if task_idx == 0:
                    # First task: initialize environment with the first prompt
                    accumulated_conversation, _ = env.init(current_prompt, **extras)
                else:
                    # Subsequent tasks: append the new prompt to the accumulated conversation
                    # Only add user messages (skip system messages as they were set in the first task)
                    for msg in current_prompt:
                        if msg.get("role") == "user":
                            accumulated_conversation.append(msg)

                # Run multi-turn interaction for this task
                task_final_answer = ""
                task_turns = 0
                task_done = False

                for turn in range(self.max_turns):
                    try:
                        # Generate response
                        response = self.backend_instance.generate(accumulated_conversation)

                        # Count tokens (approximate)
                        total_tokens += len(response.split())

                        # Step environment
                        step_output = env.step(response)
                        task_turns += 1
                        total_turns += 1
                        if step_output.get('postprocessed_action', response).strip() == "":
                            print(f"⚠️ {color.YELLOW}Task {task_idx + 1}, Turn {turn + 1}: No postprocessed action, using raw response{color.END}")
                            step_output['postprocessed_action'] = "<reasoning>no postprocessed action</reasoning>"
                            with open(f"{bak_path}/task_{task_idx + 1}_turn_{turn + 1}_no_postprocess.json", "w") as f:
                                import json
                                json.dump({
                                    'response': response,
                                    'step_output': step_output
                                }, f, indent=4)
                        # Update conversation
                        accumulated_conversation.append({
                            "role": "assistant",
                            "content": step_output.get('postprocessed_action', response)
                        })
                        if step_output['observations']:
                            accumulated_conversation.extend(step_output['observations'])
                        else:
                            print(f"{color.RED}Step output:{color.END}\n {step_output}")

                        # Check if current task is complete
                        if step_output['done']:
                            task_final_answer = step_output['metadata'].get('final_answer', response)
                            task_done = True
                            break

                    except Exception as e:
                        error_msg = f"Task {task_idx + 1}, Turn {turn + 1} failed: {e}"
                        print(f"⚠️ {color.RED}Error:{color.END}\n {error_msg}")

                        # Add error to conversation for recovery
                        accumulated_conversation.append({
                            "role": "user",
                            "content": f"Error: {error_msg}. Please try a different approach."
                        })
                        continue

                task_execution_time = time.time() - task_start_time

                # Save prediction for this task
                trajectory_id = sample.get("extra_info", {}).get("trajectory_id")
                # env.save_prediction(task_final_answer, trajectory_id=trajectory_id)
                
                task_result = {
                    'turn_id': sample.get("turn_id", task_idx),
                    'context': sample.get("context", ""),
                    'question': sample.get("question", ""),
                    'ground_truth': sample.get("answer", ""),
                    'solution': task_final_answer,
                    'success': task_done and bool(task_final_answer),
                    'turns': task_turns,
                    'error': None if bool(task_final_answer) or task_turns < self.max_turns else 'Max turns reached',
                    'execution_time': task_execution_time
                }
                all_results.append(task_result)
                if bak_path:
                    with open(f"{bak_path}/task_{task_idx + 1}_result.json", "w") as f:
                        import json
                        json.dump(accumulated_conversation, f, indent=4)
                
                if task_result['success']:
                    print(f"✅ Task {task_idx + 1}: Completed in {task_turns} turns")
                else:
                    print(f"❌ Task {task_idx + 1}: Failed - {task_result.get('error', 'No answer')}")

            execution_time = time.time() - start_time

            return {
                'solutions': all_results,
                'success': all(r['success'] for r in all_results),
                'total_turns': total_turns,
                'total_tasks': len(sample_list),
                'successful_tasks': sum(1 for r in all_results if r['success']),
                'error': None,
                'metadata': {
                    'model': self.model,
                    'backend': self.backend,
                    'max_turns_per_task': self.max_turns,
                    'total_tokens': total_tokens,
                    'execution_time': execution_time,
                    'conversation_length': len(accumulated_conversation)
                },
                'conversation': accumulated_conversation,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()

            return {
                'solutions': all_results,
                'success': False,
                'total_turns': total_turns,
                'total_tasks': len(sample_list),
                'successful_tasks': sum(1 for r in all_results if r['success']),
                'error': str(e),
                'metadata': {
                    'model': self.model,
                    'backend': self.backend,
                    'max_turns_per_task': self.max_turns,
                    'execution_time': execution_time,
                    'error_trace': error_trace
                },
                'conversation': accumulated_conversation if 'accumulated_conversation' in locals() else [],
            }
        finally:
            # Clean up environment
            if env is not None:
                env.close()

    def evaluate_batch(self, samples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of samples.

        Args:
            samples: List of samples to evaluate
            **kwargs: Additional evaluation parameters

        Returns:
            List of evaluation results
        """
        results = []
        
        for i, sample in enumerate(samples):
            print(f"🔄 Processing sample {i + 1}/{len(samples)}")
            
            try:
                result = self.solve_task(sample, **kwargs)
                results.append(result)
                
                # Print progress
                if result['success']:
                    print(f"✅ Sample {i + 1}: Success")
                else:
                    print(f"❌ Sample {i + 1}: Failed - {result.get('error', 'No answer')}")
                    
            except Exception as e:
                print(f"💥 Sample {i + 1}: Exception - {e}")
                error_result = {
                    'solution': '',
                    'success': False,
                    'turns': 0,
                    'error': str(e),
                    'metadata': {
                        'model': self.model,
                        'backend': self.backend,
                        'sample_index': i
                    },
                    'conversation': [],
                    'raw_result': None
                }
                results.append(error_result)
        
        return results
    
