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
class MultiTurnReActDSAgent(BaseAgent):
    """ReAct pattern data science agent with integrated backends and environment."""
    
    def __init__(self, backend: str, model: str, **kwargs):
        """
        Initialize MultiTurnReActDSAgent.
        
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
        self.max_steps = kwargs.get('max_steps', 40)
        self.output_dir = kwargs.get('output_dir', './outputs')
    
    def solve_task(self, sample_list: List[Dict[str, Any]], bak_path: str, reset_env_times: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Solve multiple turns sequentially, sharing conversation context across turns.

        After completing each turn, the next turn's prompt is appended to the
        accumulated conversation trajectory and execution continues.

        Args:
            sample_list: List of sample dictionaries, each with prompt, ground_truth, extra_info, etc.
            bak_path: Path to save the backup of the results
            reset_env_times: Number of times to reset the environment
            **kwargs: Additional task-specific parameters

        Returns:
            Dictionary containing all solutions and metadata for the multi-turn session
        """
        start_time = time.time()
        print(f"{color.RED}🚀 Starting multi-turn solving for {len(sample_list)} turns with model {self.model} Reset Env Times: {reset_env_times}{color.END}")
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
        total_steps = 0
        env = None
        num_turns = len(sample_list)
        if reset_env_times > 0:
            reset_points = [reset_env_times]
        else:
            reset_points = []
        print(f"{color.RED}🔧 Environment will be reset at turn indices: {reset_points}{color.END}")
        try:
            # Create a single environment for all turns
            env = AllocatedCodeEnv(
                manager_url=self.manager_url,
                max_turns=self.max_steps,  # Allow more turns for multi-turn interaction
                output_dir=self.output_dir
            )

            for  turn_idx, sample in enumerate(sample_list):
                extras = {
                    "reward_spec": sample.get("reward_spec", {"ground_truth": ""}),
                    "extra_info": sample.get("extra_info", {}),
                    "max_turns": self.max_steps
                }
                if turn_idx+1 in reset_points:
                    print(f"{color.YELLOW}🔄 Resetting environment at turn {turn_idx+1}...{color.END}")
                    env.close()
                    env.init(accumulated_conversation, **extras)
                env.reset_turns()  # Reset step count but keep conversation and environment state
                turn_start_time = time.time()
                print(f"📋 {color.PURPLE}Starting turn {turn_idx + 1}/{len(sample_list)}{color.END}")

                # Extract prompt from current sample
                current_prompt = sample.get("prompt", [])
                if not current_prompt:
                    print(f"⚠️ {color.YELLOW}Turn {turn_idx + 1}: Empty prompt, skipping{color.END}")
                    all_results.append({
                        'solution': '',
                        'success': False,
                        'steps': 0,
                        'error': 'Empty prompt'
                    })
                    continue


                if turn_idx == 0:
                    # First turn: initialize environment with the first prompt
                    accumulated_conversation, _ = env.init(current_prompt, **extras)
                else:
                    # Subsequent turns: append the new prompt to the accumulated conversation
                    # Only add user messages (skip system messages as they were set in the first turn)
                    for msg in current_prompt:
                        if msg.get("role") == "user":
                            accumulated_conversation.append(msg)

                # Run multi-turn interaction for this turn
                turn_final_answer = ""
                turn_steps = 0
                turn_done = False

                for turn in range(self.max_steps):
                    try:
                        # Generate response
                        response = self.backend_instance.generate(accumulated_conversation)

                        # Count tokens (approximate)
                        total_tokens += len(response.split())

                        # Step environment
                        step_output = env.step(response)
                        turn_steps += 1
                        total_steps += 1
                        if step_output.get('postprocessed_action', response).strip() == "":
                            print(f"⚠️ {color.YELLOW}Turn {turn_idx + 1}, Step {turn + 1}: No postprocessed action, using raw response{color.END}")
                            step_output['postprocessed_action'] = "<reasoning>no postprocessed action</reasoning>"
                            with open(f"{bak_path}/turn_{turn_idx + 1}_step_{turn + 1}_no_postprocess.json", "w") as f:
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

                        # Check if current turn is complete
                        if step_output['done']:
                            turn_final_answer = step_output['metadata'].get('final_answer', response)
                            turn_done = True
                            break

                    except Exception as e:
                        error_msg = f"Turn {turn_idx + 1}, Step {turn + 1} failed: {e}"
                        print(f"⚠️ {color.RED}Error:{color.END}\n {error_msg}")

                        # Add error to conversation for recovery
                        accumulated_conversation.append({
                            "role": "user",
                            "content": f"Error: {error_msg}. Please try a different approach."
                        })
                        continue

                turn_execution_time = time.time() - turn_start_time

                # Save prediction for this turn
                trajectory_id = sample.get("extra_info", {}).get("trajectory_id")
                # env.save_prediction(turn_final_answer, trajectory_id=trajectory_id)
                
                turn_result = {
                    'turn_id': sample.get("turn_id", turn_idx),
                    'context': sample.get("context", ""),
                    'question': sample.get("question", ""),
                    'ground_truth': sample.get("answer", ""),
                    'solution': turn_final_answer,
                    'success': turn_done and bool(turn_final_answer),
                    'steps': turn_steps,
                    'error': None if bool(turn_final_answer) or turn_steps < self.max_steps else 'Max steps reached',
                    'execution_time': turn_execution_time
                }
                all_results.append(turn_result)
                if bak_path:
                    with open(f"{bak_path}/turn_{turn_idx + 1}_result.json", "w") as f:
                        import json
                        json.dump(accumulated_conversation, f, indent=4)
                
                if turn_result['success']:
                    print(f"✅ Turn {turn_idx + 1}: Completed in {turn_steps} steps")
                else:
                    print(f"❌ Turn {turn_idx + 1}: Failed - {turn_result.get('error', 'No answer')}")

            execution_time = time.time() - start_time

            return {
                'solutions': all_results,
                'success': all(r['success'] for r in all_results),
                'total_steps': total_steps,
                'total_turns': len(sample_list),
                'successful_turns': sum(1 for r in all_results if r['success']),
                'error': None,
                'metadata': {
                    'model': self.model,
                    'backend': self.backend,
                    'max_steps_per_turn': self.max_steps,
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
                'total_steps': total_steps,
                'total_turns': len(sample_list),
                'successful_turns': sum(1 for r in all_results if r['success']),
                'error': str(e),
                'metadata': {
                    'model': self.model,
                    'backend': self.backend,
                    'max_steps_per_turn': self.max_steps,
                    'execution_time': execution_time,
                    'error_trace': error_trace
                },
                'conversation': accumulated_conversation if 'accumulated_conversation' in locals() else [],
            }
        finally:
            # Clean up environment
            if env is not None:
                env.close()
