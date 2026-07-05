"""
ReActDSAgent implementation - main data science agent using ReAct pattern.

This agent provides a self-contained implementation using the new DSGym
backend and environment components.
"""

import time
import traceback
import copy
from typing import Dict, Any, List

from .base_agent import BaseAgent
from .backends import get_backend
from .environment import AllocatedCodeEnv


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
        self.max_turns = kwargs.get('max_turns', 15)
        self.output_dir = kwargs.get('output_dir', './outputs')

        excluded_params = {'manager_url', 'max_turns', 'output_dir', 'api_key', 'base_url', 'max_model_len', 'submission_dir'}
        self.generate_kwargs = {key: value for key, value in kwargs.items() if key not in excluded_params}
    
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
            raw_conversation = copy.deepcopy(conversation)  # For debugging  
            # Run multi-turn interaction
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            final_answer = ""
            actual_turns = 0

            for turn in range(self.max_turns):
                try:
                    # Generate response
                    response, reasoning_response, token_usage = self.backend_instance.generate(conversation, **self.generate_kwargs)

                    input_tokens += token_usage.get("input_tokens", 0)
                    output_tokens += token_usage.get("output_tokens", 0)
                    total_tokens += token_usage.get("total_tokens", 0)
                    # Step environment
                    step_output = env.step(response)
                    actual_turns = turn + 1
                    
                    # Update conversation - add assistant response and new observations
                    conversation.append({"role": "assistant", "content": step_output.get('postprocessed_action', response)})
                    raw_conversation.append({"role": "assistant", "content": step_output.get('postprocessed_action', response), "reasoning_content": reasoning_response})
                    if step_output['observations']:
                        conversation.extend(step_output['observations'])
                        raw_conversation.extend(step_output['observations'])
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
                    raw_conversation.append({"role": "user", "content": f"Error: {error_msg}. Please try a different approach."})
                    continue
            

            # Check if this is part of trajectory generation
            trajectory_id = sample.get("extra_info", {}).get("trajectory_id")
            token_stats = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            env.save_prediction(final_answer, trajectory_id=trajectory_id, conversation=raw_conversation, token_usage=token_stats)

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
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'execution_time': execution_time,
                    'conversation_length': len(conversation)
                },
                'conversation': conversation,
                'raw_result': {
                    'raw_conversation': raw_conversation,
                    'prediction': final_answer,
                    'turns': actual_turns,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
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
    
