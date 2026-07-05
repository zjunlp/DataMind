"""
ReActDSAgent implementation - main data science agent using ReAct pattern.
"""

import time
import traceback
import copy

from src.runtimes.base_agent import BaseAgent
from src.runtimes.agents.backends import get_backend
from src.runtimes.agents.environment import AllocatedCodeEnv
from src.runtimes.registry import register_agent
from src.core.schema import AgentResult

AGENT_TYPE = "react"

@register_agent(AGENT_TYPE)
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
        super().__init__(AGENT_TYPE, model, backend, **kwargs)
        
        try:
            self.backend_instance = get_backend(backend, model, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {backend} backend: {e}")
        
        # Environment configuration
        self.manager_url = kwargs.get('manager_url', 'http://localhost:5000')
        self.max_turns = kwargs.get('max_turns', 15)
        self.output_dir = kwargs.get('output_dir', './outputs')

        excluded_params = {'manager_url', 'max_turns', 'output_dir', 'api_key', 'base_url', 'max_model_len', 'submission_dir', 'output_schema', 'working_dir', 'timeout'}
        self.generate_kwargs = {key: value for key, value in kwargs.items() if key not in excluded_params}
    
    def solve_task(self, prompt: str, system: str = "", **kwargs) -> AgentResult:
        """
        Solve a given task using the ReAct pattern.
        
        Args:
            prompt: query or task description
            system: system prompt or context for the agent
            **kwargs: Additional task-specific parameters
            
        Returns:
            AgentResult containing solution and metadata
        """
        start_time = time.time()
        
        try:

            # Extract conversation and metadata
            conversation = [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            extra_info = kwargs.get("extra_info", {})

            if not conversation:
                raise ValueError("Prompt must be a non-empty list of conversation messages.")
            
            extras = {
                "extra_info": extra_info,
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
            final_answer = None
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

            execution_time = time.time() - start_time

            if final_answer:
                response = {
                    "answer": final_answer,
                }
            else:
                response = {
                    "answer": None,
                }

            result_fields = {
                'conversation': raw_conversation,
                'response': response,
                'raw_response': raw_conversation[-1]['content'] if raw_conversation else '',
                'turns': actual_turns,
                'error': None,
                'metadata': {
                    'model': self.model,
                    'backend': self.backend,
                    'max_turns': self.max_turns,
                    'token_usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': total_tokens,
                    },
                    'execution_time': execution_time,
                    'conversation_length': len(conversation)
                },
                'raw_result': {
                    'raw_conversation': raw_conversation,
                    "raw_response": raw_conversation[-1]['content'] if raw_conversation else '',
                    'turns': actual_turns,
                    'token_usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': total_tokens,
                    },
                }
            }

            return AgentResult(**result_fields)
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            
            result_fields = {
                'conversation': [],
                'response': {},
                'raw_response': '',
                'turns': actual_turns if 'actual_turns' in locals() else 0,
                'error': str(e),
                'metadata': {
                    'model': self.model,
                    'backend': self.backend,
                    'max_turns': self.max_turns,
                    'execution_time': execution_time,
                    'error_trace': error_trace
                },
                'raw_result': None
            }
            return AgentResult(**result_fields)
        
        finally:
            # Clean up environment
            if 'env' in locals():
                env.close()
        