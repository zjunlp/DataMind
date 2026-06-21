"""
Allocated code environment for data science tasks.

Provides code execution capabilities with container isolation.
"""

import re
import os
import json
from typing import Any, Dict, Optional, Tuple, List
from ...base_text_env import BaseTextEnv, BaseEnvStepOutput, ConversationType
from .utils import clean_jupyter_output
import httpx
import traceback
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
class AllocatedCodeEnv(BaseTextEnv):
    """
    Environment for data science tasks with code execution capabilities.
    
    Supports multi-turn interactions with Python code execution in isolated containers.
    """
    
    def __init__(
        self, 
        manager_url: str = "http://localhost:5000",
        max_turns: int = 15,
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize allocated code environment.
        
        Args:
            manager_url: URL of the code execution manager
            max_turns: Maximum number of turns per episode
            output_dir: Directory for saving outputs
            **kwargs: Additional configuration
        """
        super().__init__()
        
        self.manager_url = manager_url
        self.max_turns = max_turns
        self.output_dir = output_dir or "./outputs"
        
        # Optional HTTP client timeout (seconds) passed via kwargs as `time_out`
        self.time_out: Optional[int] = kwargs.get("time_out")  # type: ignore[assignment]

        # Initialize tool group for code execution, honoring optional timeout
        if self.time_out is not None:
            self.tool_group = AllocatedCodeToolGroup(manager_url, timeout=self.time_out)
        else:
            self.tool_group = AllocatedCodeToolGroup(manager_url)
        
        self.chat_history: ConversationType = []
        # Environment state
        self.ground_truth = None
        self.query = None
        self.extra_info = {}
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def init(self, prompt: ConversationType, **extras) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize environment with prompt and task-specific information.
        
        Args:
            prompt: Initial conversation prompt
            **extras: Additional task information (ground_truth, extra_info, etc.)
            
        Returns:
            Tuple of (prompt, metadata)
        """
        # Reset environment state
        self.turns = 0
        self.chat_history = prompt.copy() if prompt else []
        
        # Extract task information from extras
        if "reward_spec" in extras:
            self.ground_truth = extras["reward_spec"].get("ground_truth", "")
        
        if "extra_info" in extras:
            self.extra_info = extras["extra_info"]
            self.query = self.extra_info.get("question", "")
        
        # Allocate new container
        try:
            container_id = self.tool_group.allocate_container()
            print(f"🔧 {color.GREEN}Allocated container: {container_id}{color.END}")
        except Exception as e:
            print(f"⚠️ {color.YELLOW}Warning: Failed to allocate container: {e}{color.END}")
        
        metadata = {
            "initialized": True,
            "container_allocated": True,
            "max_turns": self.max_turns
        }
        
        return prompt, metadata
    
    def _postprocess_action(self, action: str) -> str:
        """
        Post-process action to truncate at appropriate tags.
        
        Args:
            action: Raw action from LLM
            
        Returns:
            Post-processed action
        """
        if not action:
            return action
            
        # Truncate at </python> tag if present
        if "</python>" in action:
            return action.split("</python>")[0] + "</python>"
        # Truncate at </answer> tag if present
        elif "</answer>" in action:
            return action.split("</answer>")[0] + "</answer>"
        else:
            return action

    def _is_done(self, action: str) -> bool:
        """
        Check if episode should end.
        
        Args:
            action: LLM action
            
        Returns:
            True if episode should end
        """
        max_turns_reached = self.turns >= self.max_turns
        has_answer_tags = action and "<answer>" in action and "</answer>" in action
        
        return max_turns_reached or has_answer_tags
    
    def step(self, action: str) -> BaseEnvStepOutput:
        """
        Execute one environment step.
        
        Args:
            action: LLM response/action
            
        Returns:
            Step output with observations, reward, done flag, and metadata
        """
        self.turns += 1
        
        # Post-process action (truncate at appropriate tags)
        postprocessed_action = self._postprocess_action(action)
        
        print(f"{color.BLUE}[Step {self.turns}] Postprocessed action:{color.END}\n {postprocessed_action}\n")
        # Add assistant response to chat history
        if postprocessed_action:
            self.chat_history.append({"role": "assistant", "content": postprocessed_action})
        
        # Check if episode is done
        done = self._is_done(postprocessed_action)
        
        # Extract final answer if present
        final_answer = self._extract_final_answer(postprocessed_action)
        
        # Calculate reward 
        reward = 1.0 if final_answer else 0.0
        
        # If episode is done, don't execute code, just return
        if done:
            print(f"{color.RED}[Step {self.turns}] Result:{color.END}\n {final_answer}\n")
            metadata = {
                "turns": self.turns,
                "execution_output": "",
                "final_answer": final_answer,
                "code_executed": False
            }
            
            return BaseEnvStepOutput(
                observations=[],  # No new observations when done
                reward=reward,
                done=done,
                metadata=metadata,
                postprocessed_action=postprocessed_action
            )
        
        # Parse and execute any code in the action
        execution_output = ""
        new_observations = []
        
        try:
            code = self._parse_action(postprocessed_action)
            if code:
                execution_output = self.tool_group.execute_code(code)
                observation_content = "\n<information>" + execution_output + "</information>\n"
                new_obs = {"role": "user", "content": observation_content}
                self.chat_history.append(new_obs)
                new_observations.append(new_obs)
                print(f"{color.GREEN}[Step {self.turns}] Execution output:{color.END}\n {execution_output}\n")
            else:
                new_obs = {"role": "user", "content": "<information>No python code found. You should either write python code within the <python>CODE</python> tag or write the answer in the <answer>ANSWER</answer> tag.</information>"}
                self.chat_history.append(new_obs)
                new_observations.append(new_obs)
                
        except Exception as e:
            error_msg = f"<information>Code execution error: {e}</information>"
            new_obs = {"role": "user", "content": error_msg}
            print(f"{color.RED}[Step {self.turns}] Execution error:{color.END}")
            print(execution_output)
            print(traceback.format_exc())
            self.chat_history.append(new_obs)
            new_observations.append(new_obs)
        
        # Prepare metadata
        metadata = {
            "turns": self.turns,
            "execution_output": execution_output,
            "final_answer": final_answer,
            "code_executed": bool(execution_output)
        }
        
        return BaseEnvStepOutput(
            observations=new_observations,  # Return only new observations
            reward=reward,
            done=done,
            metadata=metadata,
            postprocessed_action=postprocessed_action
        )
    
    def reset(self, **kwargs) -> ConversationType:
        """
        Reset environment to initial state.
        
        Args:
            **kwargs: Reset parameters
            
        Returns:
            Initial observations (empty conversation)
        """
        self.turns = 0
        self.chat_history = []
        
        # Allocate new container
        try:
            self.tool_group.allocate_container()
        except Exception as e:
            print(f"⚠️ {color.YELLOW}Warning: Failed to allocate container during reset: {e}{color.END}")
        
        return self.chat_history

    def reset_turns(self):
        """Reset turn count to zero."""
        self.turns = 0

    def close(self):
        """Clean up environment resources."""
        try:
            self.tool_group.deallocate_container()
        except Exception as e:
            print(f"⚠️ {color.YELLOW}Warning: Failed to deallocate container: {e}{color.END}")
    
    def _parse_action(self, action: str) -> Optional[str]:
        match = None
        if action and "<python>" in action and "</python>" in action:
            match = re.search(r"<python>(.*?)</python>", action, re.DOTALL)
            if match:
                code = match.group(1)
                if "```python" in code and "```" in code:
                    inner_match = re.search(r"```python(.*?)```", code, re.DOTALL)
                    if inner_match:
                        return inner_match.group(1)
                return code
        return None

    def _extract_final_answer(self, response: str) -> str:
        """
        Extract final answer from model response.
        
        Args:
            response: Model response text
            
        Returns:
            Extracted answer or empty string if not found
        """
        if not response:
            return ""
        
        # Look for <answer> tags first (primary pattern)
        if '<answer>' in response and '</answer>' in response:
            match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def save_prediction(self, prediction: str, filename_prefix: str = "prediction", trajectory_id: Optional[int] = None, conversation: Optional[List] = None):
        """
        Save prediction to output directory.
        
        Args:
            prediction: Prediction text to save
            filename_prefix: Prefix for output filename
            trajectory_id: Optional trajectory ID for trajectory generation (adds _traj_X suffix)
            conversation: Optional conversation history to save (if None, uses self.chat_history)
        """
        # Use provided conversation or fall back to self.chat_history
        chat_to_save = conversation if conversation is not None else self.chat_history
        
        # Calculate correct number of assistant turns
        assistant_turns = sum(1 for msg in chat_to_save if msg.get("role") == "assistant") if chat_to_save else self.turns
        
        # Create prediction data
        prediction_data = {
            "prediction": prediction,
            "ground_truth": self.ground_truth,
            "query": self.query,
            "turns": assistant_turns,
            "conversation": chat_to_save,
            "extra_info": self.extra_info
        }
        
        # Generate filename
        index = self.extra_info.get("index", 0)
        if trajectory_id is not None:
            # Include trajectory ID in filename: prediction_i_traj_j.json
            filename = f"{filename_prefix}_{index}_traj_{trajectory_id}.json"
        else:
            # Original format: prediction_i.json
            filename = f"{filename_prefix}_{index}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(prediction_data, f, indent=2, ensure_ascii=False)
            print(f"💾 {color.GREEN}Saved prediction to: {filepath}{color.END}")
        except Exception as e:
            print(f"⚠️ {color.YELLOW}Warning: Failed to save prediction: {e}{color.END}")


class AllocatedCodeToolGroup:
    """Same as original AllocatedCodeToolGroup"""
    def __init__(self, manager_url: str = "http://localhost:5000", timeout: int = 1800):
        self.manager_url = manager_url
        self.allocated_container: Optional[int] = None
        self.client = httpx.Client(timeout=timeout)
    
    def allocate_container(self):
        if self.allocated_container is not None:
            return
        
        response = self.client.post(f"{self.manager_url}/allocate")
        response.raise_for_status()
        result = response.json()
        self.allocated_container = result["container_id"]
        return self.allocated_container
    
    def deallocate_container(self):
        if self.allocated_container is None:
            return
        
        self.client.post(f"{self.manager_url}/deallocate/{self.allocated_container}")
        self.allocated_container = None
    
    def execute_code(self, code: str) -> str:
        if self.allocated_container is None:
            raise RuntimeError("No container allocated")
        
        try:
            response = self.client.post(
                f"{self.manager_url}/session/{self.allocated_container}/execute",
                json={"code": code}
            )
            response.raise_for_status()
            result = response.json()
            raw_output = result.get("outputs", [])
            # Clean and format the Jupyter output
            output = clean_jupyter_output(raw_output)
            return output
        except Exception as e:
            # Convert HTTPStatusError to a more Ray-friendly exception
            error_msg = f"Failed to execute code: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" (Status: {e.response.status_code})"
            raise RuntimeError(error_msg)
    
    def get_tool_names(self):
        return ["python"]

    