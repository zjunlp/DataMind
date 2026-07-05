"""
Multi-instance vLLM backend for DSGym agents.

Automatically detects available GPUs and creates one vLLM instance per GPU
for maximum parallel inference throughput.
"""

import os
import threading
import time
import queue
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import subprocess

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AutoTokenizer = None

from .base import BaseBackend


def get_available_gpus() -> List[int]:
    """
    Detect available GPUs using nvidia-ml-py or fallback methods.
    
    Returns:
        List of GPU IDs that are available
    """
    try:
        # Try using nvidia-ml-py first (more reliable)
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        return list(range(gpu_count))
    except ImportError:
        pass
    
    try:
        # Fallback to nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_ids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
        return gpu_ids
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        # Fallback to CUDA_VISIBLE_DEVICES if set
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices:
            if cuda_devices == "-1":
                return []
            return [int(x.strip()) for x in cuda_devices.split(',') if x.strip().isdigit()]
    except ValueError:
        pass
    
    # Final fallback: assume single GPU
    print("⚠️  Could not detect GPU count, assuming 1 GPU available")
    return [0]


class VLLMInstance:
    """Single vLLM instance bound to a specific GPU."""
    
    def __init__(
        self,
        gpu_id: int,
        model_name: str,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 32768,
        trust_remote_code: bool = True,
        **kwargs
    ):
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.kwargs = kwargs
        
        self.llm = None
        self.tokenizer = None
        self.is_ready = False
        self.error = None
        self._lock = threading.Lock()
        
    def initialize(self):
        """Initialize vLLM instance on the specified GPU."""
        try:
            # Set GPU visibility for this process/thread
            original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            
            print(f"🚀 Initializing vLLM instance on GPU {self.gpu_id}")
            
            # Initialize tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=self.trust_remote_code
                )
            except Exception as e:
                print(f"Warning: Could not load tokenizer for {self.model_name}: {e}")
                self.tokenizer = None
            
            # Initialize vLLM model
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,  # Single GPU per instance
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=self.trust_remote_code,
            )
            
            self.is_ready = True
            print(f"✅ vLLM instance on GPU {self.gpu_id} ready")
            
            # Restore original CUDA_VISIBLE_DEVICES
            if original_cuda_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            
        except Exception as e:
            self.error = str(e)
            self.is_ready = False
            print(f"❌ Failed to initialize vLLM instance on GPU {self.gpu_id}: {e}")
            
            # Restore original CUDA_VISIBLE_DEVICES on error too
            if original_cuda_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1524,
        **kwargs
    ) -> str:
        """Generate response using this vLLM instance."""
        if not self.is_ready:
            raise RuntimeError(f"vLLM instance on GPU {self.gpu_id} is not ready: {self.error}")
        
        with self._lock:  # Ensure thread safety for this instance
            # Apply chat template if tokenizer is available
            if self.tokenizer is not None:
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except Exception as e:
                    print(f"Warning: Could not apply chat template: {e}")
                    prompt = self._format_messages_fallback(messages)
            else:
                prompt = self._format_messages_fallback(messages)
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            
            # Generate response
            try:
                # Set GPU context for this generation
                original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
                
                outputs = self.llm.generate([prompt], sampling_params)
                
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_devices is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].outputs[0].text
                    return generated_text.strip()
                else:
                    return ""
                    
            except Exception as e:
                # Restore original CUDA_VISIBLE_DEVICES on error
                if original_cuda_devices is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                raise RuntimeError(f"vLLM generation failed on GPU {self.gpu_id}: {e}")
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Fallback message formatting when chat template is not available."""
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            else:
                formatted_parts.append(f"{role}: {content}")
        
        formatted_parts.append("Assistant:")
        return "\n\n".join(formatted_parts)


class MultiVLLMBackend(BaseBackend):
    """Multi-instance vLLM backend for parallel inference across multiple GPUs."""
    
    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 32768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1524,
        trust_remote_code: bool = True,
        max_concurrent_requests: int = None,
        gpu_ids: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Initialize multi-instance vLLM backend.
        
        Args:
            model_name: Path to local model or HuggingFace model name
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            max_model_len: Maximum model sequence length
            temperature: Default sampling temperature
            top_p: Default top-p sampling parameter
            max_tokens: Default maximum tokens to generate
            trust_remote_code: Whether to trust remote code
            max_concurrent_requests: Maximum concurrent requests (default: 2x GPU count)
            gpu_ids: Specific GPU IDs to use (auto-detect if None)
            **kwargs: Additional configuration
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install vLLM: pip install vllm")
        
        super().__init__(model_name, **kwargs)
        
        # Auto-detect available GPUs if not specified
        if gpu_ids is None:
            self.gpu_ids = get_available_gpus()
        else:
            self.gpu_ids = gpu_ids
            
        self.num_gpus = len(self.gpu_ids)
        
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for multi-instance vLLM backend")
        
        print(f"🔍 Detected {self.num_gpus} GPUs: {self.gpu_ids}")
        
        # Set default concurrent requests based on GPU count
        if max_concurrent_requests is None:
            max_concurrent_requests = self.num_gpus * 2
        self.max_concurrent_requests = max_concurrent_requests
        
        # Default generation parameters
        self.default_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        
        # Instance configuration
        self.instance_config = {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": trust_remote_code,
            **kwargs
        }
        
        # Initialize instances
        self.instances = {}
        self.ready_instances = queue.Queue()
        
        print(f"🔧 Initializing {self.num_gpus} vLLM instances...")
        self._initialize_instances()
        
        # Start request handler
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self.is_running = True
        
    def _initialize_instances(self):
        """Initialize all vLLM instances across GPUs."""
        def init_single_instance(gpu_id):
            try:
                instance = VLLMInstance(
                    gpu_id=gpu_id,
                    model_name=self.model_name,
                    **self.instance_config
                )
                instance.initialize()
                
                if instance.is_ready:
                    self.instances[gpu_id] = instance
                    self.ready_instances.put(gpu_id)
                    print(f"✅ GPU {gpu_id} instance ready")
                else:
                    print(f"❌ GPU {gpu_id} instance failed: {instance.error}")
                    
            except Exception as e:
                print(f"❌ Failed to initialize GPU {gpu_id}: {e}")
        
        # Initialize instances in parallel
        with ThreadPoolExecutor(max_workers=self.num_gpus) as init_executor:
            futures = [
                init_executor.submit(init_single_instance, gpu_id)
                for gpu_id in self.gpu_ids
            ]
            
            # Wait for all to complete
            for future in futures:
                future.result()
        
        ready_count = len(self.instances)
        print(f"📊 Initialization complete: {ready_count}/{self.num_gpus} instances ready")
        
        if ready_count == 0:
            raise RuntimeError("No vLLM instances were successfully initialized!")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using available vLLM instance.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if messages is None:
            raise RuntimeError("Messages is None in multi-vLLM generate - this indicates an upstream error")
        elif not isinstance(messages, list):
            raise TypeError(f"Messages should be a list, got {type(messages)}")
        
        # Get generation parameters
        gen_params = self.default_params.copy()
        gen_params.update(kwargs)
        
        # Get available instance
        try:
            # Wait for available instance (with timeout)
            gpu_id = self.ready_instances.get(timeout=60)
        except queue.Empty:
            raise RuntimeError("No vLLM instances available (timeout)")
        
        try:
            # Generate using the instance
            instance = self.instances[gpu_id]
            result = instance.generate(messages, **gen_params)
            
            # Return instance to ready pool
            self.ready_instances.put(gpu_id)
            
            return result
            
        except Exception as e:
            # Return instance to ready pool even on error
            self.ready_instances.put(gpu_id)
            raise e
    
    def get_available_instances(self) -> int:
        """Get number of available instances."""
        return self.ready_instances.qsize()
    
    def get_total_instances(self) -> int:
        """Get total number of instances."""
        return len(self.instances)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get information about GPU usage."""
        return {
            "total_gpus": self.num_gpus,
            "gpu_ids": self.gpu_ids,
            "ready_instances": self.get_available_instances(),
            "total_instances": self.get_total_instances(),
        }
    
    def is_api_based(self) -> bool:
        """Multi-vLLM backend is local, not API-based."""
        return False
    
    def apply_chat_template(self, conversation: List[Dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = False) -> Optional[str]:
        """
        Apply chat template using the first available instance's tokenizer.
        
        Args:
            conversation: List of conversation messages
            add_generation_prompt: Whether to add generation prompt
            tokenize: Whether to tokenize the result
            
        Returns:
            Formatted conversation string or None if not supported
        """
        if not self.instances:
            return None
        
        # Use first available instance's tokenizer
        first_instance = next(iter(self.instances.values()))
        if first_instance.tokenizer is None:
            return None
        
        try:
            return first_instance.tokenizer.apply_chat_template(
                conversation,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt
            )
        except Exception:
            return None
    
    def __del__(self):
        """Cleanup resources."""
        self.is_running = False
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)