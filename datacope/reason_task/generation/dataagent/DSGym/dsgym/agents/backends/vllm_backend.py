"""
vLLM backend for DSGym agents.

Supports local model inference with GPU acceleration.
"""

from typing import List, Dict, Any, Optional

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


class VLLMBackend(BaseBackend):
    """vLLM backend for local model inference."""
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 32768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1524,
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Initialize vLLM backend.
        
        Args:
            model_name: Path to local model or HuggingFace model name
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            max_model_len: Maximum model sequence length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional configuration
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install vLLM: pip install vllm")
        
        super().__init__(model_name, **kwargs)
        
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {model_name}: {e}")
            self.tokenizer = None
        
        # Initialize vLLM model
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using vLLM.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if messages is None:
            raise RuntimeError("Messages is None in vLLM generate - this indicates an upstream error")
        elif not isinstance(messages, list):
            raise TypeError(f"Messages should be a list, got {type(messages)}")
        
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
                # Fallback to simple concatenation
                prompt = self._format_messages_fallback(messages)
        else:
            # Fallback formatting
            prompt = self._format_messages_fallback(messages)
        
        # Update sampling parameters with any provided kwargs
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', self.sampling_params.temperature),
            top_p=kwargs.get('top_p', self.sampling_params.top_p),
            max_tokens=kwargs.get('max_tokens', self.sampling_params.max_tokens),
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        # Generate response
        try:
            outputs = self.llm.generate([prompt], sampling_params)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                return generated_text.strip()
            else:
                return ""
                
        except Exception as e:
            raise RuntimeError(f"vLLM generation failed: {e}")
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """
        Fallback message formatting when chat template is not available.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Formatted prompt string
        """
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
        
        # Add generation prompt
        formatted_parts.append("Assistant:")
        
        return "\n\n".join(formatted_parts)
    
    def is_api_based(self) -> bool:
        """vLLM backend is local, not API-based."""
        return False
    
    def apply_chat_template(self, conversation: List[Dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = False) -> Optional[str]:
        """
        Apply chat template using the model's tokenizer.
        
        Args:
            conversation: List of conversation messages
            add_generation_prompt: Whether to add generation prompt
            tokenize: Whether to tokenize the result
            
        Returns:
            Formatted conversation string or None if not supported
        """
        if self.tokenizer is None:
            return None
        
        try:
            return self.tokenizer.apply_chat_template(
                conversation,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt
            )
        except Exception:
            return None