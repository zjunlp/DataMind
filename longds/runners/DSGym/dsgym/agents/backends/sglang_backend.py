"""
SGLang backend for DSGym agents.

Supports SGLang for efficient language model serving using offline Engine API.
"""

from typing import List, Dict, Any, Optional
import requests
import json

try:
    import sglang as sgl
    from transformers import AutoTokenizer
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    sgl = None
    AutoTokenizer = None

from .base import BaseBackend


class SGLangBackend(BaseBackend):
    """SGLang backend for language model inference using offline Engine."""
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:30000",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1524,
        timeout: int = 60,
        max_retries: int = 3,
        # Offline engine parameters
        use_offline_engine: bool = True,
        tensor_parallel_size: int = 1,
        mem_fraction_static: float = 0.8,
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Initialize SGLang backend.
        
        Args:
            model_name: Model name/path for offline engine or model identifier for API
            base_url: SGLang server base URL (used only if offline engine fails)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            use_offline_engine: Whether to try offline engine first
            tensor_parallel_size: Number of GPUs for tensor parallelism
            mem_fraction_static: Memory fraction for static allocation
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional configuration
        """
        if not SGLANG_AVAILABLE:
            raise ImportError("SGLang is not available. Please install SGLang: pip install sglang")
        
        super().__init__(model_name, **kwargs)
        
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_offline_engine = use_offline_engine
        self.tensor_parallel_size = tensor_parallel_size
        self.mem_fraction_static = mem_fraction_static
        self.trust_remote_code = trust_remote_code
        
        # Generation parameters
        self.sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        }
        
        # Initialize engine and tokenizer
        self.engine = None
        self.tokenizer = None
        
        if self.use_offline_engine:
            try:
                self._setup_offline_engine()
            except Exception as e:
                print(f"Warning: SGLang offline engine setup failed: {e}")
                print("Falling back to HTTP API")
        else:
            print("Using HTTP API mode")
    
    def _setup_offline_engine(self):
        """Setup SGLang offline engine."""
        print(f"Initializing SGLang offline engine with model: {self.model_name}")
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {self.model_name}: {e}")
            self.tokenizer = None
        
        # Initialize SGLang engine
        self.engine = sgl.Engine(
            model_path=self.model_name,
            tp_size=self.tensor_parallel_size,
            mem_fraction_static=self.mem_fraction_static,
            trust_remote_code=self.trust_remote_code,
        )
        
        print("SGLang offline engine initialized successfully")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using SGLang.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if messages is None:
            raise RuntimeError("Messages is None in SGLang generate - this indicates an upstream error")
        elif not isinstance(messages, list):
            raise TypeError(f"Messages should be a list, got {type(messages)}")
        
        # Try offline engine first, fallback to HTTP API
        if self.engine is not None:
            try:
                return self._generate_offline_engine(messages, **kwargs)
            except Exception as e:
                print(f"SGLang offline engine failed: {e}, falling back to HTTP API")
        
        return self._generate_http_api(messages, **kwargs)
    
    def _generate_offline_engine(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using SGLang offline engine."""
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
        
        # Merge sampling parameters with any provided kwargs
        sampling_params = self.sampling_params.copy()
        sampling_params.update({
            k: v for k, v in kwargs.items() 
            if k in ["temperature", "top_p", "max_new_tokens", "max_tokens"]
        })
        
        # Convert max_tokens to max_new_tokens if needed
        if "max_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = sampling_params.pop("max_tokens")
        
        # Generate response using SGLang engine
        try:
            # Try synchronous generation first
            try:
                outputs = self.engine.generate([prompt], sampling_params)
            except RuntimeError as e:
                if "event loop" in str(e).lower():
                    # Handle async requirement by creating event loop
                    import asyncio
                    
                    async def async_generate():
                        return await self.engine.async_generate([prompt], sampling_params)
                    
                    # Create new event loop if none exists
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            raise RuntimeError("Event loop is closed")
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    try:
                        outputs = loop.run_until_complete(async_generate())
                    finally:
                        # Don't close the loop as it might be needed elsewhere
                        pass
                else:
                    raise e
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["text"]
                return generated_text.strip()
            else:
                return ""
                
        except Exception as e:
            raise RuntimeError(f"SGLang offline engine generation failed: {e}")
    
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
    
    def _generate_http_api(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using HTTP API fallback."""
        # Convert sampling params to HTTP API format
        http_params = {
            "temperature": kwargs.get("temperature", self.sampling_params["temperature"]),
            "top_p": kwargs.get("top_p", self.sampling_params["top_p"]),
            "max_tokens": kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.sampling_params["max_new_tokens"])),
        }
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            **http_params
        }
        
        # Make HTTP request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return content if content is not None else ""
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                else:
                    wait_time = 2 ** attempt
                    print(f"SGLang HTTP API call failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s: {str(e)}")
                    import time
                    time.sleep(wait_time)
        
        raise RuntimeError(f"SGLang HTTP API call failed after {self.max_retries} attempts")
    
    def is_api_based(self) -> bool:
        """SGLang backend can be either local (offline engine) or API-based."""
        return self.engine is None  # If no offline engine, using HTTP API
    
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