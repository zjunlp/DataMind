from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class AgentResult:
    """Dataclass to hold the result of an agent's task solving."""
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    response: dict = field(default_factory=dict)
    raw_response: str = ""
    turns: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_result: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    # Basic identification
    sample_id: str
    dataset_name: str
    query: str
    ground_truth: Optional[str] = None  # Allow None for cases without ground truth
    
    # Predictions
    prediction: str = ""
    raw_response: str = ""

    # Execution details
    execution_time: float = 0.0
    total_turns: int = 0
    success: bool = True
    error_info: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_ground_truth(self) -> bool:
        """Check if this result has ground truth for evaluation."""
        return self.ground_truth is not None and self.ground_truth.strip() != ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "raw_response": self.raw_response,
            "execution_time": self.execution_time,
            "total_turns": self.total_turns,
            "success": self.success,
            "extra_info": self.extra_info,
        }
        
        if self.error_info:
            result["error_info"] = self.error_info
            
        return result
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for summary (without raw_response and trajectory)."""
        result = {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "execution_time": self.execution_time,
            "total_turns": self.total_turns,
            "success": self.success,
            "extra_info": self.extra_info,
        }
        
        if self.error_info:
            result["error_info"] = self.error_info
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create from dictionary."""
        return cls(
            sample_id=data["sample_id"],
            dataset_name=data["dataset_name"], 
            query=data["query"],
            ground_truth=data.get("ground_truth"),  # Allow None
            prediction=data.get("prediction", ""),
            raw_response=data.get("raw_response", ""),
            execution_time=data.get("execution_time", 0.0),
            total_turns=data.get("total_turns", 0),
            success=data.get("success", True),
            error_info=data.get("error_info"),
            extra_info=data.get("extra_info", {}),
        )

@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    
    # Model configuration
    model_name: str
    backend_type: str = "openai"
    
    # Dataset configuration  
    dataset_name: str = "dabstep"
    dataset_split: str = "explore"
    limit: Optional[int] = None
    start_index: int = 0
    
    # Evaluation protocol
    protocol: str = "multi_turn"
    max_turns: int = 20

    # Generation parameters
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1524
    
    # Output configuration
    output_dir: str = "./evaluation_results"
    run_name: Optional[str] = None
    save_trajectories: bool = True
    
    # Performance
    max_workers: Optional[int] = None
    batch_size: int = 1
    
    # Additional configuration
    seed: int = 42
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "backend_type": self.backend_type,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "limit": self.limit,
            "start_index": self.start_index,
            "protocol": self.protocol,
            "max_turns": self.max_turns,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "save_trajectories": self.save_trajectories,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "extra_config": self.extra_config,
        }