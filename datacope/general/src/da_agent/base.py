from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

class BaseDaAgent(ABC):
    def __init__(self, agent_type: str, model: str, backend: str = "", output_dir: Optional[str] = None, parallel_workers: int = 1, **kwargs):
        self.agent_type = agent_type
        self.model = model
        self.backend = backend
        self.output_dir = output_dir
        self.parallel_workers = parallel_workers

    @abstractmethod
    def generate(
        self
    ) -> Dict[str, Any]:
        """
        Generate trajectories for one task.

        Returns:
            List of generated trajectory metadata.
        """
        pass
