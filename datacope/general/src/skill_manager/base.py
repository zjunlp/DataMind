from abc import ABC, abstractmethod

class BaseSkillManager(ABC):
    """Base interface for all skill managers."""
    
    def __init__(
        self,
        model: str,
        agent_type: str,
        task: str,
        data_dir: str,
        current_skill_dir: str,
        category_list: list = None,
        **kwargs,
    ):
        """
        Initialize skill manager.
        
        Args:
            model: Name/path of the model
            agent_type: Type of agent (e.g., "react", "codex", etc.)
            task: Task description and metadata
            data_dir: Directory containing dataset files
            current_skill_dir: Directory for current skill management
            category_list: List of categories for skill management (default: None)
            **kwargs: Additional configuration parameters
        """
        self.model = model
        self.agent_type = agent_type
        self.task = task
        self.data_dir = data_dir
        self.current_skill_dir = current_skill_dir
        self.category_list = category_list
        self.kwargs = kwargs

    @abstractmethod
    def create_skill(self) -> None:
        pass

    @abstractmethod
    def modify_skill(self) -> None:
        pass