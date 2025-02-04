from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class TaskExample:
    """A single example for a task."""
    id: str
    input: str
    expected_output: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TaskResult:
    """Result of evaluating a model on a task example."""
    example_id: str
    model_name: str
    model_output: str
    is_correct: bool
    reasoning_quality: float
    metrics: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

class BaseTask(ABC):
    """Abstract base class for defining evaluation tasks."""
    
    def __init__(self, task_name: str, config: Dict[str, Any]):
        """
        Initialize a task.
        
        Args:
            task_name: Name of the task
            config: Task configuration dictionary
        """
        self.task_name = task_name
        self.config = config
    
    @abstractmethod
    async def generate_examples(self, num_examples: int) -> List[TaskExample]:
        """
        Generate synthetic examples for the task.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            List of TaskExample objects
        """
        pass
    
    @abstractmethod
    async def evaluate_response(
        self,
        example: TaskExample,
        model_response: str,
        model_name: str
    ) -> TaskResult:
        """
        Evaluate a model's response to an example.
        
        Args:
            example: The task example
            model_response: The model's response
            model_name: Name of the model
            
        Returns:
            TaskResult object containing evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_prompt(self, example: TaskExample) -> str:
        """
        Generate the prompt for a task example.
        
        Args:
            example: The task example
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def validate_example(self, example: TaskExample) -> Tuple[bool, Optional[str]]:
        """
        Validate a task example.
        
        Args:
            example: The task example to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    def get_metrics(self) -> List[str]:
        """
        Get the list of metrics this task evaluates.
        
        Returns:
            List of metric names
        """
        return self.config.get("evaluation", {}).get("metrics", []) 