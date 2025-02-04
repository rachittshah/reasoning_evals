from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class ModelResponse:
    """Container for model responses."""
    
    def __init__(
        self,
        text: str,
        model_name: str,
        tokens_used: int,
        latency: float,
        raw_response: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.model_name = model_name
        self.tokens_used = tokens_used
        self.latency = latency
        self.raw_response = raw_response or {}

class BaseModelClient(ABC):
    """Abstract base class for model API clients."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Optional stop sequences
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse object containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    async def get_token_count(self, text: str) -> int:
        """
        Get the number of tokens in the text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass
    
    @abstractmethod
    def validate_response(self, response: Any) -> bool:
        """
        Validate the model's response.
        
        Args:
            response: Raw response from the API
            
        Returns:
            True if response is valid, False otherwise
        """
        pass 