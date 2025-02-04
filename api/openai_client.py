import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
import tiktoken

from .base import BaseModelClient, ModelResponse
from ..utils.config import config

class OpenAIClient(BaseModelClient):
    """Client for interacting with OpenAI's API."""
    
    def __init__(self):
        super().__init__(config.get("models.openai.api_key"))
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using OpenAI's API."""
        try:
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs
            )
            
            end_time = time.time()
            
            if not self.validate_response(response):
                raise ValueError("Invalid response from OpenAI API")
            
            return ModelResponse(
                text=response.choices[0].message.content,
                model_name=model,
                tokens_used=response.usage.total_tokens,
                latency=end_time - start_time,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            # Log the error and re-raise
            print(f"Error generating response from OpenAI: {str(e)}")
            raise
    
    async def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def validate_response(self, response: Any) -> bool:
        """Validate the OpenAI API response."""
        try:
            return (
                hasattr(response, "choices") and
                len(response.choices) > 0 and
                hasattr(response.choices[0], "message") and
                hasattr(response.choices[0].message, "content") and
                hasattr(response, "usage") and
                hasattr(response.usage, "total_tokens")
            )
        except Exception:
            return False 