import time
from typing import Any, Dict, List, Optional
import aiohttp

from .base import BaseModelClient, ModelResponse
from ..utils.config import config

class DeepSeekClient(BaseModelClient):
    """Client for interacting with DeepSeek's API."""
    
    def __init__(self):
        super().__init__(config.get("models.deepseek.api_key"))
        self.api_base = "https://api.deepseek.com/v1"  # Example API base URL
    
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using DeepSeek's API."""
        try:
            start_time = time.time()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                **kwargs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    response_json = await response.json()
                    
            end_time = time.time()
            
            if not self.validate_response(response_json):
                raise ValueError("Invalid response from DeepSeek API")
            
            return ModelResponse(
                text=response_json["choices"][0]["text"],
                model_name=model,
                tokens_used=response_json["usage"]["total_tokens"],
                latency=end_time - start_time,
                raw_response=response_json
            )
            
        except Exception as e:
            print(f"Error generating response from DeepSeek: {str(e)}")
            raise
    
    async def get_token_count(self, text: str) -> int:
        """
        Get the number of tokens in the text.
        Note: This is a simplified implementation. Replace with actual DeepSeek tokenizer.
        """
        # TODO: Replace with actual DeepSeek tokenizer implementation
        # For now, using a simple approximation
        return len(text.split())
    
    def validate_response(self, response: Any) -> bool:
        """Validate the DeepSeek API response."""
        try:
            return (
                isinstance(response, dict) and
                "choices" in response and
                len(response["choices"]) > 0 and
                "text" in response["choices"][0] and
                "usage" in response and
                "total_tokens" in response["usage"]
            )
        except Exception:
            return False 