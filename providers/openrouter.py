"""OpenRouter provider implementation."""
import logging
from typing import Dict, List, Any, Optional
import importlib.util

import httpx

from config.settings import Settings, ProviderType, get_settings
from .base import LLMProvider, register_provider

logger = logging.getLogger(__name__)

# Check if pydantic_ai is installed with OpenAIProvider
PYDANTIC_AI_AVAILABLE = importlib.util.find_spec("pydantic_ai") is not None
PyAIOpenAIProvider = None

if PYDANTIC_AI_AVAILABLE:
    try:
        from pydantic_ai.providers.openai import OpenAIProvider as PyAIOpenAIProvider
    except ImportError:
        logger.warning("pydantic_ai installed but OpenAIProvider not available")


@register_provider(ProviderType.OPENROUTER)
class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the OpenRouter provider.
        
        Args:
            settings (Optional[Settings], optional): Application settings. Defaults to None.
        """
        self.settings = settings or get_settings()
        if not self.settings.openrouter:
            raise ValueError("OpenRouter settings not configured")
            
        self.openrouter_settings = self.settings.openrouter
        
        # Initialize the pydantic-ai OpenAI provider with OpenRouter base URL if available
        self.pydantic_provider = None
        if PyAIOpenAIProvider is not None:
            try:
                self.pydantic_provider = PyAIOpenAIProvider(
                    api_key=self.openrouter_settings.api_key.get_secret_value(),
                    base_url=self.openrouter_settings.base_url
                )
            except Exception as e:
                logger.warning(f"Failed to initialize pydantic-ai OpenAIProvider: {e}")
        
        # Initialize httpx client for direct API calls
        self.client = httpx.AsyncClient(
            base_url=self.openrouter_settings.base_url,
            headers={
                "Authorization": f"Bearer {self.openrouter_settings.api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/yourname/pydantic-rag-ollama", # Replace with your repo
                "X-Title": "Pydantic RAG"
            }
        )
    
    async def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using OpenRouter.
        
        Args:
            text (str): The text to embed.
            
        Returns:
            List[float]: The embedding vector.
        """
        try:
            response = await self.client.post(
                "/embeddings",
                json={
                    "input": text,
                    "model": self.openrouter_settings.embedding_model
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenRouter: {e}")
            raise
    
    async def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a text completion using OpenRouter.
        
        Args:
            prompt (str): The prompt text.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dict[str, Any]: The completion response.
        """
        try:
            # Convert to chat completion since OpenRouter uses chat endpoint
            return await self.generate_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error generating completion with OpenRouter: {e}")
            raise
    
    async def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a chat completion using OpenRouter.
        
        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dict[str, Any]: The chat completion response.
        """
        try:
            model = kwargs.pop("model", self.openrouter_settings.completion_model)
            
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return {
                "message": data["choices"][0]["message"],
                "model": model,
                "provider": "openrouter",
                "raw_response": data
            }
        except Exception as e:
            logger.error(f"Error generating chat completion with OpenRouter: {e}")
            raise

    def get_pydantic_provider(self):
        """Get the pydantic-ai compatible provider.
        
        Returns:
            Any: The pydantic-ai provider.
            
        Raises:
            ValueError: If pydantic-ai is not available.
        """
        if self.pydantic_provider is None:
            raise ValueError("pydantic-ai OpenAIProvider not available")
        return self.pydantic_provider

    async def close(self):
        """Close the httpx client."""
        await self.client.aclose()
