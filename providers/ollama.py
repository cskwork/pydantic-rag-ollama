"""Ollama provider implementation."""
import logging
from typing import Dict, List, Any, Optional
import importlib.util

import httpx
import ollama

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


@register_provider(ProviderType.OLLAMA)
class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the Ollama provider.
        
        Args:
            settings (Optional[Settings], optional): Application settings. Defaults to None.
        """
        self.settings = settings or get_settings()
        self.ollama_settings = self.settings.ollama
        
        # Initialize the pydantic-ai OpenAI provider with Ollama base URL if available
        self.pydantic_provider = None
        if PyAIOpenAIProvider is not None:
            try:
                self.pydantic_provider = PyAIOpenAIProvider(
                    base_url=f"{self.ollama_settings.base_url}/v1"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize pydantic-ai OpenAIProvider: {e}")
    
    async def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Ollama.
        
        Args:
            text (str): The text to embed.
            
        Returns:
            List[float]: The embedding vector.
        """
        try:
            # Use direct Ollama API
            response = ollama.embed(
                model=self.ollama_settings.embedding_model,
                input=text
            )
            return response['embeddings'][0]
        except Exception as e:
            logger.error(f"Error generating embeddings with Ollama: {e}")
            raise
    
    async def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a text completion using Ollama.
        
        Args:
            prompt (str): The prompt text.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dict[str, Any]: The completion response.
        """
        try:
            model = kwargs.pop("model", self.ollama_settings.completion_model)
            response = ollama.generate(
                model=model,
                prompt=prompt,
                **kwargs
            )
            
            # Format response to be similar to OpenAI for consistency
            return {
                "text": response["response"],
                "model": model,
                "provider": "ollama",
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Error generating completion with Ollama: {e}")
            raise
    
    async def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a chat completion using Ollama.
        
        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dict[str, Any]: The chat completion response.
        """
        try:
            model = kwargs.pop("model", self.ollama_settings.completion_model)
            
            # Ollama chat API expects messages in a specific format
            response = ollama.chat(
                model=model,
                messages=messages,
                **kwargs
            )
            
            # Format response to be similar to OpenAI for consistency
            return {
                "message": response["message"],
                "model": model,
                "provider": "ollama",
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Error generating chat completion with Ollama: {e}")
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
