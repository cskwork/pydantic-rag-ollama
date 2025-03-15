"""Embedding service."""
import logging
from functools import lru_cache
from typing import List, Optional

import pydantic_core

from config.settings import Settings, get_settings
from providers.base import LLMProvider, create_provider

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize the embedding service.
        
        Args:
            provider (Optional[LLMProvider], optional): LLM provider. Defaults to None.
            settings (Optional[Settings], optional): Settings override. Defaults to None.
        """
        self.settings = settings or get_settings()
        self.provider = provider or create_provider(settings=self.settings)
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text.
        
        Args:
            text (str): The text to embed.
            
        Returns:
            List[float]: The embedding vector.
        """
        try:
            return await self.provider.get_embeddings(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embedding_to_json(self, embedding: List[float]) -> str:
        """Convert embedding to JSON.
        
        Args:
            embedding (List[float]): The embedding vector.
            
        Returns:
            str: JSON representation of the embedding.
        """
        return pydantic_core.to_json(embedding).decode()


@lru_cache(maxsize=1)
def get_embedding_service(
    provider: Optional[LLMProvider] = None,
    settings: Optional[Settings] = None,
) -> EmbeddingService:
    """Get the embedding service singleton.
    
    Args:
        provider (Optional[LLMProvider], optional): LLM provider. Defaults to None.
        settings (Optional[Settings], optional): Settings override. Defaults to None.
            
    Returns:
        EmbeddingService: The embedding service.
    """
    return EmbeddingService(provider, settings)
