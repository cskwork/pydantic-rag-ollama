"""Base provider interface and factory."""
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, List, Optional, Union
import inspect
import logging

from config.settings import ProviderType, Settings, get_settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text.
        
        Args:
            text (str): The text to embed.
            
        Returns:
            List[float]: The embedding vector.
        """
        pass
    
    @abstractmethod
    async def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a text completion.
        
        Args:
            prompt (str): The prompt text.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dict[str, Any]: The completion response.
        """
        pass
    
    @abstractmethod
    async def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a chat completion.
        
        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dict[str, Any]: The chat completion response.
        """
        pass
    
    def compatibility_check(self) -> bool:
        """Check if this provider is compatible with pydantic-ai.
        
        Returns:
            bool: True if compatible, False otherwise.
        """
        # Check if get_pydantic_provider method exists
        return hasattr(self, 'get_pydantic_provider') and callable(getattr(self, 'get_pydantic_provider'))
    
    def get_pydantic_provider(self):
        """Get the pydantic-ai compatible provider.
        Implemented by subclasses if they support pydantic-ai.
        
        Returns:
            Any: The pydantic-ai compatible provider.
            
        Raises:
            NotImplementedError: If not implemented by the subclass.
        """
        raise NotImplementedError("This provider does not support pydantic-ai directly")


# Provider registry
_PROVIDER_REGISTRY: Dict[ProviderType, Type[LLMProvider]] = {}


def register_provider(provider_type: ProviderType):
    """Register a provider implementation.
    
    Args:
        provider_type (ProviderType): The provider type to register.
        
    Returns:
        Callable: Decorator function.
    """
    def decorator(cls):
        _PROVIDER_REGISTRY[provider_type] = cls
        return cls
    return decorator


def create_provider(
    provider_type: Optional[ProviderType] = None, 
    settings: Optional[Settings] = None
) -> LLMProvider:
    """Create a provider instance based on configuration.
    
    Args:
        provider_type (Optional[ProviderType], optional): Provider type override.
            Defaults to None.
        settings (Optional[Settings], optional): Settings override. Defaults to None.
            
    Returns:
        LLMProvider: Provider instance.
        
    Raises:
        ValueError: If the provider type is not supported.
    """
    if settings is None:
        settings = get_settings()
    
    provider_type = provider_type or settings.provider
    
    if provider_type not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported provider: {provider_type}")
    
    provider_cls = _PROVIDER_REGISTRY[provider_type]
    
    try:
        # Check if the provider constructor accepts settings parameter
        sig = inspect.signature(provider_cls.__init__)
        if 'settings' in sig.parameters:
            return provider_cls(settings)
        else:
            # Fall back to older style initialization
            logger.warning(f"Provider {provider_type} doesn't accept settings parameter. Using legacy init.")
            return provider_cls()
    except Exception as e:
        logger.error(f"Error creating provider {provider_type}: {e}")
        raise ValueError(f"Failed to create provider {provider_type}: {e}") from e
