"""LLM provider implementations module."""
import logging
import importlib.util

from .base import LLMProvider, register_provider, create_provider
from .ollama import OllamaProvider
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)

# Conditionally import OpenRouter if dependencies are met
try:
    from .openrouter import OpenRouterProvider
    HAS_OPENROUTER = True
except ImportError:
    logger.warning("OpenRouter provider not available - dependencies might be missing")
    HAS_OPENROUTER = False
    OpenRouterProvider = None

__all__ = [
    "LLMProvider",
    "register_provider",
    "create_provider",
    "OllamaProvider",
    "OpenAIProvider",
]

if HAS_OPENROUTER:
    __all__.append("OpenRouterProvider")
