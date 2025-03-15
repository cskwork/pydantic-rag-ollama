"""Configuration settings for the application."""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Optional, Dict, Any, Union

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    SecretStr,
)
from pydantic_settings import BaseSettings

# Note: Pydantic v2 still supports these settings in the BaseSettings
# We don't need pydantic_settings package for basic functionality


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class DatabaseSettings(BaseModel):
    """Database configuration."""
    host: str = Field("localhost", description="Database host")
    port: str = Field("5432", description="Database port")
    user: str = Field("postgres", description="Database user")
    password: str = Field("admin", description="Database password")
    database: str = Field("pydantic_ai_rag", description="Database name")
    
    @computed_field
    def dsn(self) -> str:
        """Get the database connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @computed_field
    def server_dsn(self) -> str:
        """Get the server connection string (without database)."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}"


class OllamaSettings(BaseModel):
    """Ollama specific settings."""
    base_url: str = Field("http://localhost:11434", description="Ollama API base URL")
    completion_model: str = Field("llama3.2:1b", description="Model for completion")
    embedding_model: str = Field("nomic-embed-text", description="Model for embeddings")
    embedding_dimensions: int = Field(768, description="Dimensions of the embedding model")


class OpenAISettings(BaseModel):
    """OpenAI specific settings."""
    api_key: SecretStr = Field(..., description="OpenAI API key")
    organization_id: Optional[str] = Field(None, description="OpenAI organization ID")
    base_url: str = Field("https://api.openai.com/v1", description="OpenAI API base URL")
    completion_model: str = Field("gpt-3.5-turbo", description="Model for completion")
    embedding_model: str = Field("text-embedding-3-small", description="Model for embeddings")
    embedding_dimensions: int = Field(1536, description="Dimensions of the embedding model")
    
    @field_validator('api_key', mode='before')
    @classmethod
    def validate_api_key(cls, v):
        """Convert empty string to None and wrap in SecretStr if needed."""
        if not v:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        return v if isinstance(v, SecretStr) else SecretStr(v)


class OpenRouterSettings(BaseModel):
    """OpenRouter specific settings."""
    api_key: SecretStr = Field(..., description="OpenRouter API key")
    base_url: str = Field("https://openrouter.ai/api/v1", description="OpenRouter API base URL")
    completion_model: str = Field("openrouter/auto", description="Model for completion")
    embedding_model: str = Field("openai/text-embedding-3-small", description="Model for embeddings")
    embedding_dimensions: int = Field(1536, description="Dimensions of the embedding model")
    
    @field_validator('api_key', mode='before')
    @classmethod
    def validate_api_key(cls, v):
        """Convert empty string to None and wrap in SecretStr if needed."""
        if not v:
            raise ValueError("OpenRouter API key is required when using OpenRouter provider")
        return v if isinstance(v, SecretStr) else SecretStr(v)


class Settings(BaseSettings):
    """Application settings."""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        extra = "ignore"
    
    # Provider selection
    provider: ProviderType = Field(
        ProviderType.OLLAMA,
        description="LLM provider to use",
    )
    
    # Database settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    
    # Provider-specific settings
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    openai: Optional[OpenAISettings] = None
    openrouter: Optional[OpenRouterSettings] = None
    
    # Application settings
    log_level: str = Field("INFO", description="Logging level")
    logfire_token: Optional[str] = Field(None, description="Logfire token")
    print("logfire", logfire_token)
    
    @computed_field
    def current_provider_settings(self) -> Union[OllamaSettings, OpenAISettings, OpenRouterSettings]:
        """Get the settings for the current provider."""
        if self.provider == ProviderType.OLLAMA:
            return self.ollama
        elif self.provider == ProviderType.OPENAI:
            if not self.openai:
                raise ValueError("OpenAI settings not configured")
            return self.openai
        elif self.provider == ProviderType.OPENROUTER:
            if not self.openrouter:
                raise ValueError("OpenRouter settings not configured")
            return self.openrouter
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    @computed_field
    def embedding_dimensions(self) -> int:
        """Get the embedding dimensions for the current provider."""
        return self.current_provider_settings.embedding_dimensions


@lru_cache()
def get_settings() -> Settings:
    """Get the application settings.
    
    Returns:
        Settings: The application settings.
    """
    return Settings()
