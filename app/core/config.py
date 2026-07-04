from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Supported deployment environments for the application."""

    local = "local"
    development = "development"
    staging = "staging"
    production = "production"


class AIProvider(str, Enum):
    """Supported AI model providers."""

    ollama = "ollama"
    openai = "openai"
    groq = "groq"
    gemini = "gemini"


class Settings(BaseSettings):
    """Application configuration settings.

    Settings are loaded from environment variables prefixed with 'AI_'.
    Infrastructure required for startup is strictly required to fail-fast.
    """

    # Application settings
    app_name: str = "FastAPI AI Engine"
    version: str = "1.0.0"
    environment: Environment = Environment.development
    debug: bool = True

    # Database settings
    postgres_url: str

    # Redis settings
    redis_url: str

    # AI Provider settings
    provider: AIProvider

    # Ollama settings
    ollama_base_url: str

    # Embedding settings
    embedding_model: str

    # LLM settings
    llm_model: str

    # File upload settings
    max_resume_size_bytes: int = 10 * 1024 * 1024  # Default to 10MB

    # AI service settings
    max_embedding_input_length: int = 100000  # Default to 100,000 characters

    # Logging settings
    log_level: str = "INFO"

    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AI_",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Load and cache the application settings.

    Returns the cached Settings instance using a process-wide lru_cache.
    """
    return Settings()
