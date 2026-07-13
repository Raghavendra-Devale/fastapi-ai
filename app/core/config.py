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
    ollama_model: str = "llama3"
    ollama_timeout: float = 300.0

    # Embedding settings
    embedding_model: str

    # LLM settings
    llm_model: str

    # File upload settings
    max_resume_size_bytes: int = 10 * 1024 * 1024  # Default to 10MB

    # AI service settings
    max_embedding_input_length: int = 100000  # Default to 100,000 characters
    enable_ai_explanations: bool = True  # Enable AI explanation generation for recommendations


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
    import sys
    sys.stderr.write("[WSL_AUTO_DETECT] get_settings() called\n")
    sys.stderr.flush()
    
    settings = Settings()

    # Auto-detect WSL if Ollama is configured on localhost/127.0.0.1 on Windows host
    if "localhost" in settings.ollama_base_url or "127.0.0.1" in settings.ollama_base_url:
        import subprocess
        is_inside_wsl = False
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    is_inside_wsl = True
        except Exception:
            pass

        if not is_inside_wsl:
            try:
                sys.stderr.write("[WSL_AUTO_DETECT] Attempting to auto-detect WSL IP for Ollama...\n")
                sys.stderr.flush()
                # Query WSL instance IP addresses
                result = subprocess.run(
                    ["wsl", "hostname", "-I"],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                if result.returncode == 0:
                    ips = result.stdout.strip().split()
                    if ips:
                        wsl_ip = ips[0]
                        original = settings.ollama_base_url
                        settings.ollama_base_url = original.replace("localhost", wsl_ip).replace("127.0.0.1", wsl_ip)
                        sys.stderr.write(f"[WSL_AUTO_DETECT] Swapped Ollama base URL from {original} to {settings.ollama_base_url}\n")
                        sys.stderr.flush()
                else:
                    sys.stderr.write(f"[WSL_AUTO_DETECT] wsl command returned non-zero code: {result.returncode}\n")
                    sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"[WSL_AUTO_DETECT] Failed to execute wsl command: {e}\n")
                sys.stderr.flush()

    return settings
