from typing import Type

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationException
from app.providers.base import AIProvider


class ProviderFactory:
    """Factory to register and resolve AI provider implementations dynamically."""

    _registry: dict[str, Type[AIProvider]] = {}
    _instances: dict[str, AIProvider] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[AIProvider]) -> None:
        """Register a provider implementation class under a unique name.

        Args:
            name (str): Identifier name of the provider.
            provider_class (Type[AIProvider]): Concrete implementation class.
        """
        cls._registry[name.lower()] = provider_class

    @classmethod
    def get_provider(cls, settings: Settings) -> AIProvider:
        """Resolve, instantiate and cache the configured AIProvider implementation.

        Args:
            settings (Settings): Active application configuration.

        Returns:
            AIProvider: Instantiated concrete provider class.

        Raises:
            ConfigurationException: If the configured provider name is missing
                                    or has no registered implementation.
        """
        provider_name = settings.provider.lower()

        # Resolve from cache if already instantiated
        if provider_name in cls._instances:
            return cls._instances[provider_name]

        provider_class = cls._registry.get(provider_name)

        if not provider_class:
            raise ConfigurationException(
                error_code="UNSUPPORTED_PROVIDER",
                message=(
                    f"AI provider '{settings.provider}' is unsupported "
                    "or has no registered implementation."
                ),
            )

        instance = provider_class(settings)
        cls._instances[provider_name] = instance
        return instance


def get_ai_provider(settings: Settings = Depends(get_settings)) -> AIProvider:
    """Dependency injection helper to obtain the configured AIProvider instance.

    This function should be the primary dependency injected into downstream services.

    Args:
        settings (Settings): Application configuration injected via FastAPI.

    Returns:
        AIProvider: Confirmed active provider implementation.
    """
    return ProviderFactory.get_provider(settings)


# Register concrete provider implementations
from app.providers.implementations.ollama_provider import OllamaProvider

ProviderFactory.register("ollama", OllamaProvider)

