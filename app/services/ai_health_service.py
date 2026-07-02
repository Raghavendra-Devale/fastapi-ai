from fastapi import Depends

from app.providers.base import AIProvider
from app.providers.factory import get_ai_provider
from app.providers.models import HealthResponse


class AIHealthService:
    """Service to monitor and query the health status of active AI providers."""

    def __init__(self, provider: AIProvider = Depends(get_ai_provider)):
        """Initialize the health service with the active AIProvider dependency."""
        self._provider = provider

    async def check_provider_health(self) -> HealthResponse:
        """Query the configured AI provider health status.

        Returns:
            HealthResponse: Standard health check response from the provider.
        """
        return await self._provider.health()
        
# class AIHealthService
