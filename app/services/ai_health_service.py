from fastapi import Depends

from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.domain.ai.providers.interfaces.embedding_provider import EmbeddingProvider
from app.domain.ai.providers.dependencies import get_llm_provider, get_embedding_provider
from app.domain.ai.providers.models import HealthResponse


class AIHealthService:
    """Service to monitor and query the health status of active AI providers."""

    def __init__(
        self,
        llm_provider: LLMProvider = Depends(get_llm_provider),
        embedding_provider: EmbeddingProvider = Depends(get_embedding_provider)
    ):
        """Initialize the health service with the active AIProvider dependencies."""
        self._llm_provider = llm_provider
        self._embedding_provider = embedding_provider

    async def check_provider_health(self) -> HealthResponse:
        """Query the configured AI provider health status.

        Returns:
            HealthResponse: Standard health check response from the provider.
        """
        llm_health = await self._llm_provider.health()
        emb_health = await self._embedding_provider.health()

        healthy = llm_health.healthy and emb_health.healthy
        message = f"LLM: {llm_health.message} | Embedding: {emb_health.message}"
        provider = f"{llm_health.provider} + {emb_health.provider}"

        return HealthResponse(
            provider=provider,
            healthy=healthy,
            message=message
        )
        
# class AIHealthService
