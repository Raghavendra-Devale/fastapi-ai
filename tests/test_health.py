import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.providers.base import AIProvider
from app.providers.factory import ProviderFactory
from app.providers.models import HealthResponse
from app.services.ai_health_service import AIHealthService
from app.domain.ai.providers.interfaces.embedding_provider import EmbeddingProvider
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.domain.ai.providers.dependencies import get_embedding_provider, get_llm_provider
from app.main import app


class SentenceTransformerProvider(EmbeddingProvider):
    def embed(self, text: str):
        return []

    def embed_batch(self, texts: list[str]):
        return []


class OllamaProvider(LLMProvider):
    async def generate(self, prompt: str, system_prompt=None, temperature=0.2):
        return ""


def test_dependency_injection_caching():
    """Test that get_ai_provider caches the instantiated provider object."""
    settings = Settings(
        postgres_url="postgresql://localhost",
        redis_url="redis://localhost",
        provider="ollama",
        ollama_base_url="http://localhost:11434",
        embedding_model="test-embed",
        llm_model="test-llm",
    )

    # Resolve provider first time
    provider_1 = ProviderFactory.get_provider(settings)
    # Resolve provider second time
    provider_2 = ProviderFactory.get_provider(settings)

    # They should be the exact same cached instance
    assert provider_1 is provider_2


@pytest.mark.asyncio
async def test_ai_health_service():
    """Test that AIHealthService properly queries provider health."""
    mock_provider = AsyncMock(spec=AIProvider)
    mock_provider.health.return_value = HealthResponse(
        provider="ollama",
        healthy=True,
        message="Ollama is online.",
    )

    service = AIHealthService(provider=mock_provider)
    result = await service.check_provider_health()

    assert result.healthy is True
    assert result.provider == "ollama"
    assert result.message == "Ollama is online."
    mock_provider.health.assert_called_once()


def test_health_endpoint_success():
    """Test the GET /api/v1/health endpoint returns UP, embeddingProvider, llmProvider, and version."""
    app.dependency_overrides[get_embedding_provider] = lambda: SentenceTransformerProvider()
    app.dependency_overrides[get_llm_provider] = lambda: OllamaProvider()

    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "UP"
        assert data["embeddingProvider"] == "SentenceTransformerProvider"
        assert data["llmProvider"] == "OllamaProvider"
        assert "version" in data

    app.dependency_overrides.clear()


def test_startup_validation_warning():
    """Test that lifespan startup validation logs a warning but allows app to boot when provider health check fails."""
    mock_provider = AsyncMock(spec=AIProvider)
    mock_provider.health.return_value = HealthResponse(
        provider="ollama",
        healthy=False,
        message="Connection refused",
    )

    ProviderFactory._instances["ollama"] = mock_provider

    # Patch the logger inside app/main.py to verify warning is logged
    with patch("app.main.logger") as mock_logger:
        with TestClient(app):
            # Assert that the warning log was triggered
            mock_logger.warn.assert_any_call(
                event="provider_health_check_warning",
                provider="ollama",
                message="Connection refused",
            )
            
        mock_logger.info.assert_any_call(
            event="application_shutdown"
        )
