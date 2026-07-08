import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.ai.providers.interfaces.embedding_provider import EmbeddingProvider
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.domain.ai.providers.dependencies import get_embedding_provider, get_llm_provider
from app.domain.ai.providers.models import HealthResponse
from app.services.ai_health_service import AIHealthService
from app.main import app


class MockSentenceTransformerProvider(EmbeddingProvider):
    def embed(self, text: str):
        return []

    def embed_batch(self, texts: list[str]):
        return []

    async def health(self) -> HealthResponse:
        return HealthResponse(provider="sentence-transformers", healthy=True, message="Active")


class MockOllamaProvider(LLMProvider):
    async def generate(self, prompt: str, system_prompt=None, temperature=0.2):
        return ""

    async def health(self) -> HealthResponse:
        return HealthResponse(provider="ollama", healthy=True, message="Active")


def test_dependency_injection_caching():
    """Test that get_embedding_provider and get_llm_provider cache their instances."""
    settings = Settings(
        postgres_url="postgresql://localhost",
        redis_url="redis://localhost",
        provider="ollama",
        ollama_base_url="http://localhost:11434",
        embedding_model="test-embed",
        llm_model="test-llm",
    )

    # Resolve providers and check singleton instance caching
    provider_1 = get_embedding_provider(settings)
    provider_2 = get_embedding_provider(settings)
    assert provider_1 is provider_2

    llm_1 = get_llm_provider(settings)
    llm_2 = get_llm_provider(settings)
    assert llm_1 is llm_2


@pytest.mark.asyncio
async def test_ai_health_service():
    """Test that AIHealthService properly queries provider health."""
    mock_llm_provider = AsyncMock(spec=LLMProvider)
    mock_llm_provider.health.return_value = HealthResponse(
        provider="ollama",
        healthy=True,
        message="Ollama is online.",
    )

    mock_emb_provider = AsyncMock(spec=EmbeddingProvider)
    mock_emb_provider.health.return_value = HealthResponse(
        provider="sentence-transformers",
        healthy=True,
        message="SentenceTransformers is active.",
    )

    service = AIHealthService(
        llm_provider=mock_llm_provider,
        embedding_provider=mock_emb_provider
    )
    result = await service.check_provider_health()

    assert result.healthy is True
    assert result.provider == "ollama + sentence-transformers"
    assert "ollama" in result.message.lower()
    mock_llm_provider.health.assert_called_once()
    mock_emb_provider.health.assert_called_once()


def test_health_endpoint_success():
    """Test the GET /api/v1/health endpoint returns UP, embeddingProvider, llmProvider, and version."""
    app.dependency_overrides[get_embedding_provider] = lambda: MockSentenceTransformerProvider()
    app.dependency_overrides[get_llm_provider] = lambda: MockOllamaProvider()

    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "UP"
        assert data["embeddingProvider"] == "MockSentenceTransformerProvider"
        assert data["llmProvider"] == "MockOllamaProvider"
        assert "version" in data

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_startup_validation_warning():
    """Test that lifespan startup validation logs a warning but allows app to boot when provider health check fails."""
    with patch("app.main.OllamaProvider") as mock_ollama_cls, \
         patch("app.main.SentenceTransformerProvider") as mock_st_cls:
        
        mock_ollama = AsyncMock()
        mock_ollama.health.return_value = HealthResponse(
            provider="ollama",
            healthy=False,
            message="Connection refused",
        )
        mock_ollama_cls.return_value = mock_ollama

        mock_st = AsyncMock()
        mock_st.health.return_value = HealthResponse(
            provider="sentence-transformers",
            healthy=True,
            message="ST is active",
        )
        mock_st_cls.return_value = mock_st

        # Patch the logger inside app/main.py to verify warning is logged
        with patch("app.main.logger") as mock_logger:
            with TestClient(app):
                # Assert that the warning log was triggered
                mock_logger.warn.assert_any_call(
                    event="provider_health_check_warning",
                    provider="ollama",
                    message="LLM: Connection refused | Embedding: ST is active",
                )
                
            mock_logger.info.assert_any_call(
                event="application_shutdown"
            )
