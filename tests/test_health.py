import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.core.config import Settings
from app.providers.base import AIProvider
from app.providers.factory import get_ai_provider, ProviderFactory
from app.providers.models import HealthResponse
from app.services.ai_health_service import AIHealthService
from app.main import app


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


def test_health_endpoint_healthy():
    """Test the GET /api/v1/health endpoint when provider is healthy."""
    mock_provider = AsyncMock(spec=AIProvider)
    mock_provider.health.return_value = HealthResponse(
        provider="ollama",
        healthy=True,
        message="Ollama is online.",
    )

    # Cache the mock provider in the factory
    ProviderFactory._instances["ollama"] = mock_provider

    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "UP"
        assert data["dependencies"]["provider"]["healthy"] is True
        assert data["dependencies"]["provider"]["provider"] == "ollama"


def test_health_endpoint_unhealthy():
    """Test the GET /api/v1/health endpoint when provider is unhealthy/offline."""
    mock_provider = AsyncMock(spec=AIProvider)
    mock_provider.health.return_value = HealthResponse(
        provider="ollama",
        healthy=False,
        message="Ollama is offline or loading.",
    )

    # Cache the mock provider in the factory
    ProviderFactory._instances["ollama"] = mock_provider

    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        # Unhealthy dependencies should still yield a HTTP 200 health check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "UP"
        assert data["dependencies"]["provider"]["healthy"] is False
        assert data["dependencies"]["provider"]["message"] == "Ollama is offline or loading."


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
