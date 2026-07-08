import pytest
from unittest.mock import AsyncMock

from app.application.ai.embedding_service import EmbeddingService
from app.core.config import Settings
from app.core.exceptions import ValidationException
from app.domain.ai.providers.interfaces.embedding_provider import EmbeddingProvider
from app.domain.ai.providers.models import EmbeddingResponse


@pytest.fixture
def mock_settings():
    return Settings(
        postgres_url="postgresql://localhost",
        redis_url="redis://localhost",
        provider="ollama",
        ollama_base_url="http://localhost:11434",
        embedding_model="test-model",
        llm_model="test-llm",
        max_embedding_input_length=50,
    )


@pytest.mark.asyncio
async def test_embedding_service_success(mock_settings):
    """Test successful embedding generation with validation and normalization."""
    mock_provider = AsyncMock(spec=EmbeddingProvider)
    mock_provider.embed.return_value = [0.1, 0.2, 0.3]

    service = EmbeddingService(ai_provider=mock_provider, settings=mock_settings)

    # Input has leading/trailing spaces to test normalization
    response = await service.generate_embedding("  some text  ")

    assert response.embedding == [0.1, 0.2, 0.3]
    assert response.model == "test-model"
    assert response.dimensions == 3
    # Check that text was normalized (stripped) before calling provider
    mock_provider.embed.assert_called_once_with("some text")


@pytest.mark.asyncio
async def test_embedding_service_validation_empty(mock_settings):
    """Test that empty or blank inputs raise a ValidationException."""
    mock_provider = AsyncMock(spec=EmbeddingProvider)
    service = EmbeddingService(ai_provider=mock_provider, settings=mock_settings)

    # Empty string
    with pytest.raises(ValidationException) as exc_info:
        await service.generate_embedding("")
    assert "empty" in str(exc_info.value).lower()

    # Blank string
    with pytest.raises(ValidationException) as exc_info:
        await service.generate_embedding("   ")
    assert "blank" in str(exc_info.value).lower()

    mock_provider.embed.assert_not_called()


@pytest.mark.asyncio
async def test_embedding_service_validation_oversized(mock_settings):
    """Test that input exceeding length limit raises a ValidationException."""
    mock_provider = AsyncMock(spec=EmbeddingProvider)
    service = EmbeddingService(ai_provider=mock_provider, settings=mock_settings)

    # limit is 50, let's send 51 characters
    long_text = "a" * 51
    with pytest.raises(ValidationException) as exc_info:
        await service.generate_embedding(long_text)
    assert "exceeds" in str(exc_info.value).lower()

    mock_provider.embed.assert_not_called()


@pytest.mark.asyncio
async def test_embedding_service_provider_failure(mock_settings):
    """Test that provider errors propagate correctly."""
    mock_provider = AsyncMock(spec=EmbeddingProvider)
    mock_provider.embed.side_effect = RuntimeError("Ollama connection failed")

    service = EmbeddingService(ai_provider=mock_provider, settings=mock_settings)

    with pytest.raises(RuntimeError) as exc_info:
        await service.generate_embedding("valid text")
    assert "Ollama connection failed" in str(exc_info.value)
