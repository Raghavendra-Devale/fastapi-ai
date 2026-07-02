import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.providers.implementations.ollama_provider import OllamaProvider


@pytest.fixture
def mock_settings():
    return Settings(
        postgres_url="postgresql://postgres:postgres@localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        provider="ollama",
        ollama_base_url="http://localhost:11434",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3",
    )


def test_health_check_success(mock_settings):
    with patch(
        "app.providers.implementations.ollama_provider.AsyncClient"
    ) as mock_async_client_cls:
        mock_client = AsyncMock()
        mock_async_client_cls.return_value = mock_client
        mock_client.list.return_value = {"models": []}

        provider = OllamaProvider(mock_settings)
        result = asyncio.run(provider.health())

        assert result.healthy is True
        assert result.provider == "ollama"
        assert "healthy" in result.message.lower()
        mock_client.list.assert_called_once()


def test_health_check_failure(mock_settings):
    with patch(
        "app.providers.implementations.ollama_provider.AsyncClient"
    ) as mock_async_client_cls:
        mock_client = AsyncMock()
        mock_async_client_cls.return_value = mock_client
        mock_client.list.side_effect = Exception("Connection error")

        provider = OllamaProvider(mock_settings)
        result = asyncio.run(provider.health())

        assert result.healthy is False
        assert result.provider == "ollama"
        assert "unreachable" in result.message.lower()
        mock_client.list.assert_called_once()


def test_generate_embedding_success(mock_settings):
    with patch(
        "app.providers.implementations.ollama_provider.AsyncClient"
    ) as mock_async_client_cls:
        mock_client = AsyncMock()
        mock_async_client_cls.return_value = mock_client
        mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        provider = OllamaProvider(mock_settings)
        result = asyncio.run(provider.generate_embedding("hello"))

        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.dimensions == 3
        assert result.model == mock_settings.embedding_model
        mock_client.embed.assert_called_once_with(
            model=mock_settings.embedding_model, input="hello"
        )


def test_generate_embedding_validation_failure(mock_settings):
    provider = OllamaProvider(mock_settings)
    with pytest.raises(ValidationException) as exc_info:
        asyncio.run(provider.generate_embedding("  "))
    assert exc_info.value.error_code == "INVALID_INPUT"


def test_generate_embedding_provider_error(mock_settings):
    with patch(
        "app.providers.implementations.ollama_provider.AsyncClient"
    ) as mock_async_client_cls:
        mock_client = AsyncMock()
        mock_async_client_cls.return_value = mock_client
        mock_client.embed.side_effect = Exception("API limit exceeded")

        provider = OllamaProvider(mock_settings)
        with pytest.raises(ExternalServiceException) as exc_info:
            asyncio.run(provider.generate_embedding("hello"))

        assert exc_info.value.error_code == "EMBEDDING_FAILED"
        assert "Failed to generate embedding" in exc_info.value.message


def test_generate_embeddings_success(mock_settings):
    with patch(
        "app.providers.implementations.ollama_provider.AsyncClient"
    ) as mock_async_client_cls:
        mock_client = AsyncMock()
        mock_async_client_cls.return_value = mock_client
        mock_client.embed.side_effect = [
            {"embeddings": [[0.1, 0.2]]},
            {"embeddings": [[0.3, 0.4]]},
        ]

        provider = OllamaProvider(mock_settings)
        results = asyncio.run(
            provider.generate_embeddings(["hello", "", "  ", "world"])
        )

        assert len(results) == 2
        assert results[0].embedding == [0.1, 0.2]
        assert results[1].embedding == [0.3, 0.4]
        assert mock_client.embed.call_count == 2


def test_chat_success(mock_settings):
    with patch(
        "app.providers.implementations.ollama_provider.AsyncClient"
    ) as mock_async_client_cls:
        mock_client = AsyncMock()
        mock_async_client_cls.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"role": "assistant", "content": "hi user"}
        }

        provider = OllamaProvider(mock_settings)
        result = asyncio.run(provider.chat("hello"))

        assert result.content == "hi user"
        assert result.model == mock_settings.llm_model
        mock_client.chat.assert_called_once_with(
            model=mock_settings.llm_model,
            messages=[{"role": "user", "content": "hello"}],
        )


def test_chat_validation_failure(mock_settings):
    provider = OllamaProvider(mock_settings)
    with pytest.raises(ValidationException) as exc_info:
        asyncio.run(provider.chat(""))
    assert exc_info.value.error_code == "INVALID_INPUT"


def test_chat_provider_error(mock_settings):
    with patch(
        "app.providers.implementations.ollama_provider.AsyncClient"
    ) as mock_async_client_cls:
        mock_client = MagicMock()
        mock_async_client_cls.return_value = mock_client
        # Use an async function instead of AsyncMock for side_effect to ensure perfect exception propagation
        async def mock_chat(*args, **kwargs):
            raise Exception("Ollama connection timeout")
        mock_client.chat = mock_chat

        provider = OllamaProvider(mock_settings)
        with pytest.raises(ExternalServiceException) as exc_info:
            asyncio.run(provider.chat("hello"))

        assert exc_info.value.error_code == "OLLAMA_UNAVAILABLE"
        assert "Failed to generate chat response" in exc_info.value.message
