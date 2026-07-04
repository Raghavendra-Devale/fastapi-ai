import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import Settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.domain.ai.providers.implementations.ollama_provider import OllamaProvider


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_model = "llama3"
    settings.ollama_timeout = 30.0
    return settings


@pytest.mark.asyncio
async def test_ollama_generate_success(mock_settings):
    with patch("app.domain.ai.providers.implementations.ollama_provider.AsyncClient") as mock_client_class:
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.chat = AsyncMock(
            return_value={
                "message": {"role": "assistant", "content": "Clean summary content."}
            }
        )

        provider = OllamaProvider(mock_settings)
        content = await provider.generate("Summarize resume.", system_prompt="Sys prompt", temperature=0.5)

        assert content == "Clean summary content."
        mock_client_class.assert_called_once_with(
            host="http://localhost:11434",
            timeout=30.0,
        )
        mock_client_instance.chat.assert_called_once_with(
            model="llama3",
            messages=[
                {"role": "system", "content": "Sys prompt"},
                {"role": "user", "content": "Summarize resume."},
            ],
            options={"temperature": 0.5},
        )


@pytest.mark.asyncio
async def test_ollama_generate_empty_prompt_validation(mock_settings):
    with patch("app.domain.ai.providers.implementations.ollama_provider.AsyncClient"):
        provider = OllamaProvider(mock_settings)
        with pytest.raises(ValidationException) as exc_info:
            await provider.generate("")
        assert exc_info.value.error_code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_ollama_generate_invalid_empty_response(mock_settings):
    with patch("app.domain.ai.providers.implementations.ollama_provider.AsyncClient") as mock_client_class:
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.chat = AsyncMock(
            return_value={"message": {"role": "assistant", "content": ""}}
        )

        provider = OllamaProvider(mock_settings)
        with pytest.raises(ExternalServiceException) as exc_info:
            await provider.generate("hello")
        assert exc_info.value.error_code == "INVALID_RESPONSE"


@pytest.mark.asyncio
async def test_ollama_generate_timeout_error(mock_settings):
    with patch("app.domain.ai.providers.implementations.ollama_provider.AsyncClient") as mock_client_class:
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.chat = AsyncMock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )

        provider = OllamaProvider(mock_settings)
        with pytest.raises(ExternalServiceException) as exc_info:
            await provider.generate("hello")
        assert exc_info.value.error_code == "OLLAMA_TIMEOUT"


@pytest.mark.asyncio
async def test_ollama_generate_connection_unavailable(mock_settings):
    with patch("app.domain.ai.providers.implementations.ollama_provider.AsyncClient") as mock_client_class:
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.chat = AsyncMock(
            side_effect=httpx.RequestError("DNS resolution failed")
        )

        provider = OllamaProvider(mock_settings)
        with pytest.raises(ExternalServiceException) as exc_info:
            await provider.generate("hello")
        assert exc_info.value.error_code == "OLLAMA_UNAVAILABLE"


@pytest.mark.asyncio
async def test_ollama_generate_general_failure(mock_settings):
    with patch("app.domain.ai.providers.implementations.ollama_provider.AsyncClient") as mock_client_class:
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.chat = AsyncMock(
            side_effect=Exception("Internal server error")
        )

        provider = OllamaProvider(mock_settings)
        with pytest.raises(ExternalServiceException) as exc_info:
            await provider.generate("hello")
        assert exc_info.value.error_code == "LLM_GENERATION_FAILED"
