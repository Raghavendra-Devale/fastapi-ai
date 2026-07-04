import pytest
from unittest.mock import AsyncMock

from app.core.config import Settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.domain.resume.services.resume_summary_service import ResumeSummaryService


@pytest.fixture
def mock_settings():
    return Settings(
        postgres_url="postgresql://localhost",
        redis_url="redis://localhost",
        provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        ollama_timeout=30.0,
        embedding_model="test-model",
        llm_model="test-llm-model",
        max_resume_size_bytes=1000000,
    )


@pytest.mark.asyncio
async def test_resume_summary_service_success(mock_settings):
    """Test successful resume summary generation."""
    mock_provider = AsyncMock(spec=LLMProvider)
    mock_provider.generate.return_value = "Generated resume summary"

    service = ResumeSummaryService(llm_provider=mock_provider, settings=mock_settings)
    summary = await service.generate_summary("John Doe Resume Text")

    assert summary == "Generated resume summary"
    mock_provider.generate.assert_called_once()
    called_kwargs = mock_provider.generate.call_args[1]
    assert "John Doe Resume Text" in called_kwargs["prompt"]
    assert "150 words" in called_kwargs["prompt"]
    assert called_kwargs["system_prompt"] == "You are a professional resume parser and writer."
    assert called_kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_resume_summary_service_validation_failures(mock_settings):
    """Test validation errors for empty or blank resume text."""
    mock_provider = AsyncMock(spec=LLMProvider)
    service = ResumeSummaryService(llm_provider=mock_provider, settings=mock_settings)

    with pytest.raises(ValidationException) as exc_info:
        await service.generate_summary("")
    assert "cannot be empty" in str(exc_info.value).lower()

    with pytest.raises(ValidationException) as exc_info:
        await service.generate_summary("    ")
    assert "cannot be empty" in str(exc_info.value).lower()

    with pytest.raises(ValidationException) as exc_info:
        await service.generate_summary(None)
    assert "cannot be empty" in str(exc_info.value).lower()

    mock_provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_resume_summary_service_provider_error_propagates(mock_settings):
    """Test that provider failures (ExternalServiceException) propagate directly."""
    mock_provider = AsyncMock(spec=LLMProvider)
    mock_provider.generate.side_effect = ExternalServiceException(
        message="Ollama chat failure",
        error_code="OLLAMA_ERROR",
    )

    service = ResumeSummaryService(llm_provider=mock_provider, settings=mock_settings)

    with pytest.raises(ExternalServiceException) as exc_info:
        await service.generate_summary("John Doe Resume")
    assert exc_info.value.error_code == "OLLAMA_ERROR"
    assert "Ollama chat failure" in exc_info.value.message


@pytest.mark.asyncio
async def test_resume_summary_service_unexpected_error_wrapped(mock_settings):
    """Test that unexpected provider exceptions are wrapped in ExternalServiceException."""
    mock_provider = AsyncMock(spec=LLMProvider)
    mock_provider.generate.side_effect = RuntimeError("Something unexpected happened")

    service = ResumeSummaryService(llm_provider=mock_provider, settings=mock_settings)

    with pytest.raises(ExternalServiceException) as exc_info:
        await service.generate_summary("John Doe Resume")
    assert exc_info.value.error_code == "SUMMARY_GENERATION_FAILED"
    assert "Failed to generate summary" in exc_info.value.message
    assert "Something unexpected happened" in exc_info.value.message
