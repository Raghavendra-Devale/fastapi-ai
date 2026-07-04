import pytest
from unittest.mock import AsyncMock, patch

from app.core.config import Settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.domain.resume.services.resume_summary_service import ResumeSummaryService
from app.providers.base import AIProvider
from app.providers.models import ChatResponse


@pytest.fixture
def mock_settings():
    return Settings(
        postgres_url="postgresql://localhost",
        redis_url="redis://localhost",
        provider="ollama",
        ollama_base_url="http://localhost:11434",
        embedding_model="test-model",
        llm_model="test-llm-model",
        max_resume_size_bytes=1000000,
    )


@pytest.mark.asyncio
async def test_resume_summary_service_success(mock_settings):
    """Test successful resume summary generation."""
    mock_provider = AsyncMock(spec=AIProvider)
    mock_provider.chat.return_value = ChatResponse(
        content="Generated resume summary",
        model="test-llm-model",
    )

    service = ResumeSummaryService(ai_provider=mock_provider, settings=mock_settings)
    summary = await service.generate_summary("John Doe Resume Text")

    assert summary == "Generated resume summary"
    # Verify mock call includes instruction to generate summary and the resume text
    mock_provider.chat.assert_called_once()
    called_prompt = mock_provider.chat.call_args[0][0]
    assert "John Doe Resume Text" in called_prompt
    assert "150 words" in called_prompt


@pytest.mark.asyncio
async def test_resume_summary_service_validation_failures(mock_settings):
    """Test validation errors for empty or blank resume text."""
    mock_provider = AsyncMock(spec=AIProvider)
    service = ResumeSummaryService(ai_provider=mock_provider, settings=mock_settings)

    with pytest.raises(ValidationException) as exc_info:
        await service.generate_summary("")
    assert "cannot be empty" in str(exc_info.value).lower()

    with pytest.raises(ValidationException) as exc_info:
        await service.generate_summary("    ")
    assert "cannot be empty" in str(exc_info.value).lower()

    with pytest.raises(ValidationException) as exc_info:
        await service.generate_summary(None)
    assert "cannot be empty" in str(exc_info.value).lower()

    mock_provider.chat.assert_not_called()


@pytest.mark.asyncio
async def test_resume_summary_service_provider_error_propagates(mock_settings):
    """Test that provider failures (ExternalServiceException) propagate directly."""
    mock_provider = AsyncMock(spec=AIProvider)
    mock_provider.chat.side_effect = ExternalServiceException(
        message="Ollama chat failure",
        error_code="OLLAMA_ERROR",
    )

    service = ResumeSummaryService(ai_provider=mock_provider, settings=mock_settings)

    with pytest.raises(ExternalServiceException) as exc_info:
        await service.generate_summary("John Doe Resume")
    assert exc_info.value.error_code == "OLLAMA_ERROR"
    assert "Ollama chat failure" in exc_info.value.message


@pytest.mark.asyncio
async def test_resume_summary_service_unexpected_error_wrapped(mock_settings):
    """Test that unexpected provider exceptions are wrapped in ExternalServiceException."""
    mock_provider = AsyncMock(spec=AIProvider)
    mock_provider.chat.side_effect = RuntimeError("Something unexpected happened")

    service = ResumeSummaryService(ai_provider=mock_provider, settings=mock_settings)

    with pytest.raises(ExternalServiceException) as exc_info:
        await service.generate_summary("John Doe Resume")
    assert exc_info.value.error_code == "SUMMARY_GENERATION_FAILED"
    assert "Failed to generate summary" in exc_info.value.message
    assert "Something unexpected happened" in exc_info.value.message
