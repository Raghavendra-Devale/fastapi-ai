import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.exceptions import ValidationException
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.application.ai.structured_output_service import StructuredOutputService
from app.application.resume.resume_analyzer_service import LLMResumeAnalyzer
from app.domain.resume.candidate_profile import CandidateProfile


@pytest.mark.asyncio
async def test_llm_resume_analyzer_success():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = '{"name": "John Doe", "headline": "Software Engineer"}'

    mock_profile = CandidateProfile(
        name="John Doe",
        headline="Software Engineer",
        summary="Summary of SWE",
    )
    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.return_value = mock_profile

    analyzer = LLMResumeAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)

    # Act
    result = await analyzer.analyze("Sample clean resume text")

    # Assert
    assert result == mock_profile
    mock_llm.generate.assert_called_once()
    called_prompt = mock_llm.generate.call_args[1]["prompt"]
    assert "Sample clean resume text" in called_prompt
    mock_structured.parse.assert_called_once_with(
        '{"name": "John Doe", "headline": "Software Engineer"}',
        CandidateProfile
    )


@pytest.mark.asyncio
async def test_llm_resume_analyzer_llm_failure():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.side_effect = RuntimeError("LLM API Timeout")

    mock_structured = MagicMock(spec=StructuredOutputService)

    analyzer = LLMResumeAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)

    # Act & Assert
    with pytest.raises(RuntimeError, match="LLM API Timeout"):
        await analyzer.analyze("Sample resume text")

    mock_structured.parse.assert_not_called()


@pytest.mark.asyncio
async def test_llm_resume_analyzer_validation_failure():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = "{invalid_json}"

    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.side_effect = ValidationException("LLM response is not valid JSON")

    analyzer = LLMResumeAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)

    # Act & Assert
    with pytest.raises(ValidationException, match="LLM response is not valid JSON"):
        await analyzer.analyze("Sample resume text")

    mock_structured.parse.assert_called_once_with("{invalid_json}", CandidateProfile)
