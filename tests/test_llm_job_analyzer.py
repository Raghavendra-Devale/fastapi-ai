import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.ai.exceptions import AIProviderException, AIValidationException
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.application.ai.structured_output_service import StructuredOutputService
from app.application.jobs.job_analyzer_service import LLMJobAnalyzer
from app.domain.jobs.models.raw_job import RawJob
from app.domain.jobs.job_profile import JobProfile


@pytest.mark.asyncio
async def test_llm_job_analyzer_success():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = '{"title": "DevOps Engineer", "company": "Stripe"}'

    mock_profile = JobProfile(
        title="DevOps Engineer",
        company="Stripe",
        summary="A Stripe SWE role",
    )
    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.return_value = mock_profile

    analyzer = LLMJobAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)
    raw_job = RawJob(
        title="DevOps Engineer",
        company="Stripe",
        description="Write Terraform scripts.",
        location="Dublin, IE"
    )

    # Act
    result = await analyzer.analyze(raw_job)

    # Assert
    assert result == mock_profile
    mock_llm.generate.assert_called_once()
    called_prompt = mock_llm.generate.call_args[1]["prompt"]
    assert "Dublin, IE" in called_prompt
    assert "Write Terraform scripts" in called_prompt
    mock_structured.parse.assert_called_once_with(
        '{"title": "DevOps Engineer", "company": "Stripe"}',
        JobProfile
    )


@pytest.mark.asyncio
async def test_llm_job_analyzer_success_fallback():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = '{"title": "", "company": ""}'  # LLM fails to extract title/company

    # The parser returns the empty/blank profile
    mock_profile = JobProfile(
        title="",
        company="",
    )
    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.return_value = mock_profile

    analyzer = LLMJobAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)
    raw_job = RawJob(
        title="SRE Engineer",
        company="Meta",
        description="Kubernetes configuration."
    )

    # Act
    result = await analyzer.analyze(raw_job)

    # Assert fallback triggers
    assert result.title == "SRE Engineer"
    assert result.company == "Meta"


@pytest.mark.asyncio
async def test_llm_job_analyzer_llm_failure():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.side_effect = RuntimeError("OpenAI Connection Refused")

    mock_structured = MagicMock(spec=StructuredOutputService)

    analyzer = LLMJobAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)
    raw_job = RawJob(title="SRE", company="Meta", description="K8s.")

    # Act & Assert
    with pytest.raises(AIProviderException, match="LLM provider generation call failed"):
        await analyzer.analyze(raw_job)

    mock_structured.parse.assert_not_called()


@pytest.mark.asyncio
async def test_llm_job_analyzer_validation_failure():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = "{bad_json}"

    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.side_effect = AIValidationException("Missing required title field")

    analyzer = LLMJobAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)
    raw_job = RawJob(title="SRE", company="Meta", description="K8s.")

    # Act & Assert
    with pytest.raises(AIValidationException, match="Missing required title field"):
        await analyzer.analyze(raw_job)

    mock_structured.parse.assert_called_once_with("{bad_json}", JobProfile)
