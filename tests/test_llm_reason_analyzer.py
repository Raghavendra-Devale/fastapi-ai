import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.ai.exceptions import AIProviderException, AIValidationException
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.application.ai.structured_output_service import StructuredOutputService
from app.application.recommendation.recommendation_reason_generator import LLMReasonAnalyzer
from app.domain.recommendation.recommendation_reason import RecommendationReason
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile


@pytest.mark.asyncio
async def test_llm_reason_analyzer_success():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = '{"bullet_points": ["Strong Java & Spring Boot alignment", "Missing Kafka"]}'

    mock_reason = RecommendationReason(
        bullet_points=["Strong Java & Spring Boot alignment", "Missing Kafka"]
    )
    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.return_value = mock_reason

    analyzer = LLMReasonAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)

    candidate = CandidateProfile(name="Alice")
    job = JobProfile(title="Java Dev", company="Uber")
    result = RecommendationResult(
        job_profile=job,
        semantic_score=0.9,
        skill_score=0.9,
        experience_score=0.9,
        location_score=1.0,
        education_score=1.0,
        final_score=0.9,
        matched_skills=["Java"],
        missing_skills=["Kafka"],
    )

    # Act
    res = await analyzer.analyze(result, candidate, job)

    # Assert
    assert res == mock_reason
    mock_llm.generate.assert_called_once()
    called_prompt = mock_llm.generate.call_args[1]["prompt"]
    assert "Semantic Cosine Match: 90%" in called_prompt
    assert "Matched Skills: Java" in called_prompt
    mock_structured.parse.assert_called_once_with(
        '{"bullet_points": ["Strong Java & Spring Boot alignment", "Missing Kafka"]}',
        RecommendationReason
    )


@pytest.mark.asyncio
async def test_llm_reason_analyzer_provider_failure():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.side_effect = RuntimeError("Service Unreachable")

    mock_structured = MagicMock(spec=StructuredOutputService)

    analyzer = LLMReasonAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)
    candidate = CandidateProfile(name="Alice")
    job = JobProfile(title="Java Dev", company="Uber")
    result = RecommendationResult(
        job_profile=job,
        semantic_score=0.9,
        skill_score=0.9,
        experience_score=0.9,
        location_score=1.0,
        education_score=1.0,
        final_score=0.9,
    )

    # Act & Assert
    with pytest.raises(AIProviderException, match="LLM provider generation call failed"):
        await analyzer.analyze(result, candidate, job)

    mock_structured.parse.assert_not_called()


@pytest.mark.asyncio
async def test_llm_reason_analyzer_validation_failure():
    # Arrange
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = "{bad_json}"

    mock_structured = MagicMock(spec=StructuredOutputService)
    mock_structured.parse.side_effect = AIValidationException("Missing bullet_points")

    analyzer = LLMReasonAnalyzer(llm_provider=mock_llm, structured_output=mock_structured)
    candidate = CandidateProfile(name="Alice")
    job = JobProfile(title="Java Dev", company="Uber")
    result = RecommendationResult(
        job_profile=job,
        semantic_score=0.9,
        skill_score=0.9,
        experience_score=0.9,
        location_score=1.0,
        education_score=1.0,
        final_score=0.9,
    )

    # Act & Assert
    with pytest.raises(AIValidationException, match="Missing bullet_points"):
        await analyzer.analyze(result, candidate, job)

    mock_structured.parse.assert_called_once_with("{bad_json}", RecommendationReason)
