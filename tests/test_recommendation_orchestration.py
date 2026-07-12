import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.recommendation.models.recommendation_request import JobDocument, RecommendationRequest
from app.domain.recommendation.models.recommendation_response import RecommendationResponse
from app.application.resume.resume_analyzer_service import ResumeAnalyzer
from app.application.jobs.job_analyzer_service import JobAnalyzer
from app.domain.jobs.job_profile import JobProfile
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.application.pipelines.recommendation_pipeline import RecommendationPipeline
from app.domain.recommendation.services.recommendation_service import RecommendationService


@pytest.mark.asyncio
async def test_recommendation_orchestration_new_pipeline():
    # Arrange
    mock_resume_analyzer = AsyncMock(spec=ResumeAnalyzer)
    mock_candidate = CandidateProfile(name="Alice")
    mock_resume_analyzer.analyze.return_value = mock_candidate

    mock_job_analyzer = AsyncMock(spec=JobAnalyzer)
    mock_job_profile = JobProfile(title="Developer", company="TechCorp")
    mock_job_analyzer.analyze.return_value = mock_job_profile

    mock_pipeline = AsyncMock(spec=RecommendationPipeline)
    mock_result = RecommendationResult(
        job_profile=mock_job_profile,
        semantic_score=0.9,
        skill_score=0.8,
        experience_score=0.7,
        location_score=1.0,
        education_score=1.0,
        final_score=0.85,
        matched_skills=["Python"],
        missing_skills=[],
        recommendation_reason="Strong Python matches",
    )
    mock_pipeline.run.return_value = [mock_result]

    service = RecommendationService(
        resume_analyzer=mock_resume_analyzer,
        job_analyzer=mock_job_analyzer,
        recommendation_pipeline=mock_pipeline,
    )

    request = RecommendationRequest(
        resume_text="Experienced developer",
        jobs=[
            JobDocument(
                title="Developer",
                company="TechCorp",
                description="Write Python code",
                apply_url="https://tech.corp/apply"
            )
        ]
    )

    # Act
    response = await service.generate_recommendations(request)

    # Assert
    assert isinstance(response, RecommendationResponse)
    assert len(response.recommendations) == 1
    item = response.recommendations[0]
    assert item.title == "Developer"
    assert item.company == "TechCorp"
    assert item.similarity_score == 0.85  # Maps to final_score
    assert item.recommendation_reason == "Strong Python matches"

    mock_resume_analyzer.analyze.assert_called_once_with("Experienced developer")
    mock_job_analyzer.analyze.assert_called_once()
    mock_pipeline.run.assert_called_once_with(
        candidate_profile=mock_candidate,
        job_profiles=[mock_job_profile],
    )
