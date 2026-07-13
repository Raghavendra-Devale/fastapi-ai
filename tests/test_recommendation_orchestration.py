import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.recommendation.models.recommendation_request import RecommendationRequest
from app.domain.recommendation.models.recommendation_response import RecommendationResponse
from app.domain.jobs.job_profile import JobProfile
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.application.pipelines.recommendation_pipeline import RecommendationPipeline
from app.domain.recommendation.services.recommendation_service import RecommendationService
from app.application.resume.candidate_profile_retrieval_service import CandidateProfileRetrievalService


@pytest.mark.asyncio
async def test_recommendation_orchestration_new_pipeline():
    # Arrange
    mock_retrieval_service = MagicMock(spec=CandidateProfileRetrievalService)
    
    mock_candidate_profile = CandidateProfile(
        name="Alice",
        headline="Software Engineer",
        experience_years=5.0,
        skills=[],
        preferred_roles=["Developer"],
    )
    mock_retrieval_service.get_candidate_profile.return_value = mock_candidate_profile

    mock_job_profile = JobProfile(title="Developer", company="TechCorp")
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
        candidate_retrieval_service=mock_retrieval_service,
        recommendation_pipeline=mock_pipeline,
    )

    request = RecommendationRequest(
        candidate_profile_id="a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
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

    mock_retrieval_service.get_candidate_profile.assert_called_once_with("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11")
    mock_pipeline.run.assert_called_once()
