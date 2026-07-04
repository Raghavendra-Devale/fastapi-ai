import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.config import Settings
from app.domain.recommendation.models.recommendation_request import JobDocument, RecommendationRequest
from app.domain.recommendation.models.recommendation_response import RecommendationItem, RecommendationResponse
from app.domain.recommendation.services.embedding_service import EmbeddingService
from app.domain.recommendation.services.similarity_service import SimilarityService
from app.domain.recommendation.services.ranking_service import RankingService
from app.domain.recommendation.services.explanation_service import ExplanationService
from app.domain.recommendation.services.recommendation_service import RecommendationService


@pytest.fixture
def mock_embedding_service():
    service = MagicMock(spec=EmbeddingService)
    service.embed_resume = AsyncMock(return_value=[0.1, 0.2])
    service.embed_jobs = AsyncMock(return_value=[[0.3, 0.4], [0.5, 0.6]])
    return service


@pytest.fixture
def mock_similarity_service():
    service = MagicMock(spec=SimilarityService)
    service.calculate_similarity = MagicMock(return_value=[0.8, 0.9])
    return service


@pytest.fixture
def mock_ranking_service():
    service = MagicMock(spec=RankingService)
    ranked_items = [
        RecommendationItem(
            title="SWE 2",
            company="Beta Corp",
            description="Coding 2",
            apply_url="https://beta.corp/2",
            similarity_score=0.9,
            recommendation_reason=None,
        ),
        RecommendationItem(
            title="SWE 1",
            company="Acme Corp",
            description="Coding 1",
            apply_url="https://acme.corp/1",
            similarity_score=0.8,
            recommendation_reason=None,
        ),
    ]
    service.rank = MagicMock(return_value=ranked_items)
    return service


@pytest.fixture
def mock_explanation_service():
    service = MagicMock(spec=ExplanationService)
    explained_items = [
        RecommendationItem(
            title="SWE 2",
            company="Beta Corp",
            description="Coding 2",
            apply_url="https://beta.corp/2",
            similarity_score=0.9,
            recommendation_reason="Excellent match based on skills.",
        ),
        RecommendationItem(
            title="SWE 1",
            company="Acme Corp",
            description="Coding 1",
            apply_url="https://acme.corp/1",
            similarity_score=0.8,
            recommendation_reason="Good match based on role.",
        ),
    ]
    service.explain_recommendations = AsyncMock(return_value=explained_items)
    return service


@pytest.fixture
def test_request():
    return RecommendationRequest(
        resume_text="Experienced Python Developer",
        jobs=[
            JobDocument(title="SWE 1", company="Acme Corp", description="Coding 1", apply_url="https://acme.corp/1"),
            JobDocument(title="SWE 2", company="Beta Corp", description="Coding 2", apply_url="https://beta.corp/2"),
        ]
    )


@pytest.mark.asyncio
async def test_recommendation_orchestration_explanations_enabled(
    mock_embedding_service,
    mock_similarity_service,
    mock_ranking_service,
    mock_explanation_service,
    test_request,
):
    settings = MagicMock(spec=Settings)
    settings.enable_ai_explanations = True

    service = RecommendationService(
        embedding_service=mock_embedding_service,
        similarity_service=mock_similarity_service,
        ranking_service=mock_ranking_service,
        explanation_service=mock_explanation_service,
        settings=settings,
    )

    response = await service.generate_recommendations(test_request)

    assert isinstance(response, RecommendationResponse)
    assert len(response.recommendations) == 2
    assert response.recommendations[0].recommendation_reason == "Excellent match based on skills."

    mock_embedding_service.embed_resume.assert_called_once_with("Experienced Python Developer")
    mock_embedding_service.embed_jobs.assert_called_once_with(test_request.jobs)
    mock_similarity_service.calculate_similarity.assert_called_once_with([0.1, 0.2], [[0.3, 0.4], [0.5, 0.6]])
    mock_ranking_service.rank.assert_called_once_with(test_request.jobs, [0.8, 0.9])
    mock_explanation_service.explain_recommendations.assert_called_once_with("Experienced Python Developer", mock_ranking_service.rank.return_value)


@pytest.mark.asyncio
async def test_recommendation_orchestration_explanations_disabled(
    mock_embedding_service,
    mock_similarity_service,
    mock_ranking_service,
    mock_explanation_service,
    test_request,
):
    settings = MagicMock(spec=Settings)
    settings.enable_ai_explanations = False

    service = RecommendationService(
        embedding_service=mock_embedding_service,
        similarity_service=mock_similarity_service,
        ranking_service=mock_ranking_service,
        explanation_service=mock_explanation_service,
        settings=settings,
    )

    response = await service.generate_recommendations(test_request)

    assert isinstance(response, RecommendationResponse)
    assert len(response.recommendations) == 2
    assert response.recommendations[0].recommendation_reason is None

    mock_embedding_service.embed_resume.assert_called_once_with("Experienced Python Developer")
    mock_embedding_service.embed_jobs.assert_called_once_with(test_request.jobs)
    mock_similarity_service.calculate_similarity.assert_called_once_with([0.1, 0.2], [[0.3, 0.4], [0.5, 0.6]])
    mock_ranking_service.rank.assert_called_once_with(test_request.jobs, [0.8, 0.9])
    mock_explanation_service.explain_recommendations.assert_not_called()
