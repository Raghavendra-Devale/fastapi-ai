import pytest
from pydantic import ValidationError

from app.domain.recommendation.models.recommendation_response import RecommendationItem, RecommendationResponse


def test_recommendation_response_model_creation_and_serialization():
    # Construct a valid item
    item = RecommendationItem(
        job_id=123,
        similarity_score=0.95,
        matching_skills=["Python", "FastAPI"],
        missing_skills=["Docker"],
        recommendation_reason="Excellent match with Python/FastAPI experience.",
    )

    assert item.job_id == 123
    assert item.similarity_score == 0.95
    assert item.recommendation_reason == "Excellent match with Python/FastAPI experience."

    # Construct response model
    response = RecommendationResponse(recommendations=[item])
    assert len(response.recommendations) == 1
    assert response.recommendations[0].job_id == 123

    # Test serialization to dict
    data = response.model_dump()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["similarity_score"] == 0.95
    assert data["recommendations"][0]["recommendation_reason"] == "Excellent match with Python/FastAPI experience."


def test_recommendation_response_model_validation_failures():
    # Missing required field job_id
    with pytest.raises(ValidationError):
        RecommendationItem(
            similarity_score=0.5,
        )

    # Missing required field similarity_score
    with pytest.raises(ValidationError):
        RecommendationItem(
            job_id=123,
        )

    # Invalid type for similarity_score
    with pytest.raises(ValidationError):
        RecommendationItem(
            job_id=123,
            similarity_score="not-a-float",
        )
