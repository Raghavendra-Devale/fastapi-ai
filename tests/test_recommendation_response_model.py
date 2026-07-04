import pytest
from pydantic import ValidationError

from app.domain.recommendation.models.recommendation_response import RecommendationItem, RecommendationResponse


def test_recommendation_response_model_creation_and_serialization():
    # Construct a valid item
    item = RecommendationItem(
        title="Software Engineer",
        company="Acme Corp",
        location="Remote",
        description="Coding in Python",
        employment_type="Full-time",
        apply_url="https://acme.corp/apply",
        similarity_score=0.95,
        recommendation_reason="Excellent match with Python/FastAPI experience.",
    )

    assert item.title == "Software Engineer"
    assert item.similarity_score == 0.95
    assert item.recommendation_reason == "Excellent match with Python/FastAPI experience."

    # Construct response model
    response = RecommendationResponse(recommendations=[item])
    assert len(response.recommendations) == 1
    assert response.recommendations[0].company == "Acme Corp"

    # Test serialization to dict
    data = response.model_dump()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["similarity_score"] == 0.95
    assert data["recommendations"][0]["recommendation_reason"] == "Excellent match with Python/FastAPI experience."


def test_recommendation_response_model_validation_failures():
    # Missing required field title
    with pytest.raises(ValidationError):
        RecommendationItem(
            company="Acme Corp",
            description="Coding",
            apply_url="https://acme.corp",
            similarity_score=0.5,
        )

    # Missing required field similarity_score
    with pytest.raises(ValidationError):
        RecommendationItem(
            title="SWE",
            company="Acme Corp",
            description="Coding",
            apply_url="https://acme.corp",
        )

    # Invalid type for similarity_score
    with pytest.raises(ValidationError):
        RecommendationItem(
            title="SWE",
            company="Acme Corp",
            description="Coding",
            apply_url="https://acme.corp",
            similarity_score="not-a-float",
        )
