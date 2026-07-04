import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.domain.recommendation.services.recommendation_service import RecommendationService
from app.domain.recommendation.models.recommendation_response import RecommendationResponse, RecommendationItem
from app.core.exceptions import ExternalServiceException, ValidationException


@pytest.fixture
def mock_recommendation_service():
    mock_service = AsyncMock(spec=RecommendationService)
    # Store standard response
    mock_service.generate_recommendations.return_value = RecommendationResponse(
        recommendations=[
            RecommendationItem(
                title="Software Engineer",
                company="Acme Corp",
                location="Remote",
                description="Coding in Python",
                employment_type="Full-time",
                apply_url="https://acme.corp/apply",
                similarity_score=0.95,
                recommendation_reason=None,
            )
        ]
    )
    return mock_service


def test_generate_recommendations_success(mock_recommendation_service):
    """Test successful job recommendations generation through the API endpoint."""
    app.dependency_overrides[RecommendationService] = lambda: mock_recommendation_service

    payload = {
        "resume_text": "Experienced Python Developer.",
        "jobs": [
            {
                "title": "Software Engineer",
                "company": "Acme Corp",
                "location": "Remote",
                "description": "Coding in Python",
                "employment_type": "Full-time",
                "apply_url": "https://acme.corp/apply",
            }
        ],
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/recommendations/generate", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["title"] == "Software Engineer"
        assert data["recommendations"][0]["similarity_score"] == 0.95

    app.dependency_overrides.clear()


def test_generate_recommendations_invalid_request_schema(mock_recommendation_service):
    """Test that schema validation fails (422) for bad payload format (e.g. missing resume_text)."""
    app.dependency_overrides[RecommendationService] = lambda: mock_recommendation_service

    # Missing resume_text
    payload = {
        "jobs": [
            {
                "title": "Software Engineer",
                "company": "Acme Corp",
                "description": "Coding in Python",
                "apply_url": "https://acme.corp/apply",
            }
        ],
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/recommendations/generate", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "resume_text" in data["error"]["message"]

    app.dependency_overrides.clear()


def test_generate_recommendations_empty_jobs_list(mock_recommendation_service):
    """Test that empty jobs list returns a 400 validation error."""
    app.dependency_overrides[RecommendationService] = lambda: mock_recommendation_service

    # Empty jobs list
    payload = {
        "resume_text": "Experienced Python Developer.",
        "jobs": [],
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/recommendations/generate", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "empty" in data["error"]["message"].lower()

    app.dependency_overrides.clear()


def test_generate_recommendations_service_exception(mock_recommendation_service):
    """Test that downstream service exceptions (e.g. ExternalServiceException) return a 500 error."""
    mock_recommendation_service.generate_recommendations.side_effect = ExternalServiceException(
        message="Embedding service failed",
        error_code="EMBEDDING_FAILED",
    )
    app.dependency_overrides[RecommendationService] = lambda: mock_recommendation_service

    payload = {
        "resume_text": "Experienced Python Developer.",
        "jobs": [
            {
                "title": "Software Engineer",
                "company": "Acme Corp",
                "location": "Remote",
                "description": "Coding in Python",
                "employment_type": "Full-time",
                "apply_url": "https://acme.corp/apply",
            }
        ],
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/recommendations/generate", json=payload)
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "AI engine failure" in data["error"]["message"]

    app.dependency_overrides.clear()
