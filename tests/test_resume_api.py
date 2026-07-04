import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.application.resume.resume_application_service import ResumeApplicationService
from app.domain.resume.schemas import ResumeProcessResponse


@pytest.fixture
def mock_app_service():
    from app.domain.resume.models import ResumeIntelligence, EmbeddingMetadata
    mock_service = AsyncMock(spec=ResumeApplicationService)
    # Store standard response
    mock_service.process_resume.return_value = ResumeProcessResponse(
        resume_text="Extracted Resume Text",
        embedding=[0.1, 0.2, 0.3],
        embedding_dimensions=3,
        embedding_model="test-model",
        processing_time_ms=12.34,
        summary="A professional summary.",
        intelligence=ResumeIntelligence(
            extracted_text="Extracted Resume Text",
            summary="A professional summary.",
            embedding=EmbeddingMetadata(model="test-model", dimensions=3),
        ),
    )
    return mock_service


def test_process_resume_success(mock_app_service):
    """Test successful resume processing through the API endpoint."""
    app.dependency_overrides[ResumeApplicationService] = lambda: mock_app_service

    pdf_content = b"%PDF-1.4 mock pdf content"
    files = {"file": ("resume.pdf", pdf_content, "application/pdf")}

    with TestClient(app) as client:
        response = client.post("/api/v1/resume/process", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["resume"]["extracted_text"] == "Extracted Resume Text"
        assert data["resume"]["embedding_model"] == "test-model"
        assert data["resume"]["embedding_dimensions"] == 3
        assert data["resume"]["summary"] == "A professional summary."
        assert data["resume"]["skills"] == []
        # Ensure raw embeddings are NOT leaked
        assert "embedding" not in data["resume"]
        assert "embedding" not in data
        
        mock_app_service.process_resume.assert_called_once_with(pdf_content)

    app.dependency_overrides.clear()


def test_process_resume_rejects_non_pdf(mock_app_service):
    """Test that non-PDF files are rejected with a 400 validation error."""
    app.dependency_overrides[ResumeApplicationService] = lambda: mock_app_service

    # Invalid extension
    files = {"file": ("resume.txt", b"some text content", "text/plain")}

    with TestClient(app) as client:
        response = client.post("/api/v1/resume/process", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "Only PDF files are accepted" in data["error"]["message"]

    # Invalid content type but PDF extension
    files_2 = {"file": ("resume.pdf", b"some pdf content", "text/plain")}
    with TestClient(app) as client:
        response = client.post("/api/v1/resume/process", files=files_2)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "Only PDF files are accepted" in data["error"]["message"]

    mock_app_service.process_resume.assert_not_called()
    app.dependency_overrides.clear()


def test_process_resume_rejects_empty_file(mock_app_service):
    """Test that empty files are rejected with a 400 validation error."""
    app.dependency_overrides[ResumeApplicationService] = lambda: mock_app_service

    files = {"file": ("resume.pdf", b"", "application/pdf")}

    with TestClient(app) as client:
        response = client.post("/api/v1/resume/process", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "empty" in data["error"]["message"].lower()

    mock_app_service.process_resume.assert_not_called()
    app.dependency_overrides.clear()


def test_process_resume_rejects_oversized_file(mock_app_service):
    """Test that files exceeding the size limit are rejected with a 400 validation error."""
    app.dependency_overrides[ResumeApplicationService] = lambda: mock_app_service

    # Override Settings to have max_resume_size_bytes = 10
    from app.core.config import get_settings, Settings
    original_settings = get_settings()
    
    test_settings = Settings(
        postgres_url=original_settings.postgres_url,
        redis_url=original_settings.redis_url,
        provider=original_settings.provider,
        ollama_base_url=original_settings.ollama_base_url,
        embedding_model=original_settings.embedding_model,
        llm_model=original_settings.llm_model,
        max_resume_size_bytes=10,
    )
    app.dependency_overrides[get_settings] = lambda: test_settings

    files = {"file": ("resume.pdf", b"%PDF-1.4 extremely long mock pdf content", "application/pdf")}

    with TestClient(app) as client:
        response = client.post("/api/v1/resume/process", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "exceeds" in data["error"]["message"].lower()

    mock_app_service.process_resume.assert_not_called()
    app.dependency_overrides.clear()
