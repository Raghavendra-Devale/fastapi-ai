import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.application.pipelines.resume_pipeline import ResumePipeline
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.resume.resume_analysis_result import ResumeAnalysisResult


@pytest.fixture
def mock_pipeline():
    mock_pipe = AsyncMock(spec=ResumePipeline)
    mock_pipe.run.return_value = ResumeAnalysisResult(
        extracted_text="Extracted Resume Text",
        embedding=[0.1, 0.2, 0.3],
        suggestions=[],
        candidate_profile=CandidateProfile(
            name="John Doe",
            headline="Software Engineer",
            summary="A professional summary.",
            experience_years=5.0,
            skills=[],
            projects=[],
            education=[],
            certifications=[],
            languages=[],
        )
    )
    return mock_pipe


def test_process_resume_success(mock_pipeline):
    """Test successful resume processing through the API endpoint."""
    app.dependency_overrides[ResumePipeline] = lambda: mock_pipeline

    pdf_content = b"%PDF-1.4 mock pdf content"
    files = {"file": ("resume.pdf", pdf_content, "application/pdf")}

    with TestClient(app) as client:
        response = client.post("/api/v1/resume/process", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["extracted_text"] == "Extracted Resume Text"
        assert data["summary"] == "A professional summary."
        assert data["skills"] == []
        assert data["education"] == []
        assert data["experience"] == []
        
        # Ensure no embedding metadata or implementation details are returned
        assert "embedding" not in data
        assert "embedding_model" not in data
        assert "embedding_dimensions" not in data
        assert "success" not in data
        assert "processing_time_ms" not in data
        
        mock_pipeline.run.assert_called_once_with(pdf_content)

    app.dependency_overrides.clear()


def test_process_resume_rejects_non_pdf(mock_pipeline):
    """Test that non-PDF files are rejected with a 400 validation error."""
    app.dependency_overrides[ResumePipeline] = lambda: mock_pipeline

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

    mock_pipeline.run.assert_not_called()
    app.dependency_overrides.clear()


def test_process_resume_rejects_empty_file(mock_pipeline):
    """Test that empty files are rejected with a 400 validation error."""
    app.dependency_overrides[ResumePipeline] = lambda: mock_pipeline

    files = {"file": ("resume.pdf", b"", "application/pdf")}

    with TestClient(app) as client:
        response = client.post("/api/v1/resume/process", files=files)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "empty" in data["error"]["message"].lower()

    mock_pipeline.run.assert_not_called()
    app.dependency_overrides.clear()


def test_process_resume_rejects_oversized_file(mock_pipeline):
    """Test that files exceeding the size limit are rejected with a 400 validation error."""
    app.dependency_overrides[ResumePipeline] = lambda: mock_pipeline

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

    mock_pipeline.run.assert_not_called()
    app.dependency_overrides.clear()
