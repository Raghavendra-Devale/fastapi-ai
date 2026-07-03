import fitz
import pytest
from unittest.mock import AsyncMock

from app.core.exceptions import ExternalServiceException, ValidationException
from app.providers.base import AIProvider
from app.providers.models import EmbeddingResponse
from app.domain.resume.services.pdf_extraction_service import PDFExtractionService
from app.domain.resume.services.text_normalization_service import TextNormalizationService
from app.domain.resume.services.resume_processing_service import ResumeProcessingService


def create_sample_pdf(text: str = "John Doe\nSoftware Engineer\nExperience.") -> bytes:
    """Helper to generate a valid PDF dynamically in-memory using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_pdf_extraction_success():
    """Test successful text extraction from a valid PDF."""
    pdf_bytes = create_sample_pdf("Page 1 text\nPage 2 info.")
    service = PDFExtractionService()
    extracted_text = service.extract_text(pdf_bytes)
    assert "Page 1 text" in extracted_text
    assert "Page 2 info." in extracted_text


def test_pdf_extraction_empty_bytes():
    """Test that empty bytes raise a ValidationException."""
    service = PDFExtractionService()
    with pytest.raises(ValidationException) as exc_info:
        service.extract_text(b"")
    assert "empty" in str(exc_info.value).lower()


def test_pdf_extraction_invalid_bytes():
    """Test that invalid PDF bytes raise a ValidationException."""
    service = PDFExtractionService()
    with pytest.raises(ValidationException) as exc_info:
        service.extract_text(b"invalid pdf data")
    assert "invalid pdf" in str(exc_info.value).lower()


def test_text_normalization():
    """Test text normalization collapses spaces, normalizes endings, and removes duplicate blank lines."""
    service = TextNormalizationService()

    raw_text = (
        "  John   Doe  \r\n"
        "Software     Engineer\r"
        "\n"
        "\n"
        "Experience:\tGoogle  \n"
        "\n"
        "\n"
        "Skills:\n"
        "Python\n"
    )

    normalized = service.normalize_text(raw_text)

    expected = (
        "John Doe\n"
        "Software Engineer\n"
        "\n"
        "Experience: Google\n"
        "\n"
        "Skills:\n"
        "Python"
    )

    assert normalized == expected


def test_text_normalization_empty():
    """Test normalization with empty/None text."""
    service = TextNormalizationService()
    assert service.normalize_text("") == ""
    assert service.normalize_text(None) == ""


@pytest.mark.asyncio
async def test_resume_processing_pipeline_success():
    """Test the complete resume processing orchestrator success path."""
    pdf_bytes = create_sample_pdf("John Doe\nSoftware Engineer")

    mock_ai_provider = AsyncMock(spec=AIProvider)
    mock_ai_provider.generate_embedding.return_value = EmbeddingResponse(
        embedding=[0.1, 0.2, 0.3],
        model="test-embed-model",
        dimensions=3,
    )

    service = ResumeProcessingService(ai_provider=mock_ai_provider)
    response = await service.process_resume(pdf_bytes)

    assert response.resume_text == "John Doe\nSoftware Engineer"
    assert response.embedding == [0.1, 0.2, 0.3]
    assert response.embedding_dimensions == 3
    assert response.embedding_model == "test-embed-model"
    assert response.processing_time_ms > 0

    mock_ai_provider.generate_embedding.assert_called_once_with("John Doe\nSoftware Engineer")


@pytest.mark.asyncio
async def test_resume_processing_invalid_pdf_propagates():
    """Test that PDF extraction validation failures propagate through orchestrator."""
    mock_ai_provider = AsyncMock(spec=AIProvider)
    service = ResumeProcessingService(ai_provider=mock_ai_provider)

    with pytest.raises(ValidationException):
        await service.process_resume(b"corrupted data")


@pytest.mark.asyncio
async def test_resume_processing_provider_error_wrapped():
    """Test that provider errors are wrapped in ExternalServiceException."""
    pdf_bytes = create_sample_pdf("Some Resume Text")

    mock_ai_provider = AsyncMock(spec=AIProvider)
    mock_ai_provider.generate_embedding.side_effect = Exception("Ollama connection timed out")

    service = ResumeProcessingService(ai_provider=mock_ai_provider)

    with pytest.raises(ExternalServiceException) as exc_info:
        await service.process_resume(pdf_bytes)

    assert exc_info.value.error_code == "EMBEDDING_GENERATION_FAILED"
    assert "Failed to generate embedding" in exc_info.value.message
