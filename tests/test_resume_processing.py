import fitz
import pytest

from app.core.exceptions import ValidationException
from app.domain.resume.services.pdf_extraction_service import PDFExtractionService
from app.domain.resume.services.text_normalization_service import TextNormalizationService


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
