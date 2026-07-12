from fastapi import Depends
from app.domain.resume.services.pdf_extraction_service import PDFExtractionService


class ResumeExtractorService:
    """Service responsible for extracting text from raw resume PDF bytes."""

    def __init__(self, pdf_extractor: PDFExtractionService = Depends(PDFExtractionService)):
        self._pdf_extractor = pdf_extractor

    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text from the PDF bytes.

        Args:
            pdf_bytes (bytes): The raw PDF document bytes.

        Returns:
            str: The extracted raw text contents.
        """
        return self._pdf_extractor.extract_text(pdf_bytes)
