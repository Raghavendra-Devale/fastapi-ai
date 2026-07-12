from fastapi import Depends
from app.domain.resume.services.text_normalization_service import TextNormalizationService


class ResumeCleanerService:
    """Service responsible for cleaning and normalizing extracted resume text."""

    def __init__(self, text_normalizer: TextNormalizationService = Depends(TextNormalizationService)):
        self._text_normalizer = text_normalizer

    def clean_text(self, extracted_text: str) -> str:
        """Clean and normalize the extracted text content.

        Args:
            extracted_text (str): Raw extracted text content.

        Returns:
            str: Cleaned, normalized text content.
        """
        return self._text_normalizer.normalize_text(extracted_text)
