import time
from fastapi import Depends

from app.core.exceptions import AppException, ExternalServiceException, ValidationException
from app.core.logging import get_logger
from app.application.ai.embedding_service import EmbeddingService
from app.domain.resume.services.pdf_extraction_service import PDFExtractionService
from app.domain.resume.services.text_normalization_service import TextNormalizationService
from app.domain.resume.schemas import ResumeProcessResponse
from app.domain.resume.services.resume_summary_service import ResumeSummaryService
from app.domain.resume.models import ResumeIntelligence, EmbeddingMetadata

logger = get_logger(__name__)


class ResumeProcessingService:
    """Orchestration service to process resume PDFs and generate embeddings."""

    def __init__(
        self,
        embedding_service: EmbeddingService = Depends(EmbeddingService),
        pdf_extractor: PDFExtractionService = Depends(PDFExtractionService),
        text_normalizer: TextNormalizationService = Depends(TextNormalizationService),
        summary_service: ResumeSummaryService = Depends(ResumeSummaryService),
    ):
        """Initialize the processing service with its dependent services."""
        self._embedding_service = embedding_service
        self._pdf_extractor = pdf_extractor
        self._text_normalizer = text_normalizer
        self._summary_service = summary_service

    async def process_resume(self, pdf_bytes: bytes) -> ResumeProcessResponse:
        """Execute the full resume processing pipeline.

        Args:
            pdf_bytes (bytes): Raw bytes of the resume PDF.

        Returns:
            ResumeProcessResponse: Strongly typed response including text, embedding, and times.

        Raises:
            ValidationException: If input is invalid, or text extraction/normalization fails.
            ExternalServiceException: If the embedding generation fails.
        """
        total_start = time.perf_counter()

        if not pdf_bytes:
            raise ValidationException("PDF bytes cannot be empty.")

        # 1. PDF Extraction
        try:
            extracted_text = self._pdf_extractor.extract_text(pdf_bytes)
        except AppException:
            raise
        except Exception as e:
            raise ValidationException(f"Failed to extract text from PDF: {str(e)}")

        # 2. Text Normalization
        try:
            normalized_text = self._text_normalizer.normalize_text(extracted_text)
        except AppException:
            raise
        except Exception as e:
            raise ValidationException(f"Failed to normalize text: {str(e)}")

        # 3. Generate Embedding via EmbeddingService
        embed_start = time.perf_counter()
        try:
            embedding_response = await self._embedding_service.generate_embedding(normalized_text)
        except AppException:
            raise
        except Exception as e:
            raise ExternalServiceException(
                message=f"Failed to generate embedding: {str(e)}",
                error_code="EMBEDDING_GENERATION_FAILED",
            )
        embed_duration_ms = (time.perf_counter() - embed_start) * 1000.0

        logger.info(
            event="embedding_generation_completed",
            duration_ms=round(embed_duration_ms, 2),
        )

        # 4. Generate Summary via ResumeSummaryService
        summary_start = time.perf_counter()
        try:
            summary = await self._summary_service.generate_summary(normalized_text)
        except AppException:
            raise
        except Exception as e:
            raise ExternalServiceException(
                message=f"Failed to generate summary: {str(e)}",
                error_code="SUMMARY_GENERATION_FAILED",
            )
        summary_duration_ms = (time.perf_counter() - summary_start) * 1000.0

        logger.info(
            event="summary_generation_completed",
            duration_ms=round(summary_duration_ms, 2),
        )

        # 5. Enrich ResumeIntelligence
        intelligence = ResumeIntelligence(
            extracted_text=normalized_text,
            summary=summary,
            embedding=EmbeddingMetadata(
                model=embedding_response.model,
                dimensions=embedding_response.dimensions,
            ),
        )

        total_duration_ms = (time.perf_counter() - total_start) * 1000.0
        logger.info(
            event="resume_processing_completed",
            total_duration_ms=round(total_duration_ms, 2),
        )

        return ResumeProcessResponse(
            resume_text=normalized_text,
            embedding=embedding_response.embedding,
            embedding_dimensions=embedding_response.dimensions,
            embedding_model=embedding_response.model,
            processing_time_ms=round(total_duration_ms, 2),
            summary=summary,
            intelligence=intelligence,
        )
