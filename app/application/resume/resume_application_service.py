from fastapi import Depends
from app.domain.resume.services.resume_processing_service import ResumeProcessingService
from app.domain.resume.schemas import ResumeProcessResponse


class ResumeApplicationService:
    """Application Service orchestrating the Resume processing pipeline and domain actions.

    This service is the primary entry point for all resume-related actions.
    Future features like skill extraction, summary generation, experience/education extraction,
    async processing, and event publishing can be orchestrated here.
    """

    def __init__(
        self,
        resume_processing_service: ResumeProcessingService = Depends(ResumeProcessingService),
    ):
        """Initialize the application service with the required domain services."""
        self._resume_processing_service = resume_processing_service

    async def process_resume(self, pdf_bytes: bytes) -> ResumeProcessResponse:
        """Receive PDF bytes, call domain resume processing, and return result.

        Args:
            pdf_bytes (bytes): Raw bytes of the resume PDF.

        Returns:
            ResumeProcessResponse: Strongly typed response including text, embedding, and times.
        """
        # Orchestrate the resume processing pipeline
        return await self._resume_processing_service.process_resume(pdf_bytes)
