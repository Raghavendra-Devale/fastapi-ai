import time
from fastapi import APIRouter, Depends, File, UploadFile

from app.application.resume.resume_application_service import ResumeApplicationService
from app.core.config import Settings, get_settings
from app.core.exceptions import ValidationException
from app.core.logging import get_logger
from app.domain.resume.schemas import ResumeIntelligence as ResumeIntelligenceSchema

router = APIRouter()
logger = get_logger("resume_endpoint")


class ResumeController:
    """Thin controller layer to validate resume upload requests, orchestrate application flow, and format responses."""

    def __init__(
        self,
        application_service: ResumeApplicationService = Depends(ResumeApplicationService),
    ):
        """Initialize ResumeController with dependencies."""
        self._application_service = application_service

    async def process_resume(self, file: UploadFile, settings: Settings) -> ResumeIntelligenceSchema:
        """Validate, parse, and process the uploaded PDF file to extract ResumeIntelligence."""
        # 1. Accept PDF only by extension
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise ValidationException("Only PDF files are accepted.")

        # 2. Reject non-PDF content type
        if file.content_type != "application/pdf":
            raise ValidationException("Only PDF files are accepted.")

        # 3. Reject empty file or oversized file using file.size if available
        if file.size is not None:
            if file.size == 0:
                raise ValidationException("Uploaded file is empty.")
            if file.size > settings.max_resume_size_bytes:
                raise ValidationException(
                    f"File size exceeds the maximum allowed limit of {settings.max_resume_size_bytes} bytes."
                )

        # 4. Read bytes and double check empty/oversized
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise ValidationException("Uploaded file is empty.")
        if len(pdf_bytes) > settings.max_resume_size_bytes:
            raise ValidationException(
                f"File size exceeds the maximum allowed limit of {settings.max_resume_size_bytes} bytes."
            )

        start_time = time.perf_counter()
        try:
            # 5. Call ResumeApplicationService
            domain_response = await self._application_service.process_resume(pdf_bytes)

            # 6. Extract ResumeIntelligence and map to business schema
            intelligence = domain_response.intelligence
            if not intelligence:
                from app.domain.resume.models import ResumeIntelligence as DomainResumeIntelligence, EmbeddingMetadata
                intelligence = DomainResumeIntelligence(
                    extracted_text=domain_response.resume_text,
                    summary=domain_response.summary,
                    embedding=EmbeddingMetadata(
                        model=domain_response.embedding_model,
                        dimensions=domain_response.embedding_dimensions,
                    ),
                )

            # Map the domain model directly to the business-only response schema
            return ResumeIntelligenceSchema(
                extracted_text=intelligence.extracted_text,
                summary=intelligence.summary,
                skills=intelligence.skills,
                education=intelligence.education,
                experience=intelligence.experience,
                certifications=intelligence.certifications,
                projects=intelligence.projects,
                languages=intelligence.languages,
            )
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            # 7. Log filename, file size, processing duration. (Never log contents, embeddings, or PII)
            logger.info(
                event="resume_processing_completed",
                filename=file.filename,
                size=len(pdf_bytes),
                duration_ms=round(duration_ms, 2),
            )


@router.post(
    "/process",
    response_model=ResumeIntelligenceSchema,
    status_code=200,
    summary="Process a resume PDF document to extract candidate intelligence",
    description=(
        "Upload a resume PDF file to extract normalized candidate information, "
        "including text content, summary, skills, work history, education, and language proficiencies. "
        "Excludes raw vector data, model details, or prompt configurations."
    ),
    responses={
        200: {
            "description": "Resume intelligence metadata extracted and returned successfully.",
            "model": ResumeIntelligenceSchema,
        },
        400: {
            "description": "Validation error (invalid file, empty file, or non-PDF content type).",
        },
        500: {
            "description": "Internal server error during resume parsing or pipeline processing.",
        },
        502: {
            "description": "AI provider service or connection error.",
        },
    },
)
async def process_resume(
    file: UploadFile = File(..., description="The resume PDF file to process."),
    settings: Settings = Depends(get_settings),
    controller: ResumeController = Depends(),
) -> ResumeIntelligenceSchema:
    """Expose the AI capability through the public business endpoint."""
    return await controller.process_resume(file, settings)
