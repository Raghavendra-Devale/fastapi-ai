import time
from fastapi import APIRouter, Depends, File, UploadFile

from app.application.resume.resume_application_service import ResumeApplicationService
from app.core.config import Settings, get_settings
from app.core.exceptions import ValidationException
from app.core.logging import get_logger
from app.domain.resume.schemas import ResumeProcessAPIResponse, ResumeDetails

router = APIRouter()
logger = get_logger("resume_endpoint")


@router.post(
    "/process",
    response_model=ResumeProcessAPIResponse,
    status_code=200,
    summary="Process a resume PDF document",
    description=(
        "Upload a resume PDF to extract its normalized text contents, "
        "generate vector embeddings, and return the metadata (excluding the raw embeddings)."
    ),
    responses={
        200: {
            "description": "Resume processed successfully.",
            "model": ResumeProcessAPIResponse,
        },
        400: {
            "description": "Validation error (empty file, oversized file, or non-PDF).",
        },
        500: {
            "description": "Internal server error.",
        },
        502: {
            "description": "External AI provider service error.",
        },
    },
)
async def process_resume(
    file: UploadFile = File(..., description="The resume PDF file to process."),
    settings: Settings = Depends(get_settings),
    application_service: ResumeApplicationService = Depends(),
) -> ResumeProcessAPIResponse:
    """Upload and process a resume PDF file."""
    start_time = time.perf_counter()

    # 1. Reject non-PDF file extension
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

    # 5. Call Application Service
    domain_response = await application_service.process_resume(pdf_bytes)

    total_duration_ms = (time.perf_counter() - start_time) * 1000.0

    # 6. Logging (filename, size, duration. NEVER log resume text, embeddings, or PII)
    logger.info(
        event="resume_upload_processed",
        filename=file.filename,
        size=len(pdf_bytes),
        duration=round(total_duration_ms, 2),
    )

    # 7. Return Response DTO (excluding raw embeddings)
    return ResumeProcessAPIResponse(
        success=True,
        processing_time_ms=round(total_duration_ms, 2),
        resume=ResumeDetails(
            extracted_text=domain_response.resume_text,
            embedding_model=domain_response.embedding_model,
            embedding_dimensions=domain_response.embedding_dimensions,
            summary=domain_response.summary,
            skills=domain_response.intelligence.skills if domain_response.intelligence else [],
            education=domain_response.intelligence.education if domain_response.intelligence else [],
            experience=domain_response.intelligence.experience if domain_response.intelligence else [],
            certifications=domain_response.intelligence.certifications if domain_response.intelligence else [],
            projects=domain_response.intelligence.projects if domain_response.intelligence else [],
            languages=domain_response.intelligence.languages if domain_response.intelligence else [],
        ),
    )
