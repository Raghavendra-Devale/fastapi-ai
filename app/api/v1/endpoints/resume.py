import time
from fastapi import APIRouter, Depends, File, UploadFile, Form

from app.application.pipelines.resume_pipeline import ResumePipeline
from app.core.config import Settings, get_settings
from app.core.exceptions import ValidationException
from app.core.logging import get_logger
from app.domain.resume.schemas import ResumeIntelligence as ResumeIntelligenceSchema

router = APIRouter()
logger = get_logger("resume_endpoint")


class ResumeController:
    """Thin controller layer to validate resume upload requests, orchestrate application flow,
    and format responses."""

    def __init__(
        self,
        resume_pipeline: ResumePipeline = Depends(ResumePipeline),
    ):
        """Initialize ResumeController with dependencies."""
        self._resume_pipeline = resume_pipeline

    async def process_resume(
        self,
        file: UploadFile,
        settings: Settings,
        resume_id: int | None = None,
        user_id: int | None = None,
    ) -> ResumeIntelligenceSchema:
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
            # 5. Call ResumePipeline
            logger.info(f"ResumeController: calling pipeline.run with resume_id={resume_id}, user_id={user_id}")
            result = await self._resume_pipeline.run(pdf_bytes, resume_id, user_id)

            # 6. Extract CandidateProfile and map to business schema
            profile = result.candidate_profile
            return ResumeIntelligenceSchema(
                extracted_text=result.extracted_text,
                summary=profile.summary,
                skills=profile.skills,
                education=profile.education,
                experience=profile.experience,
                certifications=profile.certifications,
                projects=profile.projects,
                languages=profile.languages,
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
    resumeId: int | None = Form(None, description="The Spring Boot resume database ID."),
    userId: int | None = Form(None, description="The Spring Boot user database ID."),
    settings: Settings = Depends(get_settings),
    controller: ResumeController = Depends(),
) -> ResumeIntelligenceSchema:
    """Expose the AI capability through the public business endpoint."""
    return await controller.process_resume(file, settings, resume_id=resumeId, user_id=userId)
