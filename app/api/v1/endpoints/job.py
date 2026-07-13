from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.application.pipelines.job_pipeline import JobPipeline
from app.domain.jobs.models.raw_job import RawJob
from app.core.exceptions import ValidationException, ExternalServiceException
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("job_endpoint")


class JobProcessRequest(BaseModel):
    """Pydantic schema for job processing requests."""
    job_id: int = Field(..., description="The Spring Boot database job ID.")
    title: str = Field(..., description="Raw title of the job.")
    company: str = Field(..., description="Raw company name.")
    location: str | None = Field(None, description="Raw location string.")
    description: str | None = Field(None, description="Raw description of the job.")
    apply_url: str | None = Field(None, description="URL to apply for the job.")
    salary: str | None = Field(None, description="Raw salary information.")
    employment_type: str | None = Field(None, description="Raw employment type string.")
    source: str | None = Field(None, description="The name of the external job source provider.")


class JobProcessResponse(BaseModel):
    """Pydantic schema for job processing response metadata."""
    job_id: int
    job_profile_id: str
    status: str


class JobController:
    """Controller layer to orchestrate raw job analysis, parsing, and database indexing."""

    def __init__(
        self,
        job_pipeline: JobPipeline = Depends(JobPipeline),
    ):
        """Initialize Controller with pipeline dependency."""
        self._job_pipeline = job_pipeline

    async def process_job(self, request: JobProcessRequest) -> JobProcessResponse:
        """Map request body, execute LLM job analysis, generate vectors, and persist results."""
        raw_job = RawJob(
            title=request.title,
            company=request.company,
            location=request.location,
            description=request.description,
            apply_url=request.apply_url,
            salary=request.salary,
            employment_type=request.employment_type,
            source=request.source,
        )

        try:
            result = await self._job_pipeline.run(raw_job, job_id=request.job_id)
            # Use string representation of job profile UUID
            profile_id = str(result.job_profile.id) if result.job_profile.id else ""
            return JobProcessResponse(
                job_id=request.job_id,
                job_profile_id=profile_id,
                status="COMPLETED",
            )
        except ValidationException as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ExternalServiceException as exc:
            raise HTTPException(status_code=502, detail=f"AI provider connection error: {str(exc)}")
        except Exception as exc:
            logger.exception("Unexpected error during job processing")
            raise HTTPException(status_code=500, detail="Unexpected server error processing job.")


@router.post(
    "/process",
    response_model=JobProcessResponse,
    status_code=200,
    summary="Process and persist a raw job posting",
    description="Analyze, normalize, generate embeddings, and persist a raw job posting.",
)
async def process_job(
    request: JobProcessRequest,
    controller: JobController = Depends(),
) -> JobProcessResponse:
    """REST endpoint matching Spring Boot synchronizer call."""
    return await controller.process_job(request)
