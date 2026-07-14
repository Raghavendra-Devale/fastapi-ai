from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import ValidationException, ExternalServiceException
from app.core.logging import get_logger
from app.domain.recommendation.models.recommendation_request import RecommendationRequest
from app.domain.recommendation.models.recommendation_response import RecommendationResponse, RecommendationItem
from app.domain.recommendation.services.recommendation_service import RecommendationService
from app.application.jobs.job_profile_retrieval_service import JobProfileRetrievalService
from app.application.resume.candidate_profile_retrieval_service import CandidateProfileRetrievalService
from app.application.recommendation.ranking_service import RankingService
from app.application.recommendation.recommendation_reason_service import RecommendationReasonService

logger = get_logger("recommendation_controller")

router = APIRouter()


@router.post(
    "/generate",
    response_model=RecommendationResponse,
    status_code=200,
    summary="Generate Job Recommendations",
    description="Match a stored candidate profile against all active stored job profiles.",
)
async def generate_recommendations(
    request: RecommendationRequest,
    recommendation_service: RecommendationService = Depends(),
) -> RecommendationResponse:
    """Endpoint to generate job recommendations based on candidate_profile_id.

    Invokes RecommendationService and returns ranked RecommendationResponse.
    Maps downstream domain errors to correct REST responses.
    """
    if not request.candidate_profile_id or not request.candidate_profile_id.strip():
        raise HTTPException(
            status_code=400,
            detail="candidate_profile_id cannot be blank.",
        )

    try:
        response = await recommendation_service.generate_recommendations(request)
        return response
    except ValidationException as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ExternalServiceException as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected AI engine failure: {str(exc)}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in generate_recommendations")
        raise HTTPException(
            status_code=500,
            detail="Unexpected AI engine failure.",
        ) from exc


@router.get(
    "/match/{job_id}",
    response_model=RecommendationItem,
    status_code=200,
    summary="Get Single Job Recommendation Match Details",
    description="Calculate and return detailed matching metrics for a specific job against a candidate profile without full database ranking.",
)
async def get_job_match_details(
    job_id: int,
    candidate_profile_id: str,
    job_retrieval: JobProfileRetrievalService = Depends(JobProfileRetrievalService),
    candidate_retrieval: CandidateProfileRetrievalService = Depends(CandidateProfileRetrievalService),
    ranking_service: RankingService = Depends(RankingService),
    reason_service: RecommendationReasonService = Depends(RecommendationReasonService),
) -> RecommendationItem:
    """Score a single job against the candidate profile and return its match details and explanation reason."""
    if not candidate_profile_id or not candidate_profile_id.strip():
        raise HTTPException(
            status_code=400,
            detail="candidate_profile_id cannot be blank.",
        )

    # 1. Fetch Candidate Profile
    candidate_profile = candidate_retrieval.get_candidate_profile(candidate_profile_id)
    if not candidate_profile:
        raise HTTPException(
            status_code=404,
            detail="Candidate profile not found.",
        )

    # 2. Fetch Job Profile by Spring Boot job_id
    job_profile = job_retrieval.find_by_job_id(job_id)
    if not job_profile:
        raise HTTPException(
            status_code=404,
            detail="Job profile not found.",
        )

    # 3. Calculate scores for the single job
    try:
        results = await ranking_service.rank_jobs(candidate_profile, [job_profile])
        if not results:
            raise HTTPException(
                status_code=500,
                detail="Failed to rank the job profile.",
            )
        rec = results[0]

        # 4. Generate explanation reason
        await reason_service.generate_reason(rec, candidate_profile)

        # 5. Return match DTO
        return RecommendationItem(
            job_id=job_id,
            similarity_score=rec.final_score,
            matching_skills=rec.matched_skills,
            missing_skills=rec.missing_skills,
            recommendation_reason=rec.recommendation_reason,
        )
    except Exception as exc:
        logger.exception("Unexpected error in get_job_match_details")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate match details: {str(exc)}",
        ) from exc
