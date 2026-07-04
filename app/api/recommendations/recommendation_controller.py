from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import ValidationException, ExternalServiceException
from app.core.logging import get_logger
from app.domain.recommendation.models.recommendation_request import RecommendationRequest
from app.domain.recommendation.models.recommendation_response import RecommendationResponse
from app.domain.recommendation.services.recommendation_service import RecommendationService

logger = get_logger("recommendation_controller")

router = APIRouter()


@router.post(
    "/generate",
    response_model=RecommendationResponse,
    status_code=200,
    summary="Generate Job Recommendations",
    description="Analyze resume text and match it against a list of job documents to produce ranked recommendations.",
)
async def generate_recommendations(
    request: RecommendationRequest,
    recommendation_service: RecommendationService = Depends(),
) -> RecommendationResponse:
    """Endpoint to generate job recommendations based on resume and jobs list.

    Validates that the jobs list is not empty, invokes RecommendationService,
    and returns ranked RecommendationResponse. Maps downstream domain errors
    to correct REST responses.
    """
    if not request.jobs:
        raise HTTPException(
            status_code=400,
            detail="Job list cannot be empty.",
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
    except Exception as exc:
        logger.exception("Unexpected error in generate_recommendations")
        raise HTTPException(
            status_code=500,
            detail="Unexpected AI engine failure.",
        ) from exc
