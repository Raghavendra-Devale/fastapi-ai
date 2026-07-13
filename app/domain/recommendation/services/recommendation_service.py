from fastapi import Depends, HTTPException

from app.domain.recommendation.models.recommendation_request import RecommendationRequest
from app.domain.recommendation.models.recommendation_response import RecommendationResponse, RecommendationItem
from app.domain.resume.candidate_profile import CandidateProfile
from app.application.pipelines.recommendation_pipeline import RecommendationPipeline
from app.application.resume.candidate_profile_retrieval_service import CandidateProfileRetrievalService
from app.core.logging import get_logger

logger = get_logger("recommendation_service")


class RecommendationService:
    """Orchestration service (entry point) for generating job recommendations."""

    def __init__(
        self,
        candidate_retrieval_service: CandidateProfileRetrievalService = Depends(CandidateProfileRetrievalService),
        recommendation_pipeline: RecommendationPipeline = Depends(RecommendationPipeline),
    ):
        """Initialize the orchestration service with the pipeline and retrieval service."""
        self._candidate_retrieval_service = candidate_retrieval_service
        self._recommendation_pipeline = recommendation_pipeline

    async def generate_recommendations(
        self,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        """Load candidate profile from DB, perform matching against all job profiles, and return recommendations.

        Args:
            request (RecommendationRequest): Contains candidate_profile_id.

        Returns:
            RecommendationResponse: Ranked list of recommendation items.
        """
        # 1. Load and map CandidateProfile from database via CandidateProfileRetrievalService
        candidate_profile = self._candidate_retrieval_service.get_candidate_profile(
            request.candidate_profile_id
        )

        # 3. Execute the recommendation pipeline (loads jobs from DB dynamically)
        results = await self._recommendation_pipeline.run(
            candidate_profile=candidate_profile,
        )

        # 4. Map to RecommendationResponse items
        recommendations = []
        for res in results:
            recommendations.append(
                RecommendationItem(
                    title=res.job_profile.title,
                    company=res.job_profile.company,
                    location=res.job_profile.location,
                    description=res.job_profile.summary or "",
                    employment_type=res.job_profile.employment_type,
                    apply_url=res.job_profile.apply_url or "",
                    similarity_score=res.final_score,  # Expose final score as similarity_score for contract compatibility
                    recommendation_reason=res.recommendation_reason,
                )
            )

        return RecommendationResponse(recommendations=recommendations)
