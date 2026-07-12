from fastapi import Depends
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.application.recommendation.recommendation_reason_generator import ReasonAnalyzer, get_reason_analyzer


class RecommendationReasonService:
    """Service responsible for generating recommendation reasons for Ranked Jobs."""

    def __init__(
        self,
        reason_analyzer: ReasonAnalyzer = Depends(get_reason_analyzer),
    ):
        """Initialize service with reasoning analyzer dependency."""
        self._reason_analyzer = reason_analyzer

    async def generate_reason(
        self,
        result: RecommendationResult,
        candidate_profile: CandidateProfile,
    ) -> RecommendationResult:
        """Populate the recommendation reason of a Ranked Job Recommendation using ReasonAnalyzer.

        Args:
            result (RecommendationResult): Ranked recommendation.
            candidate_profile (CandidateProfile): Candidate profile context.

        Returns:
            RecommendationResult: Enriched recommendation.
        """
        # Call the reasoning analyzer to extract bullet points
        reason_data = await self._reason_analyzer.analyze(
            result=result,
            candidate_profile=candidate_profile,
            job_profile=result.job_profile,
        )

        # Format list of bullet points as structured text block
        formatted_bullets = "\n".join(f"• {bp}" for bp in reason_data.bullet_points)
        result.recommendation_reason = formatted_bullets
        return result
