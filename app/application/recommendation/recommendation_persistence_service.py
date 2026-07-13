from app.domain.recommendation.recommendation_result import RecommendationResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class RecommendationPersistenceService:
    """Service responsible for persisting recommendations and scores."""

    def __init__(self):
        pass

    async def save_recommendations(
        self,
        recommendations: list[RecommendationResult],
    ) -> bool:
        """Persist ranked job recommendation results.

        Args:
            recommendations (list[RecommendationResult]): Ranked recommendations.

        Returns:
            bool: True if saving was successful.
        """
        # Phase 3: Architecture Only (mock persistence logic)
        for idx, rec in enumerate(recommendations):
            logger.info(
                event="persisting_recommendation",
                rank=idx + 1,
                job_id=rec.job_profile.id,
                similarity_score=rec.semantic_score,
                final_score=rec.final_score,
            )
        return True
