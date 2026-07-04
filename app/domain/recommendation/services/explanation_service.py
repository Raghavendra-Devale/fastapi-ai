from app.domain.recommendation.models.recommendation_response import RecommendationItem


class ExplanationService:
    """Domain service responsible for generating AI explanations for recommendations."""

    async def explain_recommendations(
        self,
        resume_text: str,
        items: list[RecommendationItem],
    ) -> list[RecommendationItem]:
        """Generate explanations for the top recommendations.

        Args:
            resume_text (str): The candidate's resume text.
            items (list[RecommendationItem]): Sorted recommendation items.

        Returns:
            list[RecommendationItem]: Recommendation items with updated reasons.
        """
        # Skeleton implementation
        for item in items:
            item.recommendation_reason = "Match based on semantic similarity of resume and job description."
        return items
