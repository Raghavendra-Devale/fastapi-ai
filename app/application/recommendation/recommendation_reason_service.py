from app.domain.recommendation.recommendation_result import RecommendationResult


class RecommendationReasonService:
    """Service responsible for generating recommendation reasons for Ranked Jobs."""

    def __init__(self):
        pass

    def generate_reason(self, result: RecommendationResult) -> RecommendationResult:
        """Populate the recommendation reason of a Ranked Job Recommendation.

        Args:
            result (RecommendationResult): Ranked recommendation.

        Returns:
            RecommendationResult: Enriched recommendation.
        """
        # Phase 3: Rule-based placeholder explanation logic (no LLM yet)
        title = result.job_profile.title
        company = result.job_profile.company
        
        # Determine explanation based on skill match or score
        score_percent = int(result.similarity_score * 100)
        
        if score_percent >= 80:
            reason = f"Strong match for {title} at {company} with a semantic similarity score of {score_percent}%."
        elif score_percent >= 50:
            reason = f"Good match for {title} at {company} ({score_percent}% similarity). Consider adding missing skills: {', '.join(result.missing_skills[:3])}."
        else:
            reason = f"Low similarity match ({score_percent}%) for {title} at {company}."
            
        result.recommendation_reason = reason
        return result
