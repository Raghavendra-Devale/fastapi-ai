from app.domain.jobs.job_profile import JobProfile
from app.domain.recommendation.recommendation_result import RecommendationResult


class FinalScoreService:
    """Service to calculate the weighted final recommendation score."""

    def calculate_final_score(
        self,
        job_profile: JobProfile,
        semantic_score: float,
        skill_score: float,
        experience_score: float,
        location_score: float,
        education_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> RecommendationResult:
        """Apply weights and construct the final RecommendationResult.

        Args:
            job_profile (JobProfile): Job profile.
            semantic_score (float): Cosine similarity score.
            skill_score (float): Skill matching score.
            experience_score (float): Experience match score.
            location_score (float): Location match score.
            education_score (float): Education match score.
            matched_skills (list[str]): List of matched skills.
            missing_skills (list[str]): List of missing required skills.

        Returns:
            RecommendationResult: Final scored recommendation.
        """
        # Final Score = 50% Semantic + 30% Skills + 20% Experience
        final_score = (semantic_score * 0.5) + (skill_score * 0.3) + (experience_score * 0.2)
        final_score = max(0.0, min(1.0, float(final_score)))

        return RecommendationResult(
            job_profile=job_profile,
            semantic_score=semantic_score,
            skill_score=skill_score,
            experience_score=experience_score,
            location_score=location_score,
            education_score=education_score,
            final_score=final_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            recommendation_reason=None,
        )
