from app.core.exceptions import ValidationException
from app.domain.recommendation.models.recommendation_request import JobDocument
from app.domain.recommendation.models.recommendation_response import RecommendationItem


class RankingService:
    """Domain service to rank recommendation candidates by semantic similarity scores."""

    def rank(
        self,
        jobs: list[JobDocument],
        similarity_scores: list[float],
    ) -> list[RecommendationItem]:
        """Rank job documents based on their similarity scores in descending order.

        Args:
            jobs (list[JobDocument]): The list of JobDocument items.
            similarity_scores (list[float]): Similarity scores mapping 1-to-1 with jobs.

        Returns:
            list[RecommendationItem]: List of ranked RecommendationItem objects.

        Raises:
            ValidationException: If inputs are empty or list lengths do not match.
        """
        if not jobs:
            raise ValidationException("Job list cannot be empty.")
        if not similarity_scores:
            raise ValidationException("Similarity scores list cannot be empty.")

        if len(jobs) != len(similarity_scores):
            raise ValidationException(
                f"Mismatched list sizes: got {len(jobs)} jobs and {len(similarity_scores)} similarity scores."
            )

        items = []
        for job, score in zip(jobs, similarity_scores):
            items.append(
                RecommendationItem(
                    title=job.title,
                    company=job.company,
                    location=job.location,
                    description=job.description,
                    employment_type=job.employment_type,
                    apply_url=job.apply_url,
                    similarity_score=score,
                    recommendation_reason=None,
                )
            )

        # Sort recommendations by similarity score in descending order
        items.sort(key=lambda item: item.similarity_score, reverse=True)
        return items
