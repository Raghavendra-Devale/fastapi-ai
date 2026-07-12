from fastapi import Depends
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.application.recommendation.candidate_retrieval_service import CandidateRetrievalService
from app.application.recommendation.ranking_service import RankingService
from app.application.recommendation.recommendation_reason_service import RecommendationReasonService
from app.application.recommendation.recommendation_persistence_service import RecommendationPersistenceService


class RecommendationPipeline:
    """Orchestrates the candidate retrieval, ranking, reason generation, and persistence steps of recommendations."""

    def __init__(
        self,
        retrieval: CandidateRetrievalService = Depends(CandidateRetrievalService),
        ranking: RankingService = Depends(RankingService),
        reason_service: RecommendationReasonService = Depends(RecommendationReasonService),
        persistence: RecommendationPersistenceService = Depends(RecommendationPersistenceService),
    ):
        """Initialize the recommendation pipeline with required sub-services."""
        self._retrieval = retrieval
        self._ranking = ranking
        self._reason_service = reason_service
        self._persistence = persistence

    async def run(
        self,
        candidate_profile: CandidateProfile,
        job_profiles: list[JobProfile] = None,
    ) -> list[RecommendationResult]:
        """Run the recommendation pipeline.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.
            job_profiles (list[JobProfile], optional): Custom list of job profiles to rank.
                If not provided, retrievals will be run to fetch jobs automatically.

        Returns:
            list[RecommendationResult]: Ranked recommendations list.
        """
        # 1. Retrieve Jobs (if not provided explicitly)
        if job_profiles is None:
            job_profiles = await self._retrieval.retrieve_jobs(candidate_profile)

        # 2. Ranking
        ranked_recommendations = await self._ranking.rank_jobs(candidate_profile, job_profiles)

        # 3. Recommendation Reason
        import asyncio
        await asyncio.gather(*(
            self._reason_service.generate_reason(rec, candidate_profile)
            for rec in ranked_recommendations
        ))

        # 4. Persistence
        await self._persistence.save_recommendations(ranked_recommendations)

        return ranked_recommendations
