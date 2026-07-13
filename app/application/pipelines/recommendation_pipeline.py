import time
import asyncio
from fastapi import Depends
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.application.recommendation.candidate_retrieval_service import CandidateRetrievalService
from app.application.recommendation.ranking_service import RankingService
from app.application.recommendation.recommendation_reason_service import RecommendationReasonService
from app.application.recommendation.recommendation_persistence_service import RecommendationPersistenceService
from app.core.logging import get_logger

logger = get_logger("recommendation_pipeline")


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
    ) -> list[RecommendationResult]:
        """Run the recommendation pipeline.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.

        Returns:
            list[RecommendationResult]: Ranked recommendations list.
        """
        total_start = time.perf_counter()

        # 1. Retrieve Jobs (using vector retrieval if embedding is present)
        retrieval_start = time.perf_counter()
        job_profiles = await self._retrieval.retrieve_jobs(candidate_profile)
        retrieval_duration = (time.perf_counter() - retrieval_start) * 1000

        logger.info(
            event="recommendation_retrieval_completed",
            jobs_retrieved_count=len(job_profiles),
            duration_ms=round(retrieval_duration, 2),
        )

        # 2. Ranking
        ranking_start = time.perf_counter()
        ranked_recommendations = await self._ranking.rank_jobs(candidate_profile, job_profiles)
        ranking_duration = (time.perf_counter() - ranking_start) * 1000

        logger.info(
            event="recommendation_ranking_completed",
            duration_ms=round(ranking_duration, 2),
        )

        # 3. Recommendation Reason
        reason_start = time.perf_counter()
        await asyncio.gather(*(
            self._reason_service.generate_reason(rec, candidate_profile)
            for rec in ranked_recommendations
        ))
        reason_duration = (time.perf_counter() - reason_start) * 1000

        logger.info(
            event="recommendation_reasons_completed",
            duration_ms=round(reason_duration, 2),
        )

        # 4. Persistence
        await self._persistence.save_recommendations(ranked_recommendations)

        total_duration = (time.perf_counter() - total_start) * 1000
        logger.info(
            event="recommendation_pipeline_completed",
            duration_ms=round(total_duration, 2),
        )

        return ranked_recommendations
