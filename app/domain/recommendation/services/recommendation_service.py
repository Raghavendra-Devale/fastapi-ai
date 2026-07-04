from fastapi import Depends

from app.core.config import Settings, get_settings
from app.domain.recommendation.models.recommendation_request import RecommendationRequest
from app.domain.recommendation.models.recommendation_response import RecommendationResponse
from app.domain.recommendation.services.embedding_service import EmbeddingService
from app.domain.recommendation.services.similarity_service import SimilarityService
from app.domain.recommendation.services.ranking_service import RankingService
from app.domain.recommendation.services.explanation_service import ExplanationService


class RecommendationService:
    """Orchestration service (entry point) for generating job recommendations."""

    def __init__(
        self,
        embedding_service: EmbeddingService = Depends(EmbeddingService),
        similarity_service: SimilarityService = Depends(SimilarityService),
        ranking_service: RankingService = Depends(RankingService),
        explanation_service: ExplanationService = Depends(ExplanationService),
        settings: Settings = Depends(get_settings),
    ):
        """Initialize the orchestration service with downstream recommendation services."""
        self._embedding_service = embedding_service
        self._similarity_service = similarity_service
        self._ranking_service = ranking_service
        self._explanation_service = explanation_service
        self._settings = settings

    async def generate_recommendations(
        self,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        """Coordinate embedding, similarity scoring, ranking, and explanation pipelines to generate job recommendations.

        Args:
            request (RecommendationRequest): Normalized resume text and job documents.

        Returns:
            RecommendationResponse: Ranked list of recommendation items.
        """
        # 1. Generate resume embedding
        resume_emb = await self._embedding_service.embed_resume(request.resume_text)

        # 2. Generate embeddings for all jobs
        job_embs = await self._embedding_service.embed_jobs(request.jobs)

        # 3. Calculate similarity scores
        scores = self._similarity_service.calculate_similarity(resume_emb, job_embs)

        # 4. Rank recommendations
        ranked_items = self._ranking_service.rank(request.jobs, scores)

        # 5. If explanation generation is enabled, generate explanations for the top recommendations.
        if self._settings.enable_ai_explanations and self._explanation_service:
            ranked_items = await self._explanation_service.explain_recommendations(
                request.resume_text,
                ranked_items,
            )

        # 6. Return RecommendationResponse
        return RecommendationResponse(recommendations=ranked_items)
