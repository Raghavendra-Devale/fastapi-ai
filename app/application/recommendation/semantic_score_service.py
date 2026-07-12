from fastapi import Depends
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile
from app.domain.recommendation.services.similarity_service import SimilarityService
from app.application.resume.resume_embedding_service import ResumeEmbeddingService
from app.application.jobs.job_embedding_service import JobEmbeddingService


class SemanticScoreService:
    """Service to calculate the semantic similarity score between candidate and job."""

    def __init__(
        self,
        resume_embedding_service: ResumeEmbeddingService = Depends(ResumeEmbeddingService),
        job_embedding_service: JobEmbeddingService = Depends(JobEmbeddingService),
        similarity_service: SimilarityService = Depends(SimilarityService),
    ):
        self._resume_embedding_service = resume_embedding_service
        self._job_embedding_service = job_embedding_service
        self._similarity_service = similarity_service

    async def calculate_score(
        self,
        candidate_profile: CandidateProfile,
        job_profile: JobProfile,
        candidate_embedding: list[float] | None = None,
    ) -> float:
        """Calculate the semantic cosine similarity score.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.
            job_profile (JobProfile): Job profile.
            candidate_embedding (list[float] | None): Optional precomputed candidate embedding.

        Returns:
            float: Semantic score (0.0 to 1.0).
        """
        # 1. Generate candidate embedding if not precomputed
        if candidate_embedding is None:
            candidate_text = (
                f"Headline: {candidate_profile.headline or ''}\n"
                f"Summary: {candidate_profile.summary or ''}\n"
                f"Skills: {', '.join(s.name for s in candidate_profile.skills)}"
            )
            candidate_embedding = await self._resume_embedding_service.generate_embedding(candidate_text)

        # 2. Generate embedding for JobProfile
        job_emb = await self._job_embedding_service.generate_embedding(job_profile)

        # 3. Calculate similarity using domain similarity service
        scores = self._similarity_service.calculate_similarity(candidate_embedding, [job_emb])
        return max(0.0, min(1.0, float(scores[0])))
