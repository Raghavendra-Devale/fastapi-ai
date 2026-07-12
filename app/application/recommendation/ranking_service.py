from fastapi import Depends
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.domain.recommendation.services.similarity_service import SimilarityService
from app.application.resume.resume_embedding_service import ResumeEmbeddingService
from app.application.jobs.job_embedding_service import JobEmbeddingService


class RankingService:
    """Service responsible for ranking JobProfiles based on their similarity to a CandidateProfile."""

    def __init__(
        self,
        resume_embedding_service: ResumeEmbeddingService = Depends(ResumeEmbeddingService),
        job_embedding_service: JobEmbeddingService = Depends(JobEmbeddingService),
        similarity_service: SimilarityService = Depends(SimilarityService),
    ):
        self._resume_embedding_service = resume_embedding_service
        self._job_embedding_service = job_embedding_service
        self._similarity_service = similarity_service

    async def rank_jobs(
        self,
        candidate_profile: CandidateProfile,
        job_profiles: list[JobProfile],
    ) -> list[RecommendationResult]:
        """Rank job profiles against a candidate profile using semantic similarity and skill matching.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.
            job_profiles (list[JobProfile]): List of job profiles.

        Returns:
            list[RecommendationResult]: Ranked recommendations.
        """
        if not job_profiles:
            return []

        # 1. Generate text and embedding for CandidateProfile
        candidate_text = (
            f"Headline: {candidate_profile.headline or ''}\n"
            f"Summary: {candidate_profile.summary or ''}\n"
            f"Skills: {', '.join(s.name for s in candidate_profile.skills)}"
        )
        candidate_emb = await self._resume_embedding_service.generate_embedding(candidate_text)

        # 2. Generate embeddings for all job profiles
        job_embs = []
        for jp in job_profiles:
            emb = await self._job_embedding_service.generate_embedding(jp)
            job_embs.append(emb)

        # 3. Calculate similarity scores using domain similarity service
        scores = self._similarity_service.calculate_similarity(candidate_emb, job_embs)

        # 4. Perform skill matching and construct results
        results = []
        candidate_skills = {s.name.lower() for s in candidate_profile.skills}

        for jp, score in zip(job_profiles, scores):
            # Resolve matched skills
            job_req_skills = jp.required_skills
            job_pref_skills = jp.preferred_skills
            
            matched = []
            missing = []
            
            # Check required skills
            for req in job_req_skills:
                if req.lower() in candidate_skills:
                    matched.append(req)
                else:
                    missing.append(req)
                    
            # Check preferred skills
            for pref in job_pref_skills:
                if pref.lower() in candidate_skills:
                    matched.append(pref)
            
            results.append(
                RecommendationResult(
                    job_profile=jp,
                    similarity_score=score,
                    final_score=score,  # Phase 3 uses similarity score as final score
                    matched_skills=matched,
                    missing_skills=missing,
                    recommendation_reason=None,
                )
            )

        # 5. Sort recommendations by similarity score in descending order
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results
