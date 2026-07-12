import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.resume.models import Skill as CandidateSkill
from app.domain.jobs.job_profile import JobProfile
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.domain.recommendation.services.similarity_service import SimilarityService
from app.application.resume.resume_embedding_service import ResumeEmbeddingService
from app.application.jobs.job_embedding_service import JobEmbeddingService
from app.application.recommendation.candidate_retrieval_service import CandidateRetrievalService
from app.application.recommendation.ranking_service import RankingService
from app.application.recommendation.recommendation_reason_service import RecommendationReasonService
from app.application.recommendation.recommendation_persistence_service import RecommendationPersistenceService
from app.application.pipelines.recommendation_pipeline import RecommendationPipeline


@pytest.mark.asyncio
async def test_candidate_retrieval_service():
    service = CandidateRetrievalService()
    candidate = CandidateProfile(name="John")
    jobs = await service.retrieve_jobs(candidate)
    
    assert len(jobs) == 2
    assert jobs[0].title == "Python Developer"
    assert jobs[1].title == "React Frontend Developer"


@pytest.mark.asyncio
async def test_ranking_service():
    mock_resume_embed = AsyncMock(spec=ResumeEmbeddingService)
    mock_resume_embed.generate_embedding.return_value = [0.1, 0.2]

    mock_job_embed = AsyncMock(spec=JobEmbeddingService)
    mock_job_embed.generate_embedding.return_value = [0.3, 0.4]

    mock_similarity = MagicMock(spec=SimilarityService)
    mock_similarity.calculate_similarity.return_value = [0.85, 0.95]

    service = RankingService(
        resume_embedding_service=mock_resume_embed,
        job_embedding_service=mock_job_embed,
        similarity_service=mock_similarity,
    )

    candidate = CandidateProfile(
        name="John",
        skills=[CandidateSkill(name="Python", confidence=1.0), CandidateSkill(name="FastAPI", confidence=0.9)],
    )
    jobs = [
        JobProfile(
            id="job1",
            title="Python dev",
            company="C1",
            required_skills=["Python", "Go"],
            preferred_skills=["Docker"],
        ),
        JobProfile(
            id="job2",
            title="FastAPI lead",
            company="C2",
            required_skills=["FastAPI"],
            preferred_skills=["Python"],
        ),
    ]

    results = await service.rank_jobs(candidate, jobs)
    
    # Assert ordering (similarity calculations returned 0.85 and 0.95 respectively, so job2 must be first)
    assert len(results) == 2
    assert results[0].job_profile.id == "job2"
    assert results[0].similarity_score == 0.95
    assert "FastAPI" in results[0].matched_skills
    assert "Python" in results[0].matched_skills
    assert results[0].missing_skills == []

    assert results[1].job_profile.id == "job1"
    assert results[1].similarity_score == 0.85
    assert "Python" in results[1].matched_skills
    assert "Go" in results[1].missing_skills

    mock_resume_embed.generate_embedding.assert_called_once()
    assert mock_job_embed.generate_embedding.call_count == 2
    mock_similarity.calculate_similarity.assert_called_once()


def test_recommendation_reason_service():
    service = RecommendationReasonService()
    profile = JobProfile(title="Manager", company="SuperTech")
    
    # 1. High Score
    res_high = RecommendationResult(
        job_profile=profile,
        similarity_score=0.85,
        final_score=0.85,
        matched_skills=["Python"],
    )
    enriched_high = service.generate_reason(res_high)
    assert "Strong match" in enriched_high.recommendation_reason
    assert "85%" in enriched_high.recommendation_reason

    # 2. Medium Score
    res_med = RecommendationResult(
        job_profile=profile,
        similarity_score=0.60,
        final_score=0.60,
        matched_skills=[],
        missing_skills=["Kubernetes"],
    )
    enriched_med = service.generate_reason(res_med)
    assert "Good match" in enriched_med.recommendation_reason
    assert "Kubernetes" in enriched_med.recommendation_reason


@pytest.mark.asyncio
async def test_recommendation_persistence_service():
    service = RecommendationPersistenceService()
    profile = JobProfile(id="1", title="Developer", company="DevCorp")
    res = RecommendationResult(
        job_profile=profile,
        similarity_score=0.9,
        final_score=0.9,
    )
    success = await service.save_recommendations([res])
    assert success is True


@pytest.mark.asyncio
async def test_recommendation_pipeline_run():
    mock_retrieval = AsyncMock(spec=CandidateRetrievalService)
    mock_retrieval.retrieve_jobs.return_value = [JobProfile(id="1", title="Job 1", company="C1")]
    
    mock_ranking = AsyncMock(spec=RankingService)
    mock_ranking.rank_jobs.return_value = [
        RecommendationResult(
            job_profile=JobProfile(id="1", title="Job 1", company="C1"),
            similarity_score=0.9,
            final_score=0.9,
        )
    ]
    
    mock_reason = MagicMock(spec=RecommendationReasonService)
    mock_reason.generate_reason.side_effect = lambda r: setattr(r, "recommendation_reason", "reason") or r
    
    mock_persistence = AsyncMock(spec=RecommendationPersistenceService)
    mock_persistence.save_recommendations.return_value = True
    
    pipeline = RecommendationPipeline(
        retrieval=mock_retrieval,
        ranking=mock_ranking,
        reason_service=mock_reason,
        persistence=mock_persistence,
    )
    
    candidate = CandidateProfile(name="Bob")
    results = await pipeline.run(candidate)
    
    assert len(results) == 1
    assert results[0].job_profile.title == "Job 1"
    assert results[0].recommendation_reason == "reason"
    
    mock_retrieval.retrieve_jobs.assert_called_once_with(candidate)
    mock_ranking.rank_jobs.assert_called_once_with(candidate, mock_retrieval.retrieve_jobs.return_value)
    mock_reason.generate_reason.assert_called_once()
    mock_persistence.save_recommendations.assert_called_once_with(results)
