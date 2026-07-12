import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.resume.models import Skill as CandidateSkill, Education as CandidateEdu
from app.domain.jobs.job_profile import JobProfile
from app.application.resume.resume_embedding_service import ResumeEmbeddingService
from app.application.jobs.job_embedding_service import JobEmbeddingService
from app.domain.recommendation.services.similarity_service import SimilarityService
from app.application.recommendation.semantic_score_service import SemanticScoreService
from app.application.recommendation.skill_score_service import SkillScoreService
from app.application.recommendation.experience_score_service import ExperienceScoreService
from app.application.recommendation.final_score_service import FinalScoreService


@pytest.mark.asyncio
async def test_semantic_score_service():
    # Arrange
    mock_resume_embed = AsyncMock(spec=ResumeEmbeddingService)
    mock_resume_embed.generate_embedding.return_value = [0.1, 0.2]

    mock_job_embed = AsyncMock(spec=JobEmbeddingService)
    mock_job_embed.generate_embedding.return_value = [0.3, 0.4]

    mock_similarity = MagicMock(spec=SimilarityService)
    mock_similarity.calculate_similarity.return_value = [0.88]

    service = SemanticScoreService(
        resume_embedding_service=mock_resume_embed,
        job_embedding_service=mock_job_embed,
        similarity_service=mock_similarity,
    )

    candidate = CandidateProfile(
        name="Alice",
        skills=[CandidateSkill(name="Python", confidence=1.0)]
    )
    job = JobProfile(title="Developer", company="TechInc")

    # Act
    score = await service.calculate_score(candidate, job)

    # Assert
    assert score == 0.88
    mock_resume_embed.generate_embedding.assert_called_once()
    mock_job_embed.generate_embedding.assert_called_once_with(job)
    mock_similarity.calculate_similarity.assert_called_once()


def test_skill_score_service():
    service = SkillScoreService()

    candidate = CandidateProfile(
        name="Bob",
        skills=[
            CandidateSkill(name="Python", confidence=1.0),
            CandidateSkill(name="FastAPI", confidence=0.8),
        ]
    )

    # 1. Matches required only
    job_req_only = JobProfile(
        title="Job 1",
        company="C1",
        required_skills=["Python", "Go"],
    )
    score1, matched1, missing1 = service.calculate_score(candidate, job_req_only)
    assert score1 == 0.5  # 1/2 required matched
    assert "Python" in matched1
    assert "Go" in missing1

    # 2. Matches required and preferred
    job_both = JobProfile(
        title="Job 2",
        company="C2",
        required_skills=["Python"],
        preferred_skills=["FastAPI", "AWS"],
    )
    score2, matched2, missing2 = service.calculate_score(candidate, job_both)
    # req_score = 1.0 (matched Python)
    # pref_score = 0.5 (matched FastAPI, missed AWS)
    # score = 1.0 * 0.7 + 0.5 * 0.3 = 0.75
    assert score2 == 0.85
    assert "Python" in matched2
    assert "FastAPI" in matched2
    assert missing2 == []

    # 3. Empty requirements
    job_empty = JobProfile(title="Job 3", company="C3")
    score3, matched3, missing3 = service.calculate_score(candidate, job_empty)
    assert score3 == 1.0
    assert matched3 == []
    assert missing3 == []


def test_experience_score_service():
    service = ExperienceScoreService()

    # Regex extraction check
    assert service._extract_years_required("3+ years of experience") == 3.0
    assert service._extract_years_required("5-7 years") == 5.0
    assert service._extract_years_required("No requirements") == 0.0

    # Score matches check
    candidate_exp = CandidateProfile(name="John", experience_years=4.0)

    # Required: 3 years
    job_3 = JobProfile(title="J1", company="C1", experience="3+ years")
    assert service.calculate_score(candidate_exp, job_3) == 1.0

    # Required: 5 years
    job_5 = JobProfile(title="J2", company="C2", experience="5 years")
    assert service.calculate_score(candidate_exp, job_5) == 0.8  # 4/5

    # Required: 0 years
    job_0 = JobProfile(title="J3", company="C3")
    assert service.calculate_score(candidate_exp, job_0) == 1.0


def test_final_score_service():
    service = FinalScoreService()
    job = JobProfile(title="J1", company="C1")

    # Semantic (50%), Skills (30%), Experience (20%)
    # Semantic = 0.8, Skills = 0.6, Experience = 1.0
    # Final = 0.8 * 0.5 + 0.6 * 0.3 + 1.0 * 0.2 = 0.4 + 0.18 + 0.2 = 0.78
    res = service.calculate_final_score(
        job_profile=job,
        semantic_score=0.8,
        skill_score=0.6,
        experience_score=1.0,
        location_score=1.0,
        education_score=1.0,
        matched_skills=["Python"],
        missing_skills=["Go"],
    )

    assert res.final_score == 0.78
    assert res.semantic_score == 0.8
    assert res.skill_score == 0.6
    assert res.experience_score == 1.0
    assert res.location_score == 1.0
    assert res.education_score == 1.0
    assert res.matched_skills == ["Python"]
    assert res.missing_skills == ["Go"]
