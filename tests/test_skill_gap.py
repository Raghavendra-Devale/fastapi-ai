import pytest

from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.resume.models import Skill as CandidateSkill
from app.domain.jobs.job_profile import JobProfile
from app.application.recommendation.skill_gap_service import SkillGapService


def test_skill_gap_service_exact_and_partial_matches():
    service = SkillGapService()

    # Candidate profile with a few skills
    candidate = CandidateProfile(
        name="Charlie",
        skills=[
            CandidateSkill(name="Python", confidence=1.0),
            CandidateSkill(name="FastAPI", confidence=0.8),
            CandidateSkill(name="Docker", confidence=0.9),
        ],
    )

    # Job profile with required and preferred skills
    job = JobProfile(
        title="Python Developer",
        company="FastCorp",
        required_skills=["Python", "Docker", "Kubernetes"],
        preferred_skills=["FastAPI", "AWS"],
    )

    # Calculate gap
    gap = service.calculate_gap(candidate, job)

    # Assertions
    # Required: Python (matched), Docker (matched), Kubernetes (missing)
    # Preferred: FastAPI (matched), AWS (missing)
    # Matched skills (both required & preferred): Python, Docker, FastAPI
    assert "Python" in gap.matched_skills
    assert "Docker" in gap.matched_skills
    assert "FastAPI" in gap.matched_skills
    assert len(gap.matched_skills) == 3

    assert gap.missing_required_skills == ["Kubernetes"]
    assert gap.missing_preferred_skills == ["AWS"]

    # Coverage: matched required / total required = 2 / 3 = 66.67%
    assert gap.coverage_percentage == pytest.approx(66.67)


def test_skill_gap_service_no_job_skills():
    service = SkillGapService()

    candidate = CandidateProfile(
        name="Charlie",
        skills=[CandidateSkill(name="Python", confidence=1.0)],
    )

    # Job profile with zero skill requirements
    job = JobProfile(title="General Manager", company="Anywhere")

    gap = service.calculate_gap(candidate, job)

    assert gap.matched_skills == []
    assert gap.missing_required_skills == []
    assert gap.missing_preferred_skills == []
    assert gap.coverage_percentage == 100.0
