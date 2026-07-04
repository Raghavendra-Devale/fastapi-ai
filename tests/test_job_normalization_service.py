import pytest
from app.domain.jobs.models import JobIntelligence, Skill
from app.domain.jobs.services.job_normalization_service import JobNormalizationService


@pytest.fixture
def service():
    return JobNormalizationService()


def test_normalize_whitespace(service):
    """Test whitespace normalization logic."""
    assert service.normalize_whitespace("  Software   Engineer  ") == "Software Engineer"
    assert service.normalize_whitespace("Line1\nLine2\tTab") == "Line1 Line2 Tab"
    assert service.normalize_whitespace(None) is None
    assert service.normalize_whitespace("") == ""


def test_standardize_employment_type(service):
    """Test standardizing various raw employment types to canonical categories."""
    # Full-time mappings
    assert service.standardize_employment_type("full-time") == "Full-time"
    assert service.standardize_employment_type("Full Time") == "Full-time"
    assert service.standardize_employment_type("FT") == "Full-time"
    assert service.standardize_employment_type("permanent") == "Full-time"

    # Part-time mappings
    assert service.standardize_employment_type("part time") == "Part-time"
    assert service.standardize_employment_type("pt") == "Part-time"

    # Contract mappings
    assert service.standardize_employment_type("contractor") == "Contract"
    assert service.standardize_employment_type("temp") == "Contract"
    assert service.standardize_employment_type("temporary") == "Contract"

    # Internship mappings
    assert service.standardize_employment_type("internship") == "Internship"
    assert service.standardize_employment_type("apprentice") == "Internship"

    # Co-op mappings
    assert service.standardize_employment_type("coop") == "Co-op"
    assert service.standardize_employment_type("co-op") == "Co-op"

    # Fallback mappings
    assert service.standardize_employment_type("freelance") == "Freelance"
    assert service.standardize_employment_type("seasonal") == "Seasonal"
    assert service.standardize_employment_type(None) is None
    assert service.standardize_employment_type("") is None


def test_normalize_location_and_remote_inference(service):
    """Test location normalization and remote work inference."""
    # Remote cases
    loc, remote = service.normalize_location("Remote, US")
    assert loc == "Remote, US"
    assert remote is True

    loc, remote = service.normalize_location("San Francisco (WFH)")
    assert loc == "San Francisco (WFH)"
    assert remote is True

    loc, remote = service.normalize_location("Work From Home")
    assert loc == "Work From Home"
    assert remote is True

    # Non-remote cases
    loc, remote = service.normalize_location("New York, NY")
    assert loc == "New York, NY"
    assert remote is None

    # Empty cases
    loc, remote = service.normalize_location(None)
    assert loc is None
    assert remote is None

    loc, remote = service.normalize_location("   ")
    assert loc is None
    assert remote is None


def test_deduplicate_and_normalize_skills(service):
    """Test that skills are correctly stripped, clamped, and deduplicated."""
    dirty_skills = [
        Skill(name="python", required=False, confidence=0.8),
        Skill(name="Python", required=True, confidence=0.5),
        Skill(name="PYTHON ", required=False, confidence=1.2),  # confidence should clamp to 1.0
        Skill(name="java", required=True, confidence=-0.2),      # confidence should clamp to 0.0
    ]

    normalized = service.deduplicate_and_normalize_skills(dirty_skills)

    # We expect 2 unique skills: "Python" and "Java"
    assert len(normalized) == 2
    
    python_skill = next(s for s in normalized if s.name == "Python")
    assert python_skill.required is True  # True OR False OR False = True
    assert python_skill.confidence == 1.0  # max(0.8, 0.5, 1.0) = 1.0

    java_skill = next(s for s in normalized if s.name == "Java")
    assert java_skill.required is True
    assert java_skill.confidence == 0.0  # clamped to min 0.0


def test_normalize_job_full_pipeline(service):
    """Test the end-to-end normalization of a raw JobIntelligence model."""
    raw_job = JobIntelligence(
        title="  Senior   Software   Engineer  ",
        company="  Google   Inc.  ",
        location="  San Francisco, CA (Remote)  ",
        employment_type="  full-time  ",
        experience_level="  senior  ",
        description="  Build awesome   things.  ",
        responsibilities=[
            "  Design systems.  ",
            "   ",  # should be filtered out
            "Write code.",
        ],
        required_skills=[
            Skill(name="python", required=True, confidence=0.9),
            Skill(name="c++", required=True, confidence=0.5),
        ],
        preferred_skills=[
            Skill(name="Python", required=False, confidence=0.95),  # duplicate of required python
            Skill(name="kubernetes", required=False, confidence=0.8),
        ],
        salary="  $150k - $200k  ",
        remote=None,  # should be inferred as True from location
        source="  linkedin  ",
        apply_url="  https://apply.google.com  "
    )

    normalized = service.normalize_job(raw_job)

    assert normalized.title == "Senior Software Engineer"
    assert normalized.company == "Google Inc."
    assert normalized.location == "San Francisco, CA (Remote)"
    assert normalized.employment_type == "Full-time"
    assert normalized.experience_level == "senior"
    assert normalized.description == "Build awesome things."
    assert normalized.responsibilities == ["Design systems.", "Write code."]
    assert normalized.salary == "$150k - $200k"
    assert normalized.remote is True
    assert normalized.source == "linkedin"
    assert normalized.apply_url == "https://apply.google.com"

    # Verify cross-list deduplication (Python should only be in required, c++ in required, kubernetes in preferred)
    assert len(normalized.required_skills) == 2
    assert len(normalized.preferred_skills) == 1

    py_skill = next(s for s in normalized.required_skills if s.name == "Python")
    assert py_skill.required is True
    assert py_skill.confidence == 0.95  # Max of 0.9 and 0.95

    cpp_skill = next(s for s in normalized.required_skills if s.name == "C++")
    assert cpp_skill.required is True

    k8s_skill = next(s for s in normalized.preferred_skills if s.name == "Kubernetes")
    assert k8s_skill.required is False
    assert k8s_skill.confidence == 0.8
