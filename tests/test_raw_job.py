from datetime import datetime
import pytest
from pydantic import ValidationError
from app.domain.jobs.models.raw_job import RawJob


def test_raw_job_valid_payload():
    """Test that a valid RawJob payload passes validation and is correctly parsed."""
    published_time = datetime(2026, 7, 4, 12, 0, 0)
    job = RawJob(
        title="Software Engineer",
        company="Tech Corp",
        location="Remote",
        description="Write beautiful code.",
        employment_type="Full-time",
        apply_url="https://jobs.techcorp.com/apply",
        source="Arbeitnow",
        published_at=published_time,
    )

    assert job.title == "Software Engineer"
    assert job.company == "Tech Corp"
    assert job.location == "Remote"
    assert job.description == "Write beautiful code."
    assert job.employment_type == "Full-time"
    assert job.apply_url == "https://jobs.techcorp.com/apply"
    assert job.source == "Arbeitnow"
    assert job.published_at == published_time


def test_raw_job_missing_required_fields():
    """Test that missing required fields raise a Pydantic ValidationError."""
    # Missing title
    with pytest.raises(ValidationError):
        RawJob(
            company="Tech Corp",
            description="Description",
            apply_url="https://apply.com",
            source="LinkedIn",
        )

    # Missing company
    with pytest.raises(ValidationError):
        RawJob(
            title="Engineer",
            description="Description",
            apply_url="https://apply.com",
            source="LinkedIn",
        )

    # Missing description
    with pytest.raises(ValidationError):
        RawJob(
            title="Engineer",
            company="Tech Corp",
            apply_url="https://apply.com",
            source="LinkedIn",
        )

    # Missing apply_url
    with pytest.raises(ValidationError):
        RawJob(
            title="Engineer",
            company="Tech Corp",
            description="Description",
            source="LinkedIn",
        )

    # Missing source
    with pytest.raises(ValidationError):
        RawJob(
            title="Engineer",
            company="Tech Corp",
            description="Description",
            apply_url="https://apply.com",
        )


def test_raw_job_optional_fields_defaults():
    """Test that optional fields default correctly to None when omitted."""
    job = RawJob(
        title="Software Engineer",
        company="Tech Corp",
        description="Write code.",
        apply_url="https://jobs.techcorp.com/apply",
        source="Arbeitnow",
    )

    assert job.location is None
    assert job.employment_type is None
    assert job.published_at is None


def test_raw_job_published_at_datetime_parsing():
    """Test that Pydantic automatically parses ISO strings into datetime objects."""
    job = RawJob(
        title="Software Engineer",
        company="Tech Corp",
        description="Write code.",
        apply_url="https://jobs.techcorp.com/apply",
        source="Arbeitnow",
        published_at="2026-07-04T12:00:00+00:00",
    )

    assert isinstance(job.published_at, datetime)
    assert job.published_at.year == 2026
    assert job.published_at.month == 7
    assert job.published_at.day == 4
