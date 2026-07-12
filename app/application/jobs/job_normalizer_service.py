from fastapi import Depends
from app.domain.jobs.models.raw_job import RawJob
from app.domain.jobs.services.job_normalization_service import JobNormalizationService


class JobNormalizerService:
    """Service responsible for cleaning, standardizing and normalizing raw job postings from multiple sources."""

    def __init__(self, domain_normalizer: JobNormalizationService = Depends(JobNormalizationService)):
        self._domain_normalizer = domain_normalizer

    def normalize_job(self, raw_job: RawJob) -> str:
        """Normalize a raw job to a standardized text representation.

        Args:
            raw_job (RawJob): Raw scraped job posting.

        Returns:
            str: Standardized clean text representation of the job.
        """
        title = self._domain_normalizer.normalize_whitespace(raw_job.title) or "Untitled Job"
        company = self._domain_normalizer.normalize_whitespace(raw_job.company) or "Unknown Company"
        
        # Normalize details
        location, _ = self._domain_normalizer.normalize_location(raw_job.location)
        emp_type = self._domain_normalizer.standardize_employment_type(raw_job.employment_type)
        salary = self._domain_normalizer.normalize_whitespace(raw_job.salary)
        description = self._domain_normalizer.normalize_whitespace(raw_job.description) or ""

        # Construct a future-proof, provider-agnostic unified text block
        normalized_text = (
            f"Title: {title}\n"
            f"Company: {company}\n"
            f"Location: {location or 'Not Specified'}\n"
            f"Employment Type: {emp_type or 'Not Specified'}\n"
            f"Salary: {salary or 'Not Specified'}\n\n"
            f"Description:\n{description}"
        )
        return normalized_text
