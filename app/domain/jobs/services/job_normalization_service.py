import re
import time
from app.core.logging import get_logger
from app.domain.jobs.models import JobIntelligence, Skill

logger = get_logger(__name__)


class JobNormalizationService:
    """Domain service to normalize job postings.

    Responsible for text whitespace cleanup, location normalization, remote inference,
    employment type standardization, and skill list deduplication and merging.
    Does not use AI or LLM models.
    """

    def normalize_whitespace(self, text: str | None) -> str | None:
        """Collapse multiple spaces/tabs into a single space and strip boundaries.

        Args:
            text (str | None): Input text string.

        Returns:
            str | None: Cleaned text, or None if input was None.
        """
        if text is None:
            return None
        # Replace all consecutive whitespace characters with a single space and trim
        return re.sub(r"\s+", " ", text).strip()

    def standardize_employment_type(self, emp_type: str | None) -> str | None:
        """Map raw employment type string to a canonical representation.

        Canonical categories: Full-time, Part-time, Contract, Internship, Co-op.

        Args:
            emp_type (str | None): Raw employment type string.

        Returns:
            str | None: Canonical employment type, or None if input was None.
        """
        if not emp_type:
            return None

        # Clean string for comparison
        normalized = emp_type.strip().lower()
        cleaned = re.sub(r"[\s\-_/]+", " ", normalized).strip()

        if cleaned in ("full time", "fulltime", "ft", "permanent", "full_time"):
            return "Full-time"
        if cleaned in ("part time", "parttime", "pt", "part_time"):
            return "Part-time"
        if cleaned in ("contract", "contractor", "contractual", "temp", "temporary"):
            return "Contract"
        if cleaned in ("intern", "internship", "apprentice", "apprenticeship"):
            return "Internship"
        if cleaned in ("co op", "coop", "co-op"):
            return "Co-op"

        # Fallback: title-case the trimmed original input
        return emp_type.strip().title()

    def normalize_location(self, location: str | None) -> tuple[str | None, bool | None]:
        """Normalize job location and infer remote work status from keywords.

        Args:
            location (str | None): Raw job location string.

        Returns:
            tuple[str | None, bool | None]: (normalized_location, inferred_remote)
        """
        if not location:
            return None, None

        normalized = self.normalize_whitespace(location)
        if not normalized:
            return None, None

        loc_lower = normalized.lower()
        inferred_remote = None

        # Check for remote keywords
        remote_keywords = ("remote", "wfh", "work from home", "telecommute", "anywhere")
        if any(kw in loc_lower for kw in remote_keywords):
            inferred_remote = True

        return normalized, inferred_remote

    def deduplicate_and_normalize_skills(self, skills: list[Skill]) -> list[Skill]:
        """Deduplicate and normalize a list of skills.

        Clamps confidence scores between 0.0 and 1.0, strips skill names,
        and merges case-insensitive duplicate skills.
        Duplicates are merged by keeping the maximum confidence score and
        marking the merged skill required if any duplicate was required.

        Args:
            skills (list[Skill]): List of Skill objects.

        Returns:
            list[Skill]: Deduplicated list of Skill objects.
        """
        seen_skills: dict[str, Skill] = {}

        for skill in skills:
            if not skill or not skill.name:
                continue

            normalized_name = self.normalize_whitespace(skill.name)
            if not normalized_name:
                continue

            # Clamp confidence to [0.0, 1.0]
            clamped_confidence = max(0.0, min(1.0, float(skill.confidence)))

            key = normalized_name.lower()
            if key in seen_skills:
                # Merge duplicate properties
                existing = seen_skills[key]
                existing.confidence = max(existing.confidence, clamped_confidence)
                existing.required = existing.required or skill.required
            else:
                seen_skills[key] = Skill(
                    name=normalized_name,
                    required=skill.required,
                    confidence=clamped_confidence,
                )

        return list(seen_skills.values())

    def normalize_job(self, job_data: JobIntelligence) -> JobIntelligence:
        """Run the complete normalization pipeline on JobIntelligence.

        Args:
            job_data (JobIntelligence): Input raw job intelligence data.

        Returns:
            JobIntelligence: Fully formatted and normalized job intelligence.
        """
        start_time = time.perf_counter()

        # 1. Normalize basic string fields
        title = self.normalize_whitespace(job_data.title) or "Untitled Job"
        company = self.normalize_whitespace(job_data.company) or "Unknown Company"
        description = self.normalize_whitespace(job_data.description)
        salary = self.normalize_whitespace(job_data.salary)
        source = self.normalize_whitespace(job_data.source)
        apply_url = self.normalize_whitespace(job_data.apply_url)
        experience_level = self.normalize_whitespace(job_data.experience_level)

        # 2. Standardize employment type
        employment_type = self.standardize_employment_type(job_data.employment_type)

        # 3. Normalize location and infer remote
        normalized_loc, inferred_remote = self.normalize_location(job_data.location)
        # Explicit remote takes precedence over inferred remote
        remote = job_data.remote if job_data.remote is not None else inferred_remote

        # 4. Clean lists (responsibilities)
        responsibilities = []
        for resp in job_data.responsibilities:
            cleaned_resp = self.normalize_whitespace(resp)
            if cleaned_resp:
                responsibilities.append(cleaned_resp)

        # 5. Normalize and deduplicate skills inside each list
        normalized_required = self.deduplicate_and_normalize_skills(job_data.required_skills)
        normalized_preferred = self.deduplicate_and_normalize_skills(job_data.preferred_skills)

        # 6. Deduplicate across required and preferred skill lists
        # If a skill exists in both, keep it only in required_skills with merged properties
        required_keys = {s.name.lower(): s for s in normalized_required}
        final_preferred_skills = []

        for preferred_skill in normalized_preferred:
            pref_key = preferred_skill.name.lower()
            if pref_key in required_keys:
                # Merge into the required skill: max confidence, and it must be marked required
                req_skill = required_keys[pref_key]
                req_skill.confidence = max(req_skill.confidence, preferred_skill.confidence)
                req_skill.required = True
            else:
                final_preferred_skills.append(preferred_skill)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            event="job_normalization_completed",
            title=title,
            company=company,
            duration_ms=round(duration_ms, 2),
        )

        return JobIntelligence(
            title=title,
            company=company,
            location=normalized_loc,
            employment_type=employment_type,
            experience_level=experience_level,
            description=description,
            responsibilities=responsibilities,
            required_skills=list(required_keys.values()),
            preferred_skills=final_preferred_skills,
            salary=salary,
            remote=remote,
            source=source,
            apply_url=apply_url,
        )
