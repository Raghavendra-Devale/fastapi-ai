from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile
from app.domain.recommendation.skill_gap import SkillGap


class SkillGapService:
    """Service to calculate deterministic skill gaps between candidate profiles and job requirements."""

    def calculate_gap(
        self,
        candidate_profile: CandidateProfile,
        job_profile: JobProfile,
    ) -> SkillGap:
        """Calculate the skill gap.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.
            job_profile (JobProfile): Job profile.

        Returns:
            SkillGap: Deterministic skill gap analysis.
        """
        # Lowercase candidate skills for case-insensitive lookup
        candidate_skills = {s.name.lower() for s in candidate_profile.skills}

        job_req_skills = job_profile.required_skills or []
        job_pref_skills = job_profile.preferred_skills or []

        matched_skills = []
        missing_required = []
        missing_preferred = []

        # 1. Evaluate required skills
        for req in job_req_skills:
            if req.lower() in candidate_skills:
                matched_skills.append(req)
            else:
                missing_required.append(req)

        # 2. Evaluate preferred skills
        for pref in job_pref_skills:
            if pref.lower() in candidate_skills:
                matched_skills.append(pref)
            else:
                missing_preferred.append(pref)

        # 3. Calculate coverage percentage of required skills (matched required / total required)
        if not job_req_skills:
            coverage = 100.0
        else:
            matched_req_count = len(job_req_skills) - len(missing_required)
            coverage = (matched_req_count / len(job_req_skills)) * 100.0

        coverage = max(0.0, min(100.0, float(coverage)))

        return SkillGap(
            matched_skills=matched_skills,
            missing_required_skills=missing_required,
            missing_preferred_skills=missing_preferred,
            coverage_percentage=round(coverage, 2),
        )
