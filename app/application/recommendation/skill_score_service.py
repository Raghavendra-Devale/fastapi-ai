from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile


class SkillScoreService:
    """Service to calculate skill match score, listing matched and missing skills."""

    def calculate_score(
        self,
        candidate_profile: CandidateProfile,
        job_profile: JobProfile,
    ) -> tuple[float, list[str], list[str]]:
        """Calculate the skill match score and return matches and missing lists.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.
            job_profile (JobProfile): Job profile.

        Returns:
            tuple[float, list[str], list[str]]: (skill_score, matched_skills, missing_skills)
        """
        # Resolve candidate skill names lowercased
        candidate_skills = {s.name.lower() for s in candidate_profile.skills}

        job_req_skills = job_profile.required_skills or []
        job_pref_skills = job_profile.preferred_skills or []

        matched_req = []
        missing_req = []
        matched_pref = []

        # Intersect required skills
        for req in job_req_skills:
            if req.lower() in candidate_skills:
                matched_req.append(req)
            else:
                missing_req.append(req)

        # Intersect preferred skills
        for pref in job_pref_skills:
            if pref.lower() in candidate_skills:
                matched_pref.append(pref)

        # If job profile has no skill requirements at all, default to 1.0 match
        if not job_req_skills and not job_pref_skills:
            return 1.0, [], []

        req_score = len(matched_req) / len(job_req_skills) if job_req_skills else 1.0
        pref_score = len(matched_pref) / len(job_pref_skills) if job_pref_skills else 1.0

        # Weighted calculation: 70% required skills, 30% preferred skills
        if job_req_skills and job_pref_skills:
            score = (req_score * 0.7) + (pref_score * 0.3)
        elif job_req_skills:
            score = req_score
        else:
            score = pref_score

        return float(score), matched_req + matched_pref, missing_req
