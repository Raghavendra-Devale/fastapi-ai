from fastapi import Depends
from app.domain.resume.candidate_profile import CandidateProfile
from app.domain.jobs.job_profile import JobProfile
from app.domain.recommendation.recommendation_result import RecommendationResult
from app.application.resume.resume_embedding_service import ResumeEmbeddingService
from app.application.recommendation.semantic_score_service import SemanticScoreService
from app.application.recommendation.skill_score_service import SkillScoreService
from app.application.recommendation.experience_score_service import ExperienceScoreService
from app.application.recommendation.final_score_service import FinalScoreService


class RankingService:
    """Service responsible for ranking JobProfiles based on multi-dimensional scoring."""

    def __init__(
        self,
        resume_embedding_service: ResumeEmbeddingService = Depends(ResumeEmbeddingService),
        semantic_service: SemanticScoreService = Depends(SemanticScoreService),
        skill_service: SkillScoreService = Depends(SkillScoreService),
        experience_service: ExperienceScoreService = Depends(ExperienceScoreService),
        final_service: FinalScoreService = Depends(FinalScoreService),
    ):
        self._resume_embedding_service = resume_embedding_service
        self._semantic_service = semantic_service
        self._skill_service = skill_service
        self._experience_service = experience_service
        self._final_service = final_service

    async def rank_jobs(
        self,
        candidate_profile: CandidateProfile,
        job_profiles: list[JobProfile],
    ) -> list[RecommendationResult]:
        """Rank job profiles against a candidate profile using multi-dimensional matching.

        Args:
            candidate_profile (CandidateProfile): Candidate profile.
            job_profiles (list[JobProfile]): List of job profiles.

        Returns:
            list[RecommendationResult]: Ranked recommendations.
        """
        if not job_profiles:
            return []

        # 1. Precalculate candidate embedding once to avoid redundant computations
        candidate_text = (
            f"Headline: {candidate_profile.headline or ''}\n"
            f"Summary: {candidate_profile.summary or ''}\n"
            f"Skills: {', '.join(s.name for s in candidate_profile.skills)}"
        )
        candidate_embedding = await self._resume_embedding_service.generate_embedding(candidate_text)

        results = []
        for jp in job_profiles:
            # 2. Semantic matching
            semantic = await self._semantic_service.calculate_score(
                candidate_profile,
                jp,
                candidate_embedding=candidate_embedding,
            )

            # 3. Skill matching
            skill_score, matched_skills, missing_skills = self._skill_service.calculate_score(
                candidate_profile,
                jp,
            )

            # 4. Experience matching
            experience = self._experience_service.calculate_score(
                candidate_profile,
                jp,
            )

            # 5. Location matching heuristic (Default to 1.0, remote-friendly or local fallback)
            job_loc = (jp.location or "").lower()
            location = 1.0 if not job_loc or "remote" in job_loc or "anywhere" in job_loc else 1.0

            # 6. Education matching heuristic
            education = 1.0
            job_edu = (jp.education or "").lower()
            if job_edu and candidate_profile.education:
                levels = {
                    "phd": ["phd", "ph.d", "doctorate", "doctor"],
                    "master": ["master", "m.s", "ms", "m.tech", "mba"],
                    "bachelor": ["bachelor", "b.s", "bs", "b.tech", "ba"],
                }
                # Determine required level
                required_level = None
                for level_name, keywords in levels.items():
                    if any(kw in job_edu for kw in keywords):
                        required_level = level_name
                        break

                # Extract candidate max level
                candidate_degrees = [str(edu.degree).lower() for edu in candidate_profile.education if edu.degree]
                if required_level and candidate_degrees:
                    candidate_max_level = None
                    for level_name, keywords in levels.items():
                        if any(any(kw in deg for kw in keywords) for deg in candidate_degrees):
                            candidate_max_level = level_name
                            break

                    if candidate_max_level:
                        hierarchy = ["bachelor", "master", "phd"]
                        try:
                            req_idx = hierarchy.index(required_level)
                            cand_idx = hierarchy.index(candidate_max_level)
                            education = 1.0 if cand_idx >= req_idx else 0.5
                        except ValueError:
                            education = 0.7
                    else:
                        education = 0.7
                elif required_level:
                    education = 0.5

            # 7. Compute weighted final score and construct result object
            res = self._final_service.calculate_final_score(
                job_profile=jp,
                semantic_score=semantic,
                skill_score=skill_score,
                experience_score=experience,
                location_score=location,
                education_score=education,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
            )
            results.append(res)

        # 8. Sort recommendations by final score descending
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results
