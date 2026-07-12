class PromptManager:
    """Manages raw prompt templates for all AI-related features in the application."""

    @staticmethod
    def resume_analysis() -> str:
        """Returns the prompt template for resume analysis/candidate profile extraction.

        Expected formatting keys: {resume_text}
        """
        return (
            "You are an expert resume parsing system. Analyze the following resume text and extract "
            "the structured candidate profile.\n\n"
            "Resume Text:\n{resume_text}\n\n"
            "Format the extracted information into a single valid JSON object containing exactly these fields:\n"
            "- name (string or null): Candidate's full name\n"
            "- headline (string or null): Professional title or designation\n"
            "- summary (string or null): Summary of the candidate's career/profile\n"
            "- experience_years (number or null): Total estimated years of experience\n"
            "- skills (array of objects): Extracted skills. Each skill object must have 'name' (string) and 'confidence' (number, between 0.0 and 1.0)\n"
            "- projects (array of objects): Key projects. Each project object must have 'name' (string), 'description' (string or null), and 'technologies' (array of strings)\n"
            "- education (array of objects): Educational credentials. Each education object must have 'degree' (string or null), 'institution' (string or null), 'start_date' (string or null), and 'end_date' (string or null)\n"
            "- certifications (array of objects): Certifications. Each certification object must have 'name' (string) and 'issuer' (string or null)\n"
            "- languages (array of objects): Languages. Each language object must have 'name' (string) and 'proficiency' (string or null)\n"
            "- preferred_roles (array of strings): Desired job roles or job titles\n"
            "- domains (array of strings): Industry domains (e.g. Finance, Healthcare, Cloud, Gaming)\n"
            "- strengths (array of strings): Key professional strengths\n"
            "- weaknesses (array of strings): Areas for improvement\n"
            "- experience (array of objects): Job history. Each experience object must have 'company' (string or null), 'designation' (string or null), 'start_date' (string or null), 'end_date' (string or null), and 'responsibilities' (array of strings)\n\n"
            "Return ONLY raw valid JSON text matching the schema. Do not include markdown code block blocks (such as ```json), "
            "no preamble, no conversational prose, and no explanations."
        )

    @staticmethod
    def job_analysis() -> str:
        """Returns the prompt template for job description analysis.

        Expected formatting keys: {job_description}
        """
        return (
            "Analyze the following job description and extract the structured job details. "
            "Return the output as a valid JSON object matching the JobProfile schema.\n\n"
            "Job Description:\n{job_description}"
        )

    @staticmethod
    def resume_suggestions() -> str:
        """Returns the prompt template for generating resume suggestions.

        Expected formatting keys: {candidate_profile}
        """
        return (
            "Generate actionable suggestions to improve the following candidate profile. "
            "Return the output as a valid JSON list of ResumeSuggestion objects.\n\n"
            "Candidate Profile:\n{candidate_profile}"
        )

    @staticmethod
    def recommendation_reason() -> str:
        """Returns the prompt template for generating recommendation explanations.

        Expected formatting keys: {candidate_profile}, {job_profile}
        """
        return (
            "Provide a brief, compelling, and professional reason explaining why the following job "
            "is a good match for the candidate. Return the output as a plain text reason.\n\n"
            "Candidate Profile:\n{candidate_profile}\n\n"
            "Job Profile:\n{job_profile}"
        )

    @staticmethod
    def chat() -> str:
        """Returns the prompt template for chat features.

        Expected formatting keys: {user_profile}, {message}
        """
        return (
            "You are a helpful AI assistant for a job recommendation platform. "
            "Answer the user's questions based on their profile.\n\n"
            "User Profile:\n{user_profile}\n\n"
            "User Message:\n{message}"
        )
