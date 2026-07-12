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
            "You are an expert job parsing system. Analyze the following job description and extract "
            "the structured job details.\n\n"
            "Job Description:\n{job_description}\n\n"
            "Format the extracted information into a single valid JSON object containing exactly these fields:\n"
            "- title (string): The job title\n"
            "- company (string): The hiring company name\n"
            "- summary (string or null): A brief summary of the job description\n"
            "- required_skills (array of strings): List of required skill names\n"
            "- preferred_skills (array of strings): List of preferred skill names\n"
            "- experience (string or null): Detailed professional experience requirements\n"
            "- education (string or null): Education or degree requirements\n"
            "- employment_type (string or null): Employment type (e.g., Full-time, Contract, Part-time)\n"
            "- location (string or null): Job location\n"
            "- salary (string or null): Salary or compensation package information\n"
            "- industry (string or null): The industry field of the job/company\n"
            "- responsibilities (array of strings): List of job responsibilities and duties\n"
            "- requirements (array of strings): List of core requirements for the candidate\n"
            "- benefits (array of strings): List of benefits offered by the company\n"
            "- technologies (array of strings): List of tools, languages, and technologies used\n"
            "- keywords (array of strings): Keywords/tags extracted for categorization\n\n"
            "Return ONLY raw valid JSON text matching the schema. Do not include markdown code block blocks (such as ```json), "
            "no preamble, no conversational prose, and no explanations."
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

        Expected formatting keys: {candidate_profile}, {job_profile}, {scores}
        """
        return (
            "You are an expert recruitment assistant explaining why a job is recommended for a candidate.\n\n"
            "Candidate Profile:\n{candidate_profile}\n\n"
            "Job Profile:\n{job_profile}\n\n"
            "Deterministic Match Scores:\n{scores}\n\n"
            "Generate a professional, structured explanation of the match. Focus on the factual alignment "
            "and discrepancies from the profile and match scores (e.g. matched skills, missing skills, experience comparison).\n\n"
            "Return a valid JSON object containing exactly this key:\n"
            "- bullet_points (array of strings): A list of 3-5 concise, actionable bullet points explaining the match, such as "
            "'Strong Java & Spring Boot alignment', 'PostgreSQL experience matches', 'Missing Kafka experience', 'Experience slightly below preferred'.\n\n"
            "Return ONLY raw valid JSON text. Do not include markdown code block syntax (like ```json), no preamble, and no explanations."
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
