import time
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.core.logging import get_logger
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.domain.ai.providers.dependencies import get_llm_provider

logger = get_logger(__name__)


class ResumeSummaryService:
    """Domain service to generate a concise professional summary from resume text.

    Uses a decoupled LLMProvider interface to execute the prompt, avoiding direct dependency on any model SDK.
    """

    def __init__(
        self,
        llm_provider: LLMProvider = Depends(get_llm_provider),
        settings: Settings = Depends(get_settings),
    ):
        """Initialize the resume summary service with LLM provider and settings."""
        self._llm_provider = llm_provider
        self._settings = settings

    async def generate_summary(self, normalized_text: str) -> str:
        """Generate a concise professional summary from normalized resume text.

        Args:
            normalized_text (str): The normalized resume text.

        Returns:
            str: Generated summary, limited to approximately 150 words.

        Raises:
            ValidationException: If the input text is empty or blank.
            ExternalServiceException: If the AI provider fails.
        """
        if not normalized_text or not normalized_text.strip():
            raise ValidationException("Resume text cannot be empty or blank.")

        prompt = (
            "You are a professional resume writer. Generate a concise professional summary "
            "from the following resume text. The summary should be approximately 150 words or fewer, "
            "and written in the third person. Return only the plain text summary without any markdown formatting, "
            "titles, labels, introduction, or concluding sentences.\n\n"
            f"Resume Text:\n{normalized_text.strip()}"
        )

        start_time = time.perf_counter()
        success = False
        model = self._settings.ollama_model

        try:
            summary = await self._llm_provider.generate(
                prompt=prompt,
                system_prompt="You are a professional resume parser and writer.",
                temperature=0.2,
            )
            success = True
            return summary.strip()
        except ExternalServiceException:
            # Re-raise since it's already an ExternalServiceException
            raise
        except Exception as exc:
            raise ExternalServiceException(
                message=f"Failed to generate summary: {str(exc)}",
                error_code="SUMMARY_GENERATION_FAILED",
            ) from exc
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                event="resume_summary_generation",
                model=model,
                duration_ms=round(duration_ms, 2),
                success=success,
            )
