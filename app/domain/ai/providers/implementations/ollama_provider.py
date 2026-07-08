import httpx
from ollama import AsyncClient

from app.core.config import Settings
from app.core.exceptions import ExternalServiceException, ValidationException
from app.core.logging import get_logger
from app.domain.ai.providers.interfaces.llm_provider import LLMProvider

logger = get_logger("ollama_llm_provider")


class OllamaProvider(LLMProvider):
    """Concrete implementation of LLMProvider utilizing official Ollama AsyncClient."""

    def __init__(self, settings: Settings):
        """Initialize the Ollama AsyncClient using configurations.

        Args:
            settings (Settings): Active application configuration.
        """
        self._settings = settings
        self._client = AsyncClient(
            host=settings.ollama_base_url,
            timeout=settings.ollama_timeout,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate text response using Ollama AsyncClient.

        Args:
            prompt (str): User prompt/instruction.
            system_prompt (str | None): Optional system prompt to instruct model behavior.
            temperature (float): Generation temperature. Defaults to 0.2.

        Returns:
            str: Generated text content response.

        Raises:
            ValidationException: If input prompt is empty.
            ExternalServiceException: If Ollama fails, times out, or returns invalid response.
        """
        if not prompt or not prompt.strip():
            raise ValidationException(
                error_code="INVALID_INPUT",
                message="Prompt cannot be blank.",
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat(
                model=self._settings.ollama_model,
                messages=messages,
                options={"temperature": temperature},
            )

            content = response.get("message", {}).get("content", "")
            if not content:
                raise ExternalServiceException(
                    error_code="INVALID_RESPONSE",
                    message="Ollama returned an empty response content.",
                )

            return content
        except httpx.TimeoutException as exc:
            raise ExternalServiceException(
                error_code="OLLAMA_TIMEOUT",
                message=f"Ollama request timed out: {str(exc)}",
            ) from exc
        except httpx.RequestError as exc:
            raise ExternalServiceException(
                error_code="OLLAMA_UNAVAILABLE",
                message=f"Ollama service is unreachable: {str(exc)}",
            ) from exc
        except ExternalServiceException:
            raise
        except Exception as exc:
            raise ExternalServiceException(
                error_code="LLM_GENERATION_FAILED",
                message=f"Ollama text generation failed: {str(exc)}",
            ) from exc

    async def health(self):
        """Verify the Ollama provider connection health status."""
        from app.domain.ai.providers.models import HealthResponse
        try:
            # Query downloaded models to verify server responsiveness
            await self._client.list()
            return HealthResponse(
                provider="ollama",
                healthy=True,
                message="Ollama service is reachable and healthy.",
            )
        except Exception as exc:
            logger.error(
                event="provider_health_check_failed",
                provider="ollama",
                exc_info=exc,
            )
            return HealthResponse(
                provider="ollama",
                healthy=False,
                message=f"Ollama service is unreachable: {str(exc)}",
            )
