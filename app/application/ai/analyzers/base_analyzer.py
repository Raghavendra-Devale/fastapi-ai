from typing import Generic, Type, TypeVar
from fastapi import Depends
from pydantic import BaseModel

from app.domain.ai.providers.interfaces.llm_provider import LLMProvider
from app.domain.ai.providers.dependencies import get_llm_provider
from app.application.ai.structured_output_service import StructuredOutputService
from app.domain.ai.exceptions import AIProviderException
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAIAnalyzer(Generic[T]):
    """Generic base class for all LLM-powered structured output analyzers."""

    def __init__(
        self,
        llm_provider: LLMProvider = Depends(get_llm_provider),
        structured_output: StructuredOutputService = Depends(StructuredOutputService),
    ):
        self._llm_provider = llm_provider
        self._structured_output = structured_output

    async def _analyze_structured(
        self,
        prompt: str,
        system_prompt: str,
        output_model: Type[T],
        temperature: float = 0.0,
    ) -> T:
        """Invoke the LLM provider, validate response structure, and parse into output Pydantic model.

        Args:
            prompt (str): User prompt instructions.
            system_prompt (str): System instruction prompt.
            output_model (Type[T]): Target Pydantic model class to validate and instantiate.
            temperature (float): Model temperature. Defaults to 0.0.

        Returns:
            T: Loaded and validated Pydantic model instance.

        Raises:
            AIProviderException: If the external LLM call fails.
            AIParsingException: If JSON extraction fails.
            AIValidationException: If Pydantic model validation fails.
        """
        # 1. Generate text from LLM provider
        try:
            raw_response = await self._llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(
                event="llm_generation_call_failed",
                error=str(e),
            )
            raise AIProviderException(f"LLM provider generation call failed: {str(e)}") from e

        # 2. Extract and parse structure (AIParsingException/AIValidationException will bubble up)
        return self._structured_output.parse(raw_response, output_model)
