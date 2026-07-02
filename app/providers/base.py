from abc import ABC, abstractmethod

from app.providers.models import ChatResponse, EmbeddingResponse, HealthResponse


class AIProvider(ABC):
    """Abstract Base Class (interface) for all AI engine providers.

    All downstream business logic and service dependencies must interact only
    via this interface to maintain provider independence.
    """

    @abstractmethod
    async def health(self) -> HealthResponse:
        """Perform a connection health check to the AI provider.

        Returns:
            HealthResponse: Standardized response indicating provider availability.
        """
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> EmbeddingResponse:
        """Generate a vector embedding for a single text input.

        Args:
            text (str): Input text string.

        Returns:
            EmbeddingResponse: Generated text embedding representation.
        """
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: list[str]) -> list[EmbeddingResponse]:
        """Generate vector embeddings for a list of text inputs.

        Args:
            texts (list[str]): List of input text strings.

        Returns:
            list[EmbeddingResponse]: List of generated text embedding representations.
        """
        pass

    @abstractmethod
    async def chat(self, prompt: str) -> ChatResponse:
        """Submit a prompt context to the chat LLM model.

        Args:
            prompt (str): Text prompt context.

        Returns:
            ChatResponse: Generated text output message.
        """
        pass
