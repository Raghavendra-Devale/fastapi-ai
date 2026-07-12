from fastapi import Depends
from app.application.ai.embedding_service import EmbeddingService


class ResumeEmbeddingService:
    """Service responsible for generating vector embeddings for resume text."""

    def __init__(self, embedding_service: EmbeddingService = Depends(EmbeddingService)):
        self._embedding_service = embedding_service

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate a vector embedding representing the resume text.

        Args:
            text (str): Cleaned, normalized resume text.

        Returns:
            list[float]: The generated embedding vector.
        """
        response = await self._embedding_service.generate_embedding(text)
        return response.embedding
