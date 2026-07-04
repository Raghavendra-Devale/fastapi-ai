from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface (Abstract Base Class) for generating vector embeddings.

    All embedding generation engines (e.g. SentenceTransformers, OpenAI, Ollama)
    must implement this interface to decouple downstream services from model-specific logic.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate a vector embedding for a single text input.

        Args:
            text (str): The text input to embed.

        Returns:
            list[float]: The generated embedding vector.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a batch of text inputs.

        Args:
            texts (list[str]): List of text inputs to embed.

        Returns:
            list[list[float]]: List of generated embedding vectors.
        """
        pass
