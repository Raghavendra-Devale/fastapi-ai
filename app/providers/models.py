from pydantic import BaseModel


class EmbeddingResponse(BaseModel):
    """Response model for a single text embedding request."""

    embedding: list[float]
    model: str
    dimensions: int


class HealthResponse(BaseModel):
    """Response model for an AI provider connection health check."""

    provider: str
    healthy: bool
    message: str


class ChatResponse(BaseModel):
    """Response model for chat model text generation requests."""

    content: str
    model: str
