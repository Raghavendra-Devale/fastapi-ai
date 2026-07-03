from pydantic import BaseModel, Field


class ResumeProcessResponse(BaseModel):
    """Pydantic schema representing the processed resume output."""

    resume_text: str = Field(
        ..., description="The normalized text content extracted from the resume PDF."
    )
    embedding: list[float] = Field(
        ..., description="The generated vector embedding representing the resume content."
    )
    embedding_dimensions: int = Field(
        ..., description="The length (dimensions) of the generated embedding."
    )
    embedding_model: str = Field(
        ..., description="The model identifier used to generate the embedding."
    )
    processing_time_ms: float = Field(
        ..., description="The total pipeline processing time in milliseconds."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "resume_text": "John Doe\nSoftware Engineer\nExperience...",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_dimensions": 3,
                "embedding_model": "all-MiniLM-L6-v2",
                "processing_time_ms": 123.45,
            }
        }
    }
