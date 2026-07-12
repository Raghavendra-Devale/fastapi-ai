from pydantic import BaseModel, Field


class RecommendationReason(BaseModel):
    """Pydantic model representing structured recommendation reasons."""

    bullet_points: list[str] = Field(
        ...,
        description="List of bullet points detailing matching criteria and discrepancies.",
    )
