from pydantic import BaseModel, Field


class ResumeSuggestion(BaseModel):
    """Model representing an individual suggestion or recommendation for improving a resume."""

    category: str = Field(..., description="The category of the suggestion (e.g. Formatting, Content, Skills).")
    severity: str = Field(..., description="The severity of the issue (e.g. High, Medium, Low).")
    message: str = Field(..., description="The description of the suggestion/issue.")
    recommendation: str | None = Field(None, description="Actionable advice on how to address the issue.")
