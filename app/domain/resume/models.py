from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ResumeText:
    """Value object representing the normalized resume text content."""
    content: str


@dataclass(frozen=True)
class ResumeEmbedding:
    """Value object representing a generated vector embedding of a resume."""
    vector: List[float]
    dimensions: int
    model: str


@dataclass(frozen=True)
class ResumeProcessResult:
    """Domain representation of the completed resume processing lifecycle."""
    resume_text: ResumeText
    embedding: ResumeEmbedding
    processing_time_ms: float
