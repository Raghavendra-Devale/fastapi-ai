import math
from app.core.exceptions import ValidationException
from app.core.logging import get_logger

logger = get_logger(__name__)


class SimilarityService:
    """Domain service to calculate semantic similarity scores using Cosine Similarity."""

    def calculate_similarity(
        self,
        resume_embedding: list[float],
        job_embeddings: list[list[float]],
    ) -> list[float]:
        """Calculate the cosine similarity between a resume embedding and a list of job embeddings.

        Args:
            resume_embedding (list[float]): Vector embedding of the resume.
            job_embeddings (list[list[float]]): List of vector embeddings of the jobs.

        Returns:
            list[float]: A list of similarity scores in the same order as the input jobs.

        Raises:
            ValidationException: If inputs are empty, contain zero vectors, or have mismatched dimensions.
        """
        if not resume_embedding:
            raise ValidationException("Resume embedding cannot be empty.")

        if not job_embeddings:
            raise ValidationException("Job embeddings list cannot be empty.")

        expected_dim = len(resume_embedding)

        # Calculate resume norm and protect against zero vector
        resume_sum_sq = sum(x * x for x in resume_embedding)
        if resume_sum_sq == 0.0:
            raise ValidationException("Resume embedding cannot be a zero vector.")
        resume_norm = math.sqrt(resume_sum_sq)

        scores = []
        for idx, job_emb in enumerate(job_embeddings):
            if not job_emb:
                raise ValidationException(f"Job embedding at index {idx} cannot be empty.")

            if len(job_emb) != expected_dim:
                raise ValidationException(
                    f"Dimension mismatch at index {idx}. "
                    f"Expected dimension {expected_dim}, but got {len(job_emb)}."
                )

            job_sum_sq = sum(x * x for x in job_emb)
            if job_sum_sq == 0.0:
                raise ValidationException(f"Job embedding at index {idx} cannot be a zero vector.")
            job_norm = math.sqrt(job_sum_sq)

            dot_product = sum(r * j for r, j in zip(resume_embedding, job_emb))
            similarity = dot_product / (resume_norm * job_norm)

            # Clamp the similarity to [-1.0, 1.0] due to floating-point precision bounds
            similarity = max(-1.0, min(1.0, similarity))
            scores.append(similarity)

        return scores
