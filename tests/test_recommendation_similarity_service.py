import pytest

from app.core.exceptions import ValidationException
from app.domain.recommendation.services.similarity_service import SimilarityService


def test_similarity_identical_vectors():
    service = SimilarityService()
    resume = [1.0, 2.0, -1.0, 0.5]
    jobs = [
        [1.0, 2.0, -1.0, 0.5],
    ]
    scores = service.calculate_similarity(resume, jobs)
    assert len(scores) == 1
    assert pytest.approx(scores[0]) == 1.0


def test_similarity_orthogonal_vectors():
    service = SimilarityService()
    # [1.0, 0.0] and [0.0, 1.0] are orthogonal
    resume = [1.0, 0.0]
    jobs = [
        [0.0, 1.0],
    ]
    scores = service.calculate_similarity(resume, jobs)
    assert len(scores) == 1
    assert pytest.approx(scores[0]) == 0.0


def test_similarity_multiple_job_embeddings():
    service = SimilarityService()
    resume = [1.0, 0.0]
    jobs = [
        [1.0, 0.0],   # Identical: 1.0
        [0.0, 1.0],   # Orthogonal: 0.0
        [-1.0, 0.0],  # Opposite: -1.0
        [1.0, 1.0],   # 45 degrees: 1/sqrt(2) approx 0.7071
    ]
    scores = service.calculate_similarity(resume, jobs)
    assert len(scores) == 4
    assert pytest.approx(scores[0]) == 1.0
    assert pytest.approx(scores[1]) == 0.0
    assert pytest.approx(scores[2]) == -1.0
    assert pytest.approx(scores[3]) == 1.0 / (2.0 ** 0.5)


def test_similarity_dimension_mismatch():
    service = SimilarityService()
    resume = [1.0, 2.0]
    
    # Mismatch between resume and job
    with pytest.raises(ValidationException) as exc_info:
        service.calculate_similarity(resume, [[1.0, 2.0, 3.0]])
    assert "Dimension mismatch" in str(exc_info.value)

    # Mismatch among jobs in the list
    with pytest.raises(ValidationException) as exc_info:
        service.calculate_similarity(resume, [[1.0, 2.0], [1.0]])
    assert "Dimension mismatch" in str(exc_info.value)


def test_similarity_empty_inputs():
    service = SimilarityService()
    
    # Empty resume embedding
    with pytest.raises(ValidationException) as exc_info:
        service.calculate_similarity([], [[1.0, 2.0]])
    assert "Resume embedding cannot be empty" in str(exc_info.value)

    # Empty jobs list
    with pytest.raises(ValidationException) as exc_info:
        service.calculate_similarity([1.0, 2.0], [])
    assert "Job embeddings list cannot be empty" in str(exc_info.value)

    # Empty job embedding in list
    with pytest.raises(ValidationException) as exc_info:
        service.calculate_similarity([1.0, 2.0], [[1.0, 2.0], []])
    assert "cannot be empty" in str(exc_info.value)


def test_similarity_zero_vectors():
    service = SimilarityService()
    
    # Zero vector resume
    with pytest.raises(ValidationException) as exc_info:
        service.calculate_similarity([0.0, 0.0], [[1.0, 2.0]])
    assert "zero vector" in str(exc_info.value).lower()

    # Zero vector job
    with pytest.raises(ValidationException) as exc_info:
        service.calculate_similarity([1.0, 2.0], [[0.0, 0.0]])
    assert "zero vector" in str(exc_info.value).lower()
