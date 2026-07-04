import pytest

from app.core.exceptions import ValidationException
from app.domain.recommendation.models.recommendation_request import JobDocument
from app.domain.recommendation.services.ranking_service import RankingService


def test_ranking_order_and_matching_scores():
    service = RankingService()
    
    jobs = [
        JobDocument(
            title="SWE 1",
            company="Acme Corp",
            location="Remote",
            description="Coding 1",
            apply_url="https://acme.corp/1",
        ),
        JobDocument(
            title="SWE 2",
            company="Beta Corp",
            location="Remote",
            description="Coding 2",
            apply_url="https://beta.corp/2",
        ),
        JobDocument(
            title="SWE 3",
            company="Gamma Corp",
            location="Remote",
            description="Coding 3",
            apply_url="https://gamma.corp/3",
        ),
    ]
    
    # Mapped scores where index 1 has highest relevance, index 2 has mid, index 0 has lowest
    scores = [0.45, 0.95, 0.75]
    
    ranked = service.rank(jobs, scores)
    
    # Assert return size
    assert len(ranked) == 3
    
    # Assert ordering (highest score first)
    assert ranked[0].title == "SWE 2"
    assert ranked[0].similarity_score == 0.95
    assert ranked[0].recommendation_reason is None
    
    assert ranked[1].title == "SWE 3"
    assert ranked[1].similarity_score == 0.75
    assert ranked[1].recommendation_reason is None
    
    assert ranked[2].title == "SWE 1"
    assert ranked[2].similarity_score == 0.45
    assert ranked[2].recommendation_reason is None


def test_ranking_mismatched_list_sizes():
    service = RankingService()
    
    job = JobDocument(
        title="SWE",
        company="Acme Corp",
        description="Coding",
        apply_url="https://acme.corp",
    )
    
    # More jobs than scores
    with pytest.raises(ValidationException) as exc_info:
        service.rank([job, job], [0.8])
    assert "Mismatched list sizes" in str(exc_info.value)
    
    # More scores than jobs
    with pytest.raises(ValidationException) as exc_info:
        service.rank([job], [0.8, 0.9])
    assert "Mismatched list sizes" in str(exc_info.value)


def test_ranking_empty_inputs():
    service = RankingService()
    
    job = JobDocument(
        title="SWE",
        company="Acme Corp",
        description="Coding",
        apply_url="https://acme.corp",
    )
    
    # Empty jobs list
    with pytest.raises(ValidationException) as exc_info:
        service.rank([], [0.8])
    assert "Job list cannot be empty" in str(exc_info.value)
    
    # Empty scores list
    with pytest.raises(ValidationException) as exc_info:
        service.rank([job], [])
    assert "Similarity scores list cannot be empty" in str(exc_info.value)
