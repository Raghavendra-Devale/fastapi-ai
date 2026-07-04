from app.domain.resume.models import (
    ResumeIntelligence,
    Skill,
    Education,
    Experience,
    Project,
    Certification,
    Language,
    EmbeddingMetadata,
)


def test_resume_intelligence_defaults():
    """Test that ResumeIntelligence defaults collection fields to empty lists."""
    intelligence = ResumeIntelligence(
        extracted_text="John Doe Software Engineer",
        embedding=EmbeddingMetadata(model="all-MiniLM-L6-v2", dimensions=384),
    )

    assert intelligence.extracted_text == "John Doe Software Engineer"
    assert intelligence.summary is None
    assert intelligence.skills == []
    assert intelligence.education == []
    assert intelligence.experience == []
    assert intelligence.certifications == []
    assert intelligence.projects == []
    assert intelligence.languages == []
    assert intelligence.embedding.model == "all-MiniLM-L6-v2"
    assert intelligence.embedding.dimensions == 384


def test_resume_intelligence_serialization():
    """Test that ResumeIntelligence serialization and deserialization works correctly."""
    data = {
        "extracted_text": "John Doe",
        "summary": "Experienced engineer.",
        "skills": [{"name": "Python", "confidence": 0.95}],
        "education": [
            {
                "degree": "B.S. CS",
                "institution": "MIT",
                "start_date": "2018-09",
                "end_date": "2022-06",
            }
        ],
        "experience": [
            {
                "company": "Google",
                "designation": "SWE",
                "start_date": "2022-07",
                "end_date": "Present",
                "responsibilities": ["Coding"],
            }
        ],
        "projects": [
            {
                "name": "AI Engine",
                "description": "FastAPI AI Engine",
                "technologies": ["FastAPI", "Python"],
            }
        ],
        "certifications": [{"name": "AWS Pro", "issuer": "Amazon"}],
        "languages": [{"name": "English", "proficiency": "Native"}],
        "embedding": {"model": "all-MiniLM-L6-v2", "dimensions": 384},
    }

    intelligence = ResumeIntelligence(**data)
    serialized = intelligence.model_dump()

    assert serialized["extracted_text"] == "John Doe"
    assert serialized["summary"] == "Experienced engineer."
    assert serialized["skills"][0]["name"] == "Python"
    assert serialized["skills"][0]["confidence"] == 0.95
    assert serialized["education"][0]["degree"] == "B.S. CS"
    assert serialized["education"][0]["institution"] == "MIT"
    assert serialized["experience"][0]["company"] == "Google"
    assert serialized["experience"][0]["designation"] == "SWE"
    assert serialized["experience"][0]["responsibilities"] == ["Coding"]
    assert serialized["projects"][0]["name"] == "AI Engine"
    assert serialized["projects"][0]["technologies"] == ["FastAPI", "Python"]
    assert serialized["certifications"][0]["name"] == "AWS Pro"
    assert serialized["languages"][0]["name"] == "English"
    assert serialized["embedding"]["model"] == "all-MiniLM-L6-v2"
