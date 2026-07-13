import sys
import os

# Add current directory to path so we can resolve app module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import SessionLocal
from app.core.models import JobProfileModel

print("Checking JobProfile database insertion...")
db = SessionLocal()
try:
    # 1. Clean up any existing job with id 99999
    existing = db.query(JobProfileModel).filter(JobProfileModel.job_id == 99999).first()
    if existing:
        db.delete(existing)
        db.commit()
        print("Cleaned up existing test job profile.")

    # 2. Insert new JobProfileModel
    test_profile = JobProfileModel(
        job_id=99999,
        provider="RemoteOk",
        title="Test Job Title",
        company="Test Company",
        location="Remote",
        experience="2 years",
        profile_json={"skills": ["Python", "SQL"]},
        embedding=[0.1] * 384,
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3",
        prompt_version="v1",
    )
    db.add(test_profile)
    db.commit()
    db.refresh(test_profile)
    print(f"SUCCESS: JobProfileModel created with UUID: {test_profile.id}")

    # 3. Retrieve and assert
    retrieved = db.query(JobProfileModel).filter(JobProfileModel.job_id == 99999).first()
    assert retrieved is not None
    assert retrieved.title == "Test Job Title"
    print("SUCCESS: Retrieved job profile matches inserted details!")

    # 4. Clean up
    db.delete(retrieved)
    db.commit()
    print("SUCCESS: Cleaned up test job profile successfully.")

except Exception as e:
    print(f"FAILED: Database verification failed: {e}")
    sys.exit(1)
finally:
    db.close()
