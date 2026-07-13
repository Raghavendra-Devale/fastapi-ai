from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Generator
from app.core.config import get_settings

# Retrieve application settings
settings = get_settings()

db_url = settings.postgres_url
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

# Initialize SQLAlchemy engine. pool_pre_ping checks the connection health before using it
engine = create_engine(
    db_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Set up SessionLocal class for local session creation
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Declarative base model class
Base = declarative_base()


def get_db() -> Generator:
    """Dependency injection helper yielding active database sessions.

    Ensures that database sessions are closed correctly after request lifecycle ends.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
