# FastAPI AI Platform

A backend service built with FastAPI, integrated with AI features including embeddings, vector search (using pgvector), recommendations, and task workers.

## Project Structure

```
fastapi-ai/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       └── router.py
│   ├── core/           # Configuration, security, database sessions
│   ├── providers/      # Third-party integrations (e.g., LLM APIs, email)
│   ├── embeddings/     # Custom embedding generation services
│   ├── vectorstores/   # Vector database setup and queries (pgvector)
│   ├── recommendations/# ML/rules-based recommendation engines
│   ├── services/       # Core business logic
│   ├── schemas/        # Pydantic schemas (request/response validation)
│   ├── models/         # SQLAlchemy/database models
│   ├── workers/        # Background/Celery task workers
│   ├── prompts/        # Prompt templates for LLM tasks
│   ├── utils/          # Helper/utility scripts
│   └── main.py         # Application entry point
├── tests/              # Pytest test suite
├── .env                # Local environment secrets
├── .env.example        # Reference environment variables
├── Dockerfile          # Container setup
├── docker-compose.yml  # Local orchestration (App + PostgreSQL with pgvector)
├── pyproject.toml      # Poetry/uv dependency specs
└── README.md           # Documentation
```

## Setup & Running

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```

3. **Start with Docker Compose**:
   ```bash
   docker compose up --build
   ```

4. **Access the API**:
   - API Endpoint: http://localhost:8000/
   - API Docs (Swagger UI): http://localhost:8000/docs
