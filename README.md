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

## Configuration

The application is configured using environment variables prefixed with `AI_`. See `.env.example` for details.

### Key Environment Variables
*   `AI_PROVIDER`: The selected AI provider. Currently supports `ollama`.
*   `AI_OLLAMA_BASE_URL`: Base connection URL of the running Ollama API service (e.g. `http://localhost:11434`).
*   `AI_EMBEDDING_MODEL`: Configured model used for generating embeddings.
*   `AI_LLM_MODEL`: Configured LLM model used for chat completions.

### Startup Validation
On application boot, the engine automatically resolves the configured AI provider and runs a startup validation check calling its health check API. If the provider is unreachable, a warning is printed to the structured logs, but the application continues to start up.

### Health Endpoint
A detailed health check is available at:
*   `GET /api/v1/health`

This endpoint returns HTTP 200 health status including nested dependency states even if third-party providers (like Ollama) are offline:
```json
{
  "status": "UP",
  "service": "fastapi-ai",
  "version": "1.0.0",
  "environment": "development",
  "dependencies": {
    "provider": {
      "healthy": true,
      "provider": "ollama",
      "message": "Ollama service is reachable and healthy."
    }
  }
}
```

